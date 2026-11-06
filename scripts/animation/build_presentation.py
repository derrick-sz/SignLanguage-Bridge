"""Reproducible Godette derivative and source-bound, baked pilot animations.

Host: python build_presentation.py --archive ZIP --blender EXE --output DIR
Blender subprocess executes this same file with --blender-stage BUILD_INPUT.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]
PILOTS = ['new-1918b4f42eac7d78-cap-0009', 'pilot-greeting-answer-002',
          'pilot-greeting-how-004', 'pilot-greeting-busy-005', 'pilot-hospital-pay-013',
          'new-0ad6eaefc7647487-cap-0067']
TEMPORAL_REPAIRS = {'new-0ad6eaefc7647487-cap-0067':'source-quaternion-temporal-v1'}
PROCESSING_VERSION = 'source-wrist-ik-temporal-candidate-v2'
ARCHIVE_SHA = '4d98380ac0fe08db90b042df63528ae9be5c96b93f492625ec27a015ea5edae5'


def bind_export(path: Path, binding: dict):
    data=path.read_bytes()
    length,kind=struct.unpack_from('<II',data,12)
    if data[:4]!=b'glTF' or kind!=0x4e4f534a:
        raise ValueError('Not a GLB export')
    document=json.loads(data[20:20+length])
    document['asset'].setdefault('extras',{})['slbridge_motion']=dict(binding)
    document['animations'][0]['name']=binding['asset_id']
    payload=json.dumps(document,separators=(',',':'),ensure_ascii=False).encode('utf8')
    payload+=b' '*((-len(payload))%4)
    tail=data[20+length:]
    path.write_bytes(struct.pack('<4sII',b'glTF',2,20+len(payload)+len(tail))+
                     struct.pack('<II',len(payload),0x4e4f534a)+payload+tail)


def blender_stage(input_path):
    import bpy
    import bmesh
    from mathutils import Matrix
    data = json.loads(Path(input_path).read_text(encoding='utf8'))
    output = Path(data['output'])
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=data['model'], merge_vertices=True)
    meshes = [o for o in bpy.data.objects if o.type == 'MESH' and any(m.type == 'ARMATURE' for m in o.modifiers)]
    armature = next(o for o in bpy.data.objects if o.type == 'ARMATURE')
    def material(name, color, roughness):
        m = bpy.data.materials.new(name); m.use_nodes = True
        node = m.node_tree.nodes.get('Principled BSDF')
        node.inputs['Base Color'].default_value = (*color, 1)
        node.inputs['Roughness'].default_value = roughness
        return m
    skin = material('Clear hands', (0.67, 0.43, 0.35), 0.65)
    cloth = material('Plain teal clothing', (0.027, 0.064, 0.078), 0.83)
    removed = 0
    for obj in ([] if data.get('bake_only') else meshes):
        mesh = obj.data
        skin_index = len(mesh.materials); mesh.materials.append(skin)
        cloth_index = len(mesh.materials); mesh.materials.append(cloth)
        # Assign by actual hand/finger skin weights, not texture-color guessing.
        hand_groups = {g.index for g in obj.vertex_groups if any(k in g.name.lower() for k in ('hand.', 'handl', 'handr', 'finger', 'thumb'))}
        hand_vertices = {v.index for v in mesh.vertices if sum(g.weight for g in v.groups if g.group in hand_groups) > 0.35}
        for face in mesh.polygons:
            positions = [obj.matrix_world @ mesh.vertices[v].co for v in face.vertices]
            if all(v in hand_vertices for v in face.vertices) or all(abs(p.x) > 0.48 and 1.20 < p.z < 1.36 for p in positions):
                face.material_index = skin_index
            elif all(0.74 < p.z < 1.40 and abs(p.x) < 0.29 for p in positions):
                face.material_index = cloth_index
        # Remove separately authored backpack/straps and necktie, preserving torso,
        # arms, hands, face, and the original bind/rest skeleton.
        bm = bmesh.new(); bm.from_mesh(mesh)
        seen = set(); drop = []
        for vertex in bm.verts:
            if vertex in seen: continue
            stack = [vertex]; seen.add(vertex); island = []
            while stack:
                current = stack.pop(); island.append(current)
                for edge in current.link_edges:
                    other = edge.other_vert(current)
                    if other not in seen: seen.add(other); stack.append(other)
            points = [obj.matrix_world @ v.co for v in island]
            lo = [min(p[k] for p in points) for k in range(3)]
            hi = [max(p[k] for p in points) for k in range(3)]
            backpack = lo[1] > 0.001 and lo[2] > 1.02 and hi[2] < 1.40 and len(island) < 220
            tie = lo[1] < -0.17 and lo[2] > 1.10 and hi[2] < 1.32 and hi[0]-lo[0] < 0.16
            front_strap = len(island) < 60 and lo[2] > 1.10 and hi[2] < 1.36 and lo[1] < -0.085 and (lo[0] > 0.07 or hi[0] < -0.07)
            if backpack or tie or front_strap: drop.extend(island)
        removed += len(drop)
        bmesh.ops.delete(bm, geom=drop, context='VERTS'); bm.to_mesh(mesh); bm.free(); mesh.update()
    bpy.ops.object.select_all(action='DESELECT')
    for obj in [armature, *meshes]: obj.select_set(True)
    bpy.context.view_layer.objects.active = armature
    avatar = output / 'avatar-clean.glb'
    if not data.get('bake_only'):
        bpy.ops.export_scene.gltf(filepath=str(avatar), export_format='GLB', use_selection=True, export_animations=False)
    if data.get('make_avatar_only'):
        (output/'production-report.json').write_text(json.dumps({'blender':bpy.app.version_string,
            'removed_accessory_vertices':removed,'materials':['Clear hands','Plain teal clothing'],
            'skeleton_manually_edited':False,'bind_pose_exported':True,
            'avatar':avatar.name,'language_review':'NOT_RUN'},indent=2),encoding='utf8')
        return
    normalize = lambda s: ''.join(c.lower() for c in s if c.isalnum())
    bones = {normalize(b.name): b for b in armature.pose.bones}
    # GLB is Y-up; Blender imports in Z-up. Preserve each imported bone's own axes.
    conversion = Matrix.Rotation(3.141592653589793/2, 4, 'X')
    def matrix(values):
        return Matrix([values[i::4] for i in range(4)])
    exports = []
    for pilot in data['pilots']:
        track = json.loads((output / (pilot['asset_id']+'.tracks.json')).read_text(encoding='utf8'))
        if armature.animation_data: armature.animation_data_clear()
        for bone in armature.pose.bones: bone.matrix_basis.identity(); bone.rotation_mode = 'QUATERNION'
        bpy.context.view_layer.update()
        corrections = {}
        for name, rest in track['bindMatrices'].items():
            pb = bones.get(normalize(track['boneNames'][name]))
            if pb:
                corrections[name] = (conversion @ matrix(rest)).inverted() @ armature.matrix_world @ pb.bone.matrix_local
        scene = bpy.context.scene; scene.render.fps = round(track['fps']); scene.frame_start = 0; scene.frame_end = track['frames']-1
        for index, frame in enumerate(track['worldFrames']):
            scene.frame_set(index)
            for name, transform in frame.items():
                pb = bones.get(normalize(track['boneNames'][name]))
                if pb is None: continue
                pb.matrix = armature.matrix_world.inverted() @ conversion @ matrix(transform) @ corrections[name]
                bpy.context.view_layer.update()
                for property_name in ('location', 'rotation_quaternion', 'scale'):
                    pb.keyframe_insert(data_path=property_name, frame=index, group=pb.name)
        scene.frame_set(0)
        if armature.animation_data and armature.animation_data.action:
            armature.animation_data.action.name = pilot['asset_id']
        name = pilot['asset_id']+'.glb'
        bpy.ops.export_scene.gltf(filepath=str(output/name), export_format='GLB', use_selection=True,
                                  export_animations=True, export_animation_mode='ACTIVE_ACTIONS', export_force_sampling=True)
        exports.append(name)
    production=json.loads((output/'production-report.json').read_text(encoding='utf8'))
    production.update({'exports':exports,
        'facial_tracks':'neutral; manual controls are independent and unreviewed', 'language_review':'NOT_RUN'})
    (output/'production-report.json').write_text(json.dumps(production,indent=2),encoding='utf8')


def main():
    if '--blender-stage' in sys.argv:
        blender_stage(sys.argv[sys.argv.index('--blender-stage')+1]); return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--blender',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    sys.path.insert(0,str(ROOT))
    from slbridge.animation.registry import MotionRegistry
    registry = MotionRegistry(args.archive,ARCHIVE_SHA)
    output = args.output.resolve(); output.mkdir(parents=True,exist_ok=True)
    source = output/'source'; source.mkdir(exist_ok=True)
    (source/'model.glb').write_bytes(registry.resource('model.glb')[0])
    (source/'three.mjs').write_bytes(registry.resource('three.module.js')[0])
    catalog = {a['id']:a for a in registry.catalog()['assets']}
    pilots = []
    for identity in PILOTS:
        motion = source/(identity+'.json')
        motion.write_text(json.dumps(registry.motion(identity),ensure_ascii=False),encoding='utf8')
        pilots.append({'asset_id':identity,'source_motion_sha256':catalog[identity]['sha256'],
                       'motion':str(motion), 'text':catalog[identity]['text']})
        if identity in TEMPORAL_REPAIRS:
            pilots[-1]['temporal_profile']=TEMPORAL_REPAIRS[identity]
    build = {'output':str(output),'model':str(source/'model.glb'),'three':str(source/'three.mjs'),'pilots':pilots,'make_avatar_only':True}
    input_path = source/'build-input.json'; input_path.write_text(json.dumps(build),encoding='utf8')
    blender_command=[str(args.blender.resolve()),'--background','--python',str(Path(__file__).resolve()),'--','--blender-stage',str(input_path)]
    subprocess.run(blender_command,check=True)
    build.update(model=str(output/'avatar-clean.glb'),make_avatar_only=False,bake_only=True)
    input_path.write_text(json.dumps(build),encoding='utf8')
    subprocess.run(['node',str(Path(__file__).with_name('export_tracks.mjs')),str(input_path)],check=True)
    subprocess.run(blender_command,check=True)
    avatar_sha=hashlib.sha256((output/'avatar-clean.glb').read_bytes()).hexdigest()
    for pilot in pilots:
        binding={'asset_id':pilot['asset_id'],
            'source_motion_sha256':pilot['source_motion_sha256'],'avatar_sha256':avatar_sha,
            'processing_version':PROCESSING_VERSION}
        if 'temporal_profile' in pilot:
            binding['temporal_profile']=pilot['temporal_profile']
        bind_export(output/(pilot['asset_id']+'.glb'),binding)
    files = [output/'avatar-clean.glb',*[output/(p+'.glb') for p in PILOTS]]
    manifest = {'schema_version':1,'avatar':'avatar-clean.glb','processing_version':PROCESSING_VERSION,
                'attribution':'Godette (Rigged), zahlenmaler, CC BY 4.0; presentation derivative by SignLanguage-Bridge',
                'source_archive_sha256':registry.sha256,
                'rig_profile':json.loads((output/(PILOTS[0]+'.tracks.json')).read_text(encoding='utf8'))['rigProfile'],
                'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                'pilots':[{k:v for k,v in p.items() if k!='motion'}|{'export_file':p['asset_id']+'.glb'} for p in pilots]}
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    print('Presentation ready:',output)


if __name__ == '__main__':
    main()
