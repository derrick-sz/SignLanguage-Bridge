import bpy,json,hashlib,struct
from pathlib import Path
from mathutils import Vector
O=Path('/workspace/scratch/87fc5b72c014/godette_v14_exploration')
def h(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def mx(m):return [list(x) for x in m]
def snap(path):
 bpy.ops.wm.open_mainfile(filepath=str(path),load_ui=False,use_scripts=False);r=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE');meshes={};coords={}
 for o in bpy.context.scene.objects:
  if o.type!='MESH':continue
  m=o.data;coords[o.name]=[v.co.copy() for v in m.vertices];gn=[g.name for g in o.vertex_groups]
  meshes[o.name]={'co':h([list(v.co) for v in m.vertices]),'topology':h([[list(p.vertices),p.material_index,p.use_smooth] for p in m.polygons]),'edges':h([list(e.vertices) for e in m.edges]),'uv':h([(u.name,[list(d.uv) for d in u.data]) for u in m.uv_layers]),'weights':h([[(gn[g.group],g.weight) for g in v.groups] for v in m.vertices]),'groups':h(gn),'materials':[m.name for m in m.materials],'transform':mx(o.matrix_world),'parent':o.parent.name if o.parent else None,'shape_keys':h(None if not m.shape_keys else [[k.name,k.value,[list(v.co) for v in k.data]] for k in m.shape_keys.key_blocks])}
 return {'mesh':meshes,'bones':{b.name:{'parent':b.parent.name if b.parent else None,'rest':mx(b.matrix_local),'head':list(b.head_local),'tail':list(b.tail_local)} for b in r.data.bones},'pose':{p.name:mx(p.matrix_basis) for p in r.pose.bones},'armature_world':mx(r.matrix_world)},coords
A,av=snap(O/'baseline/godette-reference-v13.blend');B,bv=snap(O/'checkpoints/godette-v14b-pressed-crease.blend');changed={}
for n in A['mesh']:
 diff=[k for k in A['mesh'][n] if A['mesh'][n][k]!=B['mesh'][n][k]]
 if diff:changed[n]={'fields':diff,'vertices_changed_above_1e8':sum((a-b).length>1e-8 for a,b in zip(av[n],bv[n])),'max_displacement_m':max((a-b).length for a,b in zip(av[n],bv[n]))}
res={'source_sha256':hashlib.sha256((O/'baseline/godette-reference-v13.blend').read_bytes()).hexdigest(),'candidate_sha256':hashlib.sha256((O/'checkpoints/godette-v14b-pressed-crease.blend').read_bytes()).hexdigest(),'bone_count':[len(A['bones']),len(B['bones'])],'rest_hierarchy_equal':A['bones']==B['bones'],'pose_equal':A['pose']==B['pose'],'armature_equal':A['armature_world']==B['armature_world'],'mesh_names_equal':set(A['mesh'])==set(B['mesh']),'protected_Object_7_equal':A['mesh']['Object_7']==B['mesh']['Object_7'],'only_mesh_coordinates_changed':all(v['fields']==['co'] for v in changed.values()),'changed':changed};(O/'validation/v13-v14b-delta.json').write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
assert all(res[k] for k in ['rest_hierarchy_equal','pose_equal','armature_equal','mesh_names_equal','protected_Object_7_equal','only_mesh_coordinates_changed'])
