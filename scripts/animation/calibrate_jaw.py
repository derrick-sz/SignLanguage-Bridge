"""Numerically inspect actual source/target skin deformation; no Blender or service.

Reports source central inner-lip landmarks and target central lip-weighted vertex
centroids separately: neither is a linguistic score or a collision certificate.
"""
from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sqlite3
import struct

import numpy as np

TEMPLATE_SHA = '376021446ddc86e99acacd795182bbef903e61d33b76b9d8b359c2b0865bd992'
AVATAR_SHA = 'b42c21dacb5d07617790c40c69f68a074aca58d74d76b898f2cb98bbbf96862c'


def rx(angle):
    c,s=np.cos(angle),np.sin(angle)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])


def local_matrix(node):
    if 'matrix' in node:
        return np.asarray(node['matrix']).reshape(4,4,order='F')
    q=np.asarray(node.get('rotation',[0,0,0,1]),dtype=float)
    x,y,z,w=q/np.linalg.norm(q)
    m=np.eye(4)
    m[:3,:3]=np.array([[1-2*y*y-2*z*z,2*x*y-2*z*w,2*x*z+2*y*w],
                      [2*x*y+2*z*w,1-2*x*x-2*z*z,2*y*z-2*x*w],
                      [2*x*z-2*y*w,2*y*z+2*x*w,1-2*x*x-2*y*y]])@np.diag(node.get('scale',[1,1,1]))
    m[:3,3]=node.get('translation',[0,0,0])
    return m


def source_scan(path):
    with np.load(path,allow_pickle=False) as z:
        vertices=z['v_template'].astype(float)
        joints=z['J_regressor']@vertices
        weights=z['weights'][:,22]
        faces=z['f'][z['lmk_faces_idx']]
        bary=z['lmk_bary_coords']
        corrective=z['posedirs'][:,:,21*9:22*9]
        assert not np.any(z['kintree_table'][0,1:]==22),'Probe assumes jaw has no descendants'
    rows=[]
    for angle in [-0.32,-0.16,0,0.08,0.16,0.32]:
        rotation=rx(angle)
        posed=vertices+np.einsum('vci,i->vc',corrective,(rotation-np.eye(3)).reshape(-1))
        deformed=posed+weights[:,None]*((posed-joints[22])@rotation.T+joints[22]-posed)
        landmarks=np.einsum('li,lic->lc',bary,deformed[faces])
        rows.append({'local_x_radians':angle,'signed_inner_lip_gap_y':float(landmarks[45,1]-landmarks[49,1]),
                     'inner_lip_distance':float(np.linalg.norm(landmarks[45]-landmarks[49])),
                     'all_vertices_finite':bool(np.isfinite(deformed).all())})
    return {'joint':22,'landmark_indices':[45,49],'source_units':'template coordinate units',
            'includes_pose_correctives':True,'samples':rows}


def target_scan(path):
    data=path.read_bytes()
    magic,version,_=struct.unpack_from('<III',data)
    assert magic==0x46546c67 and version==2
    size=struct.unpack_from('<I',data,12)[0]
    gltf=json.loads(data[20:20+size]);binary=data[28+size:]
    def accessor(index):
        a=gltf['accessors'][index];v=gltf['bufferViews'][a['bufferView']]
        width={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT4':16}[a['type']]
        dtype=np.dtype({5121:'u1',5123:'<u2',5125:'<u4',5126:'<f4'}[a['componentType']])
        value=np.ndarray((a['count'],width),dtype=dtype,buffer=binary,
            offset=v.get('byteOffset',0)+a.get('byteOffset',0),
            strides=(v.get('byteStride',dtype.itemsize*width),dtype.itemsize)).copy()
        if a.get('normalized') and dtype.kind=='u':value=value.astype(float)/np.iinfo(dtype).max
        return value
    names={n.get('name'):i for i,n in enumerate(gltf['nodes'])}
    jaw,lower=names['Jaw_4'],names['Lips_Lower_13']
    parents={c:i for i,n in enumerate(gltf['nodes']) for c in n.get('children',[])}
    assert parents[jaw]==parents[lower]
    local=[local_matrix(n) for n in gltf['nodes']]
    def worlds(mats):
        world={}
        def one(index):
            if index not in world:world[index]=(one(parents[index]) if index in parents else np.eye(4))@mats[index]
            return world[index]
        for index in range(len(mats)):one(index)
        return world
    base=worlds(local);primitives=[]
    for node in gltf['nodes']:
        if 'mesh' not in node or 'skin' not in node:continue
        skin=gltf['skins'][node['skin']]
        inverse_bind=accessor(skin['inverseBindMatrices']).reshape(-1,4,4).transpose(0,2,1)
        for primitive in gltf['meshes'][node['mesh']]['primitives']:
            a=primitive['attributes'];p=accessor(a['POSITION']);ids=accessor(a['JOINTS_0']).astype(int);weights=accessor(a['WEIGHTS_0'])
            masks=np.array([np.sum(weights*(np.asarray(skin['joints'])[ids]==names[name]),axis=1)
                            for name in ['Lip_Upper_65','Lip_Lower_6','Jaw_4']])
            primitives.append((np.column_stack([p,np.ones(len(p))]),ids,weights,np.asarray(skin['joints']),inverse_bind,masks))
    masks=np.concatenate([p[-1] for p in primitives],axis=1)
    def deform(world):
        parts=[]
        for positions,ids,weights,joints,inverse_bind,_ in primitives:
            transforms=np.asarray([world[int(j)] for j in joints])@inverse_bind
            parts.append(np.sum(np.einsum('vwij,vj->vwi',transforms[ids],positions)*weights[:,:,None],axis=1)[:,:3])
        return np.concatenate(parts)
    original=deform(base)
    def centroids(vertices):return np.asarray([np.average(vertices,axis=0,weights=m) for m in masks])
    reference=centroids(original);rows=[]
    for coupled in [False,True]:
        for angle in [-0.32,-0.16,-0.08,0,0.08,0.16,0.32]:
            mats=[m.copy() for m in local];rotation=np.eye(4);rotation[:3,:3]=rx(angle)
            mats[jaw]=mats[jaw]@rotation
            if coupled:
                delta=base[jaw]@rotation@np.linalg.inv(base[jaw])
                mats[lower]=np.linalg.inv(base[parents[lower]])@delta@base[lower]
            vertices=deform(worlds(mats));points=centroids(vertices)
            rows.append({'lower_lip_pivot_coupled':coupled,'local_x_radians':angle,
                'signed_central_lip_centroid_gap_y':float(points[0,1]-points[1,1]),
                'lower_lip_centroid_delta_y':float(points[1,1]-reference[1,1]),
                'upper_lip_centroid_displacement':float(np.linalg.norm(points[0]-reference[0])),
                'jaw_centroid_delta_y':float(points[2,1]-reference[2,1]),
                'maximum_vertex_displacement':float(np.linalg.norm(vertices-original,axis=1).max()),
                'all_vertices_finite':bool(np.isfinite(vertices).all())})
    return {'mesh_primitives_checked':len(primitives),'vertices_checked_including_material_seams':len(original),
            'jaw_and_lower_lip_are_siblings':True,'centroid_bones':['Lip_Upper_65','Lip_Lower_6','Jaw_4'],
            'weighted_vertex_counts':[int(np.count_nonzero(m>0.001)) for m in masks],
            'reference_centroids':reference.tolist(),'samples':rows}


def source_sample(library,count=40):
    with sqlite3.connect((library/'corpus.sqlite3').as_uri()+'?mode=ro',uri=True) as db:
        rows=db.execute("SELECT id,parameter_file,parameter_sha256 FROM samples WHERE state='converted_candidate' ORDER BY id LIMIT ?",(count,)).fetchall()
    values=[]
    for _,relative,digest in rows:
        path=(library/relative).resolve()
        assert path.is_relative_to(library)
        raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==digest
        with np.load(BytesIO(raw),allow_pickle=False) as z:values.append(z['effective_poses'][:,22].copy())
    joined=np.concatenate(values);steps=np.concatenate([np.linalg.norm(np.diff(v,axis=0),axis=1) for v in values])
    return {'samples':[{'sample_id':r[0],'parameter_sha256':r[2]} for r in rows],'sample_selection':'first 40 converted IDs, convenience sample; not corpus-wide',
            'frames':len(joined),'x_percentiles_0_25_50_75_95_100':np.percentile(joined[:,0],[0,25,50,75,95,100]).tolist(),
            'axis_angle_step_p95_max':np.percentile(steps,[95,100]).tolist(),
            'negative_x_frames':int((joined[:,0]<0).sum()),'norm_above_0_35_frames':int((np.linalg.norm(joined,axis=1)>0.35).sum())}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template',type=Path,required=True)
    parser.add_argument('--avatar',type=Path,required=True)
    parser.add_argument('--library',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    template_sha=hashlib.sha256(args.template.read_bytes()).hexdigest()
    avatar_sha=hashlib.sha256(args.avatar.read_bytes()).hexdigest()
    assert template_sha==TEMPLATE_SHA and avatar_sha==AVATAR_SHA,'Uncalibrated input identity'
    report={'profile':'smplx-jaw-godette-calibrated-v1','source_template_sha256':template_sha,
            'target_avatar_sha256':avatar_sha,'source':source_scan(args.template),'target':target_scan(args.avatar),
            'candidate_limits':{'angle_radians':[0,0.16],'source_norm_rejection_radians':0.35,'speed_radians_per_second':0.6},
            'language_review':'NOT_RUN','visual_acceptance':'NOT_RUN','collision_guarantee':False,
            'expression_coefficients_mapped':False,'export_supported':False}
    if args.library:report['source_sample']=source_sample(args.library.resolve())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps({'report':str(args.output),'source_samples':len(report['source']['samples']),
                      'target_samples':len(report['target']['samples']),'target_primitives':report['target']['mesh_primitives_checked']}))


if __name__=='__main__':main()
