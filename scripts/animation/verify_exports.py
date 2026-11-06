"""Check baked GLB joint trajectories against the exact runtime target tracks."""
from pathlib import Path
import argparse
import json
import struct

import numpy as np


def load_glb(path):
    data = path.read_bytes()
    size = struct.unpack_from('<I', data, 12)[0]
    gltf = json.loads(data[20:20+size])
    binary = data[28+size:]

    def accessor(index):
        item = gltf['accessors'][index]
        view = gltf['bufferViews'][item['bufferView']]
        width = {'SCALAR':1, 'VEC3':3, 'VEC4':4, 'MAT4':16}[item['type']]
        dtype = np.dtype({5126:'<f4',5123:'<u2',5125:'<u4'}[item['componentType']])
        return np.ndarray((item['count'],width),dtype=dtype,buffer=binary,
            offset=view.get('byteOffset',0)+item.get('byteOffset',0),
            strides=(view.get('byteStride',dtype.itemsize*width),dtype.itemsize)).copy()
    return gltf, accessor


def matrix(translation, rotation, scale):
    x,y,z,w = np.asarray(rotation) / np.linalg.norm(rotation)
    result = np.eye(4)
    result[:3,:3] = np.array([
        [1-2*y*y-2*z*z,2*x*y-2*z*w,2*x*z+2*y*w],
        [2*x*y+2*z*w,1-2*x*x-2*z*z,2*y*z-2*x*w],
        [2*x*z-2*y*w,2*y*z+2*x*w,1-2*x*x-2*y*y]]) @ np.diag(scale)
    result[:3,3] = translation
    return result


def verify(folder):
    normalize = lambda s: ''.join(c.lower() for c in s if c.isalnum())
    pilots=json.loads((folder/'manifest.json').read_text(encoding='utf8'))['pilots']
    expected_ids={pilot['asset_id'] for pilot in pilots}
    reports = []
    for path in sorted(folder.glob('*.tracks.json')):
        track = json.loads(path.read_text(encoding='utf8'))
        gltf, accessor = load_glb(folder/(track['asset_id']+'.glb'))
        names = {normalize(n.get('name','')):i for i,n in enumerate(gltf['nodes'])}
        animation = gltf['animations'][0]
        channels = []
        for channel in animation['channels']:
            sampler = animation['samplers'][channel['sampler']]
            if channel['target']['path'] == 'weights':
                continue
            assert sampler.get('interpolation','LINEAR') in ('LINEAR','STEP')
            channels.append((channel['target']['node'], channel['target']['path'],
                             accessor(sampler['input'])[:,0],accessor(sampler['output']),sampler.get('interpolation','LINEAR')))
        assert abs(max(float(c[2][-1]) for c in channels)-(track['frames']-1)/track['fps']) < 1e-4
        maximum = 0.0
        for sample, expected_frame in enumerate(track['worldFrames']):
            time = sample / track['fps']
            nodes = [{k:n.get(k,default) for k,default in
                      [('translation',[0,0,0]),('rotation',[0,0,0,1]),('scale',[1,1,1])]}
                     for n in gltf['nodes']]
            for node, key, times, values, interpolation in channels:
                index = min(int(np.searchsorted(times,time,side='right'))-1,len(times)-1)
                index = max(0,index)
                if index == len(times)-1 or interpolation == 'STEP':
                    value = values[index]
                else:
                    amount = max(0,min(1,(time-times[index])/(times[index+1]-times[index])))
                    a,b = values[index].astype(float),values[index+1].astype(float)
                    if key == 'rotation':
                        a/=np.linalg.norm(a); b/=np.linalg.norm(b)
                        cosine = float(np.dot(a,b))
                        if cosine < 0: b=-b; cosine=-cosine
                        angle=np.arccos(max(-1,min(1,cosine)))
                        value = a*(1-amount)+b*amount if angle < 1e-6 else (a*np.sin((1-amount)*angle)+b*np.sin(amount*angle))/np.sin(angle)
                    else:
                        value = a*(1-amount)+b*amount
                nodes[node][key] = value
            world = {}
            def visit(index, parent):
                source = gltf['nodes'][index]
                local = (np.asarray(source['matrix']).reshape(4,4,order='F') if 'matrix' in source else
                         matrix(nodes[index]['translation'],nodes[index]['rotation'],nodes[index]['scale']))
                world[index] = parent @ local
                for child in source.get('children',[]):
                    visit(child, world[index])
            for index in gltf['scenes'][gltf.get('scene',0)]['nodes']:
                visit(index, np.eye(4))
            for key, expected in expected_frame.items():
                index = names[normalize(track['boneNames'][key])]
                error = float(np.linalg.norm(world[index][:3,3]-np.asarray(expected).reshape(4,4,order='F')[:3,3]))
                maximum = max(maximum,error)
        reports.append({'asset_id':track['asset_id'],'frames_checked':track['frames'],
                        'maximum_joint_position_difference':maximum,'passed':maximum < 1e-4,
                        'language_review':'NOT_RUN'})
    assert (expected_ids and len(expected_ids)==len(pilots)==len(reports)
            and {report['asset_id'] for report in reports}==expected_ids), 'Baked tracks must match presentation pilot identities'
    (folder/'export-verification.json').write_text(json.dumps(reports,indent=2),encoding='utf8')
    print(json.dumps(reports,indent=2))
    assert all(r['passed'] for r in reports), 'Baked GLB differs from runtime target trajectory'
    return reports


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    verify(parser.parse_args().folder)
