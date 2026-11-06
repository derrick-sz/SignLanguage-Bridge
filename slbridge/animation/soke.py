"""Selective SOKE ZIP inventory and canonical upper-body motion conversion."""
from dataclasses import dataclass
from pathlib import Path
from copy import deepcopy
import hashlib
import json
import re
import struct
import zipfile
import zlib

import numpy as np

from .soke_pickle import parse_pickle

FIELDS={'smplx_root_pose':3,'smplx_body_pose':63,'smplx_lhand_pose':45,'smplx_rhand_pose':45,
        'smplx_jaw_pose':3,'smplx_shape':10,'smplx_expr':10,'cam_trans':3}
FRAME=re.compile(r'csl-daily_pose/(S\d{6}_P\d{4}_T\d{2})/(\d{6})\.pkl\Z')


def video_binding(info,pose_frames,pose_fps):
    return {'pose_frames':pose_frames,'pose_fps':pose_fps,'original_video_frames':info['frames'],
            'original_video_fps':info['fps'],'frame_count_match':info['frames']==pose_frames,
            'temporal_alignment':'unverified','pose_fps_basis':'CSL-Daily native 30 FPS; release frame clock not independently timestamp-verified'}


def select_seed_rows(rows,seeds,count):
    known={row['name']:row for row in rows};result=[];ids=set();texts=set()
    if not isinstance(seeds,list) or len(seeds)>count:raise ValueError('Seed count exceeds requested corpus')
    for seed in seeds:
        row=known.get(seed.get('name')) if isinstance(seed,dict) else None
        if (row is None or seed.get('text')!=row['text'] or seed.get('num_frames')!=row['num_frames']
                or row['name'] in ids or row['text'] in texts):
            raise ValueError('Seed corpus identity differs from released annotation')
        result.append(row);ids.add(row['name']);texts.add(row['text'])
    return result


def decode_frame(raw):
    value=parse_pickle(raw)
    if not isinstance(value,dict) or set(value)!=set(FIELDS):raise ValueError('Unexpected SOKE parameter keys')
    for key,size in FIELDS.items():
        array=value[key]
        if not isinstance(array,np.ndarray) or array.shape!=(size,) or not np.isfinite(array).all():
            raise ValueError('Invalid SOKE field '+key)
    return value


def effective_poses(parameters,hand_mean):
    mean=np.asarray(hand_mean,dtype=np.float32)
    if mean.shape!=(2,45) or not np.isfinite(mean).all():raise ValueError('Expected verified (2,45) hand mean')
    count=len(parameters['smplx_body_pose'])
    for key,size in FIELDS.items():
        if parameters[key].shape!=(count,size) or not np.isfinite(parameters[key]).all():raise ValueError('Invalid sequence field '+key)
    result=np.zeros((count,55,3),dtype=np.float32)
    result[:,12:22]=parameters['smplx_body_pose'][:,33:].reshape(count,10,3)
    result[:,25:40]=(parameters['smplx_lhand_pose']+mean[0]).reshape(count,15,3)
    result[:,40:55]=(parameters['smplx_rhand_pose']+mean[1]).reshape(count,15,3)
    result[:,22]=parameters['smplx_jaw_pose']
    return result


@dataclass
class Member:
    filename: str
    frame_number: int
    flags: int
    method: int
    crc: int
    compressed_size: int
    size: int
    offset: int


def directory_location(stream):
    stream.seek(0,2);total=stream.tell();length=min(total,131072)
    stream.seek(total-length);tail=stream.read(length);position=tail.rfind(b'PK\x05\x06')
    if position<0 or position+22>len(tail):raise ValueError('Missing ZIP end record')
    fields=struct.unpack_from('<4s4H2LH',tail,position)
    if position+22+fields[7]!=len(tail) or fields[1]!=0 or fields[2]!=0:raise ValueError('Invalid/split ZIP end record')
    entries,size,offset=fields[4:7]
    if entries==65535 or size==0xffffffff or offset==0xffffffff:
        if position<20:raise ValueError('Missing ZIP64 locator')
        locator=struct.unpack_from('<4sLQL',tail,position-20)
        if locator[0]!=b'PK\x06\x07' or locator[1]!=0 or locator[3]!=1:raise ValueError('Invalid ZIP64 locator')
        stream.seek(locator[2]);record=stream.read(56)
        if len(record)!=56:raise ValueError('Truncated ZIP64 record')
        extended=struct.unpack('<4sQ2H2L4Q',record)
        if extended[0]!=b'PK\x06\x06' or extended[4]!=0 or extended[5]!=0:raise ValueError('Invalid ZIP64 end record')
        entries,size,offset=extended[7:10]
    if offset+size>total or entries>4000000:raise ValueError('ZIP directory exceeds bounds')
    return entries,size,offset


def scan_archive(path,selected_ids,progress=None,member_callback=None):
    sequences={};selected={};unknown=0
    with Path(path).open('rb') as stream:
        declared,dir_size,dir_start=directory_location(stream);stream.seek(dir_start)
        for index in range(declared):
            header=stream.read(46)
            if len(header)!=46 or header[:4]!=b'PK\x01\x02':raise ValueError('Invalid central directory')
            values=struct.unpack('<4s6H3L5H2L',header)
            flags,method,crc,csize,size=values[3],values[4],values[7],values[8],values[9]
            nl,el,cl=values[10:13];offset=values[16]
            name=stream.read(nl).decode('utf-8' if flags&2048 else 'cp437');extra=stream.read(el);stream.seek(cl,1)
            if 0xffffffff in (size,csize,offset):
                cursor=0;zip64=None
                while cursor+4<=len(extra):
                    tag,length=struct.unpack_from('<HH',extra,cursor);cursor+=4
                    if tag==1:zip64=extra[cursor:cursor+length];break
                    cursor+=length
                if zip64 is None:raise ValueError('Missing ZIP64 member size')
                cursor=0;replacement=[]
                for value in (size,csize,offset):
                    if value==0xffffffff:
                        value=struct.unpack_from('<Q',zip64,cursor)[0];cursor+=8
                    replacement.append(value)
                size,csize,offset=replacement
            match=FRAME.fullmatch(name)
            if match:
                identity,number=match[1],int(match[2]);row=sequences.setdefault(identity,{'numbers':set(),'uncompressed_bytes':0})
                if number in row['numbers']:raise ValueError('Duplicate frame '+name)
                row['numbers'].add(number);row['uncompressed_bytes']+=size
                if identity in selected_ids or member_callback:
                    member=Member(name,number,flags,method,crc,csize,size,offset)
                    if identity in selected_ids:selected.setdefault(identity,[]).append(member)
                    if member_callback:member_callback(identity,member)
            elif not name.endswith('/'):unknown+=1
            if progress and index%250000==0:progress(index,declared)
        if stream.tell()!=dir_start+dir_size:raise ValueError('ZIP directory length mismatch')
    for identity,row in sequences.items():
        numbers=row.pop('numbers');last=max(numbers)
        if last>10000:raise ValueError('Unexpected source frame count')
        row.update(frames=len(numbers),first_frame=min(numbers),last_frame=last,
                   missing_frame_numbers=sorted(set(range(last+1))-numbers))
    for members in selected.values():members.sort(key=lambda m:m.frame_number)
    return {'archive_entries':declared,'directory_bytes':dir_size,'sequences':sequences,'unknown_files':unknown},selected


def read_member(path,member):
    with Path(path).open('rb') as stream:
        return read_member_stream(stream,member)


def read_member_stream(stream,member):
    """Read one validated member using a caller-owned archive stream."""
    if member.flags&1 or member.method not in (0,8) or not 0<member.size<=1024*1024 or not 0<member.compressed_size<=1024*1024:
        raise ValueError('Unsupported/oversized parameter member')
    stream.seek(member.offset);header=stream.read(30)
    if len(header)!=30:raise ValueError('Missing local member header')
    values=struct.unpack('<4s5H3L2H',header)
    if values[0]!=b'PK\x03\x04' or values[2]!=member.flags or values[3]!=member.method:raise ValueError('Local ZIP header mismatch')
    name=stream.read(values[-2]).decode('utf-8' if member.flags&2048 else 'cp437')
    if name!=member.filename:raise ValueError('Local ZIP member name mismatch')
    stream.seek(values[-1],1);compressed=stream.read(member.compressed_size)
    if member.method==8:
        decoder=zlib.decompressobj(-15)
        try:raw=decoder.decompress(compressed,member.size+1)
        except zlib.error as exc:raise ValueError('Invalid deflate member') from exc
        if not decoder.eof or decoder.unused_data:raise ValueError('Invalid deflate member')
    else:raw=compressed
    if len(raw)!=member.size or zlib.crc32(raw)&0xffffffff!=member.crc:raise ValueError('Member size/CRC mismatch')
    return raw


def load_sequence(path,members,expected_frames,*,stream=None):
    if stream is None:
        with Path(path).open('rb') as source:
            return load_sequence(path,members,expected_frames,stream=source)
    if [m.frame_number for m in members]!=list(range(expected_frames)):
        raise ValueError('Incomplete source sequence; no implicit frame filling')
    frames=[];digest=hashlib.sha256()
    for member in members:
        raw=read_member_stream(stream,member)
        digest.update(member.filename.encode('utf-8')+b'\0'+hashlib.sha256(raw).digest())
        frames.append(decode_frame(raw))
    return {key:np.stack([frame[key] for frame in frames]) for key in FIELDS},digest.hexdigest()


def write_candidate_archive(output,base_archive,clips,motion_attribution):
    """Package selected motions using an already verified base avatar/skeleton."""
    output=Path(output)
    if output.exists():raise ValueError('Candidate archive already exists')
    with zipfile.ZipFile(base_archive) as source:
        original=json.loads(source.read('demo/experimental-index.json'))
        skeleton={key:original[key] for key in ('parents','restJoints','representation','smplx_rest_model_sha256') if key in original}
        resources={name:source.read(name) for name in ('avatar/model.glb','avatar/three.module.js','avatar/jsm/loaders/GLTFLoader.js','avatar/jsm/utils/BufferGeometryUtils.js')}
    def encoded(value):return json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf-8')
    files=dict(resources);entries=[]
    for source in clips:
        clip=deepcopy(source)
        for key in ('motion_approval_id','review_attestation_sha256'):clip.pop(key,None)
        clip.update(human_approved=False,motion_review_status='needs_review',language_review='unverified',training_ready=False,model_training_ready=False)
        name='experimental/'+clip['id']+'.json';raw=encoded(clip);files['demo/'+name]=raw
        entries.append({key:clip[key] for key in ('id','text','frames','fps','quality_warning') if key in clip}|
                       {'asset':name,'sha256':hashlib.sha256(raw).hexdigest(),'human_approved':False})
    files['demo/experimental-index.json']=encoded(skeleton|{'clips':entries,'research_only':True})
    for name in ('demo/t08-motion.json','demo/t08-candidates.json'):files[name]=encoded(skeleton|{'clips':[]})
    manifest={'files':{name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()},
              'motion_attribution':motion_attribution,'research_only':True,'language_review':'NOT_RUN'}
    with zipfile.ZipFile(output,'x',zipfile.ZIP_DEFLATED) as archive:
        for name,raw in files.items():archive.writestr(name,raw)
        archive.writestr('BUILD.json',encoded(manifest))
