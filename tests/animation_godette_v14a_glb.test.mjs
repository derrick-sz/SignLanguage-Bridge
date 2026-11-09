// Actual GLTFLoader and CPU skinning. Texture decode is stubbed, not geometry.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {createHumanoidRig} from '../slbridge/animation/web/rig.js';
import {applyExpression} from '../slbridge/animation/web/expression.mjs';
const V14A_SHA='535fc9ab96ec3d837e2568817abc41d593fac1c4fcad8565b45b29c5f4472f5e';
const enabled=process.env.ANIMATION_THREE_MODULE&&process.env.ANIMATION_GLTF_LOADER&&process.env.ANIMATION_V14A_AVATAR_GLB;

test('frozen v14a real loader and skinned geometry remain stable through 101 manual mouth levels and neutral reset',
  {skip:!enabled&&'set ANIMATION_THREE_MODULE, ANIMATION_GLTF_LOADER and ANIMATION_V14A_AVATAR_GLB'},async()=>{
  const THREE=await import(pathToFileURL(process.env.ANIMATION_THREE_MODULE));
  const {GLTFLoader}=await import(pathToFileURL(process.env.ANIMATION_GLTF_LOADER));
  const bytes=fs.readFileSync(process.env.ANIMATION_V14A_AVATAR_GLB);
  assert.equal(crypto.createHash('sha256').update(bytes).digest('hex'),V14A_SHA,'test only frozen candidate bytes');
  const loader=new GLTFLoader();
  loader.register(parser=>{parser.loadTexture=()=>Promise.resolve(new THREE.Texture());return {name:'CPU_GEOMETRY_ONLY_TEXTURE_STUB'};});
  const gltf=await new Promise((resolve,reject)=>loader.parse(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength),'',resolve,reject));
  const root=gltf.scene,meshes=[],bones=[];
  root.traverse(node=>{if(node.isSkinnedMesh)meshes.push(node);if(node.isBone)bones.push(node);});
  const update=()=>{root.updateMatrixWorld(true);new Set(meshes.map(m=>m.skeleton)).forEach(s=>s.update());};
  const points=m=>Array.from({length:m.geometry.attributes.position.count},(_,i)=>m.getVertexPosition(i,new THREE.Vector3()).applyMatrix4(m.matrixWorld));
  const capture=()=>bones.map(b=>({name:b.name,p:b.position.toArray(),q:b.quaternion.toArray(),s:b.scale.toArray()}));
  const rig=createHumanoidRig(root,undefined,{avatarSha256:V14A_SHA});
  assert.equal(rig.isSigningReady,true);assert.equal(rig.manualMouthAvailable,true);assert.equal(rig.jawCalibrationAvailable,false);
  assert.equal(bones.length,227);update();
  const retained=meshes.filter(m=>/^Object_0(?:_\d+)?$/.test(m.name));assert.equal(retained.length,5);
  const base=new Map(meshes.map(m=>[m,points(m)])),baseBones=capture();
  const faces=[],cross=(P,t)=>P[t[1]].clone().sub(P[t[0]]).cross(P[t[2]].clone().sub(P[t[0]]));
  for(const m of retained){
    const P=base.get(m),idx=m.geometry.index?.array||Array.from({length:P.length},(_,i)=>i);
    for(let i=0;i<idx.length;i+=3){const t=Array.from(idx.slice(i,i+3)),n=cross(P,t);if(n.length()/2>1e-8)faces.push({m,t,n});}
  }
  assert.ok(faces.length>8000,'check retained face, hand and skin triangles, not just a sparse facial fixture');
  const lower=root.getObjectByName('Teeth_Lower_12'),upper=root.getObjectByName('Teeth_Upper_71');
  const lo=lower.getWorldPosition(new THREE.Vector3()),up=upper.getWorldPosition(new THREE.Vector3());let previous=lo.y;
  const report={candidate_sha256:V14A_SHA,three_revision:THREE.REVISION,intensity_samples:101,bones:bones.length,
    retained_meshes:retained.map(m=>({name:m.name,vertices:m.geometry.attributes.position.count})),triangle_count:faces.length,
    relative_normal_reversal_samples:0,minimum_area_ratio:1,maximum_area_ratio:1,maximum_edge_ratio:1,maximum_upper_teeth_motion_m:0};
  for(let k=0;k<=100;k++){
    applyExpression(rig,{jawOpen:k/100});update();const y=lower.getWorldPosition(new THREE.Vector3()).y;
    assert.ok(y<=previous+1e-10,'lower teeth should open monotonically downward');previous=y;
    report.maximum_upper_teeth_motion_m=Math.max(report.maximum_upper_teeth_motion_m,upper.getWorldPosition(new THREE.Vector3()).distanceTo(up));
    const curr=new Map(retained.map(m=>[m,points(m)]));
    for(const face of faces){
      const P=curr.get(face.m),n=cross(P,face.t),ratio=n.length()/face.n.length();
      assert.ok(Number.isFinite(ratio));report.minimum_area_ratio=Math.min(report.minimum_area_ratio,ratio);report.maximum_area_ratio=Math.max(report.maximum_area_ratio,ratio);
      if(n.dot(face.n)<0)report.relative_normal_reversal_samples++;
      for(let j=0;j<3;j++){const a=face.t[j],b=face.t[(j+1)%3],L=base.get(face.m)[a].distanceTo(base.get(face.m)[b]);
        if(L>1e-9)report.maximum_edge_ratio=Math.max(report.maximum_edge_ratio,P[a].distanceTo(P[b])/L);}
    }
    const current=capture();applyExpression(rig,{jawOpen:1});applyExpression(rig,{jawOpen:k/100});assert.deepEqual(capture(),current);
  }
  report.lower_teeth_drop_m=lo.y-previous;
  report.maximum_nonface_mesh_motion_m=0;
  for(const m of meshes.filter(m=>!retained.includes(m))){const P=points(m),B=base.get(m);for(let i=0;i<P.length;i++)report.maximum_nonface_mesh_motion_m=Math.max(report.maximum_nonface_mesh_motion_m,P[i].distanceTo(B[i]));}
  const openBones=capture();
  applyExpression(rig,{});update();assert.deepEqual(capture(),baseBones);
  report.maximum_reset_vertex_error_m=0;
  for(const m of meshes){const P=points(m),B=base.get(m);for(let i=0;i<P.length;i++)report.maximum_reset_vertex_error_m=Math.max(report.maximum_reset_vertex_error_m,P[i].distanceTo(B[i]));}
  applyExpression(rig,{jawOpen:1});rig.reset();assert.deepEqual(capture(),baseBones);
  applyExpression(rig,{}, {jawRadians:.16});assert.deepEqual(capture(),baseBones,'automatic jaw must not move v14a');
  assert.equal(report.relative_normal_reversal_samples,0);
  assert.ok(report.minimum_area_ratio>.8);assert.ok(report.maximum_area_ratio<3.1);assert.ok(report.maximum_edge_ratio<1.75);
  assert.ok(report.lower_teeth_drop_m>.003);assert.equal(report.maximum_upper_teeth_motion_m,0);
  assert.equal(report.maximum_nonface_mesh_motion_m,0);assert.equal(report.maximum_reset_vertex_error_m,0);
  report.reset_node_component_exact=true;report.repeated_backward_node_component_exact=true;
  report.limits=['CPU GLTFLoader with stubbed texture decode, not WebGL/browser visual certification.',
    'Normal and area checks are relative to the neutral baseline, not proof of absolute manifoldness or absence of collision.',
    'No semantic mouthing, lip sync, corpus, GPU generation or mobile performance certification.'];
  if(process.env.ANIMATION_V14A_REPORT_DIR){
    const dir=process.env.ANIMATION_V14A_REPORT_DIR;fs.mkdirSync(dir,{recursive:true});
    fs.writeFileSync(path.join(dir,`mouth-skin-three-${THREE.REVISION}.json`),JSON.stringify(report,null,2)+'\n');
    fs.writeFileSync(path.join(dir,'mouth-open1-nodes.json'),JSON.stringify(openBones,null,2)+'\n');
  }
});
