// Optional real-asset contract. Run with ANIMATION_THREE_MODULE and ANIMATION_AVATAR_GLB.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {createHumanoidRig} from '../slbridge/animation/web/rig.js';
import {applyExpression} from '../slbridge/animation/web/expression.mjs';
import {JAW_AVATAR_SHA256} from '../slbridge/animation/web/nonmanual.mjs';

test('calibrated real jaw opens downward with its sibling lower lip and resets without drift',
  {skip:!(process.env.ANIMATION_THREE_MODULE&&process.env.ANIMATION_AVATAR_GLB)},async()=>{
  const THREE=await import(pathToFileURL(process.env.ANIMATION_THREE_MODULE));
  const bytes=fs.readFileSync(process.env.ANIMATION_AVATAR_GLB);
  assert.equal(crypto.createHash('sha256').update(bytes).digest('hex'),JAW_AVATAR_SHA256);
  const gltf=JSON.parse(bytes.subarray(20,20+bytes.readUInt32LE(12)));
  const joints=new Set(gltf.skins.flatMap(s=>s.joints));
  const nodes=gltf.nodes.map((n,i)=>{
    const o=joints.has(i)?new THREE.Bone():new THREE.Group();o.name=n.name;
    if(n.matrix)new THREE.Matrix4().fromArray(n.matrix).decompose(o.position,o.quaternion,o.scale);
    else {if(n.translation)o.position.fromArray(n.translation);if(n.rotation)o.quaternion.fromArray(n.rotation);if(n.scale)o.scale.fromArray(n.scale);}
    return o;
  });
  gltf.nodes.forEach((n,i)=>(n.children||[]).forEach(c=>nodes[i].add(nodes[c])));
  const root=new THREE.Group();gltf.scenes[gltf.scene||0].nodes.forEach(i=>root.add(nodes[i]));root.updateMatrixWorld(true);
  const rig=createHumanoidRig(root,undefined,{avatarSha256:JAW_AVATAR_SHA256});
  assert.equal(rig.jawCalibrationAvailable,true);
  const lower=root.getObjectByName('Lip_Lower_6'),upper=root.getObjectByName('Lip_Upper_65');
  const rest=lower.getWorldPosition(new THREE.Vector3()),upperRest=upper.getWorldPosition(new THREE.Vector3());
  const jawRest=rig.bones.jaw.quaternion.clone();
  applyExpression(rig,{jawOpen:1});root.updateMatrixWorld(true);
  assert.ok(lower.getWorldPosition(new THREE.Vector3()).y<rest.y-0.01,'positive manual open must lower the actual lip');
  assert.ok(upper.getWorldPosition(new THREE.Vector3()).distanceTo(upperRest)<1e-8);
  const open=lower.getWorldPosition(new THREE.Vector3());
  for(let i=0;i<20;i++)applyExpression(rig,{jawOpen:1});root.updateMatrixWorld(true);
  assert.ok(lower.getWorldPosition(new THREE.Vector3()).distanceTo(open)<1e-8);
  applyExpression(rig,{});root.updateMatrixWorld(true);
  assert.ok(lower.getWorldPosition(new THREE.Vector3()).distanceTo(rest)<1e-8);
  assert.ok(rig.bones.jaw.quaternion.clone().normalize().angleTo(jawRest.clone().normalize())<1e-7);
  applyExpression(rig,{}, {jawRadians:0.08});root.updateMatrixWorld(true);
  assert.ok(lower.getWorldPosition(new THREE.Vector3()).y<rest.y-0.005);
  rig.reset();root.updateMatrixWorld(true);
  assert.ok(lower.getWorldPosition(new THREE.Vector3()).distanceTo(rest)<1e-8);
  assert.equal(createHumanoidRig(root,undefined,{avatarSha256:'unknown'}).jawCalibrationAvailable,false);
});
