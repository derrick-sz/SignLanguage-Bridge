// Optional frozen-asset test: no textures/WebGL; uses actual GLB node transforms.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {createHumanoidRig} from '../slbridge/animation/web/rig.js';
import {applyExpression} from '../slbridge/animation/web/expression.mjs';
const V11_SHA='e4a5766d98ab53871415da3ef4609c4848386c8b4de2958d5e420a8d7524d476';

test('frozen v11 GLB opens its real lower mouth, fixes the upper mouth and restores every bone',
  {skip:!(process.env.ANIMATION_THREE_MODULE&&process.env.ANIMATION_V11_AVATAR_GLB)},async()=>{
  const THREE=await import(pathToFileURL(process.env.ANIMATION_THREE_MODULE));
  const bytes=fs.readFileSync(process.env.ANIMATION_V11_AVATAR_GLB);
  assert.equal(crypto.createHash('sha256').update(bytes).digest('hex'),V11_SHA);
  const doc=JSON.parse(bytes.subarray(20,20+bytes.readUInt32LE(12)));
  const joints=new Set(doc.skins.flatMap(s=>s.joints));
  const nodes=doc.nodes.map((n,i)=>{
    const o=joints.has(i)?new THREE.Bone():new THREE.Group();o.name=THREE.PropertyBinding.sanitizeNodeName(n.name||'');
    if(n.matrix)new THREE.Matrix4().fromArray(n.matrix).decompose(o.position,o.quaternion,o.scale);
    else{if(n.translation)o.position.fromArray(n.translation);if(n.rotation)o.quaternion.fromArray(n.rotation);if(n.scale)o.scale.fromArray(n.scale);}
    return o;
  });
  doc.nodes.forEach((n,i)=>(n.children||[]).forEach(c=>nodes[i].add(nodes[c])));
  const root=new THREE.Group();doc.scenes[doc.scene||0].nodes.forEach(i=>root.add(nodes[i]));root.updateMatrixWorld(true);
  const rig=createHumanoidRig(root,undefined,{avatarSha256:V11_SHA});
  assert.equal(rig.isSigningReady,true);assert.equal(rig.manualMouthAvailable,true);assert.equal(rig.jawCalibrationAvailable,false);
  const bones=nodes.filter(n=>n.isBone),capture=()=>bones.map(b=>({p:b.position.toArray(),q:b.quaternion.toArray()}));
  const base=capture(),lower=root.getObjectByName('Teeth_Lower_12'),upper=root.getObjectByName('Teeth_Upper_71');
  const lo=lower.getWorldPosition(new THREE.Vector3()),up=upper.getWorldPosition(new THREE.Vector3());
  let previous=lo.y;
  for(let k=0;k<=100;k++){
    applyExpression(rig,{jawOpen:k/100});root.updateMatrixWorld(true);
    const y=lower.getWorldPosition(new THREE.Vector3()).y;
    assert.ok(y<=previous+1e-10);previous=y;
    assert.ok(upper.getWorldPosition(new THREE.Vector3()).distanceTo(up)<1e-9);
    const pose=capture();applyExpression(rig,{jawOpen:1});applyExpression(rig,{jawOpen:k/100});
    assert.deepEqual(capture(),pose,'backward/repeated samples must be deterministic');
  }
  assert.ok(previous<lo.y-.003,'actual lower teeth must open downward by >3 mm');
  applyExpression(rig,{});assert.deepEqual(capture(),base);
  applyExpression(rig,{jawOpen:1});rig.reset();assert.deepEqual(capture(),base);
  applyExpression(rig,{}, {jawRadians:.08});assert.deepEqual(capture(),base,'source jaw cannot drive v11');
});
