import test from 'node:test';
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
import {createHumanoidRig, GODETTE_PROFILE} from '../slbridge/animation/web/rig.js';
import {applyExpression} from '../slbridge/animation/web/expression.mjs';
import {JAW_AVATAR_SHA256, JAW_TEMPLATE_SHA256, JAW_PROFILE, jawCandidateForClip} from '../slbridge/animation/web/nonmanual.mjs';
const V11_SHA='e4a5766d98ab53871415da3ef4609c4848386c8b4de2958d5e420a8d7524d476';
const THREE=process.env.ANIMATION_THREE_MODULE?await import(pathToFileURL(process.env.ANIMATION_THREE_MODULE)):null;
const options={skip:!THREE&&'set ANIMATION_THREE_MODULE to the deployed Three.js module'};
function fixture(avatarSha256=V11_SHA, profile=GODETTE_PROFILE){
  const root=new THREE.Group(),head=new THREE.Bone(),jaw=new THREE.Bone(),lower=new THREE.Bone();
  head.name='Head_129';jaw.name='Jaw_4';lower.name='Lips_Lower_13';root.add(head);head.add(jaw,lower);
  for(const name of ['Lip_L007_8','Lip_R007_11']){
    const b=new THREE.Bone();b.name=name;b.position.set(name.includes('L')?.02:-.02,0,.01);lower.add(b);
  }
  return {root,jaw,lower,rig:createHumanoidRig(root,profile,{avatarSha256})};
}
function snapshot(root){return root.children.flatMap(function walk(n){return [{name:n.name,p:n.position.toArray(),q:n.quaternion.toArray()},...n.children.flatMap(walk)];});}

test('v11 enables only its SHA-bound manual driver and cannot activate the legacy automatic calibration',options,()=>{
  const {rig,jaw,lower}=fixture();
  assert.equal(rig.manualMouthAvailable,true);assert.equal(rig.jawCalibrationAvailable,false);
  assert.equal(rig.manualJawMaxAngle,.04);
  assert.equal(rig.applyCalibratedJaw(.08),false);
  applyExpression(rig,{}, {jawRadians:.08});
  assert.deepEqual(jaw.quaternion.toArray(),[0,0,0,1]);
  assert.deepEqual(lower.quaternion.toArray(),[0,0,0,1]);
  applyExpression(rig,{jawOpen:1},{jawRadians:.16});
  const expected=new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1,0,0),.04);
  assert.ok(jaw.quaternion.angleTo(expected)<1e-7);assert.ok(lower.quaternion.angleTo(expected)<1e-7);
});

test('v11 manual mouth remains deterministic through 101 levels, repeats, backward seeks and reset',options,()=>{
  const {root,rig}=fixture(),base=snapshot(root);
  for(let i=0;i<=100;i++){
    applyExpression(rig,{jawOpen:i/100});const first=snapshot(root);
    applyExpression(rig,{jawOpen:1});applyExpression(rig,{jawOpen:i/100});
    assert.deepEqual(snapshot(root),first);
  }
  applyExpression(rig,{});assert.deepEqual(snapshot(root),base);
  applyExpression(rig,{jawOpen:1});rig.reset();assert.deepEqual(snapshot(root),base);
});

test('matching names or rig profile alone never enable v11 manual controls',options,()=>{
  for(const sha of [null,'unknown',JAW_AVATAR_SHA256]){
    const {rig}=fixture(sha);assert.equal(rig.manualMouthAvailable,false);
    assert.equal(rig.jawCalibrationAvailable,sha===JAW_AVATAR_SHA256);
  }
  assert.equal(fixture(V11_SHA,{...GODETTE_PROFILE,id:'other'}).rig.manualMouthAvailable,false);
  const {root,lower}=fixture();lower.remove(lower.children[0]);
  assert.equal(createHumanoidRig(root,GODETTE_PROFILE,{avatarSha256:V11_SHA}).manualMouthAvailable,false);
});

test('v11 cannot reuse or relabel legacy source-target nonmanual candidates',()=>{
  const candidate={profile:JAW_PROFILE,source_joint:22,source_template_sha256:JAW_TEMPLATE_SHA256,
    target_avatar_sha256:JAW_AVATAR_SHA256,status:'research_candidate',export_supported:false};
  const clip={template_sha256:JAW_TEMPLATE_SHA256,nonmanual_candidate:candidate};
  assert.equal(jawCandidateForClip(clip,V11_SHA),null);
  assert.equal(jawCandidateForClip({...clip,nonmanual_candidate:{...candidate,target_avatar_sha256:V11_SHA}},V11_SHA),null);
});
