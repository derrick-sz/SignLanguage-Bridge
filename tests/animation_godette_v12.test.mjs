import test from 'node:test';
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
import fs from 'node:fs';
import vm from 'node:vm';
import {PlaybackClock} from '../slbridge/animation/web/player-core.mjs';
import {createHumanoidRig, GODETTE_PROFILE} from '../slbridge/animation/web/rig.js';
import {applyExpression} from '../slbridge/animation/web/expression.mjs';
import {JAW_AVATAR_SHA256, JAW_TEMPLATE_SHA256, JAW_PROFILE, jawCandidateForClip} from '../slbridge/animation/web/nonmanual.mjs';
// Replaced only after the model author freezes and independently hashes the bytes.
const V12_SHA='a7942f7a69e8fb078d50767e85944c0e59b47dc20b40479ab86d8c16513717c4';
const V11_SHA='e4a5766d98ab53871415da3ef4609c4848386c8b4de2958d5e420a8d7524d476';
const THREE=process.env.ANIMATION_THREE_MODULE?await import(pathToFileURL(process.env.ANIMATION_THREE_MODULE)):null;
const options={skip:!THREE&&'set ANIMATION_THREE_MODULE to the deployed Three.js module'};
function fixture(avatarSha256=V12_SHA, profile=GODETTE_PROFILE){
  const root=new THREE.Group(),head=new THREE.Bone(),jaw=new THREE.Bone(),lower=new THREE.Bone();
  head.name='Head_129';jaw.name='Jaw_4';lower.name='Lips_Lower_13';root.add(head);head.add(jaw,lower);
  for(const name of ['Lip_L007_8','Lip_R007_11']){
    const bone=new THREE.Bone();bone.name=name;bone.position.set(name.includes('L')?.02:-.02,.0015,.01);lower.add(bone);
  }
  return {root,jaw,lower,rig:createHumanoidRig(root,profile,{avatarSha256})};
}
function snapshot(root){const nodes=[];root.traverse(n=>nodes.push({name:n.name,p:n.position.toArray(),q:n.quaternion.toArray(),s:n.scale.toArray()}));return nodes;}

test('frozen v12 activates coordinated manual mouth only and never automatic jaw',options,()=>{
  const {root,rig,jaw,lower}=fixture(),base=snapshot(root);
  assert.equal(rig.manualMouthAvailable,true);assert.equal(rig.jawCalibrationAvailable,false);
  assert.equal(rig.applyCalibratedJaw(.08),false);
  applyExpression(rig,{}, {jawRadians:.08});assert.deepEqual(snapshot(root),base);
  applyExpression(rig,{jawOpen:1},{jawRadians:.16});
  const expected=new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1,0,0),.04);
  assert.ok(jaw.quaternion.angleTo(expected)<1e-7);assert.ok(lower.quaternion.angleTo(expected)<1e-7);
});

test('v12 101 levels, backward samples and both resets restore the authored neutral baseline',options,()=>{
  const {root,rig}=fixture(),base=snapshot(root);
  for(let i=0;i<=100;i++){
    applyExpression(rig,{jawOpen:i/100});const first=snapshot(root);
    applyExpression(rig,{jawOpen:1});applyExpression(rig,{jawOpen:i/100});assert.deepEqual(snapshot(root),first);
  }
  applyExpression(rig,{});assert.deepEqual(snapshot(root),base);
  applyExpression(rig,{jawOpen:1});rig.reset();assert.deepEqual(snapshot(root),base);
});

test('v12 with invalid profile or mouth hierarchy stays closed instead of unsafe jaw-only fallback',options,()=>{
  const invalidProfile=fixture(V12_SHA,{...GODETTE_PROFILE,id:'other'});
  const missingCorner=fixture();missingCorner.lower.remove(missingCorner.lower.children[0]);
  const reparented=fixture();reparented.root.add(reparented.lower);
  for(const {root} of [invalidProfile,missingCorner,reparented]){
    const profile=root===invalidProfile.root?{...GODETTE_PROFILE,id:'other'}:GODETTE_PROFILE;
    const rig=createHumanoidRig(root,profile,{avatarSha256:V12_SHA}),base=snapshot(root);
    assert.equal(rig.manualMouthAvailable,false);assert.equal(rig.jawCalibrationAvailable,false);
    assert.equal(rig.manualJawMaxAngle,0);
    applyExpression(rig,{jawOpen:1});assert.deepEqual(snapshot(root),base);
  }
});

test('unknown bytes and legacy identity cannot gain the frozen manual driver',options,()=>{
  for(const sha of [null,'unknown','f'.repeat(64),JAW_AVATAR_SHA256]){
    const {rig}=fixture(sha);assert.equal(rig.manualMouthAvailable,false);
    assert.equal(rig.jawCalibrationAvailable,sha===JAW_AVATAR_SHA256);
  }
  assert.equal(fixture(V11_SHA).rig.manualMouthAvailable,true);
});

test('v12 cannot relabel the legacy automatic candidate',()=>{
  const candidate={profile:JAW_PROFILE,source_joint:22,source_template_sha256:JAW_TEMPLATE_SHA256,
    target_avatar_sha256:JAW_AVATAR_SHA256,status:'research_candidate',export_supported:false};
  const clip={template_sha256:JAW_TEMPLATE_SHA256,nonmanual_candidate:candidate};
  assert.equal(jawCandidateForClip(clip,V12_SHA),null);
  assert.equal(jawCandidateForClip({...clip,nonmanual_candidate:{...candidate,target_avatar_sha256:V12_SHA}},V12_SHA),null);
});


test('stage playback keeps frozen v12 automatic jaw unavailable and reports only manual opening',options,async()=>{
  const source=fs.readFileSync(new URL('../slbridge/animation/web/stage.js',import.meta.url),'utf8')
    .replace(/^import .*;\r?\n/gm,'').replace('export class MotionStage','class MotionStage')+'\nMotionStage;';
  const Stage=vm.runInNewContext(source,{performance:{now:()=>1000},PlaybackClock,applyExpression,jawCandidateForClip,MAX_JAW_ANGLE:.16});
  const {root,rig,jaw,lower}=fixture(),base=snapshot(root),stage=Object.create(Stage.prototype);
  const tracks={original:[{},{}],improved:[{},{}],bounds:{},hands:{}};
  Object.assign(stage,{clock:new PlaybackClock(),ready:Promise.resolve(),cache:new Map([['clip',tracks]]),profile:'improved',
    presentation:{avatar_sha256:V12_SHA,pilots:[]},renderer:{render(){}},rig,model:root,
    originalRig:{interpolate(){},reset(){}},originalModel:{updateMatrixWorld(){}},poseNeutral(){}});
  const clip={frames:2,fps:30,poses:Array.from({length:2},()=>Array.from({length:55},()=>[0,0,0])),
    template_sha256:JAW_TEMPLATE_SHA256,nonmanual_candidate:{profile:JAW_PROFILE,source_joint:22,
      source_template_sha256:JAW_TEMPLATE_SHA256,target_avatar_sha256:JAW_AVATAR_SHA256,status:'research_candidate',export_supported:false}};
  await stage.loadMotion({clip,parents:[]},'clip',()=>true,'source');
  assert.equal(stage.nonmanualAvailable,false);assert.equal(stage.nonmanualEnabled,false);assert.deepEqual(snapshot(root),base);
  stage.setNonmanualEnabled(true);assert.equal(stage.nonmanualEnabled,false);
  stage.setExpression({jawOpen:.5});assert.equal(stage.jawAngle(0),.02);
  const expected=new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1,0,0),.02);
  assert.ok(jaw.quaternion.angleTo(expected)<1e-7);assert.ok(lower.quaternion.angleTo(expected)<1e-7);
  const review=stage.exportReview();assert.equal(review.nonmanual.angle_radians,.02);assert.equal(review.nonmanual.profile,null);
  assert.equal(review.nonmanual.available,false);assert.equal(review.nonmanual.export_supported,false);
  stage.stop();assert.deepEqual(snapshot(root),base);
});
