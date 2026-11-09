import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createHumanoidRig} from '../slbridge/animation/web/rig.js';
import {createT08Retarget} from '../slbridge/animation/web/retarget.js';
import {selectContactTrack} from '../slbridge/animation/web/body-clearance.js';

const enabled=process.env.ANIMATION_THREE_MODULE&&process.env.ANIMATION_GLTF_LOADER&&process.env.ANIMATION_V14A_AVATAR_GLB&&process.env.ANIMATION_WORD_MOTION_DIR;
const options={skip:!enabled&&'configure deployed Three.js, frozen v14A and real word motion files'};
let fixture;
async function setup(){
  if(fixture)return fixture;
  const THREE=await import(pathToFileURL(process.env.ANIMATION_THREE_MODULE));
  const {GLTFLoader}=await import(pathToFileURL(process.env.ANIMATION_GLTF_LOADER));
  const bytes=fs.readFileSync(process.env.ANIMATION_V14A_AVATAR_GLB),loader=new GLTFLoader();
  loader.register(parser=>{parser.loadTexture=()=>Promise.resolve(new THREE.Texture());return {name:'CPU_TEXTURE_STUB'};});
  const gltf=await new Promise((resolve,reject)=>loader.parse(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength),'',resolve,reject));
  fixture={THREE,root:gltf.scene,rig:createHumanoidRig(gltf.scene)};return fixture;
}
function loadMotion(id){return JSON.parse(fs.readFileSync(path.join(process.env.ANIMATION_WORD_MOTION_DIR,id+'.json')));}

for(const [id,label] of [['S000788_P0004_T00','daughter homework'],['pilot-hospital-allergy-011','drug allergy']]){
  test(`${label} clearance does not add an abrupt wrist-offset transition`,options,async()=>{
    const {THREE,rig,root}=await setup(),data=loadMotion(id),copy=JSON.stringify(data);
    const mapper=createT08Retarget(rig,root,data),profile=data.clip.gesture_refinement?.profile||'quaternion-temporal-adaptive-v1';
    const before=mapper.compile(data.clip,{stabilize:true,stabilizationProfile:profile,positionConstraints:true});
    const after=mapper.compile(data.clip,{stabilize:true,stabilizationProfile:profile,positionConstraints:true,bodyClearance:true});
    const wrists=frames=>frames.map(frame=>{rig.interpolate(frame,frame,1);root.updateMatrixWorld(true);return ['left','right'].map(side=>rig.bones[side+'Hand'].getWorldPosition(new THREE.Vector3()));});
    const a=wrists(before),b=wrists(after);let largest=0;
    for(let side=0;side<2;side++){
      const offsets=b.map((p,i)=>p[side].clone().sub(a[i][side]));
      for(let i=1;i<offsets.length;i++)largest=Math.max(largest,offsets[i].distanceTo(offsets[i-1]));
    }
    assert.ok(largest<=.015,`${label} adds an offset step of ${largest}`);
    assert.equal(mapper.diagnostics.body_clearance.remaining_core_hits,0);
    assert.equal(JSON.stringify(data),copy);
  });
}

test('watermelon motion clears the remaining head overlap within existing translation and bounded wrist rotation',options,async()=>{
  const {THREE,rig,root}=await setup(),data=loadMotion('S000931_P0000_T00'),copy=JSON.stringify(data);
  const mapper=createT08Retarget(rig,root,data),raw=mapper.compile(data.clip,{stabilize:false});
  const before=mapper.compile(data.clip,{stabilize:true,stabilizationProfile:data.clip.gesture_refinement.profile,positionConstraints:true});
  const after=mapper.compile(data.clip,{stabilize:true,stabilizationProfile:data.clip.gesture_refinement.profile,positionConstraints:true,bodyClearance:true});
  const d=mapper.diagnostics.body_clearance;
  assert.equal(d.contact_track,'v14a-S000931-contact-v1');
  assert.equal(d.remaining_core_hits,0,'the formerly unresolved watermelon contact must clear');
  assert.ok(d.max_offset<=.060001,'do not solve contact by expanding the translation budget');
  assert.ok(d.max_offset_step<=.015,'the corrected contact must not jump to a different exit direction');
  assert.ok(d.max_wrist_rotation<=Math.PI/12+1e-6);
  assert.ok(d.max_wrist_rotation_step<=Math.PI/60+1e-6);
  const wrists=frames=>frames.map(frame=>{
    rig.interpolate(frame,frame,1);root.updateMatrixWorld(true);
    return ['left','right'].map(side=>({p:rig.bones[side+'Hand'].getWorldPosition(new THREE.Vector3()),q:rig.bones[side+'Hand'].getWorldQuaternion(new THREE.Quaternion()).normalize()}));
  });
  const a=wrists(before),b=wrists(after);let maxOffset=0,maxStep=0,maxAngle=0,maxAngleStep=0;
  for(let side=0;side<2;side++){
    const offsets=b.map((p,i)=>p[side].p.clone().sub(a[i][side].p));
    const rotations=b.map((p,i)=>p[side].q.clone().multiply(a[i][side].q.clone().invert()).normalize());
    for(let i=0;i<after.length;i++){
      maxOffset=Math.max(maxOffset,offsets[i].length());
      maxAngle=Math.max(maxAngle,rotations[i].angleTo(new THREE.Quaternion()));
      if(i){maxStep=Math.max(maxStep,offsets[i].distanceTo(offsets[i-1]));maxAngleStep=Math.max(maxAngleStep,rotations[i].angleTo(rotations[i-1]));}
    }
    assert.ok(offsets[0].length()<1e-6&&offsets.at(-1).length()<1e-6,'clip endpoints keep the baseline wrist positions');
    assert.ok(rotations[0].angleTo(new THREE.Quaternion())<1e-6&&rotations.at(-1).angleTo(new THREE.Quaternion())<1e-6);
  }
  assert.ok(maxOffset<=.060001);
  assert.ok(maxStep<=.015);
  assert.ok(maxAngle<=Math.PI/12+1e-6);
  assert.ok(maxAngleStep<=Math.PI/60+1e-6);
  for(const [reported,measured] of [[d.max_offset,maxOffset],[d.max_offset_step,maxStep],[d.max_wrist_rotation,maxAngle],[d.max_wrist_rotation_step,maxAngleStep]])assert.ok(Math.abs(reported-measured)<1e-6,'diagnostics match actual world-space corrections');
  for(let i=0;i<raw.length;i++)for(const name of Object.keys(raw[i]).filter(n=>/Thumb|Index|Middle|Ring|Little/.test(n)))assert.deepEqual(after[i][name].toArray(),raw[i][name].toArray());
  assert.equal(JSON.stringify(data),copy);
});

test('watermelon playback remains outside the independently measured head core at sub-frame samples',options,async()=>{
  const {THREE,rig,root}=await setup(),data=loadMotion('S000931_P0000_T00');
  rig.reset();root.updateMatrixWorld(true);const restInverse=rig.bones.head.matrixWorld.clone().invert();
  const center=new THREE.Vector3(-0.005373135408294705,1.5162406815296796,0.061410043014662984);
  const radius=new THREE.Vector3(0.0993432834537267,0.12657852957911095,0.08303607353268709);
  const mapper=createT08Retarget(rig,root,data);
  const frames=mapper.compile(data.clip,{stabilize:true,stabilizationProfile:data.clip.gesture_refinement.profile,positionConstraints:true,bodyClearance:true});
  let minimum=Infinity;
  for(let tick=0;tick<=(frames.length-1)*16;tick++){
    const index=Math.floor(tick/16),next=Math.min(index+1,frames.length-1);
    rig.interpolate(frames[index],frames[next],tick/16-index);root.updateMatrixWorld(true);
    const undo=rig.bones.head.matrixWorld.clone().multiply(restInverse).invert();
    for(const [name,bone] of Object.entries(rig.bones))if(name.startsWith('right')&&/Hand$|Thumb|Index|Middle|Ring|Little/.test(name)){
      const point=bone.getWorldPosition(new THREE.Vector3()),points=[point];
      if(name!=='rightHand')points.push(point.clone().lerp(bone.parent.getWorldPosition(new THREE.Vector3()),.5));
      for(const p of points)minimum=Math.min(minimum,p.applyMatrix4(undo).sub(center).divide(radius).length());
    }
  }
  assert.ok(minimum>=1-1e-5,`interpolation re-enters the head core: ${minimum}`);
});

test('the calibrated contact track rejects different source identity, timing and repair configuration',options,()=>{
  const data=loadMotion('S000931_P0000_T00'),profile=data.clip.gesture_refinement.profile;
  assert.ok(selectContactTrack(data.clip,data.sha256,profile));
  assert.equal(selectContactTrack(data.clip,'0'.repeat(64),profile),null);
  assert.equal(selectContactTrack(data.clip,data.sha256,'quaternion-temporal-adaptive-v1'),null);
  for(const [key,value] of [['id','another-clip'],['source_id','another-source'],['parameter_sha256','0'.repeat(64)],['processing_version','future-version'],['fps',60],['frames',161]]){
    assert.equal(selectContactTrack({...data.clip,[key]:value},data.sha256,profile),null,key);
  }
  assert.equal(selectContactTrack({...data.clip,refined_poses:data.clip.refined_poses.slice(1)},data.sha256,profile),null);
});
