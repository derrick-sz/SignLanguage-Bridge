import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {pathToFileURL} from 'node:url';
import {createHumanoidRig} from '../slbridge/animation/web/rig.js';
import {createT08Retarget} from '../slbridge/animation/web/retarget.js';

const enabled=process.env.ANIMATION_THREE_MODULE&&process.env.ANIMATION_GLTF_LOADER&&process.env.ANIMATION_V14A_AVATAR_GLB&&process.env.ANIMATION_HAND_MOTION_JSON;
const options={skip:!enabled&&'configure deployed Three.js, v14A and a real source motion'};
let fixture;
async function setup(){
  if(fixture)return fixture;
  const THREE=await import(pathToFileURL(process.env.ANIMATION_THREE_MODULE));
  const {GLTFLoader}=await import(pathToFileURL(process.env.ANIMATION_GLTF_LOADER));
  const bytes=fs.readFileSync(process.env.ANIMATION_V14A_AVATAR_GLB),loader=new GLTFLoader();
  loader.register(parser=>{parser.loadTexture=()=>Promise.resolve(new THREE.Texture());return {name:'CPU_GEOMETRY_ONLY_TEXTURE_STUB'};});
  const gltf=await new Promise((resolve,reject)=>loader.parse(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength),'',resolve,reject));
  const data=JSON.parse(fs.readFileSync(process.env.ANIMATION_HAND_MOTION_JSON));
  const rig=createHumanoidRig(gltf.scene);
  fixture={THREE,rig,root:gltf.scene,data};return fixture;
}

for(const profile of ['quaternion-temporal-adaptive-v1','source-quaternion-temporal-v1','soke-gesture-refinement-v1']){
  test(`display repair preserves all finger joint rotations under ${profile}`,options,async()=>{
    const {rig,root,data}=await setup();
    const pose=()=>Array.from({length:55},()=>[0,0,0]);
    const poses=Array.from({length:9},pose);
    // A real handshape change must not become an independently filtered finger.
    poses[4][25]=[0.8,0,0];poses[4][26]=[0.6,0,0];poses[4][37]=[0,0,0.4];
    poses[4][16]=[0,0,0.5];
    const refined=structuredClone(poses);refined[4][25]=[0.2,0,0];
    const clip={fps:30,frames:9,poses,refined_poses:refined,gesture_refinement:{profile:'soke-gesture-refinement-v1'}};
    const before=structuredClone(clip),mapper=createT08Retarget(rig,root,data);
    const baseline=mapper.compile(clip,{stabilize:false});
    const repaired=mapper.compile(clip,{stabilize:true,stabilizationProfile:profile,positionConstraints:true});
    let checked=0;
    for(let i=0;i<poses.length;i++)for(const name of Object.keys(baseline[i]).filter(n=>/Thumb|Index|Middle|Ring|Little/.test(n))){
      assert.deepEqual(repaired[i][name].toArray(),baseline[i][name].toArray(),`${name} changed at frame ${i}`);checked++;
    }
    assert.equal(checked,270);
    assert.deepEqual(clip,before,'source and explicitly stored refined candidates remain unchanged');
    assert.ok(baseline[4].leftIndexProximal.clone().normalize().angleTo(baseline[3].leftIndexProximal.clone().normalize())>0.5,'the source handshape was not flattened');
  });
}

test('v14A hospital motion clears deep head penetration while retaining its source handshape',options,async()=>{
  const {THREE,rig,root,data}=await setup();
  rig.reset();root.updateMatrixWorld(true);
  const restHeadInverse=rig.bones.head.matrixWorld.clone().invert();
  // Independently measured retained face/head mesh core in the frozen v14A GLB.
  const center=new THREE.Vector3(-0.005373135408294705,1.5162406815296796,0.061410043014662984);
  const radius=new THREE.Vector3(0.0993432834537267,0.12657852957911095,0.08303607353268709);
  const mapper=createT08Retarget(rig,root,data);
  const raw=mapper.compile(data.clip,{stabilize:false});
  const corrected=mapper.compile(data.clip,{stabilize:true,stabilizationProfile:'quaternion-temporal-adaptive-v1',positionConstraints:true,bodyClearance:true});
  for(const [index,name] of [[25,'rightIndexDistal'],[48,'rightThumbDistal']]){
    rig.interpolate(corrected[index],corrected[index],1);root.updateMatrixWorld(true);
    const undo=rig.bones.head.matrixWorld.clone().multiply(restHeadInverse).invert();
    const point=rig.bones[name].getWorldPosition(new THREE.Vector3()).applyMatrix4(undo).sub(center).divide(radius);
    assert.ok(point.length()>=0.999,`${name} still inside measured head core at ${index}: ${point.length()}`);
    for(const key of Object.keys(raw[index]).filter(n=>/Thumb|Index|Middle|Ring|Little/.test(n)))assert.deepEqual(corrected[index][key].toArray(),raw[index][key].toArray());
  }
  assert.ok(mapper.diagnostics.body_clearance.adjusted_hands>0);
  assert.ok(mapper.diagnostics.body_clearance.max_offset<=0.060001);
});

test('body clearance enters gradually rather than adding a one-frame wrist pop',options,async()=>{
  const {THREE,rig,root,data}=await setup(),mapper=createT08Retarget(rig,root,data);
  const baseline=mapper.compile(data.clip,{stabilize:true,stabilizationProfile:'quaternion-temporal-adaptive-v1',positionConstraints:true});
  const repaired=mapper.compile(data.clip,{stabilize:true,stabilizationProfile:'quaternion-temporal-adaptive-v1',positionConstraints:true,bodyClearance:true});
  const positions=frames=>frames.map(frame=>{rig.interpolate(frame,frame,1);root.updateMatrixWorld(true);return rig.bones.rightHand.getWorldPosition(new THREE.Vector3());});
  const a=positions(baseline),b=positions(repaired),offsets=b.map((p,i)=>p.sub(a[i]));
  let largest=0;for(let i=1;i<offsets.length;i++)largest=Math.max(largest,offsets[i].distanceTo(offsets[i-1]));
  assert.ok(largest<.012,`avoidance adds a wrist offset jump of ${largest}`);
  assert.ok(Math.abs(mapper.diagnostics.body_clearance.max_offset_step-largest)<.00001,'review diagnostics must report the actual added wrist-offset step');
});
