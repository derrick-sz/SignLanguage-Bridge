// Real Three.js required. Removing rig.interpolate's equal-key copy must fail
// the first two tests: Three r170 slerp perturbs equal under-unit quaternions.
import test from 'node:test';
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
import {createHumanoidRig} from '../slbridge/animation/web/rig.js';
const THREE=process.env.ANIMATION_THREE_MODULE?await import(pathToFileURL(process.env.ANIMATION_THREE_MODULE)):null;
const options={skip:!THREE&&'set ANIMATION_THREE_MODULE to the deployed Three.js module'};
// Frozen v13 Spine_2_198 local quaternion, loaded using Three r170 GLTFLoader.
const imported=[-0.05910952761769295,1.1900088026095546e-7,7.046435879232149e-9,0.9982514977455139];
function fixture(){
 const root=new THREE.Group(),bone=new THREE.Bone();bone.name='Spine_2_198';bone.quaternion.fromArray(imported);root.add(bone);
 return createHumanoidRig(root);
}
test('equal under-unit imported resting keys are component-exact at every tested fraction',options,()=>{
 const rig=fixture(),rest=rig.rest.chest.quaternion.toArray();
 assert.ok(rig.rest.chest.quaternion.lengthSq()<1);
 for(const t of [0,.1,.25,.5,.75,.9,1,.5,0]){
  rig.interpolate({}, {},t);assert.deepEqual(rig.bones.chest.quaternion.toArray(),rest);
  assert.deepEqual(rig.rest.chest.quaternion.toArray(),rest);
 }
});
test('separate equal nonunit input keys preserve exact components and source objects',options,()=>{
 const rig=fixture();
 for(const scale of [1-1e-8,1+1e-8]){
  const a=new THREE.Quaternion().fromArray(imported).normalize();a.set(...a.toArray().map(v=>v*scale));
  const b=a.clone(),before=[a.toArray(),b.toArray()];
  for(const t of [0,.1,.25,.5,.75,.9,1]){
   rig.interpolate({chest:a},{chest:b},t);assert.deepEqual(rig.bones.chest.quaternion.toArray(),before[0]);
   assert.deepEqual([a.toArray(),b.toArray()],before);
  }
 }
});
test('changing and antipodal nonunit keys preserve exact endpoints and normalized intermediate behavior',options,()=>{
 const rig=fixture();
 for(const antipodal of [false,true]){
  const a=new THREE.Quaternion(.1,.2,.3,.9),b=antipodal?new THREE.Quaternion(-.1,-.2,-.3,-.9):new THREE.Quaternion(.5,.2,-.1,.8);
  const before=[a.toArray(),b.toArray()];
  for(const t of [0,.1,.5,.9,1,.5,0]){
   const expected=t===0?a:t===1?b:a.clone().normalize().slerp(b.clone().normalize(),t);
   rig.interpolate({chest:a},{chest:b},t);assert.deepEqual(rig.bones.chest.quaternion.toArray(),expected.toArray());
   assert.deepEqual([a.toArray(),b.toArray()],before);
  }
 }
});
