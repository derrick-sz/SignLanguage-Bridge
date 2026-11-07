// Run with ANIMATION_THREE_MODULE pointing to the actual deployed Three.js module.
// No downloaded dependency, vendored asset or historical checkout is assumed.
import test from 'node:test';
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
import {createHumanoidRig} from '../slbridge/animation/web/rig.js';

const THREE = process.env.ANIMATION_THREE_MODULE
  ? await import(pathToFileURL(process.env.ANIMATION_THREE_MODULE)) : null;
const options = {skip: !THREE && 'set ANIMATION_THREE_MODULE to the deployed Three.js module'};
const q0=[0.10117690911166022,0.0720743789298999,0.13291790942788692,0.9833115093085794];
const q1=[0.10100603284386261,0.07251659369772764,0.132617443788785,0.9833371325653655];
function fixture() {
  const root=new THREE.Group(),hand=new THREE.Bone();hand.name='HandL_154';
  hand.quaternion.fromArray(q0);root.add(hand);
  return createHumanoidRig(root);
}
function keys() {return [new THREE.Quaternion().fromArray(q0),new THREE.Quaternion().fromArray(q1)];}

test('nonunit answer002 hand keys interpolate instead of freezing when the dot exceeds one',options,()=>{
  const rig=fixture(),[a,b]=keys();assert.ok(a.dot(b)>1);
  const wanted=a.clone().normalize().slerp(b.clone().normalize(),.5);
  rig.interpolate({leftHand:a},{leftHand:b},.5);
  const got=rig.bones.leftHand.quaternion.clone().normalize();
  assert.ok(got.angleTo(wanted)<1e-7);
  assert.ok(got.angleTo(a.clone().normalize())>1e-4,'must not hold the previous key');
});

test('negative encoding takes the same shortest arc without mutating either source',options,()=>{
  const rig=fixture(),[a]=keys(),b=new THREE.Quaternion().fromArray(q1.map(x=>-x));
  const before=[a.toArray(),b.toArray()];
  const wanted=a.clone().normalize().slerp(b.clone().normalize(),.5);
  rig.interpolate({leftHand:a},{leftHand:b},.5);
  assert.ok(rig.bones.leftHand.quaternion.clone().normalize().angleTo(wanted)<1e-7);
  assert.deepEqual([a.toArray(),b.toArray()],before);
});

test('nominal integer time with floating-point residue reaches the intended key',options,()=>{
  const rig=fixture(),[a,b]=keys(),position=1.16*25,amount=position-Math.floor(position);
  assert.ok(amount<1&&amount>.999999999);
  rig.interpolate({leftHand:a},{leftHand:b},amount);
  assert.ok(rig.bones.leftHand.quaternion.clone().normalize().angleTo(b.clone().normalize())<1e-7);
});

test('authored endpoints and missing-bone resting samples remain component-exact',options,()=>{
  const rig=fixture(),[a,b]=keys();
  for(const amount of [0,1]){
    rig.interpolate({leftHand:a},{leftHand:b},amount);
    assert.deepEqual(rig.bones.leftHand.quaternion.toArray(),amount?b.toArray():a.toArray());
  }
  const rest=rig.rest.leftHand.quaternion;
  rig.interpolate({}, {}, .5);
  assert.deepEqual(rig.bones.leftHand.quaternion.toArray(),rest.clone().slerp(rest,.5).toArray());
});

test('identical nonunit keys retain exact values rather than modifying a resting pose',options,()=>{
  const rig=fixture(),[a]=keys();
  rig.interpolate({leftHand:a},{leftHand:a.clone()},.5);
  assert.deepEqual(rig.bones.leftHand.quaternion.toArray(),q0);
});

test('unit keys and backward seeks preserve ordinary shortest-arc rotations',options,()=>{
  const rig=fixture(),a=new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1,0,0),.4);
  const b=new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0,1,0),-.8);
  for(const amount of [.25,.5,.75,.5,0,1]){
    rig.interpolate({leftHand:a},{leftHand:b},amount);
    assert.ok(rig.bones.leftHand.quaternion.angleTo(a.clone().slerp(b,amount))<1e-7);
  }
});
