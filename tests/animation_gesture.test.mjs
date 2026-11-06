import test from 'node:test';
import assert from 'node:assert/strict';
import * as geometry from '../slbridge/animation/web/motion-geometry.mjs';

const pose=()=>Array.from({length:55},()=>[0,0,0]);
const clip=()=>({fps:30,poses:[pose(),pose(),pose()],refined_poses:[pose(),pose(),pose()],gesture_refinement:{profile:'soke-gesture-refinement-v1'}});

test('gesture profile keeps raw mode and explicitly selects the refined source for rotations and FK',()=>{
  assert.equal(typeof geometry.selectSourcePoses,'function');
  const value=clip();value.refined_poses[1][20]=[0,0,0.8];
  const copy=structuredClone(value);
  assert.equal(geometry.selectSourcePoses(value,{stabilize:false,profile:'soke-gesture-refinement-v1'}),value.poses);
  assert.equal(geometry.selectSourcePoses(value,{stabilize:true,profile:'soke-gesture-refinement-v1'}),value.refined_poses);
  assert.deepEqual(value,copy);
});

test('refined source rejects incomplete, nonfinite or wrong-profile data',()=>{
  assert.equal(typeof geometry.selectSourcePoses,'function');
  for(const alter of [c=>c.refined_poses.pop(),c=>c.refined_poses[1][25][0]=NaN,c=>c.gesture_refinement.profile='unknown']){
    const value=clip();alter(value);
    assert.throws(()=>geometry.selectSourcePoses(value,{stabilize:true,profile:'soke-gesture-refinement-v1'}));
  }
});
