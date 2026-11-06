import test from 'node:test';
import assert from 'node:assert/strict';
import * as geometry from '../slbridge/animation/web/motion-geometry.mjs';

const pose=()=>Array.from({length:55},()=>[0,0,0]);
function filter(frames,fps=25){
  assert.equal(typeof geometry.stabilizeSourcePoses,'function','source-space temporal processing is missing');
  return geometry.stabilizeSourcePoses(frames,fps);
}
const degrees=v=>Math.hypot(...v)*180/Math.PI;

test('source filtering reduces a one-frame finger impulse before kinematic targets are computed',()=>{
  const frames=Array.from({length:9},pose);frames[4][40]=[Math.PI/3,0,0];
  const snapshot=structuredClone(frames),result=filter(frames);
  assert.ok(degrees(result[4][40])<35,'60-degree impulse should be reduced below 35 degrees');
  assert.ok(degrees(result[4][40])>20,'bounded repair must not flatten the pose to neutral');
  assert.deepEqual(frames,snapshot);assert.deepEqual(result[0],frames[0]);assert.deepEqual(result.at(-1),frames.at(-1));
});

test('centered filtering retains sustained fast rotation and long held handshapes',()=>{
  const moving=Array.from({length:9},(_,i)=>{const p=pose();p[16]=[i*0.2,0,0];return p;});
  const result=filter(moving);
  result.forEach((p,i)=>assert.ok(Math.abs(p[16][0]-i*0.2)<1e-9));
  const holds=Array.from({length:21},(_,i)=>{const p=pose();p[25]=[i<10?0:1,0,0];return p;});
  const smoothed=filter(holds);
  assert.deepEqual(smoothed[2][25],[0,0,0]);assert.ok(Math.abs(smoothed[18][25][0]-1)<1e-9);
});

test('source filtering follows the shortest arc across equivalent axis-angle encodings',()=>{
  const frames=Array.from({length:5},pose);
  for(const [i,deg] of [175,177,179,-179,-177].entries())frames[i][20]=[0,0,deg*Math.PI/180];
  const result=filter(frames);
  assert.ok(Math.abs(degrees(result[2][20])-179)<1e-8);
  assert.ok(Math.abs(degrees(result[3][20])-179)<1e-8);
});

test('source repair bounds a large correction and validates complete pose frames',()=>{
  const frames=Array.from({length:9},pose);frames[4][40]=[170*Math.PI/180,0,0];
  const result=filter(frames);assert.ok(degrees(result[4][40])>=139.999999);
  assert.ok(result.flat(2).every(Number.isFinite));
  assert.deepEqual(filter([pose()]),[pose()]);assert.deepEqual(filter([pose(),pose()]),[pose(),pose()]);
  assert.throws(()=>filter(frames,0));assert.throws(()=>filter([pose().slice(1)]));
  const invalid=[pose()];invalid[0][20][0]=NaN;assert.throws(()=>filter(invalid));
});
