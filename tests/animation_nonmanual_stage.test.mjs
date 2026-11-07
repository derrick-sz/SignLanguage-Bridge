import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {PlaybackClock} from '../slbridge/animation/web/player-core.mjs';
import {applyExpression} from '../slbridge/animation/web/expression.mjs';
import * as jaw from '../slbridge/animation/web/nonmanual.mjs';

const source=fs.readFileSync(new URL('../slbridge/animation/web/stage.js',import.meta.url),'utf8')
  .replace(/^import .*;\r?\n/gm,'').replace('export class MotionStage','class MotionStage')+'\nMotionStage;';
const Stage=vm.runInNewContext(source,{performance:{now:()=>1000},PlaybackClock,applyExpression,...jaw});
function harness(){
  const stage=Object.create(Stage.prototype),tracks={original:[{},{}],improved:[{},{}],bounds:{},hands:{}};
  Object.assign(stage,{clock:new PlaybackClock(),ready:Promise.resolve(),cache:new Map([['clip',tracks]]),profile:'improved',
    presentation:{avatar_sha256:jaw.JAW_AVATAR_SHA256,pilots:[]},renderer:{render(){}},
    rig:{jawCalibrationAvailable:true,bones:{},reset(){},resetExpression(){},interpolate(){},applyCalibratedJaw(angle){stage.angle=angle;}},
    originalRig:{interpolate(){},reset(){}},originalModel:{updateMatrixWorld(){}},model:{updateMatrixWorld(){}},poseNeutral(){}});
  const clip={frames:2,fps:30,poses:Array.from({length:2},()=>Array.from({length:55},(_,i)=>i===22?[0.08,0,0]:[0,0,0])),
    template_sha256:jaw.JAW_TEMPLATE_SHA256,nonmanual_candidate:{profile:jaw.JAW_PROFILE,source_joint:22,
    source_template_sha256:jaw.JAW_TEMPLATE_SHA256,target_avatar_sha256:jaw.JAW_AVATAR_SHA256,status:'research_candidate',export_supported:false}};
  return {stage,data:{clip,parents:[]}};
}

test('automatic jaw starts only for explicit clips; manual override, brows, reset and toggles preserve time',async()=>{
  const {stage,data}=harness();
  await stage.loadMotion(data,'clip',()=>true,'source');
  assert.equal(stage.nonmanualAvailable,true);
  assert.equal(stage.nonmanualEnabled,true);
  assert.ok(Math.abs(stage.angle-0.08)<1e-10);
  stage.seek(0.02);const time=stage.clock.position;
  stage.setExpression({browRaise:0.5});assert.ok(Math.abs(stage.angle-0.08)<1e-10);
  stage.setExpression({jawOpen:0.75});assert.ok(Math.abs(stage.angle-0.12)<1e-10);
  stage.setExpression({blink:0.1});assert.ok(Math.abs(stage.angle-0.12)<1e-10);
  stage.setExpression({});assert.ok(Math.abs(stage.angle-0.08)<1e-10);
  stage.setNonmanualEnabled(false);assert.equal(stage.angle,0);
  stage.setNonmanualEnabled(true);assert.ok(Math.abs(stage.angle-0.08)<1e-10);
  assert.equal(stage.clock.position,time);
  const report=stage.exportReview();assert.equal(report.nonmanual.export_supported,false);
  stage.stop();assert.equal(stage.nonmanualAvailable,false);assert.equal(stage.nonmanualEnabled,false);
  const old={...data,clip:{...data.clip,nonmanual_candidate:undefined}};
  await stage.loadMotion(old,'clip',()=>true,'old');
  assert.equal(stage.nonmanualAvailable,false);assert.equal(stage.angle,0);
});

test('cancelled initialization cannot restore a pending jaw track after Stop',async()=>{
  const {stage,data}=harness();let resolve,current=true;
  stage.ready=new Promise(r=>{resolve=r;});
  const pending=stage.loadMotion(data,'clip',()=>current,'source');
  current=false;stage.stop();resolve();
  assert.equal(await pending,false);
  assert.equal(stage.nonmanualAvailable,false);assert.equal(stage.nonmanualEnabled,false);
  assert.equal(stage.clock.duration,0);
});

test('manual v11 summary records its own bounded angle while automatic source jaw stays unavailable',async()=>{
  const {stage,data}=harness();
  stage.presentation.avatar_sha256='e4a5766d98ab53871415da3ef4609c4848386c8b4de2958d5e420a8d7524d476';
  stage.rig.jawCalibrationAvailable=false;stage.rig.manualJawMaxAngle=.04;
  await stage.loadMotion(data,'clip',()=>true,'source');
  assert.equal(stage.nonmanualAvailable,false);assert.equal(stage.nonmanualEnabled,false);
  stage.setNonmanualEnabled(true);assert.equal(stage.nonmanualEnabled,false);
  stage.expression={jawOpen:.5};
  assert.equal(stage.jawAngle(0),.02);
  assert.equal(stage.exportReview().nonmanual.angle_radians,.02);
  assert.equal(stage.exportReview().nonmanual.profile,null);
});
