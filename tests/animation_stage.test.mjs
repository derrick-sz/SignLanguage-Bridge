import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import vm from "node:vm";
import { PlaybackClock, comparisonViewports } from "../slbridge/animation/web/player-core.mjs";

// Exercise the real stage controls without WebGL or requestAnimationFrame ticks.
const source = (await readFile(new URL("../slbridge/animation/web/stage.js", import.meta.url), "utf8"))
  .replace(/^import .*;\r?\n/gm, "").replace("export class MotionStage", "class MotionStage") + "\nMotionStage;";
const Stage = vm.runInNewContext(source, { performance: { now: () => 1000 }, PlaybackClock, comparisonViewports });
test("scrubbing and pausing publish a rendered frame even when RAF is throttled", () => {
  const stage = Object.create(Stage.prototype), frames = [], updates = [];
  stage.clock = new PlaybackClock(); stage.clock.load(8);
  stage.renderer = { render: () => frames.push(true) };
  stage.applyTime = time => { stage.appliedTime = time; };
  stage.onTick = state => updates.push(state);
  stage.rig = { reset() {} };
  stage.seek(2.4);
  assert.equal(stage.appliedTime, 2.4);
  assert.equal(updates.at(-1)?.time, 2.4);
  assert.equal(frames.length, 1);
  stage.play();
  assert.equal(updates.at(-1).playing, true);
  stage.pause();
  assert.equal(updates.at(-1).playing, false);
  stage.tracks = { original: ["old frame"], improved: ["old frame"] };
  stage.stop();
  assert.equal(updates.at(-1).time, 0);
  assert.equal(updates.at(-1).duration, 0);
  assert.equal(stage.tracks, null, "display switches must not restore a stopped clip");
});

test('controls remain actionable while avatar initialization has not completed', () => {
  const stage=Object.create(Stage.prototype),updates=[];
  stage.clock=new PlaybackClock();stage.clock.load(8);stage.clock.play(0);
  stage.renderer={setScissorTest(){},setViewport(){},setScissor(){},render(){throw new Error('incomplete scene must not render');}};
  stage.camera={isPerspectiveCamera:true};stage.container={clientWidth:900,clientHeight:500,dataset:{}};
  stage.onTick=state=>updates.push(state);
  assert.doesNotThrow(()=>stage.setComparison(true));
  assert.doesNotThrow(()=>stage.setView('side'));
  assert.doesNotThrow(()=>stage.stop());
  assert.equal(stage.comparison,true);assert.equal(stage.view,'side');
  assert.equal(stage.clock.duration,0);assert.equal(updates.at(-1).playing,false);
});

test('source temporal repair belongs to its declared clip and is recorded in the review', async()=>{
  const calls=[];
  const TestStage=vm.runInNewContext(source,{performance:{now:()=>1000},PlaybackClock,comparisonViewports,
    createT08Retarget:()=>({compile:(_clip,options)=>{calls.push(options);return [{},{}];},mappedBones:40,diagnostics:null})});
  const stage=Object.create(TestStage.prototype),profile='source-quaternion-temporal-v1';
  Object.assign(stage,{ready:Promise.resolve(),cache:new Map(),profile:'improved',
    presentation:{processing_version:'repair-candidate-v2',pilots:[{asset_id:'repair',temporal_profile:profile}]},
    rig:{capture:()=>({}),interpolate(){},bones:{}},originalRig:{capture:()=>({}),interpolate(){},bones:{}},
    model:{updateMatrixWorld(){}},originalModel:{updateMatrixWorld(){}},clock:new PlaybackClock(),refresh(){}});
  const data={clip:{frames:2,fps:25,poses:Array.from({length:2},()=>Array.from({length:55},()=>[0,0,0]))},parents:[-1,...Array(54).fill(0)],restJoints:Array.from({length:55},()=>[0,0,0])};
  await stage.loadMotion(data,'repair',()=>true,'a'.repeat(64));
  assert.equal(calls[1].stabilizationProfile,profile);
  assert.equal(stage.exportReview().temporal_profile,profile);
  assert.equal(stage.exportReview().processing_version,'repair-candidate-v2');
  await stage.loadMotion(data,'ordinary',()=>true,'b'.repeat(64));
  assert.equal(calls[3].stabilizationProfile,'quaternion-temporal-adaptive-v1','other clips retain their existing filter');
  assert.equal(stage.exportReview().temporal_profile,'quaternion-temporal-adaptive-v1');
});
