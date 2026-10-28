import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import vm from "node:vm";
import { PlaybackClock } from "../slbridge/animation/web/player-core.mjs";

// Exercise the real stage controls without WebGL or requestAnimationFrame ticks.
const source = (await readFile(new URL("../slbridge/animation/web/stage.js", import.meta.url), "utf8"))
  .replace(/^import .*;\r?\n/gm, "").replace("export class MotionStage", "class MotionStage") + "\nMotionStage;";
const Stage = vm.runInNewContext(source, { performance: { now: () => 1000 }, PlaybackClock });
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
  stage.stop();
  assert.equal(updates.at(-1).time, 0);
  assert.equal(updates.at(-1).duration, 0);
});
