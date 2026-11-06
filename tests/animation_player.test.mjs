import test from "node:test";
import assert from "node:assert/strict";
import { PlaybackClock, PlaybackGeneration, signingCamera } from "../slbridge/animation/web/player-core.mjs";

test("pause, speed changes and seeking preserve the same timeline", () => {
  const clock = new PlaybackClock();
  clock.load(10);
  clock.play(1000);
  assert.equal(clock.time(3000), 2);
  clock.pause(3000);
  assert.equal(clock.time(9000), 2);
  clock.setSpeed(0.5, 9000);
  clock.play(9000);
  assert.equal(clock.time(11000), 3);
  clock.seek(7, 11000);
  assert.equal(clock.time(13000), 8);
  clock.seek(-5, 13000);
  assert.equal(clock.time(13000), 0);
  clock.seek(100, 13000);
  assert.equal(clock.time(13000), 10);
});

test("completion, replay and stop have explicit behavior", () => {
  const clock = new PlaybackClock();
  clock.load(2);
  clock.play(0);
  assert.equal(clock.time(3000), 2);
  assert.equal(clock.playing, false);
  clock.play(4000);
  assert.equal(clock.time(4500), 0.5);
  clock.stop();
  assert.equal(clock.time(8000), 0);
  assert.equal(clock.playing, false);
  assert.throws(() => clock.setSpeed(0, 0));
  assert.throws(() => clock.load(NaN));
});

test("edit, stop and superseding identical plays invalidate late callbacks", () => {
  const gate = new PlaybackGeneration();
  const first = gate.begin();
  assert.ok(gate.current(first));
  const second = gate.begin();
  assert.ok(!gate.current(first));
  assert.ok(gate.current(second));
  gate.cancel();
  assert.ok(!gate.current(second));
  assert.ok(gate.current(gate.begin()));
});

test("camera fitting retains raised hands and wide signing space at narrow aspect", () => {
  const bounds = { minX: -1.2, maxX: 1.2, minY: 0.5, maxY: 2.1 };
  const camera = signingCamera(bounds, 0.7, 34);
  const halfHeight = camera.distance * Math.tan(34 * Math.PI / 360);
  assert.ok(halfHeight >= (bounds.maxY - bounds.minY) / 2);
  assert.ok(halfHeight * 0.7 >= (bounds.maxX - bounds.minX) / 2);
  assert.equal(camera.centerY, 1.3);
  assert.throws(() => signingCamera(bounds, 0, 34));
});
