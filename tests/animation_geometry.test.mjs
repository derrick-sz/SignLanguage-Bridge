import test from 'node:test';
import assert from 'node:assert/strict';

const geometry = await import('../slbridge/animation/web/motion-geometry.mjs').catch(() => ({}));
const near = (a, b) => a.forEach((x, i) => assert.ok(Math.abs(x - b[i]) < 1e-7, `${a} != ${b}`));
const length = (a, b) => Math.hypot(...a.map((x, i) => x - b[i]));

test('source FK rotates child positions around their actual parent and leaves source intact', () => {
  assert.equal(typeof geometry.forwardKinematics, 'function');
  const pose = [[0, 0, Math.PI / 2], [0, 0, 0], [0, 0, 0]];
  const rest = [[2, 0, 0], [3, 0, 0], [4, 0, 0]];
  const snapshot = JSON.stringify({ pose, rest });
  const result = geometry.forwardKinematics(pose, [-1, 0, 1], rest);
  near(result.positions[0], [2, 0, 0]);
  near(result.positions[1], [2, 1, 0]);
  near(result.positions[2], [2, 2, 0]);
  assert.equal(JSON.stringify({ pose, rest }), snapshot);
});

test('source FK rejects forward parents and non-finite rotations instead of corrupting a track', () => {
  assert.equal(typeof geometry.forwardKinematics, 'function');
  assert.throws(() => geometry.forwardKinematics([[0, 0, 0], [0, 0, 0]], [-1, 2], [[0, 0, 0], [1, 0, 0]]));
  assert.throws(() => geometry.forwardKinematics([[0, NaN, 0]], [-1], [[0, 0, 0]]));
});

test('two-bone IK reaches an attainable goal and preserves both segment lengths and bend side', () => {
  assert.equal(typeof geometry.solveTwoBone, 'function');
  const goal = [1, 1, 0];
  const result = geometry.solveTwoBone([0, 0, 0], [0, 1, 0], [1, 1, 0], goal);
  near(result.wrist, goal);
  assert.ok(Math.abs(length([0, 0, 0], result.elbow) - 1) < 1e-7);
  assert.ok(Math.abs(length(result.elbow, result.wrist) - 1) < 1e-7);
  assert.ok(result.elbow[1] > result.elbow[0]);
  assert.equal(result.clamped, false);
});

test('unreachable and coincident goals remain finite without stretching bones', () => {
  assert.equal(typeof geometry.solveTwoBone, 'function');
  for (const target of [[8, 0, 0], [0, 0, 0]]) {
    const result = geometry.solveTwoBone([0, 0, 0], [0, 1, 0], [1, 1, 0], target);
    assert.ok([...result.elbow, ...result.wrist].every(Number.isFinite));
    assert.ok(Math.abs(length([0, 0, 0], result.elbow) - 1) < 1e-7);
    assert.ok(Math.abs(length(result.elbow, result.wrist) - 1) < 1e-7);
    assert.equal(result.clamped, true);
  }
  assert.throws(() => geometry.solveTwoBone([0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]));
});
