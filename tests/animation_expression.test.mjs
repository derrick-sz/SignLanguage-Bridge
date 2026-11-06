import test from 'node:test';
import assert from 'node:assert/strict';
const expression = await import('../slbridge/animation/web/expression.mjs').catch(() => ({}));

test('missing facial data stays neutral instead of deriving an emotion from text', () => {
  assert.equal(typeof expression.sampleExpression, 'function');
  assert.deepEqual(expression.sampleExpression(null, 2, 5), { browRaise:0, browLower:0, blink:0, jawOpen:0 });
});

test('expression sampling is deterministic on forward and backward seeks', () => {
  assert.equal(typeof expression.sampleExpression, 'function');
  const track = [{time:0,channels:{browRaise:0}},{time:2,channels:{browRaise:0.8}},{time:4,channels:{browRaise:0}}];
  assert.equal(expression.sampleExpression(track, 1, 4).browRaise, 0.4);
  expression.sampleExpression(track, 3, 4);
  assert.equal(expression.sampleExpression(track, 1, 4).browRaise, 0.4);
  assert.equal(expression.sampleExpression(track, 5, 4).browRaise, 0);
});

test('untrusted face tracks cannot apply NaN, unknown channels or excessive intensities', () => {
  assert.equal(typeof expression.sampleExpression, 'function');
  for (const channels of [{browRaise:NaN},{jawOpen:2},{unknown:1}]) {
    assert.throws(() => expression.sampleExpression([{time:0,channels}], 0, 4));
  }
  assert.throws(() => expression.sampleExpression([{time:2,channels:{}},{time:1,channels:{}}], 1, 4));
});

test('rig profiles reject invalid bindings and large axis corrections', () => {
  assert.equal(typeof expression.validateRigProfile, 'function');
  const profile = {id:'test',bones:{leftHand:['Hand.L']},corrections:{leftHand:[0,0.1,0]}};
  assert.equal(expression.validateRigProfile(profile), profile);
  assert.throws(() => expression.validateRigProfile({...profile,bones:{leftHand:['']}}));
  assert.throws(() => expression.validateRigProfile({...profile,corrections:{leftHand:[0,9,0]}}));
});
