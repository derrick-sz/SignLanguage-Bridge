import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { setImmediate } from 'node:timers/promises';

const root = fileURLToPath(new URL('../', import.meta.url));
const { PlaybackGeneration } = await import(pathToFileURL(path.join(root, 'slbridge/animation/web/player-core.mjs')));
const source = fs.readFileSync(path.join(root, 'slbridge/animation/web/workspace.js'), 'utf8')
  .replace(/^import .*;\r?\n/gm, '').replace(/\binit\(\);\s*$/, '');
const deferred = () => { let resolve; const promise = new Promise(r => { resolve = r; }); return { promise, resolve }; };
const plan = { can_play: true, items: [{ asset_id: 'hello' }] };
const asset = { id: 'hello', text: 'hello', fps: 25, group: 'reference', sha256: 'hash', aliases: [], duration: 1 };
function createHarness(respond) {
  const nodes = new Map();
  const makeNode = (id = '') => ({ disabled: false, value: id === 'speed' ? '1' : 'hello', dataset: {}, hidden: false,
    listeners: {}, classList: { toggle() {} }, appendChild() {}, replaceChildren() {},
    addEventListener(type, callback) { this.listeners[type] = callback; },
    click() { if (!this.disabled) return this.listeners.click?.(); },
    dispatch(type) { return this.listeners[type]?.({}); },
  });
  const element = id => { if (!nodes.has(id)) nodes.set(id, makeNode(id)); return nodes.get(id); };
  class MotionStage {
    constructor() { this.clock = { playing: false }; this.tracks = { mappedBones: 40, corrected: 0 }; this.plays = 0; this.ready = Promise.resolve(); }
    stop() { this.clock.playing = false; } async loadMotion() { return true; }
    setSpeed() {} play() { this.plays++; this.clock.playing = true; } pause() { this.clock.playing = false; }
    setProfile() {} setView() {} seek() {}
  }
  const context = vm.createContext({ PlaybackGeneration, MotionStage, AbortController,
    document: { getElementById: element, querySelectorAll: () => [], createElement: () => makeNode() },
    window: { addEventListener() {} },
    fetch: async (url, options) => ({ ok: true, json: async () => respond(url, options) }),
  });
  vm.runInContext(source + '\nglobalThis.review = {preview, stage, gate, seed: a => {assets = a;}, setOrigin: o => {origin = o; updateInput();}};', context);
  context.review.seed([asset]);
  return { ...context.review, element };
}

test('Stop remains actionable while plan is pending and rejects its late response', async () => {
  const response = deferred();
  const h = createHarness(url => url.endsWith('/plans') ? response.promise : { clip: { frames: 2, fps: 25 } });
  const pending = h.preview();
  const stopDisabled = h.element('stop').disabled;
  const generation = h.gate.value;
  h.element('stop').click();
  response.resolve(plan);
  await pending;
  assert.equal(stopDisabled, false, 'user must be able to stop an in-flight playback request');
  assert.notEqual(h.gate.value, generation, 'Stop must invalidate the in-flight generation');
  assert.equal(h.stage.plays, 0, 'a late plan must not start playback after Stop');
});

test('Stop remains actionable during motion fetch and rejects the late motion', async () => {
  const response = deferred();
  const h = createHarness(url => url.endsWith('/plans') ? plan : response.promise);
  const pending = h.preview();
  await setImmediate();
  const stopDisabled = h.element('stop').disabled;
  h.element('stop').click();
  response.resolve({ clip: { frames: 2, fps: 25 } });
  await pending;
  assert.equal(stopDisabled, false);
  assert.equal(h.stage.plays, 0);
});

test('Editing input invalidates a late plan independently of transport abort', async () => {
  const response = deferred();
  const h = createHarness(url => url.endsWith('/plans') ? response.promise : { clip: { frames: 2, fps: 25 } });
  const pending = h.preview();
  h.element('input').value = 'hello 2';
  h.element('input').dispatch('input');
  response.resolve(plan);
  await pending;
  assert.equal(h.stage.plays, 0);
});

test('Scrubbing keeps requested time when pausing publishes the current frame', async () => {
  const h = createHarness(url => url.endsWith('/plans') ? plan : { clip: { frames: 2, fps: 25 } });
  await h.preview();
  h.element('timeline').value = '2.4';
  h.stage.pause = () => { h.element('timeline').value = '0.08'; };
  h.stage.seek = time => { h.stage.requested = time; };
  h.element('timeline').dispatch('input');
  assert.equal(h.stage.requested, 2.4);
});

test('Editing a recognition preserves raw evidence; a new catalog origin clears it', () => {
  const h = createHarness(() => plan);
  h.setOrigin({kind: 'recognition', request_id: 'r1', raw_text: 'raw prediction'});
  h.element('input').value = 'corrected input';
  h.element('input').dispatch('input');
  assert.equal(h.element('origin-label').textContent, '已编辑 · 原始输出保留');
  assert.equal(h.element('raw-result').textContent, '原始输出：raw prediction');
  assert.equal(h.element('raw-result').hidden, false);
  h.setOrigin({kind: 'catalog', asset_id: 'hello'});
  assert.equal(h.element('raw-result').hidden, true);
  assert.equal(h.element('raw-result').textContent, '');
});
