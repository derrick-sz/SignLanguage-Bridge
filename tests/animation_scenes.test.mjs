import test from 'node:test';
import assert from 'node:assert/strict';
import { ScenePreview, appendExpression, renderSource } from '../slbridge/animation/web/scenes-controller.mjs';

const deferred = () => { let resolve, reject; const promise = new Promise((a,b) => {resolve=a;reject=b;}); return {promise,resolve,reject}; };
const plan = {plan_id:'p',can_play:true,items:[{text:'真实原句',sample_id:'train/1',frames:30}],unmatched:[]};
const response = value => ({ok:true,json:async()=>value});
function setup(fetcher) {
  const calls=[];
  const stage={stop:()=>calls.push('stop'),loadMotion:async()=>true,play:()=>calls.push('play')};
  const notices=[];
  return {calls,notices,stage,controller:new ScenePreview({stage,fetcher,onState:s=>notices.push(s)})};
}
test('input change aborts old plan; ignored abort response never plays',async()=>{
  const pending=deferred();let signal;
  const s=setup(async(url,options)=>{signal=options.signal;return pending.promise;});
  const run=s.controller.preview('原句',true);s.controller.invalidate();
  assert.equal(signal.aborted,true);pending.resolve(response(plan));await run;
  assert.equal(s.calls.includes('play'),false);
});
test('stop while motion loads prevents late playback and old failure notices',async()=>{
  const pending=deferred();let count=0;
  const s=setup(async()=>++count===1?response(plan):pending.promise);
  const run=s.controller.preview('原句',true);await new Promise(r=>setImmediate(r));
  s.controller.invalidate();const before=s.notices.length;
  pending.reject(new Error('old failure'));await run;
  assert.equal(s.calls.includes('play'),false);assert.equal(s.notices.length,before);
});
test('newer preview wins when old stage load finishes',async()=>{
  const pending=deferred();let loads=0;
  const s=setup(async url=>response(url.endsWith('/motion')?{clip:{id:'clip'},sha256:'sha'}:plan));
  s.stage.loadMotion=async(data,id,current)=>{if(++loads===1){await pending.promise;return current();}return current();};
  const old=s.controller.preview('旧',true);await new Promise(r=>setImmediate(r));
  await s.controller.preview('新',true);pending.resolve();await old;
  assert.equal(s.calls.filter(c=>c==='play').length,1);
});
test('missing segment blocks entire preview and retains exact source text',async()=>{
  let count=0;const blocked={...plan,can_play:false,unmatched:['不匹配！'],reason:'缺少原句'};
  const s=setup(async()=>{count++;return response(blocked);});
  await s.controller.preview('真实原句\n不匹配！',true);
  assert.equal(count,1);assert.equal(s.calls.includes('play'),false);
  assert.equal(s.controller.plan,blocked);
  const node={textContent:''};renderSource(node,{text:'<script>原句？</script>'});
  assert.equal(node.textContent,'<script>原句？</script>');
});
test('confirmation required and append uses newline with six sentence bound',async()=>{
  const s=setup(()=>{throw new Error('must not fetch');});await s.controller.preview('原句',false);
  assert.equal(s.calls.includes('play'),false);
  assert.equal(appendExpression('你好。','谢谢！'),'你好。\n谢谢！');
  assert.throws(()=>appendExpression('一\n二\n三\n四\n五\n六','七'),/6/);
});
test('explicit preview sends both confirmations and plays only after guarded motion load',async()=>{
  const requests=[];
  const motion={clip:{id:'scene/p'},sha256:'source-sha'};
  const s=setup(async(url,options)=>{requests.push({url,options});return response(url.endsWith('/motion')?motion:plan);});
  let loaded;
  s.stage.loadMotion=async(data,id,current,sha)=>{loaded={data,id,current:current(),sha};return true;};
  await s.controller.preview('真实原句',true);
  assert.deepEqual(JSON.parse(requests[0].options.body),{text:'真实原句',confirmed:true,research_preview:true});
  assert.equal(requests[1].url,'/v1/animation/scenes/plans/p/motion');
  assert.deepEqual(loaded,{data:motion,id:'scene/p',current:true,sha:'source-sha'});
  assert.equal(s.calls.at(-1),'play');assert.equal(s.notices.at(-1).status,'playing');
});
test('overlong input is rejected before any network request', async () => {
  const states = [];
  let calls = 0;
  const preview = new ScenePreview({stage:{stop(){}},fetcher:async()=>{calls++;throw new Error('should not fetch');},onState:s=>states.push(s)});
  await preview.preview('字'.repeat(501),true);
  assert.equal(calls,0);
  assert.equal(states.at(-1).status,'error');
  assert.match(states.at(-1).message,/500/);
});
test('browser fetch keeps its global receiver', async () => {
  let calls=0;
  function fetcher() {
    assert.equal(this,globalThis);
    calls++;
    return Promise.resolve({ok:true,json:async()=>({can_play:false,reason:'blocked',unmatched:['unknown']})});
  }
  const states=[];
  const preview=new ScenePreview({stage:{stop(){}},fetcher,onState:s=>states.push(s)});
  await preview.preview('unknown',true);
  assert.equal(calls,1);
  assert.equal(states.at(-1).status,'blocked');
});
