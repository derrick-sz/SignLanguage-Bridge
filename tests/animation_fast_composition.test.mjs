import test from 'node:test';
import assert from 'node:assert/strict';
import { FastCompositionController } from '../slbridge/animation/web/fast-composition-controller.mjs';

const text='明天我去医院。';
const digest='a'.repeat(64);
const response=(data,ok=true)=>({ok,json:async()=>data});
const ready=(extra={})=>({status:'ready',input_text:text,generation_text:text,glosses:['明天','我','去','医院'],
  timings_ms:{planning:40,composition:15},language_review:'NOT_RUN',
  motion:{clip:{id:'word-composition-1',text,source_kind:'retrieved_word_composition'},sha256:digest},...extra});
const deferred=()=>{let resolve,reject;const promise=new Promise((yes,no)=>{resolve=yes;reject=no;});return {promise,resolve,reject};};
function setup(fetcher,now=()=>100){
  const states=[];
  const stage={plays:0,loads:[],stop(){},pause(){},seek(value){this.time=value;},play(){this.plays++;},
    async loadMotion(...args){this.loads.push(args);return args[2]();}};
  const controller=new FastCompositionController({stage,fetcher,onState:state=>states.push(state),now});
  return {controller,stage,states};
}

test('free text requests composition once and current verified result automatically plays',async()=>{
  const calls=[];let time=100;
  const {controller,stage,states}=setup(async function(url,options){
    assert.equal(this,globalThis);calls.push([url,options]);time=245;return response(ready());
  },()=>time);
  await controller.generate(text);
  assert.equal(calls.length,1);assert.equal(calls[0][0],'/v1/animation/composition');
  assert.deepEqual(JSON.parse(calls[0][1].body),{text});assert.equal(calls[0][1].method,'POST');
  assert.ok(calls[0][1].signal instanceof AbortSignal);
  assert.equal(stage.loads.length,1);assert.equal(stage.loads[0][3],digest);assert.equal(stage.plays,1);
  assert.equal(states.at(-1).status,'ready');assert.equal(states.at(-1).subtitle,text);
  assert.deepEqual(states.at(-1).glosses,ready().glosses);
  assert.deepEqual(states.at(-1).timings_ms,ready().timings_ms);
  assert.equal(states.at(-1).client_play_request_ms,145);
});
test('invalid text never requests and blocks previous playback',async()=>{
  let requests=0;const {controller,stage,states}=setup(async()=>{requests++;return response(ready());});
  for(const value of ['', ' ', 'a'.repeat(201), 'a\u0000b', 12])await controller.generate(value);
  assert.equal(requests,0);assert.equal(stage.plays,0);assert.equal(states.at(-1).status,'error');
});
test('multiline original input is retained while verified normalized caption is used',async()=>{
  const original='你好\n世界';const caption='你好 世界';
  const {controller,states}=setup(async()=>response(ready({input_text:original,generation_text:caption,
    motion:{clip:{id:'words-2',text:caption,source_kind:'retrieved_word_composition'},sha256:digest}})));
  await controller.generate(original);assert.equal(states.at(-1).subtitle,caption);
});
test('missing words block the whole result and expose gaps without loading or partial playback',async()=>{
  const {controller,stage,states}=setup(async()=>response(ready({status:'blocked',motion:undefined,missing_glosses:['医院']})));
  await controller.generate(text);assert.equal(stage.loads.length,0);assert.equal(stage.plays,0);
  assert.equal(controller.loaded,false);assert.equal(states.at(-1).status,'blocked');
  assert.deepEqual(states.at(-1).missing_glosses,['医院']);assert.equal(states.at(-1).subtitle,undefined);
});
test('blocked response containing motion is rejected instead of playing it',async()=>{
  const {controller,stage,states}=setup(async()=>response(ready({status:'blocked',missing_glosses:['医院']})));
  await controller.generate(text);assert.equal(stage.loads.length,0);assert.equal(states.at(-1).status,'error');
});
test('omitted meaning needs review without motion and cannot replay a previous successful result',async()=>{
  let calls=0;
  const {controller,stage,states}=setup(async()=>response(++calls===1?ready():ready({
    status:'needs_review',motion:undefined,glosses:['明天','我','去'],omitted_terms:['医院']})));
  await controller.generate(text);assert.equal(stage.plays,1);
  await controller.generate(text);controller.play();
  assert.equal(stage.loads.length,1);assert.equal(stage.plays,1);assert.equal(controller.loaded,false);
  assert.equal(states.at(-1).status,'needs_review');assert.deepEqual(states.at(-1).omitted_terms,['医院']);
  assert.equal(states.at(-1).subtitle,undefined);
  assert.equal(states.at(-1).message,'文字规划可能遗漏：医院。请核对或缩短输入。');
  assert.equal(states.at(-1).missing_glosses,undefined);
});
test('review response with motion or malformed omitted terms is rejected rather than partially played',async()=>{
  for(const extra of [{motion:ready().motion,omitted_terms:['医院']},
    {omitted_terms:[]},{omitted_terms:['']},{omitted_terms:'医院'},{omitted_terms:[null]}]) {
    const {controller,stage,states}=setup(async()=>response(ready({status:'needs_review',motion:undefined,...extra})));
    await controller.generate(text);assert.equal(stage.loads.length,0);assert.equal(stage.plays,0);
    assert.equal(states.at(-1).status,'error');
  }
});
test('late review response cannot replace a newer complete composition',async()=>{
  const older=deferred();let calls=0;
  const {controller,stage,states}=setup(async()=>++calls===1?older.promise:response(ready()));
  const pending=controller.generate(text);await controller.generate(text);
  older.resolve(response(ready({status:'needs_review',motion:undefined,omitted_terms:['医院']})));
  await pending;assert.equal(stage.plays,1);assert.equal(states.at(-1).status,'ready');
});
test('editing or switching mode during request aborts and ignores late HTTP result',async()=>{
  const pending=deferred();let signal;
  const {controller,stage,states}=setup(async(url,options)=>{signal=options.signal;return pending.promise;});
  const work=controller.generate(text);controller.invalidate('模式已改变。');assert.equal(signal.aborted,true);
  pending.resolve(response(ready()));await work;
  assert.equal(stage.loads.length,0);assert.equal(stage.plays,0);assert.equal(states.at(-1).message,'模式已改变。');
});
test('cancel during loading invalidates the stage guard and cannot publish or play late',async()=>{
  const loading=deferred(),entered=deferred();const {controller,stage,states}=setup(async()=>response(ready()));
  let current;
  stage.loadMotion=async(data,id,guard)=>{current=guard;entered.resolve();await loading.promise;return guard();};
  const work=controller.generate(text);await entered.promise;controller.cancel();assert.equal(current(),false);
  loading.resolve();await work;assert.equal(stage.plays,0);assert.equal(states.at(-1).status,'cancelled');
});
test('stop during loading prevents late autoplay while completed stop preserves replay',async()=>{
  const loading=deferred(),entered=deferred();const {controller,stage,states}=setup(async()=>response(ready()));
  stage.loadMotion=async()=>{entered.resolve();return loading.promise;};
  const work=controller.generate(text);await entered.promise;controller.stop();loading.resolve(true);await work;
  assert.equal(stage.plays,0);assert.equal(states.at(-1).status,'idle');
  stage.loadMotion=async()=>true;await controller.generate(text);assert.equal(stage.plays,1);
  controller.stop();assert.equal(stage.time,0);assert.equal(states.at(-1).status,'ready');
  assert.deepEqual(states.at(-1).glosses,ready().glosses);controller.play();assert.equal(stage.plays,2);
});
test('late older errors cannot overwrite newer successful state',async()=>{
  const pending=deferred();let calls=0;
  const {controller,stage,states}=setup(async()=>++calls===1?pending.promise:response(ready()));
  const old=controller.generate(text);await controller.generate(text);
  pending.reject(new Error('older failure'));await old;
  assert.equal(states.at(-1).status,'ready');assert.equal(stage.plays,1);
});
test('invalid identities, provenance, captions and glosses never reach stage',async()=>{
  for(const bad of [ready({input_text:'其他文字'}),ready({generation_text:'错误字幕'}),
    ready({motion:{clip:{id:'old',text,source_kind:'pretrained_text_conditioned_generation'},sha256:digest}}),
    ready({motion:{clip:{id:'words',text,source_kind:'retrieved_word_composition'}}}),
    ready({glosses:[]}),ready({glosses:['']}),ready({glosses:Array(33).fill('我')})]) {
    const {controller,stage,states}=setup(async()=>response(bad));await controller.generate(text);
    assert.equal(stage.loads.length,0);assert.equal(stage.plays,0);assert.equal(states.at(-1).status,'error');
  }
});
test('HTTP and unreadable responses remain actionable without stale subtitle',async()=>{
  for(const fetcher of [async()=>response({detail:'规划器正在启动，请稍后再试。'},false),
    async()=>({ok:true,json:async()=>{throw new Error('parse');}})]) {
    const {controller,states}=setup(fetcher);await controller.generate(text);
    assert.equal(states.at(-1).status,'error');assert.equal(states.at(-1).subtitle,undefined);
    assert.match(states.at(-1).message,/启动|读取/);
  }
});
test('newer preview wins when an older stage load finishes last',async()=>{
  const loading=deferred(),entered=deferred();let loads=0;
  const {controller,stage,states}=setup(async()=>response(ready()));
  stage.loadMotion=async(data,id,current)=>{if(++loads===1){entered.resolve();await loading.promise;}return current();};
  const old=controller.generate(text);await entered.promise;await controller.generate(text);
  loading.resolve();await old;assert.equal(stage.plays,1);assert.equal(states.at(-1).status,'ready');
});
