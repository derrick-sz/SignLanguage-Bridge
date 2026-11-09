import test from 'node:test';
import assert from 'node:assert/strict';
import { GenerationController } from '../slbridge/animation/web/generate-controller.mjs';

const response = data => ({ok:true,json:async()=>data});
const job = (state='complete', extra={}) => ({job_id:'new-1',text:'明天我们去公园散步。',generation_text:'明天我们去公园散步。',state,phase:'body',quality_warning:'手形与语义待核对',language_review:'NOT_RUN',...extra});
const motion = text => ({clip:{id:'generated-1',text},sha256:'digest'});
const deferred = () => {let resolve;const promise=new Promise(r=>resolve=r);return {promise,resolve};};
function setup(fetcher, sleep=async()=>{}) {
  const states=[],stage={stop(){},play(){this.plays++},pause(){},seek(time){this.time=time;},plays:0,loads:[],async loadMotion(...args){this.loads.push(args);return args[2]();}};
  const controller=new GenerationController({stage,fetcher,sleep,onState:s=>states.push(s)});
  return {controller,stage,states};
}
test('free text posts directly, polls actual state, loads verified caption, explicit play',async()=>{
  const calls=[],delays=[];
  const {controller,stage,states}=setup(async function(url,options){
    assert.equal(this,globalThis);calls.push([url,options]);
    if(options.method==='POST')return response(job('queued'));
    if(url.endsWith('/motion'))return response(motion(job().generation_text));
    return response(job());
  },async ms=>delays.push(ms));
  await controller.generate(job().text,6);
  assert.deepEqual(JSON.parse(calls[0][1].body),{text:job().text,seconds:6});
  assert.deepEqual(delays,[1500]);
  assert.equal(calls.some(([url])=>url.includes('/scenes')),false);
  assert.equal(stage.loads.length,1);assert.equal(stage.plays,0);
  assert.equal(states.at(-1).status,'ready');assert.equal(states.at(-1).subtitle,job().generation_text);
  controller.play();assert.equal(stage.plays,1);
});
test('invalid input and unsupported duration never submit',async()=>{
  let calls=0;const {controller,states}=setup(async()=>{calls++;});
  for(const [text,seconds] of [['',6],['x'.repeat(201),6],['a\u0000b',6],['你好',5]]) await controller.generate(text,seconds);
  assert.equal(calls,0);assert.equal(states.at(-1).status,'error');
});
test('multiline text is sent unchanged and server normalized text is checked',async()=>{
  const {controller,states}=setup(async(url)=>response(url.endsWith('/motion')?motion('你好 世界'):job('complete',{text:'你好\n世界',generation_text:'你好 世界'})));
  await controller.generate('你好\n世界',4);assert.equal(states.at(-1).subtitle,'你好 世界');
});
test('editing during submit cancels late assigned job without waiting',async()=>{
  const post=deferred(),cancel=deferred(),calls=[];
  const {controller,stage,states}=setup(async(url,options)=>{calls.push(options.method);if(options.method==='POST')assert.equal(options.signal,undefined);return options.method==='DELETE'?cancel.promise:post.promise;});
  const pending=controller.generate('你好',6);controller.invalidate();post.resolve(response(job('running')));
  await pending;assert.equal(calls.includes('DELETE'),true);assert.equal(stage.loads.length,0);assert.equal(states.at(-1).status,'idle');
  cancel.resolve(response(job('cancelled')));
});
test('cancel invalidates pending polling and ignores late completion',async()=>{
  const poll=deferred(),cancel=deferred();let started;
  const entered=new Promise(r=>started=r);
  const {controller,stage,states}=setup(async(url,options)=>{
    if(options.method==='POST')return response(job('running'));
    if(options.method==='DELETE')return cancel.promise;
    started();return poll.promise;
  });
  const pending=controller.generate(job().text);await entered;controller.cancel();poll.resolve(response(job()));await pending;
  assert.equal(stage.loads.length,0);assert.equal(states.at(-1).status,'cancelled');cancel.resolve(response(job('cancelled')));
});
test('older stage load cannot publish ready or play after new input',async()=>{
  const load=deferred();const {controller,stage,states}=setup(async url=>response(url.endsWith('/motion')?motion(job().text):job()));
  let entered;const started=new Promise(r=>entered=r);
  stage.loadMotion=async(data,id,current)=>{entered();await load.promise;return current();};
  const pending=controller.generate(job().text);await started;controller.stop();load.resolve();await pending;controller.play();
  assert.equal(stage.plays,0);assert.equal(states.at(-1).status,'idle');
});
test('old errors and cancellation responses cannot overwrite newer state',async()=>{
  const first=deferred();let count=0;
  const {controller,states}=setup(async(url,options)=>{
    if(options.method==='DELETE')throw new Error('late cancel failure');
    if(options.method==='POST' && ++count===1)return first.promise;
    return response(url.endsWith('/motion')?motion(job().text):job());
  });
  const old=controller.generate(job().text);await controller.generate(job().text);
  first.resolve({ok:false,json:async()=>({detail:'old failure'})});await old;
  assert.equal(states.at(-1).status,'ready');
});
test('failed server jobs and caption mismatch never load or play',async()=>{
  for(const failed of [true,false]) {
    const {controller,stage,states}=setup(async url=>response(url.endsWith('/motion')?motion('另一句话'):job(failed?'failed':'complete',{error:'生成失败'})));
    await controller.generate(job().text);assert.equal(stage.loads.length,0);assert.equal(states.at(-1).status,'error');
  }
});
test('HTTP errors remain readable',async()=>{
  const {controller,states}=setup(async()=>({ok:false,json:async()=>({detail:'队列已满，请稍后重试。'})}));
  await controller.generate('你好');assert.equal(states.at(-1).message,'队列已满，请稍后重试。');
});
test('stopping completed playback preserves result for replay without generating again',async()=>{
  let calls=0;const {controller,stage,states}=setup(async url=>{calls++;return response(url.endsWith('/motion')?motion(job().text):job());});
  await controller.generate(job().text);controller.play();controller.stop();controller.play();
  assert.equal(stage.time,0);assert.equal(stage.plays,2);assert.equal(calls,2);assert.equal(states.at(-1).status,'ready');
});
