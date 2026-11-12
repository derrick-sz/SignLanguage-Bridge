import test from 'node:test';
import assert from 'node:assert/strict';
import {CaptionSession} from '../slbridge/animation/web/communication/caption-session.mjs';

const tick=()=>new Promise(resolve=>setImmediate(resolve));
const clip=()=>new File(['real-encoded-clip-placeholder'],'segment.webm',{type:'video/webm'});
function harness({maxQueued=2}={}){
  const calls=[],states=[];let id=0;
  const session=new CaptionSession({uid:()=>String(++id),maxQueued,onUpdate:state=>states.push(state),fetcher:(url,options)=>new Promise(resolve=>{
    const requestId=new URL(url,'http://local').searchParams.get('request_id');
    calls.push({options,finish:(text,kind='text')=>resolve({ok:true,json:async()=>({request_id:requestId,sign_language:'csl',recognized:!!text,
      text,output_kind:kind,provenance:{model_id:'sentence-model'},review_status:'unverified'})})});
  })});
  return {session,calls,states};
}
test('separate segments with the same recognized sentence stay separate and retain raw results',async()=>{
  const {session,calls}=harness();session.start({language:'csl',sourceLabel:'会议画面'});
  session.enqueue(clip(),{startSeconds:0,endSeconds:3});
  session.enqueue(clip(),{startSeconds:3,endSeconds:6});
  assert.equal(calls.length,1);
  calls[0].finish('我想睡觉。');await tick();assert.equal(calls.length,2);
  calls[1].finish('我想睡觉。');await tick();
  assert.equal(session.snapshot().captions.length,2);
  assert.equal(session.snapshot().captions[0].output_kind,'text');
  const first=session.snapshot().captions[0];session.edit(first.id,'我想睡一会儿。');
  assert.equal(session.snapshot().captions[0].original_text,'我想睡觉。');
  assert.match(session.transcript(),/识别原文：我想睡觉。/);
});
test('paused or replaced sources cannot publish a late model response',async()=>{
  const {session,calls}=harness();const epoch=session.start({language:'csl'});
  session.enqueue(clip(),{startSeconds:0,endSeconds:3,epoch});session.pause();
  session.start({language:'csl',sourceLabel:'新选区'});
  assert.equal(session.enqueue(clip(),{startSeconds:3,endSeconds:6,epoch}),false);
  session.enqueue(clip(),{startSeconds:0,endSeconds:2});
  assert.equal(calls.length,1,'one HTTP request remains in flight even when abort is ignored');
  calls[0].finish('旧结果');await tick();assert.equal(calls.length,2);
  calls[1].finish('新结果');await tick();
  assert.deepEqual(session.snapshot().captions.map(x=>x.text),['新结果']);
});
test('slow inference has a bounded visible queue and empty results are not captions',async()=>{
  const {session,calls}=harness({maxQueued:1});session.start({language:'csl'});
  assert.equal(session.enqueue(clip(),{startSeconds:0,endSeconds:2}),true);
  assert.equal(session.enqueue(clip(),{startSeconds:2,endSeconds:4}),true);
  assert.equal(session.enqueue(clip(),{startSeconds:4,endSeconds:6}),false);
  assert.equal(session.snapshot().skipped,1);assert.match(session.snapshot().notice,/跟不上/);
  calls[0].finish('');await tick();calls[1].finish('你好','gloss_sequence');await tick();
  assert.equal(session.snapshot().captions.length,1);assert.equal(session.snapshot().empty,1);
  session.clear();assert.equal(session.snapshot().captions.length,0);
  assert.equal(session.undoClear(),true);assert.equal(session.snapshot().captions.length,1);
});
test('video completion drains a final delayed blob before stopping the caption session',async()=>{
  const {session,calls}=harness();session.start({language:'csl'});
  session.enqueue(clip(),{startSeconds:0,endSeconds:6});calls[0].finish('前半句');await tick();
  assert.equal(session.snapshot().status,'running');
  session.enqueue(clip(),{startSeconds:6,endSeconds:8.2});session.finishInput();
  assert.equal(session.snapshot().status,'running');
  assert.equal(session.enqueue(clip(),{startSeconds:8.2,endSeconds:10}),false);
  calls[1].finish('后半句');await tick();
  assert.equal(session.snapshot().status,'stopped');
  assert.deepEqual(session.snapshot().captions.map(x=>x.text),['前半句','后半句']);
});
