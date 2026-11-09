import test from 'node:test';
import assert from 'node:assert/strict';
import {Conversation,readableTranscript} from '../slbridge/animation/web/communication/conversation.mjs';
import {RecognitionController} from '../slbridge/animation/web/communication/recognition-controller.mjs';
import {CameraCapture} from '../slbridge/animation/web/communication/capture.mjs';

const deferred=()=>{let resolve;const promise=new Promise(r=>resolve=r);return {promise,resolve};};
const recognized=(id='r1')=>({request_id:id,recognized:true,text:'你 好',sign_language:'csl',output_kind:'gloss_sequence',review_status:'unverified',provenance:{model_generation:1}});
const video=()=>new File(['video'],'clip.mp4',{type:'video/mp4'});

test('recognition becomes a conversation turn only after explicit text confirmation',()=>{
  const conversation=new Conversation();const result=recognized();
  assert.equal(conversation.messages.length,0);
  assert.equal(conversation.confirmRecognition(result,'你好。'),true);
  const message=conversation.messages[0];
  assert.equal(message.text,'你好。');assert.equal(message.original_text,'你 好');
  assert.equal(message.provenance.model_generation,1);
  assert.equal(message.review_status,'unverified','editing text does not approve the underlying sign model');
  assert.equal(conversation.confirmRecognition(result,'你好。'),false,'double confirmation cannot add duplicate turns');
});

test('replay/ready notifications do not duplicate replies, and export preserves both directions',()=>{
  const c=new Conversation();c.confirmRecognition(recognized(),'你好。');
  c.addReply('我想喝水。','reply1');assert.equal(c.addReply('我想喝水。','reply1'),false);
  const exported=c.export();assert.equal(exported.messages.length,2);
  assert.deepEqual(exported.messages.map(x=>x.direction),['received','sent']);
  exported.messages[0].text='changed';assert.equal(c.messages[0].text,'你好。');
  c.clear();assert.equal(c.messages.length,0);assert.equal(c.undoClear(),true);assert.equal(c.messages.length,2);
});

test('cancel and source/language changes reject a late recognition result',async()=>{
  const pending=deferred(),states=[];
  const controller=new RecognitionController({fetcher:()=>pending.promise,onState:s=>states.push(s),uid:()=> 'r1'});
  const task=controller.recognize(video(),'csl');controller.invalidate();
  pending.resolve({ok:true,json:async()=>recognized()});await task;
  assert.equal(states.some(x=>x.status==='ready'),false);
  assert.equal(states.at(-1).status,'idle');
});

test('a response for another request or sign language never becomes editable text',async()=>{
  for(const data of [recognized('other'),{...recognized(),sign_language:'libras'}]){
    const states=[];const c=new RecognitionController({fetcher:async()=>({ok:true,json:async()=>data}),onState:s=>states.push(s),uid:()=> 'r1'});
    await c.recognize(video(),'csl');assert.equal(states.at(-1).status,'error');
  }
});

test('raw video transport sends only the selected file and declared language',async()=>{
  const calls=[],states=[];const file=video();
  const c=new RecognitionController({uid:()=> 'r1',onState:s=>states.push(s),fetcher:async(url,options)=>{calls.push({url,options});return {ok:true,json:async()=>recognized()};}});
  await c.recognize(file,'csl');assert.equal(calls[0].options.body,file);
  assert.equal(new URL(calls[0].url,'http://local').searchParams.get('sign_language'),'csl');
  assert.equal(states.at(-1).result.text,'你 好');
});

function stream(){const tracks=[{stops:0,stop(){this.stops++;}},{stops:0,stop(){this.stops++;}}];return {tracks,getTracks:()=>tracks};}

test('a camera permission result arriving after close releases every track',async()=>{
  const pending=deferred(),s=stream(),states=[];
  const camera=new CameraCapture({getUserMedia:()=>pending.promise,onState:s=>states.push(s)});
  const opening=camera.open();camera.close();pending.resolve(s);await opening;
  assert.deepEqual(s.tracks.map(x=>x.stops),[1,1]);assert.equal(states.some(x=>x.status==='preview'),false);
});

test('closing during recording discards its output and stops the owned camera',async()=>{
  class Recorder{static isTypeSupported(){return true;}constructor(){this.state='inactive';this.mimeType='video/webm';}start(){this.state='recording';}stop(){this.state='inactive';this.ondataavailable?.({data:new Blob(['frames'])});this.onstop?.();}}
  const s=stream(),states=[];const camera=new CameraCapture({getUserMedia:async()=>s,Recorder,onState:s=>states.push(s)});
  await camera.open();camera.start();camera.close();
  assert.equal(states.some(x=>x.status==='recorded'),false);assert.deepEqual(s.tracks.map(x=>x.stops),[1,1]);
});

test('an old recorder stop callback cannot clear a new recorder time limit or tracks',async()=>{
  const instances=[];
  class Recorder{static isTypeSupported(){return true;}constructor(){this.state='inactive';this.mimeType='video/webm';instances.push(this);}start(){this.state='recording';}stop(){this.state='inactive';}}
  const first=stream(),second=stream(),timers=new Map();let next=0;
  const camera=new CameraCapture({getUserMedia:async()=>instances.length?second:first,Recorder,
    setTimer:callback=>{timers.set(++next,callback);return next;},clearTimer:id=>timers.delete(id)});
  await camera.open();camera.start();camera.close();await camera.open();camera.start();
  instances[0].onstop();assert.ok(timers.has(2),'the second recording keeps its own deadline');
  assert.deepEqual(second.tracks.map(x=>x.stops),[0,0]);camera.close();
});

test('export is a readable dialogue and identifies human edits',()=>{
  const c=new Conversation();c.confirmRecognition(recognized(),'你好。');c.addReply('请等一下。','reply1');
  const transcript=readableTranscript(c.export());
  assert.match(transcript,/对方的手语（中国手语）/);assert.match(transcript,/你好。/);
  assert.match(transcript,/识别原文：你 好/);assert.match(transcript,/使用者核对修改/);
  assert.match(transcript,/我  /);assert.match(transcript,/请等一下。/);
  assert.doesNotMatch(transcript,/schema_version|model_generation/);
});

test('recorder construction and startup failures release owned camera tracks',async()=>{
  for(const failConstructor of [true,false]){
    class Recorder{static isTypeSupported(){return true;}constructor(){if(failConstructor)throw new Error('unsupported');this.state='inactive';}start(){throw new Error('cannot start');}}
    const s=stream(),states=[];const camera=new CameraCapture({getUserMedia:async()=>s,Recorder,onState:s=>states.push(s)});
    await camera.open();assert.equal(camera.start(),false);
    assert.deepEqual(s.tracks.map(x=>x.stops),[1,1]);assert.equal(states.at(-1).status,'error');
  }
});
