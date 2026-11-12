import test from 'node:test';
import assert from 'node:assert/strict';
import * as module from '../slbridge/animation/web/communication/realtime-client.mjs';

function harness({fetcher,now=()=>200,onUpdate}={}){
  const sockets=[];
  class Socket{
    constructor(){this.readyState=0;this.bufferedAmount=0;this.sent=[];sockets.push(this);}
    send(text){this.sent.push(JSON.parse(text));}close(){this.readyState=3;this.onclose?.();}
    open(){this.readyState=1;this.onopen?.();}
    event(value){this.onmessage?.({data:JSON.stringify(value)});}
  }
  const c=new module.RealtimeClient({url:'ws://local',makeSocket:()=>new Socket(),now,fetcher,onUpdate});
  let sequence=0;
  const event=(type,data={})=>({type,protocol:'csl-realtime-v1',session_id:'session',source_epoch:c.epoch,sequence:sequence++,...data});
  async function start(){const pending=c.start({fps:30});const socket=sockets.at(-1);socket.open();socket.event(event('ready',{capabilities:{}}));await pending;return socket;}
  return {c,sockets,event,start};
}

test('an empty native summary ends with a retry message rather than successful translation',async()=>{
  const h=harness(),s=await h.start();
  s.event(h.event('summary',{frames_processed:0,dropped_input_frames:0,sentences:0}));
  s.event(h.event('finished'));
  assert.equal(h.c.snapshot().status,'error');
  assert.match(h.c.snapshot().message,/未采集.*画面/);
  assert.equal(h.c.snapshot().currentCaption,null);
});

test('frame counts distinguish processed input without recognized text from empty input',async()=>{
  const h=harness(),s=await h.start();
  for(const index of [0,4]){
    assert.equal(h.c.sendFrame({index,timestamp:index/30,capture_ms:100,jpeg:'fixture'}),true);
    s.event(h.event('ack',{capture_ms:100,queue_ms:0,processing_ms:1}));
  }
  assert.equal(h.c.snapshot().framesSent,2);assert.equal(h.c.snapshot().framesProcessed,2);
  s.event(h.event('summary',{frames_processed:2}));s.event(h.event('finished'));
  assert.equal(h.c.snapshot().status,'stopped');assert.match(h.c.snapshot().message,/未识别到/);
  assert.equal(h.c.snapshot().currentCaption,null);
});

test('disconnect before the first ACK still publishes the final sent-frame count',async()=>{
  const h=harness(),metrics=[];h.c.onMetrics=value=>metrics.push(value);const socket=await h.start();
  h.c.sendFrame({index:0,timestamp:0,capture_ms:100,jpeg:'fixture'});socket.onerror();
  assert.equal(h.c.snapshot().framesSent,1);assert.equal(metrics.at(-1).framesSent,1);
  assert.equal(metrics.at(-1).framesProcessed,0);
});

test('successful finish retains only the current caption until an explicit new attempt',async()=>{
  const h=harness(),s=await h.start();
  s.event(h.event('sentence',{utterance_id:0,revision:1,status:'committed',text:'本轮译文'}));
  s.event(h.event('summary',{frames_processed:20}));s.event(h.event('finished'));
  assert.equal(h.c.snapshot().currentCaption.text,'本轮译文');
  assert.equal(h.c.snapshot().status,'stopped');h.c.stop();assert.equal(h.c.snapshot().currentCaption,null);
  assert.equal(h.c.snapshot().captions[0].original_text,'本轮译文');
});

test('new attempts and cancellation hide the previous main caption while preserving reviewed history',async()=>{
  const h=harness(),s=await h.start();
  s.event(h.event('sentence',{utterance_id:0,revision:1,status:'committed',text:'上一轮译文',predicted_gloss:['原词']}));
  const id=h.c.snapshot().captions[0].id;h.c.confirm(id);
  h.c.stop();
  assert.equal(h.c.snapshot().captions[0].confirmed,true);
  assert.equal(h.c.snapshot().currentCaption,null);
  await h.start();
  assert.equal(h.c.snapshot().currentCaption,null);
  assert.equal(h.c.snapshot().captions[0].original_text,'上一轮译文');h.c.stop();
});

test('live words update before any sentence and remain distinct from Chinese',async()=>{
  const h=harness(),s=await h.start();
  s.event(h.event('word',{utterance_id:0,event_id:'a',status:'partial',gloss:'我'}));
  assert.equal(h.c.snapshot().words[0].text,'我');assert.equal(h.c.snapshot().captions.length,0);
  s.event(h.event('sentence',{utterance_id:0,revision:1,status:'committed',text:'我想睡觉。'}));
  assert.equal(h.c.snapshot().captions[0].text,'我想睡觉。');h.c.stop();
});

test('stopped source cannot publish late words or overwrite a newer session',async()=>{
  const h=harness(),old=await h.start(),stale=h.event('word',{utterance_id:0,event_id:'x',status:'final',gloss:'旧'});
  h.c.stop();const current=await h.start();old.event(stale);
  assert.equal(h.c.snapshot().words.length,0);
  current.event(h.event('word',{utterance_id:0,event_id:'n',status:'partial',gloss:'新'}));
  assert.equal(h.c.snapshot().words[0].text,'新');h.c.stop();
});

test('different sentences survive and LLM advice cannot replace an edited original',async()=>{
  const h=harness(),s=await h.start();
  s.event(h.event('sentence',{utterance_id:0,revision:1,status:'committed',text:'第一句'}));
  s.event(h.event('sentence',{utterance_id:1,revision:1,status:'committed',text:'第二句'}));
  const first=h.c.snapshot().captions[0];h.c.edit(first.id,'人工核对');
  s.event(h.event('assistance',{utterance_id:0,revision:1,status:'suggestion',text:'模型建议'}));
  const state=h.c.snapshot();assert.equal(state.captions.length,2);
  assert.equal(state.captions[0].text,'人工核对');assert.equal(state.captions[0].original_text,'第一句');h.c.stop();
});

test('outbound congestion fails visibly instead of renumbering dropped frames',async()=>{
  const h=harness(),s=await h.start();s.bufferedAmount=300000;
  assert.equal(h.c.sendFrame({index:9,timestamp:.3,capture_ms:100,jpeg:'abc'},h.c.epoch),false);
  assert.equal(h.c.snapshot().status,'error');assert.match(h.c.snapshot().message,/跟不上|积压/);
});

test('advice can arrive after native finish and annotates its own sentence only',async()=>{
  let resolve;const calls=[];
  const h=harness({fetcher:async url=>{calls.push(url);return new Promise(r=>resolve=r);}}),s=await h.start();
  const epoch=h.c.epoch;
  s.event(h.event('sentence',{utterance_id:0,revision:1,status:'committed',text:'原译文'}));
  s.event(h.event('assistance',{utterance_id:0,revision:1,status:'pending',text:'',job_id:'a'.repeat(32)}));
  s.event(h.event('finished'));
  assert.equal(calls.length,1);
  resolve({ok:true,json:async()=>({job_id:'a'.repeat(32),session_id:'session',source_epoch:epoch,utterance_id:0,revision:1,status:'risk',text:'核对否定'})});
  await new Promise(r=>setImmediate(r));await new Promise(r=>setImmediate(r));
  const state=h.c.snapshot();assert.equal(state.status,'stopped');assert.equal(state.captions[0].text,'原译文');
  assert.equal(state.captions[0].advice.text,'核对否定');h.c.stop();
});

test('a new source rejects late advice and invalid job IDs never become URLs',async()=>{
  let resolve,calls=0;const h=harness({fetcher:async()=>{calls++;return new Promise(r=>resolve=r);}}),s=await h.start();
  s.event(h.event('sentence',{utterance_id:0,revision:1,status:'committed',text:'原译文'}));
  s.event(h.event('assistance',{utterance_id:0,revision:1,status:'pending',text:'',job_id:'../../secret'}));
  assert.equal(calls,0);
  s.event(h.event('assistance',{utterance_id:0,revision:1,status:'pending',text:'',job_id:'a'.repeat(32)}));
  const oldEpoch=h.c.epoch;h.c.stop();
  resolve({ok:true,json:async()=>({job_id:'a'.repeat(32),session_id:'session',source_epoch:oldEpoch,utterance_id:0,revision:1,status:'risk',text:'旧建议'})});
  await new Promise(r=>setImmediate(r));assert.equal(h.c.snapshot().captions[0].advice?.status,'cancelled');
});

test('deferred previous sentence and a rejected next sentence preserve earlier history',async()=>{
  const h=harness(),s=await h.start();
  s.event(h.event('sentence',{utterance_id:0,revision:1,status:'pending',text:''}));
  s.event(h.event('word',{utterance_id:1,event_id:'next',status:'partial',gloss:'新词'}));
  s.event(h.event('sentence',{utterance_id:0,revision:1,status:'committed',text:'上一句'}));
  assert.equal(h.c.snapshot().words[0].text,'新词');
  s.event(h.event('sentence',{utterance_id:1,revision:1,status:'rejected',text:'',message:'请分句重试'}));
  assert.equal(h.c.snapshot().captions[0].original_text,'上一句');
  assert.equal(h.c.snapshot().message,'请分句重试');h.c.stop();
});

test('frame acknowledgements update latency without repainting words and history',async()=>{
  let updates=0;const h=harness({onUpdate:()=>updates++}),s=await h.start();const before=updates;
  for(let i=0;i<300;i++)s.event(h.event('ack',{capture_ms:100,queue_ms:30,processing_ms:20}));
  assert.equal(updates,before);assert.equal(h.c.snapshot().lastFrameRoundTripMs,100);
  s.event(h.event('word',{utterance_id:0,event_id:'a',status:'partial',gloss:'即时词'}));
  assert.equal(updates,before+1);h.c.stop();
});

test('first-word total includes service preparation and capture startup',async()=>{
  let clock=100;const h=harness({now:()=>clock});
  const pending=h.c.start({fps:30}),s=h.sockets.at(-1);s.open();clock=400;
  s.event(h.event('ready',{capabilities:{}}));await pending;
  clock=450;h.c.noteCaptureStart();clock=1150;
  s.event(h.event('word',{utterance_id:0,event_id:'a',status:'partial',gloss:'即时词'}));
  const state=h.c.snapshot();assert.equal(state.firstWordMs,700);
  assert.equal(state.readyMs,300);assert.equal(state.firstWordTotalMs,1050);h.c.stop();
});

test('explicit completion drains final words and sentence without accepting more frames',async()=>{
  const h=harness(),s=await h.start();h.c.finish();
  assert.equal(h.c.status,'finishing');
  assert.equal(h.c.sendFrame({index:1,timestamp:1/30,capture_ms:100,jpeg:'abc'}),false);
  s.event(h.event('word',{utterance_id:0,event_id:'a',status:'final',gloss:'学校'}));
  s.event(h.event('sentence',{utterance_id:0,revision:1,status:'committed',text:'明天我想去学校看书。'}));
  s.event(h.event('finished'));
  assert.equal(h.c.snapshot().status,'stopped');
  assert.equal(h.c.snapshot().captions[0].original_text,'明天我想去学校看书。');
  assert.equal(h.c.snapshot().words[0].text,'学校');h.c.stop();
});

test('each caption keeps its own bounded word evidence while the next utterance is running',async()=>{
  const h=harness(),s=await h.start();
  s.event(h.event('word',{utterance_id:1,event_id:'next',status:'partial',gloss:'新句'}));
  s.event(h.event('sentence',{utterance_id:0,revision:1,status:'committed',text:'没钱，我出去玩。',
    predicted_gloss:['没有'],review_hint:{reason:'single_word_expansion',gloss_count:1,text_units:6,gloss_units:2}}));
  const item=h.c.snapshot().captions[0];
  assert.deepEqual(item.word_evidence,['没有']);assert.equal(item.review_hint.reason,'single_word_expansion');
  assert.equal(h.c.snapshot().words[0].text,'新句');assert.equal(item.confirmed,false);h.c.stop();
});

test('manual confirmation is preserved against late native revisions and advice',async()=>{
  const h=harness(),s=await h.start();
  s.event(h.event('sentence',{utterance_id:0,revision:1,status:'committed',text:'本人已核对',predicted_gloss:['核对']}));
  const id=h.c.snapshot().captions[0].id;h.c.confirm(id);
  s.event(h.event('sentence',{utterance_id:0,revision:2,status:'committed',text:'迟到机器改写',predicted_gloss:['其他']}));
  s.event(h.event('assistance',{utterance_id:0,revision:2,status:'risk',text:'针对迟到机器句子的建议'}));
  const state=h.c.snapshot();assert.equal(state.captions[0].text,'本人已核对');
  assert.equal(state.captions[0].advice,undefined);
  assert.equal(state.captions[0].confirmed,true);assert.equal(state.captions[0].edited,false);
  assert.equal(state.draft.text,'本人已核对');h.c.stop();
});

test('malformed evidence cannot create a one-word review action',async()=>{
  const h=harness(),s=await h.start();
  s.event(h.event('sentence',{utterance_id:0,revision:1,status:'committed',text:'原译文',
    predicted_gloss:[''],review_hint:{reason:'single_word_expansion'}}));
  assert.deepEqual(h.c.snapshot().captions[0].word_evidence,[]);
  assert.equal(h.c.snapshot().captions[0].review_hint,null);h.c.stop();
});

test('manual use of word evidence updates the visible sentence and retains the machine original',async()=>{
  const h=harness(),s=await h.start();
  s.event(h.event('sentence',{utterance_id:0,revision:1,status:'committed',text:'没钱，我出去玩。',predicted_gloss:['没有']}));
  const id=h.c.snapshot().captions[0].id;h.c.edit(id,'没有');
  const state=h.c.snapshot();assert.equal(state.captions[0].original_text,'没钱，我出去玩。');
  assert.equal(state.captions[0].text,'没有');assert.equal(state.draft.text,'没有');
  assert.equal(state.captions[0].edited,true);assert.equal(state.captions[0].confirmed,true);h.c.stop();
});

test('a revised machine sentence removes old advice and aborts its old-revision polling',async()=>{
  let resolve;const signals=[];
  const h=harness({fetcher:async(url,options)=>{signals.push(options.signal);return new Promise(r=>resolve=r);}}),s=await h.start();
  s.event(h.event('sentence',{utterance_id:0,revision:1,status:'committed',text:'第一版',predicted_gloss:['旧词']}));
  s.event(h.event('assistance',{utterance_id:0,revision:1,status:'pending',text:'',job_id:'c'.repeat(32)}));
  s.event(h.event('assistance',{utterance_id:0,revision:1,status:'risk',text:'旧句子的提示'}));
  assert.equal(h.c.snapshot().captions[0].advice.text,'旧句子的提示');
  s.event(h.event('sentence',{utterance_id:0,revision:2,status:'committed',text:'第二版',predicted_gloss:['新词']}));
  assert.equal(h.c.snapshot().captions[0].advice,undefined);
  assert.equal(signals[0].aborted,true);assert.equal(h.c.adviceJobs.size,0);
  resolve({ok:true,json:async()=>({job_id:'c'.repeat(32),session_id:'session',source_epoch:h.c.epoch,
    utterance_id:0,revision:1,status:'risk',text:'迟到旧提示'})});
  await new Promise(r=>setImmediate(r));
  assert.equal(h.c.snapshot().captions[0].advice,undefined);
  s.event(h.event('assistance',{utterance_id:0,revision:2,status:'risk',text:'新提示'}));
  assert.equal(h.c.snapshot().captions[0].advice.revision,2);h.c.stop();
});

test('a committed revision cannot be rewritten without increasing its revision',async()=>{
  const h=harness(),s=await h.start();
  s.event(h.event('sentence',{utterance_id:0,revision:1,status:'committed',text:'已交付版本'}));
  s.event(h.event('sentence',{utterance_id:0,revision:1,status:'committed',text:'同版本不同文字'}));
  assert.equal(h.c.snapshot().captions[0].text,'已交付版本');h.c.stop();
});

test('rejecting a native revision preserves the original advice task for a confirmed caption',async()=>{
  for(const edit of [false,true]){
    let resolve;const signals=[];
    const h=harness({fetcher:async(url,options)=>{signals.push(options.signal);return new Promise(r=>resolve=r);}}),s=await h.start();
    s.event(h.event('sentence',{utterance_id:0,revision:1,status:'committed',text:'原译文',predicted_gloss:['原词']}));
    s.event(h.event('assistance',{utterance_id:0,revision:1,status:'pending',text:'',job_id:'d'.repeat(32)}));
    const id=h.c.snapshot().captions[0].id;if(edit)h.c.edit(id,'人工修改');else h.c.confirm(id);
    s.event(h.event('sentence',{utterance_id:0,revision:2,status:'committed',text:'已拒绝的新机器句'}));
    assert.equal(signals[0].aborted,false);assert.equal(h.c.adviceJobs.size,1);
    resolve({ok:true,json:async()=>({job_id:'d'.repeat(32),session_id:'session',source_epoch:h.c.epoch,
      utterance_id:0,revision:1,status:'risk',text:'针对保留原译文的提示'})});
    await new Promise(r=>setImmediate(r));await new Promise(r=>setImmediate(r));
    const item=h.c.snapshot().captions[0];assert.equal(item.text,edit?'人工修改':'原译文');
    assert.equal(item.advice.status,'risk');assert.equal(item.advice.revision,1);
    assert.equal(h.c.adviceJobs.size,0);h.c.stop();
  }
});

test('source provenance stays local and each caption retains the source of its own session',async()=>{
  const h=harness();
  const begin=async(source)=>{const p=h.c.start({fps:30,source});const s=h.sockets.at(-1);s.open();s.event(h.event('ready',{capabilities:{},session_id:'session-'+h.sockets.length}));await p;return s;};
  let s=await begin({kind:'training_replay',label:'训练示范回放',id:'school',annotation:'目标文字不能发给模型'});
  assert.deepEqual(Object.keys(s.sent[0]).sort(),['fps','protocol','sign_language','source_epoch','type']);
  assert.equal(JSON.stringify(s.sent).includes('目标文字'),false);
  s.event(h.event('sentence',{utterance_id:0,revision:1,status:'committed',text:'实际机器结果',session_id:h.c.sessionId}));
  s=await begin({kind:'camera',label:'摄像头实时采集'});
  s.event(h.event('sentence',{utterance_id:0,revision:1,status:'committed',text:'第二次机器结果',session_id:h.c.sessionId}));
  const rows=h.c.snapshot().captions;assert.equal(rows[0].source.kind,'training_replay');
  assert.equal(rows[1].source.kind,'camera');assert.equal(rows[0].source.annotation,undefined);
  assert.equal(rows[0].text,'实际机器结果');h.c.stop();
});
