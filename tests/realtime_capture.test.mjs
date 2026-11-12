import test from 'node:test';
import assert from 'node:assert/strict';
import * as module from '../slbridge/animation/web/communication/realtime-capture.mjs';
const tick=()=>new Promise(resolve=>setImmediate(resolve));

function harness({contextMissing=false}={}){
  let clock=0,next,blobCallback,timerId=0;const sent=[],draws=[],metrics=[],errors=[],timers=new Map();
  const canvas={width:0,height:0,getContext:()=>contextMissing?null:({fillRect(){},drawImage(...args){draws.push(args);}}),toBlob:callback=>blobCallback=callback};
  const capture=new module.LiveFrameCapture({video:{videoWidth:640,videoHeight:320,readyState:4},
    makeCanvas:()=>canvas,now:()=>clock,requestFrame:fn=>{next=fn;return 1;},cancelFrame(){},
    encodeBlob:async()=> 'jpeg-data',onFrame:frame=>{sent.push(frame);return true;},onMetrics:value=>metrics.push(value),
    onError:value=>errors.push(value),setTimer:fn=>{timers.set(++timerId,fn);return timerId;},clearTimer:id=>timers.delete(id)});
  return {capture,canvas,sent,draws,metrics,errors,callback:()=>blobCallback,
    step(time){clock=time;next?.();},watch(time){clock=time;const jobs=[...timers.values()];timers.clear();for(const fn of jobs)fn();},
    deliver(){blobCallback?.(new Blob(['frame']));}};
}

test('capture reads the current first frame without depending on an animation callback',async()=>{
  const h=harness();h.capture.start({fps:30,region:{x:0,y:0,width:1,height:1}});
  assert.equal(h.draws.length,1);h.deliver();await tick();assert.equal(h.sent[0].index,0);h.capture.stop();
});

test('stalled animation callbacks fail clearly instead of fabricating repeated frames',async()=>{
  const h=harness();h.capture.start({fps:30,region:{x:0,y:0,width:1,height:1}});h.step(0);h.deliver();await tick();
  h.watch(2001);
  assert.equal(h.capture.active,false);assert.equal(h.errors.length,1);assert.match(h.errors[0],/采集.*停滞/);
  assert.equal(h.sent.length,1);h.step(2500);h.deliver();await tick();assert.equal(h.sent.length,1);
});

test('EOF cannot wait forever when image encoding never calls back',async()=>{
  const h=harness();h.capture.start({fps:30,region:{x:0,y:0,width:1,height:1}});h.step(0);
  const ending=h.capture.finish();h.watch(2001);
  assert.equal(await Promise.race([ending,tick().then(()=> 'hung')]),false);
  assert.equal(h.errors.length,1);h.deliver();await tick();assert.equal(h.sent.length,0);
});

test('cancelled drain resolves and an old callback cannot release a newer encoding',async()=>{
  const h=harness();h.capture.start({fps:30,region:{x:0,y:0,width:1,height:1}});h.step(0);
  const old=h.callback(),oldEnd=h.capture.finish();h.capture.stop();
  h.capture.start({fps:30,region:{x:0,y:0,width:1,height:1}});h.step(0);
  assert.equal(await Promise.race([oldEnd,tick().then(()=> 'hung')]),false);
  const newEnd=h.capture.finish();old(new Blob(['old']));await tick();
  assert.equal(await Promise.race([newEnd,tick().then(()=> 'still-waiting')]),'still-waiting');
  h.deliver();assert.equal(await newEnd,true);assert.equal(h.sent.length,1);
});

test('an unavailable canvas reports a retry error without escaping the frame callback',()=>{
  const h=harness({contextMissing:true});
  assert.doesNotThrow(()=>{h.capture.start({fps:30,region:{x:0,y:0,width:1,height:1}});h.step(0);});
  assert.equal(h.capture.active,false);assert.equal(h.errors.length,1);
});

test('a stopped encoder cannot send a late frame',async()=>{
  const h=harness();h.capture.start({fps:30,region:{x:0,y:0,width:1,height:1}});h.step(0);
  h.capture.stop();h.deliver();await tick();assert.equal(h.sent.length,0);
});

test('missed sampling slots are explicit instead of renumbered',async()=>{
  const h=harness();h.capture.start({fps:30,region:{x:0,y:0,width:1,height:1}});h.step(0);h.step(40);
  h.deliver();await tick();h.step(100);h.deliver();await tick();
  assert.deepEqual(h.sent.map(x=>x.index),[0,3]);assert.equal(h.sent[1].timestamp,.1);h.capture.stop();
});

test('wide source is fitted into a square without stretching',()=>{
  const h=harness();h.capture.start({fps:30,region:{x:0,y:0,width:1,height:1}});h.step(0);
  assert.deepEqual(h.draws[0].slice(-4),[0,128,512,256]);assert.equal(h.canvas.width,512);h.capture.stop();
});

test('natural end delivers an encoding frame before requesting sentence flush',async()=>{
  const h=harness();h.capture.start({fps:30,region:{x:0,y:0,width:1,height:1}});h.step(0);
  const ending=h.capture.finish();h.deliver();assert.equal(await ending,true);
  assert.equal(h.sent.length,1);assert.equal(h.capture.active,false);
});

test('capture reports encoding and skipped slots without renumbering or late-source metrics',async()=>{
  const h=harness();h.capture.start({fps:30,region:{x:0,y:0,width:1,height:1}});
  h.step(0);h.step(40);h.deliver();await tick();
  assert.equal(h.metrics[0].encodeMs,40);assert.equal(h.metrics[0].capturedFrames,1);
  h.step(100);h.step(120);h.deliver();await tick();
  assert.equal(h.metrics.length,1);
  h.step(600);h.deliver();await tick();
  assert.equal(h.metrics.length,2);assert.equal(h.metrics[1].missedSlots,16);
  assert.deepEqual(h.sent.map(x=>x.index),[0,3,18]);
  h.step(1200);h.capture.stop();const finalCount=h.metrics.length;h.deliver();await tick();
  assert.equal(finalCount,3);assert.equal(h.metrics.length,finalCount);
});

test('EOF and stop publish the final current-session metrics before invalidation',async()=>{
  for(const ending of ['finish','stop']){
    const h=harness();h.capture.start({fps:30,region:{x:0,y:0,width:1,height:1}});
    h.step(0);h.deliver();await tick();h.step(100);h.deliver();await tick();
    assert.equal(h.metrics.at(-1).missedSlots,0);
    await h.capture[ending]();
    assert.equal(h.metrics.at(-1).missedSlots,2);assert.equal(h.metrics.at(-1).capturedFrames,2);
  }
});
