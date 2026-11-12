import test from 'node:test';
import assert from 'node:assert/strict';
import {ScreenSource,PauseSegmenter,RegionRecorder} from '../slbridge/animation/web/communication/screen-capture.mjs';
import {CameraCapture} from '../slbridge/animation/web/communication/capture.mjs';

test('closing a pending screen chooser releases a late stream without enabling audio',async()=>{
  let finish,constraints,stopped=0;const states=[];
  const source=new ScreenSource({getDisplayMedia:options=>{constraints=options;return new Promise(r=>finish=r);},onState:s=>states.push(s.status)});
  const opening=source.open();source.close();
  finish({getTracks:()=>[{stop(){stopped++;}}]});assert.equal(await opening,null);
  assert.equal(constraints.audio,false);assert.equal(stopped,1);assert.equal(states.includes('ready'),false);
});
test('stopping shared video closes its owned stream',async()=>{
  let ended,stopped=0;
  const track={addEventListener:(event,cb)=>ended=cb,stop:()=>stopped++};
  const source=new ScreenSource({getDisplayMedia:async()=>({getTracks:()=>[track],getVideoTracks:()=>[track]})});
  await source.open();ended();assert.equal(source.stream,null);assert.equal(stopped,1);
});
test('pause segmentation waits for real activity while fixed windows and manual finishing remain explicit',()=>{
  const cuts=new PauseSegmenter({maxSeconds:6});cuts.begin(0);
  for(let t=0;t<6;t+=.1)assert.equal(cuts.feed(t,0),null);
  assert.deepEqual(cuts.feed(6,0),{publish:false,reason:'idle'});
  cuts.begin(6);cuts.feed(6.1,.04);cuts.feed(6.2,.04);cuts.feed(6.3,.04);
  assert.equal(cuts.feed(6.8,0),null);
  assert.deepEqual(cuts.feed(7.3,0),{publish:true,reason:'pause'});
  const fixed=new PauseSegmenter({mode:'interval',maxSeconds:6});fixed.begin(0);
  assert.deepEqual(fixed.feed(6,0),{publish:true,reason:'interval'});
});
function recorderHarness({nativeTimers=false}={}){
  let nextFrame,stopped=0,timerId=0;const segments=[],states=[],recorders=[],timers=new Map();
  const video={videoWidth:640,videoHeight:480,currentTime:0,readyState:4};
  class Recorder{
    static isTypeSupported(){return true;}
    constructor(){this.state='inactive';this.mimeType='video/webm';recorders.push(this);}
    start(){this.state='recording';}
    stop(){this.state='inactive';this.ondataavailable?.({data:new Blob(['encoded'])});this.onstop?.();}
  }
  const context={drawImage(){},getImageData:()=>({data:new Uint8ClampedArray(32*24*4)})};
  const makeCanvas=()=>({width:640,height:480,getContext:()=>context,captureStream:()=>({getTracks:()=>[{stop:()=>stopped++}]})});
  const capture=new RegionRecorder({video,makeCanvas,Recorder,onSegment:(file,meta)=>segments.push({file,meta}),
    onState:s=>states.push(s),requestFrame:cb=>{nextFrame=cb;return 1;},cancelFrame(){},
    ...(!nativeTimers?{setTimer:cb=>{timers.set(++timerId,cb);return timerId;},clearTimer:id=>timers.delete(id)}:{})});
  return {capture,video,segments,states,recorders,timers,step:t=>{video.currentTime=t;nextFrame?.(t*1000);},get stopped(){return stopped;}};
}
test('each timed segment is a complete independent video and pause discards unfinished encoding',()=>{
  const h=recorderHarness();h.capture.start({region:{x:0,y:0,width:1,height:1},mode:'interval',maxSeconds:2});
  h.step(.5);h.step(1);h.step(2);
  assert.equal(h.segments.length,1);assert.equal(h.recorders.length,2);
  assert.equal(h.segments[0].file.name.endsWith('.webm'),true);
  assert.equal(h.segments[0].meta.startSeconds,0);assert.equal(h.segments[0].meta.endSeconds,2);
  h.capture.pause();assert.equal(h.segments.length,1);assert.equal(h.stopped,1);
});
test('changed video dimensions pause capture and require selection confirmation',()=>{
  const h=recorderHarness();h.capture.start({region:{x:0,y:0,width:1,height:1}});
  h.video.videoWidth=1280;h.step(.5);
  assert.equal(h.capture.active,false);assert.equal(h.states.at(-1).status,'reselect');
  assert.equal(h.segments.length,0);
});
test('a stalled frame loop cannot record unbounded video in the background',()=>{
  const h=recorderHarness();h.capture.start({region:{x:0,y:0,width:1,height:1}});
  assert.equal(h.timers.size,1);[...h.timers.values()][0]();
  assert.equal(h.capture.active,false);assert.equal(h.stopped,1);
  assert.equal(h.timers.size,0);assert.equal(h.segments.length,0);
  assert.match(h.states.at(-1).message,/暂停|中断/);
});
test('default browser timers retain their Window receiver for screen and camera recordings',async()=>{
  const savedSet=globalThis.setTimeout,savedClear=globalThis.clearTimeout;
  globalThis.setTimeout=function(){assert.equal(this,globalThis,'native browser timer receiver');return 7;};
  globalThis.clearTimeout=function(){assert.equal(this,globalThis,'native browser timer receiver');};
  try{
    const h=recorderHarness({nativeTimers:true});
    assert.doesNotThrow(()=>h.capture.start({region:{x:0,y:0,width:1,height:1}}));h.capture.pause();
    class Recorder{static isTypeSupported(){return true;}start(){this.state='recording';}stop(){this.state='inactive';}}
    const camera=new CameraCapture({Recorder,getUserMedia:async()=>({getTracks:()=>[{stop(){}}]})});
    await camera.open();assert.equal(camera.start(),true);camera.close();
  }finally{globalThis.setTimeout=savedSet;globalThis.clearTimeout=savedClear;}
});
test('manual segmentation releases its canvas track when the next recorder fails to start',()=>{
  const h=recorderHarness();h.capture.start({region:{x:0,y:0,width:1,height:1}});h.step(.5);
  const first=h.recorders[0];const prototype=Object.getPrototypeOf(first),original=prototype.start;
  prototype.start=function(){throw new Error('device refused next recorder');};
  try{assert.doesNotThrow(()=>h.capture.submit());}finally{prototype.start=original;}
  assert.equal(h.capture.active,false);assert.equal(h.stopped,1);assert.equal(h.timers.size,0);
  assert.equal(h.states.at(-1).status,'error');
});
