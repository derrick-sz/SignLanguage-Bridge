import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import vm from 'node:vm';
import {RealtimeClient as ActualClient} from '../slbridge/animation/web/communication/realtime-client.mjs';
import {LiveFrameCapture as ActualCapture} from '../slbridge/animation/web/communication/realtime-capture.mjs';
import {fullRegion,videoRect} from '../slbridge/animation/web/communication/selection.mjs';
import {demoItems} from '../slbridge/animation/web/communication/realtime-demo.mjs';

async function page({captureError,realCapture=false}={}){
  const source=await readFile(new URL('../slbridge/animation/web/communication/realtime-app.mjs',import.meta.url),'utf8');
  const nodes=new Map(),sockets=[],timers=new Map(),transfers=[];let client,capture,handoff,clock=0,timerId=0;
  const node=()=>({dataset:{},style:{},classList:{toggle(){}},children:[],textContent:'',videoWidth:512,videoHeight:512,readyState:4,
    clientWidth:640,clientHeight:480,currentTime:0,checked:false,events:{},
    addEventListener(type,fn){this.events[type]=fn;},replaceChildren(...children){this.children=children;},
    append(...children){this.children.push(...children);},pause(){},play:async()=>{},removeAttribute(){},
    load(){this.onloadedmetadata?.();},getBoundingClientRect(){return{x:0,y:0};}});
  const get=id=>{if(!nodes.has(id))nodes.set(id,node());return nodes.get(id);};
  class Socket{
    constructor(){this.readyState=0;this.bufferedAmount=0;this.sent=[];this.sequence=0;this.id='run-'+sockets.length;sockets.push(this);}
    send(value){this.sent.push(JSON.parse(value));}close(){this.readyState=3;this.onclose?.();}
    open(){this.readyState=1;this.onopen?.();}
    event(type,value={}){this.onmessage?.({data:JSON.stringify({type,protocol:'csl-realtime-v1',session_id:this.id,
      source_epoch:this.sent[0].source_epoch,sequence:this.sequence++,...value})});}
  }
  class Client extends ActualClient{constructor(options){super({...options,url:'ws://fixture',makeSocket:()=>new Socket()});client=this;}}
  class Capture{
    constructor(options){
      if(realCapture)return capture=new ActualCapture({...options,now:()=>clock,requestFrame:()=>1,cancelFrame(){},
        setTimer:fn=>{timers.set(++timerId,fn);return timerId;},clearTimer:id=>timers.delete(id),
        makeCanvas:()=>({getContext:()=>({fillRect(){},drawImage(){}}),toBlob(){}})});
      this.options=options;this.active=false;capture=this;
    }
    start(){this.active=!captureError;if(captureError)this.options.onError(captureError);}
    stop(){this.active=false;}async finish(){this.active=false;return true;}
  }
  class Sender{
    constructor(options){this.options=options;this.busy=false;handoff=this;}
    send(caption){transfers.push(caption);this.busy=true;this.options.onState({status:'waiting',message:'交接中'});
      return new Promise(resolve=>{this.complete=()=>{this.busy=false;this.options.onState({status:'sent',message:'已加入对话'});resolve({added:true});};});}
    dispose(){}
  }
  vm.runInNewContext(source.replace(/^import .*;\r?\n/gm,''),{RealtimeClient:Client,LiveFrameCapture:Capture,
    RealtimeHandoffSender:Sender,
    ScreenSource:class{close(){}},fullRegion,videoRect,demoItems,selectRegion(){},
    document:{getElementById:get,createElement:node,hidden:false,addEventListener(){}},navigator:{mediaDevices:{}},
    window:{location:{origin:'http://127.0.0.1:8879'},addEventListener(){}},ResizeObserver:class{observe(){}},URL,console,setTimeout:()=>1,clearTimeout(){},
    fetch:async path=>({ok:true,json:async()=>path.endsWith('/demos')?{enabled:false}:{ready:true}})});
  await new Promise(resolve=>setImmediate(resolve));
  return {get,client,transfers,get handoff(){return handoff;},choose(name){const file=new Blob(['fixture']);file.name=name;get('file').files=[file];get('file').onchange();},
    async start(){const pending=get('start').onclick();const socket=sockets.at(-1);socket.open();socket.event('ready');await pending;return socket;},
    end:()=>get('video').events.ended(),capture,
    watch(time){clock=time;const jobs=[...timers.values()];timers.clear();for(const fn of jobs)fn();}};
}

test('real page uses the current attempt while reviewed older captions remain in history',async()=>{
  const h=await page();h.choose('first.mp4');const old=await h.start();
  old.event('sentence',{utterance_id:0,revision:1,status:'committed',text:'上一轮确认文本',predicted_gloss:['原词']});
  h.client.confirm(h.client.snapshot().captions[0].id);
  assert.equal(h.get('sentence').textContent,'上一轮确认文本');
  h.choose('next.mp4');assert.equal(h.get('sentence').textContent,'完整表达会在这里出现');
  await h.start();assert.equal(h.get('sentence').textContent,'完整表达会在这里出现');
  assert.equal(h.client.snapshot().captions[0].confirmed,true);
  assert.equal(h.get('history').children.length,1);h.client.stop();
});

test('actual EOF handler and zero-frame native events leave a clear retry state',async()=>{
  const h=await page();h.choose('empty.mp4');const socket=await h.start();await h.end();
  assert.equal(socket.sent.at(-1).type,'finish');
  socket.event('summary',{frames_processed:0});socket.event('finished');
  assert.equal(h.get('status').dataset.error,'true');assert.match(h.get('status').textContent,/未采集.*画面/);
  assert.equal(h.get('sentence').textContent,'完整表达会在这里出现');
});

test('a synchronous first-frame error cannot be overwritten by the start continuation',async()=>{
  const h=await page({captureError:'无法读取视频画面，请重新载入后重试。'});h.choose('blocked.mp4');await h.start();
  assert.match(h.get('status').textContent,/无法读取视频画面/);
  assert.equal(h.get('status').dataset.error,'true');assert.equal(h.get('stop').disabled,true);
});

test('native error during a real pending EOF drain is not replaced by the later watchdog',async()=>{
  const h=await page({realCapture:true});h.choose('pending.mp4');const socket=await h.start();
  const ending=h.end();assert.equal(h.capture.draining,true);
  socket.event('error',{message:'服务原始错误'});h.watch(2001);await ending;
  assert.equal(h.get('status').textContent,'服务原始错误');assert.equal(h.get('status').dataset.error,'true');
  assert.equal(h.capture.draining,false);
});

test('real realtime handlers require paused confirmed captions and transfer the latest text only once at a time',async()=>{
  const h=await page();h.choose('dialogue.mp4');const socket=await h.start();
  socket.event('sentence',{utterance_id:0,revision:1,status:'committed',text:'机器原文',predicted_gloss:['原词']});
  const id=h.client.snapshot().captions[0].id;
  const join=()=>h.get('history').children[0].children[0].children.find(x=>x.textContent==='加入对话并回复');
  assert.equal(join().disabled,true);await join().onclick();assert.equal(h.transfers.length,0);
  h.client.confirm(id);assert.equal(join().disabled,true);await join().onclick();assert.equal(h.transfers.length,0);
  await h.end();assert.equal(join().disabled,true);
  socket.event('summary',{frames_processed:20});socket.event('finished');assert.equal(join().disabled,false);
  const oldButton=join();h.client.edit(id,'人工更新的文字');assert.equal(join().disabled,false);
  const first=oldButton.onclick();assert.equal(h.transfers.length,1);
  assert.equal(h.transfers[0].text,'人工更新的文字');assert.equal(h.transfers[0].original_text,'机器原文');
  assert.equal(join().disabled,true);assert.equal(h.get('start').disabled,true);
  await oldButton.onclick();assert.equal(h.transfers.length,1);h.handoff.complete();await first;
  assert.equal(join().disabled,false);assert.equal(h.client.snapshot().captions.length,1);
  assert.equal(h.client.snapshot().captions[0].confirmed,true);assert.equal(h.get('handoff-status').textContent,'已加入对话');
});
