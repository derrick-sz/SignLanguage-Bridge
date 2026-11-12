import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import vm from 'node:vm';
import {fullRegion,videoRect} from '../slbridge/animation/web/communication/selection.mjs';
import {demoItems} from '../slbridge/animation/web/communication/realtime-demo.mjs';

// Exercise the real page handler with isolated device/network collaborators.
test('page start forwards the current assistance opt-in',async()=>{
  const source=await readFile(new URL('../slbridge/animation/web/communication/realtime-app.mjs',import.meta.url),'utf8');
  const nodes=new Map(),calls=[];
  const node=()=>({dataset:{},style:{},classList:{toggle(){}},videoWidth:640,videoHeight:480,
    clientWidth:640,clientHeight:480,checked:false,addEventListener(){},replaceChildren(){},pause(){},
    removeAttribute(){},load(){},getBoundingClientRect(){return{x:0,y:0};}});
  const get=id=>{if(!nodes.has(id))nodes.set(id,node());return nodes.get(id);};
  class Client{
    status='idle';snapshot(){return{status:this.status,words:[],captions:[],firstWordMs:null};}
    async start(options){calls.push(options);throw new Error('isolated before device capture');}
    stop(){}clear(){}
  }
  const context={RealtimeClient:Client,LiveFrameCapture:class{stop(){}},ScreenSource:class{close(){}},
    fullRegion:()=>({x:0,y:0,width:1,height:1}),videoRect:()=>({x:0,y:0,width:640,height:480}),selectRegion(){},
    document:{getElementById:get,createElement:node,hidden:false,addEventListener(){}},
    navigator:{mediaDevices:{}},window:{addEventListener(){}},ResizeObserver:class{observe(){}},
    fetch:async()=>({ok:true,json:async()=>({ready:true,features:{llm_assistance:true}})}),
    setTimeout:()=>1,clearTimeout(){},URL,console};
  vm.runInNewContext(source.replace(/^import .*;\r?\n/gm,''),context);
  get('full').onclick();
  for(const checked of [false,true]){
    get('assistance').checked=checked;await get('start').onclick();
    assert.equal(calls.at(-1).assistance,checked);
  }
});

test('the real resize callback hides selection through a zero-size layout and restores it',async()=>{
  const source=await readFile(new URL('../slbridge/animation/web/communication/realtime-app.mjs',import.meta.url),'utf8');
  const nodes=new Map();let resize;
  const node=()=>({dataset:{},style:{},classList:{toggle(){}},videoWidth:640,videoHeight:480,
    clientWidth:640,clientHeight:480,checked:false,addEventListener(){},replaceChildren(){},pause(){},
    removeAttribute(){},load(){},getBoundingClientRect(){return{x:0,y:0};}});
  const get=id=>{if(!nodes.has(id))nodes.set(id,node());return nodes.get(id);};
  class Client{
    status='idle';snapshot(){return{status:this.status,words:[],captions:[],firstWordMs:null,firstWordTotalMs:null};}
    stop(){}clear(){}
  }
  const context={RealtimeClient:Client,LiveFrameCapture:class{stop(){}},ScreenSource:class{close(){}},
    fullRegion,videoRect,selectRegion(){},document:{getElementById:get,createElement:node,hidden:false,addEventListener(){}},
    navigator:{mediaDevices:{}},window:{addEventListener(){}},ResizeObserver:class{constructor(fn){resize=fn;}observe(){}},
    fetch:async()=>({ok:true,json:async()=>({ready:true})}),setTimeout:()=>1,clearTimeout(){},URL,console};
  vm.runInNewContext(source.replace(/^import .*;\r?\n/gm,''),context);
  get('full').onclick();assert.equal(get('selection').hidden,false);
  for(const dimensions of [[0,480],[640,0],[0,0]]){
    [get('preview').clientWidth,get('preview').clientHeight]=dimensions;
    assert.doesNotThrow(()=>resize());assert.equal(get('selection').hidden,true);
  }
  get('preview').clientWidth=640;get('preview').clientHeight=480;resize();
  assert.equal(get('selection').hidden,false);assert.equal(get('selection').style.width,'640px');
});

test('demo starts at zero and a late old playback cannot cancel a newer file source',async()=>{
  const source=await readFile(new URL('../slbridge/animation/web/communication/realtime-app.mjs',import.meta.url),'utf8');
  const nodes=new Map(),calls=[],playbacks=[],captures=[];let stops=0;
  const node=key=>({dataset:{},style:{},classList:{toggle(){}},videoWidth:512,videoHeight:512,
    clientWidth:640,clientHeight:480,checked:false,currentTime:0,addEventListener(){},
    replaceChildren(...children){this.children=children;if(key==='demo-choice')this.value=children[0]?.value;},
    pause(){},removeAttribute(){},load(){this.onloadedmetadata?.();},
    play(){return new Promise(resolve=>playbacks.push(resolve));},getBoundingClientRect(){return{x:0,y:0};}});
  const get=id=>{if(!nodes.has(id))nodes.set(id,node(id));return nodes.get(id);};
  class Client{
    status='idle';snapshot(){return{status:this.status,words:[],captions:[],firstWordMs:null,firstWordTotalMs:null};}
    async start(options){calls.push(options);this.status='running';}
    stop(){stops++;this.status='stopped';}clear(){}noteCaptureStart(){}
  }
  const entry={id:'school',title:'校园',annotation:'仅作人参考的标注',note:'训练回放',kind:'training_replay',source_split:'train',
    fps:30,frames:85,width:512,height:512,duration_seconds:85/30,sha256:'a'.repeat(64),media_url:'/v1/communication/demos/school/video?sha256='+'a'.repeat(64)};
  const context={RealtimeClient:Client,LiveFrameCapture:class{stop(){}start(options){captures.push(options);}},ScreenSource:class{close(){}},
    fullRegion,videoRect,demoItems,selectRegion(){},document:{getElementById:get,createElement:node,hidden:false,addEventListener(){}},
    navigator:{mediaDevices:{}},window:{addEventListener(){}},ResizeObserver:class{observe(){}},
    fetch:async path=>({ok:true,json:async()=>path.endsWith('/demos')?{enabled:true,schema_version:1,sign_language:'csl',catalog_sha256:'b'.repeat(64),items:[entry]}:{ready:true}}),
    setTimeout:()=>1,clearTimeout(){},URL,console};
  vm.runInNewContext(source.replace(/^import .*;\r?\n/gm,''),context);
  await new Promise(r=>setImmediate(r));get('demo-load').onclick();get('video').currentTime=1;
  const oldStart=get('start').onclick();await new Promise(r=>setImmediate(r));
  assert.equal(get('video').currentTime,0);assert.equal(calls[0].source.kind,'training_replay');
  assert.equal(calls[0].source.annotation,undefined);
  const oldMetadata=get('video').onloadedmetadata;
  const file=new Blob(['local']);file.name='own.mp4';get('file').files=[file];get('file').onchange();
  const newStart=get('start').onclick();await new Promise(r=>setImmediate(r));
  playbacks[1]();await newStart;const afterNew=stops;
  playbacks[0]();await oldStart;
  assert.equal(stops,afterNew);assert.equal(captures.length,1);
  assert.equal(calls[1].source.kind,'file');assert.match(get('source-label').textContent,/本地视频/);
  oldMetadata();assert.match(get('source-label').textContent,/本地视频/);
});
