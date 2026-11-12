import test from 'node:test';
import assert from 'node:assert/strict';
const tick=()=>new Promise(resolve=>setImmediate(resolve));

async function pageHarness({deferPlay=false}={}){
  const elements=new Map(),handlers={},recorders=[],requests=[];let nextFrame,resolvePlay;
  const context={drawImage(){},clearRect(){},getImageData:()=>({data:new Uint8ClampedArray(32*24*4)})};
  class Element{
    constructor(){this.listeners={};this.dataset={};this.style={};this.children=[];this.value='';this.hidden=false;this.clientWidth=800;this.clientHeight=450;this.width=160;this.height=100;}
    addEventListener(name,fn){(this.listeners[name]??=[]).push(fn);}
    async fire(name,event={}){for(const fn of this.listeners[name]||[])await fn({preventDefault(){},...event});}
    setAttribute(name,value){this[name]=value;}removeAttribute(){}replaceChildren(...values){this.children=values;}append(...values){this.children.push(...values);}
    getContext(){return context;}captureStream(){return {getTracks:()=>[{stop(){}}]};}
    getBoundingClientRect(){return {x:0,y:0,left:0,top:0,width:800,height:450};}
    pause(){this.paused=true;}play(){this.paused=false;return deferPlay?new Promise(resolve=>resolvePlay=resolve):Promise.resolve();}
    load(){}showModal(){}close(){}focus(){}
  }
  const el=id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);};
  const video=el('meeting-video');Object.assign(video,{videoWidth:640,videoHeight:480,currentTime:0,readyState:4});
  el('sign-language').value='csl';el('segment-mode').value='interval';
  class Recorder{
    static isTypeSupported(){return true;}
    constructor(){this.state='inactive';this.mimeType='video/webm';recorders.push(this);}
    start(){this.state='recording';}stop(){this.state='inactive';}
    deliver(){this.ondataavailable?.({data:new Blob(['independent video'])});this.onstop?.();}
  }
  const replacements={document:{hidden:false,getElementById:el,querySelector:el,createElement:()=>new Element(),addEventListener:(name,fn)=>handlers['document:'+name]=fn},
    window:{addEventListener:(name,fn)=>handlers[name]=fn},navigator:{mediaDevices:{}},
    ResizeObserver:class{observe(){}},MediaRecorder:Recorder,
    requestAnimationFrame:fn=>{nextFrame=fn;return 1;},cancelAnimationFrame(){},
    fetch:async(url,options)=>{
      if(!url.includes('/recognitions?'))return {ok:true,json:async()=>({ready:true,languages:[{id:'csl',ready:true,label:'中国手语',scope:'句子识别'}]})};
      return new Promise(resolve=>requests.push({finish:text=>resolve({ok:true,json:async()=>({request_id:new URL(url,'http://local').searchParams.get('request_id'),recognized:true,sign_language:'csl',text,output_kind:'text'})})}));
    }};
  const saved=new Map(Object.keys(replacements).map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));
  for(const [key,value] of Object.entries(replacements))Object.defineProperty(globalThis,key,{configurable:true,writable:true,value});
  await import('../slbridge/animation/web/communication/meeting-app.js?scenario='+Math.random());await tick();
  el('source-file').files=[new File(['source'],'meeting.mp4')];await el('source-file').fire('change');video.onloadeddata();
  return {el,video,recorders,requests,resolvePlay:()=>resolvePlay(),hide:()=>{document.hidden=true;handlers['document:visibilitychange']?.();},step:time=>{video.currentTime=time;nextFrame?.(time*1000);},
    cleanup(){handlers.pagehide?.();for(const [key,descriptor] of saved){if(descriptor)Object.defineProperty(globalThis,key,descriptor);else delete globalThis[key];}}};
}

test('reselection cancels a pending playback start without allowing its continuation to resume capture',async()=>{
  const page=await pageHarness({deferPlay:true});
  try{
    const starting=page.el('start-captions').fire('click');
    await page.el('reselect').fire('click');page.resolvePlay();await starting;
    assert.equal(page.recorders.length,0);assert.equal(page.el('pause-captions').hidden,true);
    assert.match(page.el('capture-status').textContent,/暂停/);
  }finally{page.cleanup();}
});
test('a pending playback start cannot resume capture after the page becomes hidden',async()=>{
  const page=await pageHarness({deferPlay:true});
  try{
    const starting=page.el('start-captions').fire('click');page.hide();page.resolvePlay();await starting;
    assert.equal(page.recorders.length,0);assert.match(page.el('capture-status').textContent,/后台/);
  }finally{page.cleanup();}
});
test('a previous HTTP completion during final recorder shutdown cannot discard the final subtitle',async()=>{
  const page=await pageHarness();
  try{
    await page.el('start-captions').fire('click');assert.equal(page.el('caption-state').textContent,'正在采集画面');
    page.step(1);page.step(3);page.step(6);page.recorders[0].deliver();
    assert.equal(page.requests.length,1);page.step(7);page.video.currentTime=8.2;
    const ending=page.video.fire('ended');page.requests[0].finish('前半句');await tick();
    page.recorders[1].deliver();await ending;await tick();assert.equal(page.requests.length,2);
    page.requests[1].finish('后半句');await tick();assert.equal(Number(page.el('caption-count').textContent),2);
  }finally{page.cleanup();}
});
