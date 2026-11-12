import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import vm from 'node:vm';
import {webcrypto} from 'node:crypto';
import {Conversation as ActualConversation,readableTranscript} from '../slbridge/animation/web/communication/conversation.mjs';
import {RealtimeHandoffReceiver as ActualReceiver,createRealtimeTransfer,TRANSFER_PROTOCOL} from '../slbridge/animation/web/communication/conversation-transfer.mjs';

const origin='http://127.0.0.1:8879';
const caption=()=>({id:'session:0',revision:1,status:'committed',confirmed:true,text:'人工核对的句子。',original_text:'机器原译文。',
  word_evidence:['原词'],review_hint:{reason:'single_word_expansion'},source:{kind:'training_replay',label:'训练示范回放 · CSL-Daily train',id:'school-reading'}});
const allText=node=>[node.textContent,...node.children.map(allText)].join('\n');

async function page({hash='#realtime-dialogue',hasOpener=true}={}){
  const source=await readFile(process.env.SLBRIDGE_APP_TEST_SOURCE||new URL('../slbridge/animation/web/communication/app.js',import.meta.url),'utf8');
  const nodes=new Map(),windowEvents={},generated=[],posted=[],exports=[];let conversation,receiver;
  const node=()=>({value:'',dataset:{},children:[],events:{},attributes:{},lastChild:{textContent:''},textContent:'',
    classList:{toggle(){},remove(){}},hidden:false,disabled:false,scrollHeight:0,
    addEventListener(type,fn){this.events[type]=fn;},append(...children){this.children.push(...children);},
    replaceChildren(...children){this.children=children;},setAttribute(key,value){this.attributes[key]=value;},
    getAttribute(key){return this.attributes[key];},removeAttribute(){},scrollIntoView(){},focus(){this.focused=true;},click(){}});
  const get=id=>{if(!nodes.has(id))nodes.set(id,node());return nodes.get(id);};
  const opener=hasOpener?{postMessage(data,target){posted.push({data,target});}}:null;
  class Conversation extends ActualConversation{constructor(){super();conversation=this;}}
  class Receiver extends ActualReceiver{constructor(options){super({...options,listen:()=>()=>{}});receiver=this;}}
  class Controller{constructor(options){this.options=options;}invalidate(){}cancel(){}pause(){}play(){}stop(){}
    generate(text){generated.push(text);this.options.onState({status:'ready',subtitle:text});}}
  class MotionStage{constructor(){this.ready=Promise.resolve();this.renderer={domElement:node()};this.clock={playing:false};this.tracks={};}dispose(){}}
  class ExportURL extends URL{static createObjectURL(blob){exports.push(blob);return 'blob:fixture';}static revokeObjectURL(){}}
  vm.runInNewContext(source.replace(/^import .*;\r?\n/gm,''),{Conversation,readableTranscript,RealtimeHandoffReceiver:Receiver,
    MotionStage,FastCompositionController:Controller,GenerationController:Controller,
    RecognitionController:class{invalidate(){}},CameraCapture:class{close(){}},
    document:{getElementById:get,createElement:node,querySelectorAll:()=>[]},
    window:{opener,location:{origin,hash},addEventListener(type,fn){windowEvents[type]=fn;}},navigator:{},
    crypto:webcrypto,AbortSignal,Blob,URL:ExportURL,setTimeout:()=>1,clearTimeout(){},clearInterval(){},
    fetch:async path=>({ok:true,json:async()=>path.endsWith('/recognition')?
      {ready:true,languages:[{id:'csl',ready:true,label:'中国手语',scope:'fixture'}]}:
      path.endsWith('/composition')?{enabled:true,planner:{state:'ready'}}:{enabled:false}})});
  await new Promise(resolve=>setImmediate(resolve));
  return {get,conversation,receiver,generated,posted,exports,windowEvents,async receive(payload){
    assert.ok(receiver,'the actual app must install the realtime receiver');
    const base={protocol:TRANSFER_PROTOCOL,nonce:'nonce-app-abcdefgh',expiresAt:Date.now()+9000};
    await receiver.receive({origin,source:opener,data:{...base,kind:'hello'}});
    await receiver.receive({origin,source:opener,data:{...base,kind:'transfer',payload}});
  }};
}

test('actual app receives reviewed text without overwriting a draft or starting generation',async()=>{
  const h=await page();h.get('reply-text').value='我已有一条回复草稿。';h.get('reply-text').events.input();
  const payload=await createRealtimeTransfer(caption());await h.receive(payload);
  assert.equal(h.get('reply-text').value,'我已有一条回复草稿。');assert.deepEqual(h.generated,[]);
  assert.equal(h.conversation.messages.length,1);assert.equal(h.get('conversation-list').children.length,1);
  const rendered=allText(h.get('conversation-list'));assert.match(rendered,/训练示范手语/);
  assert.match(rendered,/机器原译文/);assert.match(rendered,/文字已核对/);assert.match(rendered,/人工核对的句子/);
  assert.equal(h.conversation.messages[0].review_status,'unverified');
  await h.receive(payload);assert.equal(h.conversation.messages.length,1);assert.equal(h.posted.at(-1).data.added,false);
  await h.receive(await createRealtimeTransfer({...caption(),text:'再次人工修订。'}));
  assert.equal(h.conversation.messages.length,2);assert.equal(h.conversation.messages[0].text,'人工核对的句子。');
  assert.deepEqual(h.generated,[]);assert.equal(h.get('send-reply').disabled,false);
  h.get('reply-form').events.submit({preventDefault(){}});
  assert.deepEqual(h.generated,['我已有一条回复草稿。']);assert.equal(h.conversation.messages.length,3);
  assert.equal(h.conversation.messages.at(-1).direction,'sent');
  h.get('export-conversation').events.click();const exported=await h.exports[0].text();
  assert.match(exported,/训练示范回放/);assert.match(exported,/机器原译文/);assert.match(exported,/我已有一条回复草稿/);
  assert.ok(h.posted.every(x=>x.target===origin));h.windowEvents.pagehide();assert.equal(h.receiver.disposed,true);
});

test('normal app visits do not attach a cross-window receiver',async()=>{
  for(const options of [{hash:''},{hasOpener:false}]){
    const h=await page(options);assert.equal(h.receiver,undefined);assert.deepEqual(h.generated,[]);h.windowEvents.pagehide();
  }
});
