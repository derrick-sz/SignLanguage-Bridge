import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {setImmediate} from 'node:timers/promises';
import {PlaybackGeneration} from '../slbridge/animation/web/player-core.mjs';

const source=fs.readFileSync(new URL('../slbridge/animation/web/workspace.js',import.meta.url),'utf8')
  .replace(/^import .*;\r?\n/gm,'').replace(/\binit\(\);\s*$/,'');
const deferred=()=>{let resolve;const promise=new Promise(r=>{resolve=r;});return {promise,resolve};};
const entry=(sample,text='今天开会')=>({id:'soke-full-'+sample+'-signature',sample_id:sample,text,
  aliases:[],fps:30,frames:5,duration:5/30,group:'corpus',split:'train',signer:1,state:'converted_candidate',playable:true});
const page=assets=>({assets,total:assets.length,offset:0,limit:30,has_more:false});
function harness(respond){
  const nodes=new Map();
  const make=(id='')=>({value:id==='speed'?'1':id==='corpus-quality'?'all':'',disabled:false,hidden:false,dataset:{},
    children:[],listeners:{},classList:{toggle(){}},appendChild(child){this.children.push(child);},replaceChildren(){this.children=[];},
    addEventListener(type,fn){this.listeners[type]=fn;},setAttribute(){},
    click(){return this.listeners.click?.();},dispatch(type){return this.listeners[type]?.({});}});
  const element=id=>{if(!nodes.has(id))nodes.set(id,make(id));return nodes.get(id);};
  class MotionStage{
    constructor(){this.ready=Promise.resolve();this.tracks={mappedBones:40,corrected:0};this.clock={playing:false};this.plays=0;}
    stop(){this.clock.playing=false;} async loadMotion(data,id,current,sha){this.last={data,id,sha};return current();}
    setSpeed(){} play(){this.plays++;this.clock.playing=true;}
  }
  const context=vm.createContext({PlaybackGeneration,MotionStage,AbortController,URLSearchParams,setTimeout,clearTimeout,
    document:{getElementById:element,querySelectorAll:()=>[],createElement:()=>make()},window:{addEventListener(){}},
    fetch:async(url,options)=>({ok:true,json:async()=>respond(url,options)})});
  vm.runInContext(source+`\nglobalThis.testApi={preview,stage,gate,
    available:typeof loadCorpusPage==='function'&&typeof selectCorpusAsset==='function',
    load:(...a)=>loadCorpusPage(...a),select:a=>selectCorpusAsset(a),
    enable:()=>{library='corpus';},
  };`,context);
  return {...context.testApi,element};
}

test('a late search page cannot overwrite newer corpus results',async()=>{
  const old=deferred(),fresh=deferred();let calls=0;
  const h=harness(()=>++calls===1?old.promise:fresh.promise);
  assert.equal(h.available,true,'corpus page controller must exist');h.enable();
  h.element('corpus-search').value='旧';const a=h.load(0);
  h.element('corpus-search').value='新';const b=h.load(0);
  fresh.resolve(page([entry('S000002_P0000_T00','新结果')]));await b;
  old.resolve(page([entry('S000001_P0000_T00','旧结果')]));await a;
  assert.equal(h.element('corpus-list').children[0].textContent,'新结果');
});

test('paging does not discard the selected sample and playback binds its derived SHA',async()=>{
  const chosen=entry('S000001_P0000_T00');const requests=[];
  const h=harness((url,options)=>{
    requests.push({url,options});
    if(url.endsWith('/corpus/plans'))return {can_play:true,items:[{sample_id:chosen.sample_id,asset_id:chosen.id,sha256:'a'.repeat(64),asset:chosen}]};
    if(url.includes('/corpus/motions/'))return {clip:{id:chosen.id,source_id:chosen.sample_id},sha256:'a'.repeat(64)};
    return page([entry('S000002_P0000_T00','另一页')]);
  });
  assert.equal(h.available,true);h.enable();h.select(chosen);await h.load(30);await h.preview();
  assert.equal(h.stage.plays,1);
  const body=JSON.parse(requests.find(r=>r.url.endsWith('/corpus/plans')).options.body);
  assert.equal(body.sample_id,chosen.sample_id);assert.equal(body.text,chosen.text);
  assert.ok(requests.some(r=>r.url.includes('/corpus/motions/'+chosen.sample_id+'?expected_sha256='+'a'.repeat(64))));
  assert.equal(h.stage.last.sha,'a'.repeat(64));
});

test('Stop rejects a late corpus motion and mismatched motion responses never play',async()=>{
  const chosen=entry('S000001_P0000_T00'),late=deferred();
  const h=harness(url=>url.endsWith('/corpus/plans')?{can_play:true,items:[{sample_id:chosen.sample_id,asset_id:chosen.id,sha256:'a'.repeat(64),asset:chosen}]}:late.promise);
  assert.equal(h.available,true);h.enable();h.select(chosen);const pending=h.preview();await setImmediate();
  h.element('stop').click();late.resolve({clip:{id:chosen.id},sha256:'a'.repeat(64)});await pending;assert.equal(h.stage.plays,0);
  const wrong=harness(url=>url.endsWith('/corpus/plans')?{can_play:true,items:[{sample_id:chosen.sample_id,asset_id:chosen.id,sha256:'a'.repeat(64),asset:chosen}]}:{clip:{id:'wrong'},sha256:'b'.repeat(64)});
  wrong.enable();wrong.select(chosen);await wrong.preview();assert.equal(wrong.stage.plays,0);
});
