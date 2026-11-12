import test from 'node:test';
import assert from 'node:assert/strict';
import {demoItems} from '../slbridge/animation/web/communication/realtime-demo.mjs';

const item=()=>({id:'school',title:'校园交流',annotation:'明天我想去学校看书。',kind:'training_replay',
  source_split:'train',sample_id:'S000062_P0000_T00',sha256:'a'.repeat(64),fps:30,frames:85,width:512,height:512,
  duration_seconds:85/30,note:'已测训练回放。',media_url:'/v1/communication/demos/school/video?sha256='+'a'.repeat(64)});
const catalog=entry=>({enabled:true,schema_version:1,sign_language:'csl',catalog_sha256:'b'.repeat(64),items:[entry]});

test('only fixed same-origin train media become selectable demos',()=>{
  assert.equal(demoItems(catalog(item()))[0].annotation,'明天我想去学校看书。');
  assert.deepEqual(demoItems({enabled:false,items:[]}),[]);
  for(const change of [{media_url:'https://elsewhere.invalid/video.mp4'},{source_split:'test'},
      {fps:Infinity},{width:4096},{kind:'camera'},{id:'../outside'}]){
    assert.throws(()=>demoItems(catalog({...item(),...change})));
  }
});

test('duplicate and oversized catalogs are rejected as a whole',()=>{
  const data=catalog(item());data.items.push(item());assert.throws(()=>demoItems(data));
  data.items=Array.from({length:17},()=>item());assert.throws(()=>demoItems(data));
});
