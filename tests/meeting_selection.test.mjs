import test from 'node:test';
import assert from 'node:assert/strict';
import {videoRect,selectRegion,cropRect,encodeSize} from '../slbridge/animation/web/communication/selection.mjs';

test('a letterboxed 16:9 meeting frame maps a reverse drag to real source pixels',()=>{
  assert.deepEqual(videoRect(1920,1080,800,600),{x:0,y:75,width:800,height:450});
  const region=selectRegion({x:700,y:450},{x:100,y:150},1920,1080,800,600);
  assert.deepEqual(cropRect(region,1920,1080),{x:240,y:180,width:1440,height:720});
});
test('dragging from blank bars or selecting a tiny patch cannot create a crop',()=>{
  assert.equal(selectRegion({x:200,y:20},{x:500,y:400},1920,1080,800,600),null);
  assert.equal(selectRegion({x:100,y:100},{x:102,y:102},1920,1080,800,600),null);
});
test('crop bounds reject invalid numbers and encoding stays within an even video size',()=>{
  for(const region of [{x:-.1,y:0,width:1,height:1},{x:0,y:0,width:NaN,height:1},{x:.8,y:0,width:.4,height:1}]){
    assert.throws(()=>cropRect(region,1920,1080));
  }
  assert.deepEqual(encodeSize({width:1920,height:1080}),{width:960,height:540});
  assert.deepEqual(encodeSize({width:1080,height:1920}),{width:404,height:720});
});
