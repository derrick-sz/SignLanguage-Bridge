import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {PlaybackGeneration} from '../slbridge/animation/web/player-core.mjs';

test('brow/eyelid inputs never synthesize a manual jaw override; reset releases it',()=>{
  const nodes=new Map();
  const element=id=>{
    if(!nodes.has(id))nodes.set(id,{value:'0',checked:false,listeners:{},addEventListener(type,fn){this.listeners[type]=fn;}});
    return nodes.get(id);
  };
  class MotionStage {
    constructor(){this.expression={};this.nonmanualAvailable=true;this.nonmanualEnabled=true;}
    setExpression(channels){this.last=channels;this.expression=Object.keys(channels).length?{...this.expression,...channels}:{};}
    setNonmanualEnabled(value){this.nonmanualEnabled=value;}
  }
  const context=vm.createContext({PlaybackGeneration,MotionStage,AbortController,
    document:{getElementById:element,querySelectorAll:()=>[]},window:{addEventListener(){}}});
  const source=fs.readFileSync(new URL('../slbridge/animation/web/workspace.js',import.meta.url),'utf8')
    .replace(/^import .*;\r?\n/gm,'').replace(/\binit\(\);\s*$/,'');
  vm.runInContext(source+'\nglobalThis.stageReview=stage;',context);
  element('face-brow').value='0.5';element('face-brow').listeners.input();
  assert.equal(Object.hasOwn(context.stageReview.last,'jawOpen'),false);
  element('face-jaw').value='0.75';element('face-jaw').listeners.input();
  assert.equal(context.stageReview.expression.jawOpen,0.75);
  element('face-blink').value='0.1';element('face-blink').listeners.input();
  assert.equal(Object.hasOwn(context.stageReview.last,'jawOpen'),false);
  element('face-reset').listeners.click();assert.equal(Object.hasOwn(context.stageReview.expression,'jawOpen'),false);
  element('nonmanual-jaw').checked=false;element('nonmanual-jaw').listeners.change();
  assert.equal(context.stageReview.nonmanualEnabled,false);
});
