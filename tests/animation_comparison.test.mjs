import test from 'node:test';
import assert from 'node:assert/strict';
import * as player from '../slbridge/animation/web/player-core.mjs';

test('split comparison preserves both panes and stacks them on a narrow stage', () => {
  assert.equal(typeof player.comparisonViewports,'function');
  const wide=player.comparisonViewports(1000,500,true);
  assert.deepEqual(wide.map(v=>[v.x,v.y,v.width,v.height]),[[0,0,500,500],[500,0,500,500]]);
  const narrow=player.comparisonViewports(400,600,true);
  assert.deepEqual(narrow.map(v=>[v.x,v.y,v.width,v.height]),[[0,300,400,300],[0,0,400,300]]);
  assert.equal(player.comparisonViewports(400,600,false).length,1);
});

test('camera includes off-center hands extending toward the viewer', () => {
  const bounds={minX:-0.1,maxX:1.1,minY:0.9,maxY:1.9,minZ:-0.1,maxZ:0.8};
  const c=player.signingCamera(bounds,0.7,34);
  assert.equal(c.centerX,0.5);
  assert.ok(Number.isFinite(c.centerZ));
  for(const x of [bounds.minX,bounds.maxX])for(const z of [bounds.minZ,bounds.maxZ]){
    const depth=c.distance+c.centerZ-z;
    assert.ok(Math.abs(x-c.centerX)<=depth*Math.tan(34*Math.PI/360)*0.7);
  }
});
