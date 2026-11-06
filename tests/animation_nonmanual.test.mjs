import test from 'node:test';
import assert from 'node:assert/strict';
const jaw = await import('../slbridge/animation/web/nonmanual.mjs').catch(() => ({}));
const poses = values => values.map(v => Array.from({length:55},(_,i)=>i===22?[...v]:[0,0,0]));

test('jaw candidate follows measured positive X opening without mutating source poses', () => {
  assert.equal(typeof jaw.compileJawCandidate,'function');
  const input=poses(Array.from({length:5},()=>[0.08,0,0]));
  const before=JSON.stringify(input);
  const track=jaw.compileJawCandidate(input,30,'smplx-jaw-godette-calibrated-v1');
  assert.ok(track.angles.every(a=>Math.abs(a-0.08)<1e-10));
  assert.equal(JSON.stringify(input),before);
  for(const rotation of [[-0.08,0,0],[0,0.08,0],[0,0,0]]) {
    assert.ok(jaw.compileJawCandidate(poses([rotation,rotation,rotation]),30,track.profile).angles.every(a=>a===0));
  }
});

test('automatic jaw requires explicit calibrated source and avatar identities and no baked export', () => {
  assert.equal(typeof jaw.jawCandidateForClip,'function');
  const descriptor={profile:'smplx-jaw-godette-calibrated-v1',source_joint:22,
    source_template_sha256:jaw.JAW_TEMPLATE_SHA256,target_avatar_sha256:jaw.JAW_AVATAR_SHA256,
    status:'research_candidate',export_supported:false};
  const clip={poses:poses([[0.08,0,0],[0.08,0,0]]),fps:30,template_sha256:jaw.JAW_TEMPLATE_SHA256,nonmanual_candidate:descriptor};
  assert.ok(jaw.jawCandidateForClip(clip,jaw.JAW_AVATAR_SHA256,false));
  assert.equal(jaw.jawCandidateForClip({...clip,nonmanual_candidate:undefined},jaw.JAW_AVATAR_SHA256,false),null);
  assert.equal(jaw.jawCandidateForClip(clip,jaw.JAW_AVATAR_SHA256,true),null);
  assert.equal(jaw.jawCandidateForClip(clip,'unknown-avatar',false),null);
  for(const patch of [{profile:'unknown'},{source_joint:21},{source_template_sha256:'wrong'},
      {target_avatar_sha256:'wrong'},{status:'approved'},{export_supported:true}]) {
    assert.equal(jaw.jawCandidateForClip({...clip,nonmanual_candidate:{...descriptor,...patch}},jaw.JAW_AVATAR_SHA256,false),null);
  }
  assert.equal(jaw.jawCandidateForClip({...clip,template_sha256:'other'},jaw.JAW_AVATAR_SHA256,false),null);
});

test('source spikes cannot make the mouth jump; amplitude and speed remain bounded on seeks', () => {
  const input=poses([[0,0,0],[0,0,0],[3.12,0,0],[0,0,0],[0,0,0]]);
  const spike=jaw.compileJawCandidate(input,30,'smplx-jaw-godette-calibrated-v1');
  assert.deepEqual(spike.angles,[0,0,0,0,0]);
  assert.equal(spike.diagnostics.rejected_frames,1);
  const step=jaw.compileJawCandidate(poses(Array.from({length:20},(_,i)=>[i<5?0:0.3,0,0])),30,spike.profile);
  assert.ok(step.angles.every(a=>a>=0&&a<=0.16));
  for(let i=1;i<step.angles.length;i++)assert.ok(Math.abs(step.angles[i]-step.angles[i-1])<=0.0200000001);
  assert.equal(typeof jaw.sampleJawCandidate,'function');
  const value=jaw.sampleJawCandidate(step,0.25);
  jaw.sampleJawCandidate(step,0.5);
  assert.equal(jaw.sampleJawCandidate(step,0.25),value);
  assert.equal(jaw.sampleJawCandidate(step,-1),step.angles[0]);
  assert.equal(jaw.sampleJawCandidate(step,100),step.angles.at(-1));
  assert.throws(()=>jaw.compileJawCandidate(poses([[NaN,0,0]]),30,spike.profile));
  assert.throws(()=>jaw.compileJawCandidate(input,0,spike.profile));
  assert.throws(()=>jaw.compileJawCandidate(input,30,'unknown-profile'));
});
