export const EXPRESSION_CHANNELS = Object.freeze(['browRaise', 'browLower', 'blink', 'jawOpen']);
const neutral = () => Object.fromEntries(EXPRESSION_CHANNELS.map(k => [k, 0]));
function channels(value = {}) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Invalid expression channels');
  for (const [key, amount] of Object.entries(value)) {
    if (!EXPRESSION_CHANNELS.includes(key) || !Number.isFinite(amount) || amount < 0 || amount > 1) throw new Error('Invalid expression intensity');
  }
  return { ...neutral(), ...value };
}
export function sampleExpression(track, time, duration) {
  if (!Number.isFinite(time) || !Number.isFinite(duration) || duration < 0) throw new Error('Invalid expression time');
  if (track == null || (Array.isArray(track) && !track.length)) return neutral();
  if (!Array.isArray(track) || track.length > 512) throw new Error('Invalid expression track');
  let previous = -1;
  const frames = track.map(frame => {
    if (!Number.isFinite(frame.time) || frame.time < 0 || frame.time > duration || frame.time <= previous) throw new Error('Invalid expression key time');
    previous = frame.time;
    return {time: frame.time, channels: channels(frame.channels)};
  });
  if (frames[0].time > 0) frames.unshift({time:0, channels:neutral()});
  const at = Math.max(0, Math.min(duration, time));
  const next = frames.findIndex(f => f.time > at);
  if (next < 0) return {...frames.at(-1).channels};
  const a = frames[Math.max(0,next-1)], b = frames[next];
  const amount = b.time === a.time ? 0 : (at-a.time)/(b.time-a.time);
  return Object.fromEntries(EXPRESSION_CHANNELS.map(k => [k, a.channels[k]+(b.channels[k]-a.channels[k])*amount]));
}
export function validateRigProfile(profile) {
  if (!profile || !/^[a-zA-Z0-9_-]{1,80}$/.test(profile.id) || !profile.bones || Array.isArray(profile.bones)) throw new Error('Invalid rig profile');
  for (const [key, candidates] of Object.entries(profile.bones)) {
    if (!/^[a-zA-Z][a-zA-Z0-9]{0,79}$/.test(key) || !Array.isArray(candidates) || !candidates.length || candidates.length > 8 ||
        candidates.some(x => typeof x !== 'string' || !x || x.length > 128)) throw new Error('Invalid rig bone binding');
  }
  for (const [key, correction] of Object.entries(profile.corrections || {})) {
    if (!Object.hasOwn(profile.bones,key) || !Array.isArray(correction) || correction.length !== 3 ||
        correction.some(x => !Number.isFinite(x) || Math.abs(x) > Math.PI/4)) throw new Error('Invalid rig axis correction');
  }
  return profile;
}
export function applyExpression(rig, input = {}, {jawRadians=null} = {}) {
  const values = channels(input);
  rig.resetExpression();
  for (const control of rig.faceControls || []) {
    const amount = control.kind === 'brow' ? values.browRaise - values.browLower*0.6 : values.blink;
    control.bone.position.addScaledVector(control.offset, amount);
  }
  if(rig.jawCalibrationAvailable)rig.applyCalibratedJaw(jawRadians??values.jawOpen*0.16);
  else if(rig.manualMouthAvailable)rig.applyManualMouth(values.jawOpen);
  else if (rig.bones.jaw) rig.bones.jaw.rotateX(-values.jawOpen * 0.16);
  return values;
}
