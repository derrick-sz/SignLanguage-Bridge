// Source-space kinematics; no renderer dependency and no mutation of source fits.
const add = (a, b) => a.map((x, i) => x + b[i]);
const sub = (a, b) => a.map((x, i) => x - b[i]);
const mul = (a, s) => a.map(x => x * s);
const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
const norm = a => Math.hypot(...a);
const cross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
const unit = a => mul(a, 1 / norm(a));
function vector(a) {
  if (!Array.isArray(a) || a.length !== 3 || !a.every(Number.isFinite)) throw new Error('Invalid motion vector');
}
function quaternion(v) {
  const angle = norm(v);
  return angle < 1e-12 ? [0, 0, 0, 1] : [...mul(v, Math.sin(angle/2)/angle), Math.cos(angle/2)];
}
function compose(a, b) {
  const xyz = add(add(mul(a.slice(0,3), b[3]), mul(b.slice(0,3), a[3])), cross(a, b));
  const q = [...xyz, a[3]*b[3]-dot(a.slice(0,3), b.slice(0,3))];
  const length = Math.hypot(...q);
  return q.map(x => x / length);
}
function rotate(v, q) {
  const t = mul(cross(q, v), 2);
  return add(v, add(mul(t, q[3]), cross(q, t)));
}

export const SOURCE_TEMPORAL_PROFILE = 'source-quaternion-temporal-v1';
export const GESTURE_REFINEMENT_PROFILE = 'soke-gesture-refinement-v1';
export function selectSourcePoses(clip,{stabilize,profile}) {
  if(!stabilize)return clip.poses;
  if(profile===SOURCE_TEMPORAL_PROFILE)return stabilizeSourcePoses(clip.poses,clip.fps);
  if(profile!==GESTURE_REFINEMENT_PROFILE)return clip.poses;
  const poses=clip.refined_poses;
  if(clip.gesture_refinement?.profile!==GESTURE_REFINEMENT_PROFILE || !Array.isArray(poses) || poses.length!==clip.poses.length)
    throw new Error('Invalid refined gesture source');
  for(const pose of poses){
    if(!Array.isArray(pose)||pose.length!==55)throw new Error('Invalid refined gesture source');
    for(const v of pose)vector(v);
  }
  return poses;
}
function slerp(a, b, amount) {
  let cosine = dot(a, b), target = b;
  if (cosine < 0) {cosine = -cosine;target = mul(b,-1);}
  const theta = Math.acos(Math.min(1,cosine));
  const result = theta < 1e-7 ? add(mul(a,1-amount),mul(target,amount)) :
    add(mul(a,Math.sin((1-amount)*theta)/Math.sin(theta)),mul(target,Math.sin(amount*theta)/Math.sin(theta)));
  return unit(result);
}

// A reversible display candidate. Process rotations before FK as well as retargeting,
// so the IK goals cannot reintroduce the unfiltered source's frame-to-frame noise.
export function stabilizeSourcePoses(poses, fps) {
  if (!Array.isArray(poses) || !poses.length || !Number.isFinite(fps) || fps<=0 || fps>120)
    throw new Error('Invalid motion track');
  const source = poses.map(pose=>{
    if (!Array.isArray(pose) || pose.length!==55) throw new Error('Invalid motion skeleton');
    return pose.map(v=>{vector(v);if(!Number.isFinite(norm(v)))throw new Error('Invalid motion vector');return quaternion(v);});
  });
  const output = poses.map(pose=>pose.map(v=>[...v]));
  for (let joint=0;joint<55;joint++) {
    const sigma=fps*(joint>=25?0.030:0.045),radius=Math.ceil(2.5*sigma);
    for (let frame=1;frame<source.length-1;frame++) {
      const current=source[frame][joint];let result=current,total=1;
      const count=Math.min(radius,frame,source.length-frame-1);
      for(let offset=1;offset<=count;offset++) {
        const weight=2*Math.exp(-0.5*(offset/sigma)**2);
        const pair=slerp(source[frame-offset][joint],source[frame+offset][joint],0.5);
        result=slerp(result,pair,weight/(total+weight));total+=weight;
      }
      const correction=2*Math.acos(Math.min(1,Math.abs(dot(current,result))));
      if(correction>Math.PI/6)result=slerp(current,result,(Math.PI/6)/correction);
      if(result[3]<0)result=mul(result,-1);
      const length=norm(result.slice(0,3)),angle=2*Math.atan2(length,result[3]);
      output[frame][joint]=length<1e-12?[0,0,0]:mul(result.slice(0,3),angle/length);
    }
  }
  return output;
}

export function forwardKinematics(pose, parents, restJoints) {
  if (!Array.isArray(pose) || !pose.length || !Array.isArray(parents) || !Array.isArray(restJoints)
      || pose.length !== parents.length || pose.length !== restJoints.length) throw new Error('Invalid motion skeleton');
  const positions = [], rotations = [];
  for (let i = 0; i < pose.length; i++) {
    vector(pose[i]); vector(restJoints[i]);
    const parent = parents[i], local = quaternion(pose[i]);
    if (!Number.isInteger(parent) || (i === 0 ? parent !== -1 : parent < 0 || parent >= i)) throw new Error('Invalid motion topology');
    rotations.push(i === 0 ? local : compose(rotations[parent], local));
    positions.push(i === 0 ? [...restJoints[i]] : add(positions[parent], rotate(sub(restJoints[i], restJoints[parent]), rotations[parent])));
  }
  return { positions, rotations };
}

export function solveTwoBone(root, elbow, wrist, target) {
  [root, elbow, wrist, target].forEach(vector);
  const upper = norm(sub(elbow, root)), lower = norm(sub(wrist, elbow));
  if (upper < 1e-8 || lower < 1e-8) throw new Error('Zero-length IK segment');
  const offset = sub(target, root), requested = norm(offset);
  const fallback = sub(wrist, root);
  const direction = requested > 1e-9 ? unit(offset) : norm(fallback) > 1e-9 ? unit(fallback) : [0, 1, 0];
  const distance = Math.max(Math.abs(upper-lower)+1e-9, Math.min(upper+lower, requested));
  let pole = sub(sub(elbow, root), mul(direction, dot(sub(elbow, root), direction)));
  if (norm(pole) < 1e-8) pole = cross(direction, Math.abs(direction[1]) < 0.9 ? [0,1,0] : [0,0,1]);
  pole = unit(pole);
  const along = (upper*upper-lower*lower+distance*distance)/(2*distance);
  const height = Math.sqrt(Math.max(0, upper*upper-along*along));
  return {
    elbow: add(root, add(mul(direction, along), mul(pole, height))),
    wrist: add(root, mul(direction, distance)),
    clamped: Math.abs(distance-requested) > 1e-10,
  };
}
