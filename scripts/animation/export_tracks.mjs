// Compile the same runtime retargeter against a real GLB skeleton without WebGL.
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';

const input = JSON.parse(await fs.readFile(process.argv[2], 'utf8'));
const out = path.resolve(input.output), runtime = path.join(out, 'runtime');
await fs.mkdir(runtime, {recursive:true});
const web = fileURLToPath(new URL('../../slbridge/animation/web/', import.meta.url));
await fs.copyFile(input.three, path.join(runtime,'three.mjs'));
for (const name of ['rig.js','retarget.js','stabilization.js','expression.mjs','motion-geometry.mjs','nonmanual.mjs']) {
  let source = await fs.readFile(path.join(web,name),'utf8');
  source = source.replaceAll('/animation/vendor/three.module.js','./three.mjs').replaceAll('./stabilization.js','./stabilization.mjs');
  await fs.writeFile(path.join(runtime,name.replace(/\.js$/,'.mjs')),source);
}
const THREE = await import(pathToFileURL(path.join(runtime,'three.mjs')));
const {createHumanoidRig} = await import(pathToFileURL(path.join(runtime,'rig.mjs')));
const {createT08Retarget} = await import(pathToFileURL(path.join(runtime,'retarget.mjs')));
const bytes = await fs.readFile(input.model), size=bytes.readUInt32LE(12);
const gltf = JSON.parse(bytes.subarray(20,20+size).toString());
const joints = new Set(gltf.skins.flatMap(s=>s.joints));
const sourceBind = new Map();
for (const skin of gltf.skins) {
  const accessor=gltf.accessors[skin.inverseBindMatrices], view=gltf.bufferViews[accessor.bufferView];
  const offset=28+size+(view.byteOffset||0)+(accessor.byteOffset||0), stride=view.byteStride||64;
  skin.joints.forEach((node,i)=>sourceBind.set(node,new THREE.Matrix4().fromArray(
    Array.from({length:16},(_,k)=>bytes.readFloatLE(offset+i*stride+k*4))).invert()));
}
const objects = gltf.nodes.map((node,i)=>{
  const object=joints.has(i)?new THREE.Bone():new THREE.Group();
  object.name=THREE.PropertyBinding.sanitizeNodeName(node.name||`node_${i}`);
  if (node.matrix) new THREE.Matrix4().fromArray(node.matrix).decompose(object.position,object.quaternion,object.scale);
  else {
    if(node.translation)object.position.fromArray(node.translation);
    if(node.rotation)object.quaternion.fromArray(node.rotation);
    if(node.scale)object.scale.fromArray(node.scale);
  }
  return object;
});
gltf.nodes.forEach((n,i)=>(n.children||[]).forEach(c=>objects[i].add(objects[c])));
const root=new THREE.Group();
gltf.scenes[gltf.scene||0].nodes.forEach(i=>root.add(objects[i]));root.updateMatrixWorld(true);
const rig=createHumanoidRig(root);
if(!rig.isSigningReady)throw new Error('Incomplete real target rig: '+rig.missing);
const restMatrices=Object.fromEntries(Object.entries(rig.bones).map(([name,bone])=>[name,bone.matrixWorld.toArray()]));
const bindMatrices=Object.fromEntries(Object.entries(rig.bones).map(([name,bone])=>[name,sourceBind.get(objects.indexOf(bone)).toArray()]));
const reports=[];
for(const pilot of input.pilots){
  const data=JSON.parse(await fs.readFile(pilot.motion,'utf8'));
  const retarget=createT08Retarget(rig,root,data);
  const original=retarget.compile(data.clip,{stabilize:false});
  const temporalProfile=pilot.temporal_profile||data.clip.gesture_refinement?.profile||'quaternion-temporal-adaptive-v1';
  const frames=retarget.compile(data.clip,{stabilize:true,stabilizationProfile:temporalProfile,positionConstraints:true});
  let maxFingerDelta=0,maxBoneLengthError=0;
  const restLengths=Object.fromEntries(Object.entries(rig.bones).map(([name,bone])=>[name,bone.position.length()]));
  const worldFrames=frames.map((frame,i)=>{
    for(const name of Object.keys(frame))if(/Thumb|Index|Middle|Ring|Little/.test(name)){
      // glTF bind rotations have float32 rounding; angleTo assumes unit inputs.
      maxFingerDelta=Math.max(maxFingerDelta,frame[name].clone().normalize().angleTo(original[i][name].clone().normalize()));
    }
    rig.interpolate(frame,frame,1);root.updateMatrixWorld(true);
    for(const [name,bone] of Object.entries(rig.bones))maxBoneLengthError=Math.max(maxBoneLengthError,Math.abs(bone.position.length()-restLengths[name]));
    return Object.fromEntries(Object.keys(frame).map(name=>[name,rig.bones[name].matrixWorld.toArray()]));
  });
  const fingerLimit=temporalProfile==='soke-gesture-refinement-v1'?Math.PI:
    temporalProfile==='source-quaternion-temporal-v1'?Math.PI/6:Math.PI/60;
  if(maxBoneLengthError>1e-8 || maxFingerDelta>fingerLimit+1e-6)throw new Error('Retarget preservation failed: '+JSON.stringify({asset:pilot.asset_id,maxBoneLengthError,maxFingerDeltaDegrees:maxFingerDelta*180/Math.PI}));
  const report={asset_id:pilot.asset_id,frames:frames.length,fps:data.clip.fps,source_motion_sha256:pilot.source_motion_sha256,
    ...retarget.diagnostics,temporal_profile:temporalProfile,max_finger_change_degrees:maxFingerDelta*180/Math.PI,max_bone_length_change:maxBoneLengthError,language_review:'NOT_RUN'};
  const body={...report,restMatrices,bindMatrices,rigProfile:rig.profile,boneNames:Object.fromEntries(Object.entries(rig.bones).map(([name,bone])=>[name,bone.name])),worldFrames};
  await fs.writeFile(path.join(out,pilot.asset_id+'.tracks.json'),JSON.stringify(body));
  reports.push(report);rig.reset();
}
await fs.writeFile(path.join(out,'geometry-report.json'),JSON.stringify(reports,null,2));
console.log(JSON.stringify(reports,null,2));
