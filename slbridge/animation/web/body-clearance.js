import * as THREE from '/animation/vendor/three.module.js';

// Conservative mesh-derived cores detect deep overlap, not every cloth contact.
// Applied only to an explicitly enabled, verified avatar in the stage.
export function createBodyClearance(rig, root) {
  const meshes=[];
  root.updateMatrixWorld(true);
  root.traverse(node=>{if(node.isSkinnedMesh)meshes.push(node);});
  new Set(meshes.map(mesh=>mesh.skeleton)).forEach(skeleton=>skeleton.update());
  const boxes={head:new THREE.Box3(),torso:new THREE.Box3()};
  const counts={head:0,torso:0};
  const torsoBones=new Set([rig.bones.hips,rig.bones.spine,rig.bones.chest,rig.bones.upperChest]);
  const headBones=new Set();rig.bones.head?.traverse(bone=>{if(bone.isBone)headBones.add(bone);});
  for(const mesh of meshes){
    const retained=/^Object_0(?:_\d+)?$/.test(mesh.name);
    const garment=/Mandarin_jacket|Ivory_inner_panel/.test(mesh.name);
    if(!retained&&!garment)continue;
    const {skinIndex,skinWeight}=mesh.geometry.attributes;
    if(!skinIndex||!skinWeight)continue;
    const indexes=new Set(mesh.geometry.index?.array||Array.from({length:skinIndex.count},(_,i)=>i));
    for(const index of indexes){
      let head=0,torso=0;
      for(let k=0;k<4;k++){
        const weight=skinWeight.getComponent(index,k),bone=mesh.skeleton.bones[skinIndex.getComponent(index,k)];
        if(headBones.has(bone))head+=weight;
        if(torsoBones.has(bone))torso+=weight;
      }
      if(!(retained&&head>.75)&&!(garment&&torso>.75))continue;
      const point=mesh.getVertexPosition(index,new THREE.Vector3()).applyMatrix4(mesh.matrixWorld);
      if(retained&&head>.75){boxes.head.expandByPoint(point);counts.head++;}
      if(garment&&torso>.75){boxes.torso.expandByPoint(point);counts.torso++;}
    }
  }
  const colliders=[];
  for(const name of ['head','torso']){
    const anchor=rig.bones[name==='head'?'head':'upperChest'];
    const radius=boxes[name].getSize(new THREE.Vector3()).multiplyScalar(.5).multiply(new THREE.Vector3(.88,.94,.88));
    if(!anchor||counts[name]<20||radius.toArray().some(v=>!Number.isFinite(v)||v<1e-5))continue;
    colliders.push({anchor,restInverse:anchor.matrixWorld.clone().invert(),center:boxes[name].getCenter(new THREE.Vector3()),radius});
  }
  function samples(side){
    const points=[];
    for(const [name,bone] of Object.entries(rig.bones)){
      if(!name.startsWith(side)||!(/Hand$|Thumb|Index|Middle|Ring|Little/.test(name)))continue;
      const point=bone.getWorldPosition(new THREE.Vector3());points.push(point);
      if(bone.parent?.isBone&&bone!==rig.bones[side+'Hand'])points.push(point.clone().lerp(bone.parent.getWorldPosition(new THREE.Vector3()),.5));
    }
    return points;
  }
  function spaces(){return colliders.map(collider=>{
    const world=collider.anchor.matrixWorld.clone().multiply(collider.restInverse);
    return {...collider,world,undo:world.clone().invert()};
  });}
  function coreHits(side,translation=null){
    const points=samples(side);let count=0;
    for(const space of spaces())for(const point of points){
      const shifted=point.clone();if(translation)shifted.add(translation);
      if(shifted.applyMatrix4(space.undo).sub(space.center).divide(space.radius).length()<1-1e-6)count++;
    }
    return count;
  }
  function offset(side,maximum=.06,preferred=null){
    const points=samples(side),total=preferred?preferred.clone():new THREE.Vector3();
    if(total.length()>maximum)total.setLength(maximum);
    if(!points.length)return total;
    const active=spaces().map(space=>({...space,origin:new THREE.Vector3().applyMatrix4(space.undo),
      points:points.map(point=>point.clone().applyMatrix4(space.undo).sub(space.center).divide(space.radius))}));
    const localVector=(space,value)=>value.clone().applyMatrix4(space.undo).sub(space.origin).divide(space.radius);
    function score(value){
      let hits=0,depth=0;
      for(const space of active){
        const shift=localVector(space,value);
        for(const point of space.points){const radius=point.clone().add(shift).length();if(radius<1-1e-6){hits++;depth+=(1-radius)**2;}}
      }
      return {value,hits,depth,cost:value.distanceToSquared(total)+.01*value.lengthSq()};
    }
    let best=score(total);if(!best.hits)return total;
    const directions=[];
    for(let x=-1;x<=1;x++)for(let y=-1;y<=1;y++)for(let z=-1;z<=1;z++)if(x||y||z)directions.push(new THREE.Vector3(x,y,z).normalize());
    for(const space of active){
      const shift=localVector(space,total),normalMatrix=new THREE.Matrix3().getNormalMatrix(space.world);
      for(const point of space.points){
        const q=point.clone().add(shift);
        if(q.lengthSq()<1&&q.lengthSq()>1e-12)directions.push(q.divide(space.radius).applyMatrix3(normalMatrix).normalize());
      }
    }
    // Search an exit close to the anticipated offset. A single radial normal
    // can flip across the head midline and introduce a new wrist jump.
    for(const direction of directions){
      const intervals=[];
      for(const space of active){
        const shift=localVector(space,total),d=localVector(space,direction),a=d.lengthSq();
        if(a<1e-12)continue;
        for(const point of space.points){
          const q=point.clone().add(shift),b=2*q.dot(d),c=q.lengthSq()-1,discriminant=b*b-4*a*c;
          if(discriminant>0){const root=Math.sqrt(discriminant);intervals.push([(-b-root)/(2*a),(-b+root)/(2*a)]);}
        }
      }
      let distance=0;
      for(let i=0;i<=intervals.length;i++){
        let exit=distance;
        for(const [enter,end] of intervals)if(distance>enter&&distance<end)exit=Math.max(exit,end+.002);
        if(exit===distance)break;distance=exit;
      }
      const candidate=total.clone().addScaledVector(direction,distance);
      if(candidate.length()>maximum)candidate.setLength(maximum);
      const result=score(candidate);
      if(result.hits<best.hits||result.hits===best.hits&&(result.depth<best.depth-1e-12||Math.abs(result.depth-best.depth)<=1e-12&&result.cost<best.cost))best=result;
    }
    return best.value;
  }
  return {available:colliders.length>0,offset,coreHits,profile:'mesh-core-clearance-directional-v3'};
}

// A calibrated display candidate for one frozen source/configuration. This is
// not a reusable word sign and cannot inherit approval from the original clip.
const source = {
  id: 'soke-full-S000931_P0000_T00-e61082953513a62a',
  source_id: 'S000931_P0000_T00',
  parameter_sha256: '133ae12272843871572cdc449d67706c84314c595802d00ae5f658556472e5cd',
  processing_version: 'soke-full-lazy-gesture-jaw-v2',
  fps: 30,
  frames: 162,
};
const resourceSha = '1de2fba63654bdd2f0f2984d5b2c09638e8f446ff755d4c3d46b3f8b024d46cd';
const profile = 'soke-gesture-refinement-v1';
const smooth = (start, end, frame) => {
  const t = Math.max(0, Math.min(1, (frame-start)/(end-start)));
  return t*t*(3-2*t);
};

export function selectContactTrack(clip, bundleSha, stabilizationProfile) {
  if (bundleSha !== resourceSha || stabilizationProfile !== profile ||
      clip.gesture_refinement?.profile !== profile ||
      Object.entries(source).some(([key, value]) => clip[key] !== value) ||
      clip.poses?.length !== source.frames || clip.refined_poses?.length !== source.frames) return null;
  return {
    id: 'v14a-S000931-contact-v1',
    frames: Array.from({length: source.frames}, (_, i) => {
      // A later fingertip arc crosses the head between keys 44 and 45. Hold a
      // separate, smaller exit through that entire contact; no wrist tilt.
      if (i >= 36 && i <= 54) {
        const distance = .018*Math.max(0, Math.min(1, (i-36)/7, (53-i)/7));
        return {right: {offset: [-distance/Math.SQRT2,0,distance/Math.SQRT2],rotation: [0,0,0]}};
      }
      // Lock one exit for the entire first contact, rather than switching to
      // whichever direction happens to be shortest in the current frame.
      if (i > 30) return null;
      // Hold the exit through frames 10–14. Fitting only the minimum offset at
      // the integer keys lets the actual arm/finger slerp re-enter the head.
      const distance = .06*Math.max(0, Math.min(1, (i-2)/8, (22-i)/8));
      const angle = Math.PI/12 * smooth(0,10,i) * (1-smooth(15,25,i));
      return {right: {
        offset: [-distance/Math.SQRT2,-distance/Math.SQRT2,0],
        rotation: [-angle/Math.SQRT2,angle/Math.SQRT2,0],
      }};
    }),
  };
}
