import * as THREE from "/animation/vendor/three.module.js";
import { GLTFLoader } from "/animation/vendor/jsm/loaders/GLTFLoader.js";
import { createHumanoidRig } from "./rig.js";
import { createT08Retarget } from "./retarget.js";
import { applyExpression } from "./expression.mjs";
import { jawCandidateForClip, sampleJawCandidate, MAX_JAW_ANGLE } from "./nonmanual.mjs";
import { PlaybackClock, signingCamera, comparisonViewports } from "./player-core.mjs";

export class MotionStage {
  constructor(container, onTick) {
    this.container=container; this.onTick=onTick; this.clock=new PlaybackClock();
    this.profile="improved"; this.view="front"; this.comparison=false;
    this.cache=new Map(); this.disposed=false; this.expression={}; this.skeletonEnabled=false;
    this.jawTrack=null;this.nonmanualEnabled=false;
    this.ready=this.init();
  }
  async init() {
    this.renderer=new THREE.WebGLRenderer({antialias:true,alpha:false});
    this.renderer.setPixelRatio(Math.min(devicePixelRatio,2));
    this.renderer.outputColorSpace=THREE.SRGBColorSpace;
    this.renderer.toneMapping=THREE.ACESFilmicToneMapping;
    this.renderer.shadowMap.enabled=true; this.renderer.shadowMap.type=THREE.PCFSoftShadowMap;
    this.renderer.domElement.setAttribute("aria-label","原始与制作修正版的手语三维画面");
    this.container.appendChild(this.renderer.domElement);
    this.scene=new THREE.Scene(); this.scene.background=new THREE.Color(0xdbe4e7);
    this.camera=new THREE.PerspectiveCamera(34,1,0.05,30);
    this.scene.add(new THREE.HemisphereLight(0xffffff,0x8d9eaa,1.8));
    const key=new THREE.DirectionalLight(0xfff6ed,2.3); key.position.set(2.5,4,4); key.castShadow=true;
    key.shadow.mapSize.set(1024,1024); this.scene.add(key);
    const fill=new THREE.DirectionalLight(0xe8f2ff,1.3); fill.position.set(-3,2,3); this.scene.add(fill);
    const rim=new THREE.DirectionalLight(0xffffff,0.8); rim.position.set(0,3,-3); this.scene.add(rim);
    const loader=new GLTFLoader();
    this.originalModel=(await loader.loadAsync("/animation/vendor/model.glb")).scene;
    try {
      const response=await fetch("/v1/animation/presentation");
      this.presentation=response.ok?await response.json():{};
    } catch {this.presentation={};}
    const avatar=this.presentation?.avatar;
    this.model=(await loader.loadAsync(avatar?"/animation/presentation/"+encodeURIComponent(avatar):"/animation/vendor/model.glb")).scene;
    if(this.disposed){this.disposeModel(this.originalModel);this.disposeModel(this.model);return;}
    for(const model of [this.originalModel,this.model]) {
      model.traverse(node=>{if(node.isMesh){node.castShadow=true;node.receiveShadow=true;node.frustumCulled=false;}});
      this.scene.add(model);
    }
    this.originalRig=createHumanoidRig(this.originalModel); this.rig=createHumanoidRig(this.model,this.presentation?.rig_profile||undefined,{avatarSha256:this.presentation?.avatar_sha256});
    if(!this.originalRig.isSigningReady || !this.rig.isSigningReady)throw new Error("角色骨架缺少必要手语关节");
    this.sourceLine=new THREE.LineSegments(new THREE.BufferGeometry(),new THREE.LineBasicMaterial({color:0x007c91,transparent:true,opacity:0.8,depthTest:false}));
    this.sourceLine.visible=false;this.sourceLine.renderOrder=3;this.scene.add(this.sourceLine);
    this.poseNeutral();
    this.resizeObserver=new ResizeObserver(()=>this.resize());this.resizeObserver.observe(this.container);
    this.resize();
    const render=now=>{if(this.disposed)return;this.refresh(now);this.raf=requestAnimationFrame(render);};
    this.raf=requestAnimationFrame(render);
  }
  resize(){
    const {clientWidth:w,clientHeight:h}=this.container;if(!w||!h||!this.renderer)return;
    this.renderer.setSize(w,h);this.refresh();
  }
  poseNeutral(){
    for(const [rig,model] of [[this.originalRig,this.originalModel],[this.rig,this.model]]){
      rig.reset();model.updateMatrixWorld(true);
      for(const side of ["left","right"]){
        const arm=rig.bones[side+"UpperArm"],forearm=rig.bones[side+"LowerArm"];
        const direction=forearm.getWorldPosition(new THREE.Vector3()).sub(arm.getWorldPosition(new THREE.Vector3())).normalize();
        const world=new THREE.Quaternion().setFromUnitVectors(direction,new THREE.Vector3(0,-1,0)).multiply(arm.getWorldQuaternion(new THREE.Quaternion()));
        arm.quaternion.copy(arm.parent.getWorldQuaternion(new THREE.Quaternion()).invert().multiply(world));arm.updateWorldMatrix(false,true);
      }
    }
  }
  cameraFor(aspect){
    const bounds=this.view==="hands" && this.handBounds?this.handBounds:this.bounds;
    const frame=bounds?signingCamera(bounds,aspect):{centerX:0,centerY:1.4,centerZ:0,distance:2.1};
    const center=new THREE.Vector3(frame.centerX,frame.centerY,frame.centerZ);
    const distance=this.view==="side"?frame.distance*1.15:frame.distance;
    const offset=this.view==="side"?new THREE.Vector3(distance*0.65,0,distance*0.76):new THREE.Vector3(0,0,distance);
    this.camera.aspect=aspect;this.camera.position.copy(center).add(offset);this.camera.lookAt(center);this.camera.updateProjectionMatrix();
  }
  setView(view){if(!["front","side","hands"].includes(view))throw new Error("未知观察视角");this.view=view;this.refresh();}
  setProfile(profile){if(!["original","improved"].includes(profile))throw new Error("未知显示版本");this.profile=profile;this.track=this.tracks?.[profile];this.refresh();}
  setComparison(value){this.comparison=Boolean(value);this.refresh();}
  setSkeletonVisible(value){this.skeletonEnabled=Boolean(value);this.refresh();}
  setExpression(channels){
    const next=Object.keys(channels).length?{...this.expression,...channels}:{};
    if(this.rig)applyExpression(this.rig,next);this.expression=next;this.refresh();
  }
  get nonmanualAvailable(){return Boolean(this.jawTrack&&this.rig?.jawCalibrationAvailable);}
  setNonmanualEnabled(value){this.nonmanualEnabled=Boolean(value)&&this.nonmanualAvailable;this.refresh();}
  jawAngle(time){
    if(Object.hasOwn(this.expression,'jawOpen'))return this.expression.jawOpen*MAX_JAW_ANGLE;
    return this.nonmanualEnabled?sampleJawCandidate(this.jawTrack,time):0;
  }
  async loadMotion(data,id,isCurrent=()=>true,sourceSha=null){
    await this.ready;if(this.disposed||!isCurrent())return false;
    let tracks=this.cache.get(id);
    if(!tracks){
      const raw=createT08Retarget(this.originalRig,this.originalModel,data);
      const corrected=createT08Retarget(this.rig,this.model,data);
      const temporalProfile=this.presentation?.pilots?.find(p=>p.asset_id===id)?.temporal_profile||data.clip.gesture_refinement?.profile||"quaternion-temporal-adaptive-v1";
      const original=raw.compile(data.clip,{stabilize:false});
      const improved=corrected.compile(data.clip,{stabilize:true,stabilizationProfile:temporalProfile,positionConstraints:true});
      let changed=0;
      original.forEach((frame,i)=>{if(Object.keys(frame).some(k=>frame[k].clone().normalize().angleTo(improved[i][k].clone().normalize())>1e-6))changed++;});
      const bounds={minX:-0.20,maxX:0.20,minY:0.98,maxY:1.86,minZ:-0.12,maxZ:0.12};
      const hands={minX:Infinity,maxX:-Infinity,minY:Infinity,maxY:-Infinity,minZ:Infinity,maxZ:-Infinity};
      const extend=(b,p,padding=0.04)=>{for(const [axis,key] of [["x","X"],["y","Y"],["z","Z"]]){b["min"+key]=Math.min(b["min"+key],p[axis]-padding);b["max"+key]=Math.max(b["max"+key],p[axis]+padding);}};
      for(const [rig,model,frames] of [[this.originalRig,this.originalModel,original],[this.rig,this.model,improved]]){
        const snapshot=rig.capture();
        for(const frame of frames){
          rig.interpolate(frame,frame,1);model.updateMatrixWorld(true);
          for(const [name,bone] of Object.entries(rig.bones)){
            if(!/Hand$|Distal$/.test(name))continue;
            const p=bone.getWorldPosition(new THREE.Vector3());extend(bounds,p);extend(hands,p,0.07);
          }
        }
        rig.interpolate(snapshot,snapshot,1);model.updateMatrixWorld(true);
      }
      tracks={original,improved,temporalProfile,corrected:changed,mappedBones:corrected.mappedBones,bounds,hands,mapper:corrected,diagnostics:corrected.diagnostics};
      this.cache.set(id,tracks);if(this.cache.size>6)this.cache.delete(this.cache.keys().next().value);
    }
    if(!isCurrent())return false;
    this.tracks=tracks;this.track=tracks[this.profile];this.clip=data.clip;this.parents=data.parents;
    this.assetId=id;this.sourceSha=sourceSha;this.fps=data.clip.fps;this.expression={};
    const baked=Boolean(this.presentation?.pilots?.find(p=>p.asset_id===id)?.export_file);
    this.jawTrack=this.rig.jawCalibrationAvailable?jawCandidateForClip(data.clip,this.presentation?.avatar_sha256,baked):null;
    this.nonmanualEnabled=this.nonmanualAvailable;
    this.bounds=tracks.bounds;this.handBounds=tracks.hands;this.clock.load(data.clip.frames/data.clip.fps);
    this.refresh();return true;
  }
  applyTime(time){
    if(!this.tracks)return;
    const position=Math.min(time*this.fps,this.tracks.original.length-1);
    const from=Math.floor(position),to=Math.min(from+1,this.tracks.original.length-1),amount=position-from;
    this.originalRig.interpolate(this.tracks.original[from],this.tracks.original[to],amount);
    this.rig.interpolate(this.tracks.improved[from],this.tracks.improved[to],amount);
    applyExpression(this.rig,this.expression,{jawRadians:this.jawAngle(time)});
    this.originalModel.updateMatrixWorld(true);this.model.updateMatrixWorld(true);
    if(this.skeletonEnabled){
      const a=this.tracks.mapper.sourcePositions(this.clip.poses[from]),b=this.tracks.mapper.sourcePositions(this.clip.poses[to]);
      const points=a.map((p,i)=>p.lerp(b[i],amount)),segments=[];
      this.parents.forEach((parent,i)=>{if(parent>=0)segments.push(...points[parent].toArray(),...points[i].toArray());});
      this.sourceLine.geometry.setAttribute("position",new THREE.Float32BufferAttribute(segments,3));
      this.sourceLine.geometry.computeBoundingSphere();
    }
  }
  refresh(now=performance.now()){
    if(this.disposed||!this.renderer)return;
    const time=this.clock.time(now);this.applyTime(time);
    if(this.camera?.isPerspectiveCamera && (!this.originalModel || !this.model || !this.sourceLine)){
      this.onTick?.({time,duration:this.clock.duration,playing:this.clock.playing,comparison:this.comparison});
      return;
    }
    if(!this.camera?.isPerspectiveCamera){this.renderer.render(this.scene,this.camera);}
    else {
      const w=this.container.clientWidth,h=this.container.clientHeight;if(!w||!h)return;
      const panels=comparisonViewports(w,h,this.comparison);this.container.dataset.layout=this.comparison?(panels[0].x===panels[1].x?"stacked":"split"):"single";
      this.renderer.setScissorTest(true);
      for(const panel of panels){
        const profile=panel.profile||this.profile;
        this.originalModel.visible=profile==="original";this.model.visible=profile==="improved";
        this.sourceLine.visible=this.skeletonEnabled&&Boolean(this.tracks)&&profile==="improved";
        this.cameraFor(panel.width/panel.height);
        this.renderer.setViewport(panel.x,panel.y,panel.width,panel.height);this.renderer.setScissor(panel.x,panel.y,panel.width,panel.height);
        this.renderer.render(this.scene,this.camera);
      }
      this.renderer.setScissorTest(false);
    }
    this.onTick?.({time,duration:this.clock.duration,playing:this.clock.playing,comparison:this.comparison});
  }
  play(){this.clock.play(performance.now());this.refresh();}
  pause(){this.clock.pause(performance.now());this.refresh();}
  seek(time){this.clock.seek(time,performance.now());this.refresh();}
  setSpeed(speed){this.clock.setSpeed(speed,performance.now());this.refresh();}
  stop(){
    this.clock.stop();this.clock.duration=0;this.track=null;this.tracks=null;this.clip=null;this.expression={};
    this.jawTrack=null;this.nonmanualEnabled=false;
    this.rig?.reset();this.originalRig?.reset();this.refresh();
    if(this.model && this.originalModel){this.poseNeutral();this.refresh();}
  }
  exportReview(){
    return {schema_version:1,asset_id:this.assetId,source_motion_sha256:this.sourceSha,
      avatar_sha256:this.presentation?.avatar_sha256||null,processing_version:this.presentation?.processing_version||"source-wrist-ik-v1",
      temporal_profile:this.tracks?.temporalProfile||null,
      display_profile:this.profile,comparison:Boolean(this.comparison),
      time:this.clock.time(performance.now()),expression:{...this.expression},expression_review:"unverified",
      nonmanual:{available:this.nonmanualAvailable,enabled:Boolean(this.nonmanualEnabled),
        manual_override:Object.hasOwn(this.expression,'jawOpen'),angle_radians:this.jawAngle(this.clock.time(performance.now())),
        profile:this.jawTrack?.profile||null,source_joint:22,source_template_sha256:this.clip?.nonmanual_candidate?.source_template_sha256||null,
        target_avatar_sha256:this.presentation?.avatar_sha256||null,diagnostics:this.jawTrack?.diagnostics||null,
        status:"research_candidate",language_review:"NOT_RUN",export_supported:false},
      diagnostics:this.tracks?.diagnostics||null,language_review:"NOT_RUN"};
  }
  disposeModel(model){
    model?.traverse(node=>{if(!node.isMesh)return;node.geometry.dispose();
      (Array.isArray(node.material)?node.material:[node.material]).forEach(m=>{for(const v of Object.values(m))if(v?.isTexture)v.dispose();m.dispose();});});
  }
  dispose(){
    this.disposed=true;cancelAnimationFrame(this.raf);this.resizeObserver?.disconnect();
    this.disposeModel(this.model);this.disposeModel(this.originalModel);this.sourceLine?.geometry.dispose();this.sourceLine?.material.dispose();this.renderer?.dispose();
  }
}
