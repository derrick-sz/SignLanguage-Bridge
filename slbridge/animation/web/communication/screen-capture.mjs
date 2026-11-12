import {cropRect,encodeSize} from './selection.mjs';

const release=stream=>stream?.getTracks().forEach(track=>track.stop());

export class ScreenSource {
  constructor({getDisplayMedia=options=>navigator.mediaDevices.getDisplayMedia(options),onState=()=>{}}={}){
    this.getDisplayMedia=getDisplayMedia;this.onState=onState;this.epoch=0;this.stream=null;
  }
  close(){this.epoch++;const owned=this.stream;this.stream=null;release(owned);this.onState({status:'closed'});}
  async open(){
    this.close();const epoch=this.epoch;this.onState({status:'opening'});
    try{
      const stream=await this.getDisplayMedia({audio:false,video:{frameRate:{ideal:25,max:30}}});
      if(epoch!==this.epoch){release(stream);return null;}
      const track=stream.getVideoTracks()[0];
      if(!track){release(stream);throw new Error('共享来源没有视频画面。');}
      this.stream=stream;
      track.addEventListener('ended',()=>{if(this.stream===stream){this.close();this.onState({status:'ended'});}});
      this.onState({status:'ready',stream});return stream;
    }catch(error){
      if(epoch===this.epoch)this.onState({status:'error',message:error?.name==='NotAllowedError'?'已取消共享。你也可以上传视频演示。':'屏幕共享未能开启，请使用桌面Chrome或选择视频演示。'});
      return null;
    }
  }
}

export class PauseSegmenter {
  constructor({mode='pause',maxSeconds=8,minSeconds=1.2,pauseSeconds=.75,threshold=.012}={}){
    if(!['pause','interval'].includes(mode)||!Number.isFinite(maxSeconds)||maxSeconds<1||maxSeconds>12)throw new RangeError('Invalid segment duration');
    Object.assign(this,{mode,maxSeconds,minSeconds,pauseSeconds,threshold});this.begin(0);
  }
  begin(time){this.started=time;this.lastActive=time;this.activeFrames=0;}
  feed(time,activity){
    if(activity>=this.threshold){this.lastActive=time;this.activeFrames++;}
    const elapsed=time-this.started;
    if(this.mode==='pause'&&elapsed>=this.minSeconds&&this.activeFrames>=3&&time-this.lastActive>=this.pauseSeconds)return {publish:true,reason:'pause'};
    if(elapsed>=this.maxSeconds)return {publish:this.mode==='interval'||this.activeFrames>=3,reason:this.mode==='interval'?'interval':this.activeFrames>=3?'limit':'idle'};
    return null;
  }
}

/** Owns only the cropped canvas video track; the source stream has a separate owner. */
export class RegionRecorder {
  constructor({video,makeCanvas=()=>document.createElement('canvas'),Recorder=globalThis.MediaRecorder,
    requestFrame=callback=>requestAnimationFrame(callback),cancelFrame=id=>cancelAnimationFrame(id),
    setTimer=(callback,ms)=>globalThis.setTimeout(callback,ms),clearTimer=id=>globalThis.clearTimeout(id),
    onSegment=()=>{},onState=()=>{}}={}){
    Object.assign(this,{video,makeCanvas,Recorder,requestFrame,cancelFrame,setTimer,clearTimer,onSegment,onState});
    this.epoch=0;this.active=false;this.stream=null;this.take=null;this.frameId=null;this.segmentCount=0;
  }
  start({region,mode='pause',maxSeconds=8,sourceEpoch}={}){
    this.pause({notify:false});
    if(!this.Recorder||this.video.readyState<2)throw new Error('画面尚未就绪，或当前浏览器不支持录制。');
    this.crop=cropRect(region,this.video.videoWidth,this.video.videoHeight);
    this.dimensions=[this.video.videoWidth,this.video.videoHeight];this.sourceEpoch=sourceEpoch;
    this.cuts=new PauseSegmenter({mode,maxSeconds});this.cuts.begin(this.video.currentTime);
    this.canvas=this.makeCanvas();Object.assign(this.canvas,encodeSize(this.crop));
    this.context=this.canvas.getContext('2d');this.analysis=this.makeCanvas();this.analysis.width=32;this.analysis.height=24;
    this.analysisContext=this.analysis.getContext('2d',{willReadFrequently:true});
    if(!this.context||!this.analysisContext||!this.canvas.captureStream)throw new Error('当前浏览器无法采集选区，请换用桌面Chrome。');
    this.previous=null;this.lastAnalysis=-Infinity;this.lastNotice=-Infinity;
    this.paint();this.stream=this.canvas.captureStream(25);this.active=true;
    try{this.beginTake();}catch(error){this.pause({notify:false});throw error;}
    this.onState({status:'recording'});const epoch=this.epoch;
    const next=()=>{if(this.active&&epoch===this.epoch){this.frame();if(this.active)this.frameId=this.requestFrame(next);}};
    this.frameId=this.requestFrame(next);return this.canvas;
  }
  paint(){
    const {x,y,width,height}=this.crop;
    this.context.drawImage(this.video,x,y,width,height,0,0,this.canvas.width,this.canvas.height);
  }
  beginTake(){
    const mime=['video/webm;codecs=vp8','video/webm','video/mp4'].find(type=>this.Recorder.isTypeSupported?.(type));
    const recorder=new this.Recorder(this.stream,mime?{mimeType:mime,videoBitsPerSecond:1800000}:{});
    const take={recorder,epoch:this.epoch,start:this.video.currentTime,end:this.video.currentTime,chunks:[],frames:1,publish:false,finished:false};
    take.done=new Promise(resolve=>take.resolve=resolve);this.take=take;
    const finalize=()=>{
      if(take.finished)return;take.finished=true;
      try{
        if(take.publish&&take.epoch===this.epoch&&take.frames>=2&&take.end-take.start>=.4){
          const type=recorder.mimeType||mime||'video/webm';
          const file=new File(take.chunks,`meeting-${++this.segmentCount}.${type.includes('mp4')?'mp4':'webm'}`,{type});
          if(file.size)this.onSegment(file,{epoch:this.sourceEpoch,startSeconds:take.start,endSeconds:take.end,reason:take.reason});
          else this.onState({status:'error',message:'这段选区视频为空，请重新开始。'});
        }
      }finally{take.chunks=[];take.resolve();}
    };
    recorder.ondataavailable=event=>{if(!take.finished&&take.epoch===this.epoch&&event.data?.size)take.chunks.push(event.data);};
    recorder.onstop=finalize;
    recorder.onerror=()=>{if(take.epoch===this.epoch){this.pause({notify:false});this.onState({status:'error',message:'选区录制中断，请重新开始。'});}finalize();};
    recorder.start();
    take.watchdog=this.setTimer(()=>{
      if(this.take===take&&take.epoch===this.epoch){this.pause({notify:false});this.onState({status:'error',message:'画面更新中断，采集已暂停。请保持本页可见并重新开始。'});}
    },(this.cuts.maxSeconds+3)*1000);
  }
  cut(publish,reason,restart=true){
    const take=this.take;this.take=null;
    if(!take)return Promise.resolve();
    this.clearTimer(take.watchdog);
    take.publish=publish;take.reason=reason;take.end=this.video.currentTime;
    try{if(take.recorder.state==='recording')take.recorder.stop();else take.resolve();}
    catch{take.publish=false;take.chunks=[];take.resolve();this.onState({status:'error',message:'视频片段未能完成，请重新开始。'});}
    if(restart&&this.active){
      this.cuts.begin(this.video.currentTime);
      try{this.beginTake();}catch{this.pause({notify:false});this.onState({status:'error',message:'下一段视频录制未能启动，采集已暂停。请重新开始。'});}
    }
    return take.done;
  }
  frame(){
    if(this.video.videoWidth!==this.dimensions[0]||this.video.videoHeight!==this.dimensions[1]){
      this.pause({notify:false});this.onState({status:'reselect',message:'共享画面尺寸改变，请重新确认选区。'});return;
    }
    if(this.video.readyState<2)return;
    try{
      this.paint();if(this.take)this.take.frames++;
      const time=this.video.currentTime;
      if(time-this.lastAnalysis>=.1){
        this.lastAnalysis=time;this.analysisContext.drawImage(this.canvas,0,0,32,24);
        const pixels=this.analysisContext.getImageData(0,0,32,24).data;
        const values=new Float32Array(pixels.length/4);let sum=0;
        for(let i=0;i<values.length;i++){
          values[i]=(pixels[i*4]+pixels[i*4+1]+pixels[i*4+2])/3;
          if(this.previous)sum+=Math.abs(values[i]-this.previous[i]);
        }
        const activity=this.previous?sum/(values.length*255):0;this.previous=values;
        const cut=this.cuts.feed(time,activity);if(cut)this.cut(cut.publish,cut.reason);
        if(time-this.lastNotice>=.4){this.lastNotice=time;this.onState({status:'recording',time,activity,seconds:time-this.cuts.started});}
      }
    }catch(error){this.pause({notify:false});this.onState({status:'error',message:error.message||'无法读取选区画面，请重新选择来源。'});}
  }
  submit(){
    if(!this.active||!this.take)return false;
    if(this.video.currentTime-this.take.start<.4){this.onState({status:'short',message:'请完成一个词或短句后再提交。'});return false;}
    this.cut(true,'manual');return true;
  }
  pause({notify=true}={}){
    this.epoch++;this.active=false;if(this.frameId!==null)this.cancelFrame(this.frameId);this.frameId=null;
    this.cut(false,'cancelled',false);const owned=this.stream;this.stream=null;release(owned);
    if(notify)this.onState({status:'paused'});
  }
  async finish(){
    this.active=false;if(this.frameId!==null)this.cancelFrame(this.frameId);this.frameId=null;
    const done=this.cut(true,'end',false);const owned=this.stream;this.stream=null;release(owned);await done;
  }
}
