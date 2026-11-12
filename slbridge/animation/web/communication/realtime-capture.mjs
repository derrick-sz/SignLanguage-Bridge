// Capture one bounded JPEG at a time; missed sampling slots retain their indices.
import {cropRect} from './selection.mjs';

async function encodeBlob(blob){
  const bytes=new Uint8Array(await blob.arrayBuffer());let text='';
  for(let i=0;i<bytes.length;i+=8192)text+=String.fromCharCode(...bytes.subarray(i,i+8192));
  return btoa(text);
}

export class LiveFrameCapture{
  constructor({video,makeCanvas=()=>document.createElement('canvas'),now=()=>performance.now(),
    requestFrame=fn=>requestAnimationFrame(fn),cancelFrame=id=>cancelAnimationFrame(id),
    setTimer=(fn,ms)=>setTimeout(fn,ms),clearTimer=id=>clearTimeout(id),
    encodeBlob:encode=encodeBlob,onFrame=()=>false,onError=()=>{},onPreview=()=>{},onMetrics=()=>{}}){
    Object.assign(this,{video,makeCanvas,now,requestFrame,cancelFrame,setTimer,clearTimer,encode,onFrame,onError,onPreview,onMetrics});
    this.version=0;this.active=false;this.encoding=false;this.frameHandle=this.watchHandle=null;
  }
  start({fps=30,region}){
    this.stop();
    if(!Number.isFinite(fps)||fps<10||fps>60||!region||!this.video.videoWidth||!this.video.videoHeight)throw new Error('请先确认视频与识别选区。');
    this.region={...region};this.fps=fps;this.started=this.now();this.lastIndex=-1;
    this.capturedFrames=0;this.missedSlots=0;this.lastMetricsAt=-Infinity;this.lastEncodeMs=null;this.lastJpegBytes=null;
    this.width=this.video.videoWidth;this.height=this.video.videoHeight;
    this.canvas=this.makeCanvas();this.canvas.width=this.canvas.height=512;
    this.active=true;this.encoding=false;this.lastProgress=this.now();const version=this.version;
    const fail=message=>{if(version===this.version&&(this.active||this.draining)){this.stop();this.onError(message);}};
    const watch=()=>{
      if(version!==this.version||!this.active&&!this.draining)return;
      if(this.now()-this.lastProgress>=2000){fail('视频画面采集停滞，请保持页面在前台，确认预览正在播放后重试。');return;}
      this.watchHandle=this.setTimer(watch,250);
    };
    const loop=()=>{
      if(!this.active||version!==this.version)return;
      try{
      this.frameHandle=this.requestFrame(loop);
      if(this.video.videoWidth!==this.width||this.video.videoHeight!==this.height){
        this.stop();this.onError('画面尺寸发生变化，请重新确认选区。');return;
      }
      const captureMs=this.now(),timestamp=(captureMs-this.started)/1000;
      const index=Math.floor(timestamp*this.fps+1e-6);
      if(this.encoding||index<=this.lastIndex||this.video.readyState<2)return;
      this.encoding=true;this.missedSlots+=Math.max(0,index-this.lastIndex-1);this.lastIndex=index;
      const crop=cropRect(this.region,this.width,this.height),context=this.canvas.getContext('2d');
      if(!context)throw new Error('Video canvas unavailable');
      context.fillStyle='#000';context.fillRect(0,0,512,512);
      const scale=Math.min(512/crop.width,512/crop.height),w=crop.width*scale,h=crop.height*scale;
      context.drawImage(this.video,crop.x,crop.y,crop.width,crop.height,(512-w)/2,(512-h)/2,w,h);
      this.onPreview(this.canvas);
      let resolve;const pending=new Promise(done=>{resolve=done;});
      this.pending=pending;this.resolvePending=resolve;
      this.canvas.toBlob(async blob=>{
        try{
          if(version!==this.version||!this.active&&!this.draining)return;
          if(!blob||blob.size>180*1024)throw new Error('当前画面数据过大，请调整选区。');
          const jpeg=await this.encode(blob);
          if(version!==this.version||!this.active&&!this.draining)return;
          this.encoding=false;
          if(this.onFrame({index,timestamp,capture_ms:captureMs,jpeg})===false){this.stop();return;}
          ++this.capturedFrames;
          this.lastProgress=this.now();this.lastEncodeMs=Math.max(0,this.lastProgress-captureMs);this.lastJpegBytes=blob.size;this.publishMetrics();
        }catch(error){if(version===this.version){this.stop();this.onError(error.message||'无法采集当前画面。');}}
        finally{if(this.pending===pending){this.pending=null;this.resolvePending=null;}resolve();}
      },'image/jpeg',.9);
      }catch{fail('无法读取视频画面，请重新载入或调整选区后重试。');}
    };
    this.watchHandle=this.setTimer(watch,250);
    loop();
  }
  async finish(){
    const version=this.version;this.draining=true;this.active=false;
    if(this.frameHandle!==null)this.cancelFrame(this.frameHandle);this.frameHandle=null;
    await this.pending;
    if(version!==this.version)return false;
    this.stop();return true;
  }
  publishMetrics(force=false){
    const now=this.now();if(!force&&(!this.capturedFrames||now-this.lastMetricsAt<500))return;
    this.lastMetricsAt=now;this.onMetrics({encodeMs:this.lastEncodeMs,jpegBytes:this.lastJpegBytes,
      capturedFrames:this.capturedFrames,missedSlots:this.missedSlots,fps:this.fps});
  }
  stop(){
    if(this.active||this.draining)this.publishMetrics(true);
    ++this.version;this.active=false;this.draining=false;
    if(this.frameHandle!==null)this.cancelFrame(this.frameHandle);this.frameHandle=null;this.encoding=false;
    if(this.watchHandle!==null)this.clearTimer(this.watchHandle);this.watchHandle=null;
    const resolve=this.resolvePending;this.resolvePending=null;this.pending=null;resolve?.();
  }
}
