export class CameraCapture {
  constructor({getUserMedia=constraints=>navigator.mediaDevices.getUserMedia(constraints),Recorder=globalThis.MediaRecorder,
    onState=()=>{},maxSeconds=12,setTimer=setTimeout,clearTimer=clearTimeout}={}){
    this.getUserMedia=getUserMedia;this.Recorder=Recorder;this.onState=onState;this.maxSeconds=maxSeconds;
    this.setTimer=setTimer;this.clearTimer=clearTimer;this.epoch=0;this.stream=null;this.recorder=null;this.timer=null;
  }
  close(){
    this.epoch++;if(this.timer)this.clearTimer(this.timer);this.timer=null;
    const recorder=this.recorder;this.recorder=null;
    const stream=this.stream;this.stream=null;
    if(recorder?.state==='recording')recorder.stop();
    stream?.getTracks().forEach(track=>track.stop());this.onState({status:'closed'});
  }
  async open(){
    this.close();const epoch=this.epoch;this.onState({status:'opening'});
    try{
      const stream=await this.getUserMedia({audio:false,video:{facingMode:'user',width:{ideal:1280},height:{ideal:720}}});
      if(epoch!==this.epoch){stream.getTracks().forEach(track=>track.stop());return false;}
      this.stream=stream;this.onState({status:'preview',stream});return true;
    }catch{if(epoch===this.epoch)this.onState({status:'error',message:'摄像头未开启。请允许摄像头权限，或上传已有视频。'});return false;}
  }
  start(){
    if(!this.stream||!this.Recorder||this.recorder)return false;
    const epoch=this.epoch,stream=this.stream,chunks=[];
    const mime=['video/webm;codecs=vp8','video/mp4','video/webm'].find(type=>this.Recorder.isTypeSupported?.(type));
    let recorder;
    try{recorder=new this.Recorder(stream,mime?{mimeType:mime,videoBitsPerSecond:1500000}:{});}catch{
      this.close();this.onState({status:'error',message:'录制未能启动，请重新开启摄像头或上传视频。'});return false;
    }
    this.recorder=recorder;let ownedTimer=null;
    recorder.ondataavailable=event=>{if(event.data?.size)chunks.push(event.data);};
    recorder.onstop=()=>{
      if(ownedTimer)this.clearTimer(ownedTimer);if(this.timer===ownedTimer)this.timer=null;
      if(this.recorder===recorder)this.recorder=null;
      if(this.stream===stream){stream.getTracks().forEach(track=>track.stop());this.stream=null;}
      if(epoch!==this.epoch)return;
      const type=recorder.mimeType||mime||'video/webm',extension=type.includes('mp4')?'mp4':'webm';
      const file=new File(chunks,'手语录制.'+extension,{type});
      this.onState(file.size?{status:'recorded',file}:{status:'error',message:'这次没有录到视频，请重新录制。'});
    };
    recorder.onerror=()=>{if(epoch===this.epoch){this.close();this.onState({status:'error',message:'录制中断，请重新录制或上传视频。'});}};
    try{recorder.start();}catch{this.close();this.onState({status:'error',message:'录制未能启动，请重新开启摄像头或上传视频。'});return false;}
    this.onState({status:'recording'});ownedTimer=this.setTimer(()=>{if(this.recorder===recorder)this.finish();},this.maxSeconds*1000);this.timer=ownedTimer;return true;
  }
  finish(){if(this.recorder?.state==='recording')this.recorder.stop();}
}
