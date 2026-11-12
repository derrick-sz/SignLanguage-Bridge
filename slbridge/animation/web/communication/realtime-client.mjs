// Versioned live session controller; inference stays in the model service.
export const LIVE_PROTOCOL='csl-realtime-v1';

function localSource(value){
  const kinds=['camera','screen','file','training_replay','diagnostic_replay'];
  if(!value||!kinds.includes(value.kind))return{kind:'unknown',label:'来源未标记'};
  const result={kind:value.kind,label:typeof value.label==='string'?value.label.slice(0,160):'来源未标记'};
  if(typeof value.id==='string'&&/^[a-z0-9][a-z0-9_-]{0,63}$/.test(value.id))result.id=value.id;
  return result;
}

export class RealtimeClient{
  constructor({url,makeSocket=value=>new WebSocket(value),now=()=>performance.now(),onUpdate=()=>{},
    fetcher=(...args)=>globalThis.fetch(...args),onMetrics=()=>{}}={}){
    this.url=url;this.makeSocket=makeSocket;this.now=now;this.onUpdate=onUpdate;
    this.epoch=0;this.socket=null;this.status='idle';this.message='';this.words=[];this.captions=[];
    this.draft=null;this.advice=null;this.sessionId=null;this.lastSequence=-1;
    this.versions=new Map();this.utterance=-1;this.lastFrameRoundTripMs=null;this.firstWordMs=null;this.captureStartedAt=null;
    this.fetcher=fetcher;this.adviceJobs=new Map();
    this.onMetrics=onMetrics;this.lastMetricsAt=-Infinity;this.startRequestedAt=null;
    this.readyMs=null;this.firstWordTotalMs=null;this.lastQueueMs=null;this.lastProcessingMs=null;
    this.source=localSource(null);
    this.framesSent=0;this.framesProcessed=0;this.summaryReceived=false;
  }
  get currentCaption(){return this.draft?.epoch===this.epoch?this.draft:this.captions.filter(item=>item.epoch===this.epoch).at(-1)||null;}
  snapshot(){return structuredClone({status:this.status,message:this.message,epoch:this.epoch,words:this.words,
    captions:this.captions,draft:this.draft,currentCaption:this.currentCaption,advice:this.advice,lastFrameRoundTripMs:this.lastFrameRoundTripMs,firstWordMs:this.firstWordMs,
    framesSent:this.framesSent,framesProcessed:this.framesProcessed,
    readyMs:this.readyMs,firstWordTotalMs:this.firstWordTotalMs,lastQueueMs:this.lastQueueMs,lastProcessingMs:this.lastProcessingMs});}
  emit(){this.onUpdate(this.snapshot());}
  emitMetrics(force=false){
    const now=this.now();if(!force&&now-this.lastMetricsAt<500)return;
    this.lastMetricsAt=now;this.onMetrics({epoch:this.epoch,status:this.status,readyMs:this.readyMs,
      framesSent:this.framesSent,framesProcessed:this.framesProcessed,
      frameRoundTripMs:this.lastFrameRoundTripMs,queueMs:this.lastQueueMs,processingMs:this.lastProcessingMs});
  }
  async start({fps=30,assistance=false,source}={}){
    this.stop();const epoch=++this.epoch;
    this.startRequestedAt=this.now();this.readyMs=this.firstWordTotalMs=null;
    this.source=localSource(source);
    this.framesSent=this.framesProcessed=0;this.summaryReceived=false;
    this.lastQueueMs=this.lastProcessingMs=this.lastFrameRoundTripMs=null;this.lastMetricsAt=-Infinity;
    this.status='connecting';this.message='正在准备实时识别…';this.words=[];this.draft=null;this.advice=null;
    this.sessionId=null;this.lastSequence=-1;this.utterance=-1;this.versions.clear();this.firstWordMs=null;this.captureStartedAt=null;
    const url=this.url||`${location.protocol==='https:'?'wss:':'ws:'}//${location.host}/v1/communication/realtime`;
    const socket=this.makeSocket(url);this.socket=socket;this.emit();this.emitMetrics(true);
    return new Promise((resolve,reject)=>{
      this.pending={resolve,reject};
      this.timer=setTimeout(()=>{if(epoch===this.epoch)this.fail('实时服务准备超时，请稍后重试。');},15000);
      socket.onopen=()=>{if(epoch===this.epoch)socket.send(JSON.stringify({type:'start',protocol:LIVE_PROTOCOL,
        sign_language:'csl',source_epoch:epoch,fps,...(assistance?{assistance:true}:{})}));};
      socket.onmessage=event=>{
        if(epoch!==this.epoch)return;
        try{this.receive(JSON.parse(event.data));}catch{this.fail('实时服务返回了无法读取的结果。');}
      };
      socket.onerror=()=>{if(epoch===this.epoch)this.fail('无法连接实时识别服务。');};
      socket.onclose=()=>{if(epoch===this.epoch&&!['stopped','error'].includes(this.status))this.fail('实时连接已结束，请重新开始。');};
    });
  }
  receive(data){
    if(data.protocol!==LIVE_PROTOCOL)throw new Error('Protocol changed');
    if(data.type==='error'){this.fail(data.message||'实时识别暂不可用，请重新开始。');return;}
    if(data.source_epoch!==this.epoch||typeof data.session_id!=='string'||!Number.isInteger(data.sequence)||data.sequence<=this.lastSequence)return;
    if(data.type==='ready'){
      this.readyMs=Math.max(0,this.now()-this.startRequestedAt);
      this.sessionId=data.session_id;this.lastSequence=data.sequence;this.status='running';this.message='实时词识别已开始';
      clearTimeout(this.timer);this.pending?.resolve(data);this.pending=null;this.emit();this.emitMetrics(true);return;
    }
    if(data.session_id!==this.sessionId)return;
    this.lastSequence=data.sequence;
    if(data.type==='word'){
      if(!Number.isInteger(data.utterance_id)||data.utterance_id<this.utterance||typeof data.gloss!=='string'||data.gloss.length>128)return;
      if(data.utterance_id!==this.utterance){this.words=[];this.advice=null;this.utterance=data.utterance_id;}
      const id=String(data.event_id);const index=this.words.findIndex(x=>x.id===id);
      if(data.status==='retracted'){if(index>=0)this.words.splice(index,1);}
      else{
        const word={id,text:data.gloss,status:data.status==='final'?'final':'partial'};
        if(index>=0)this.words[index]=word;else this.words.push(word);
        if(this.words.length>128){this.fail('当前句子过长，请暂停后重新开始。');return;}
        if(this.firstWordMs===null&&this.captureStartedAt!==null){
          this.firstWordMs=this.now()-this.captureStartedAt;
          this.firstWordTotalMs=Math.max(0,this.now()-this.startRequestedAt);
        }
      }
    }else if(data.type==='sentence'){
      if(!Number.isInteger(data.utterance_id)||!Number.isInteger(data.revision)||data.revision<1||typeof data.text!=='string'||data.text.length>8000)return;
      const id=`${data.session_id}:${data.utterance_id}`,old=this.versions.get(id)||0;
      if(data.revision<old)return;
      const found=this.captions.find(x=>x.id===id);
      // Human review pins this caption and its original advice provenance.
      // A refused revision must not cancel a still-valid original advice job.
      if(found&&(found.confirmed||found.edited||data.revision<=found.revision))return;
      if(data.revision>old){
        for(const [jobId,job] of this.adviceJobs){
          if(job.session_id===data.session_id&&job.utterance_id===data.utterance_id&&job.revision<data.revision){
            clearTimeout(job.timer);job.controller?.abort();this.adviceJobs.delete(jobId);
          }
        }
      }
      this.versions.set(id,data.revision);if(this.versions.size>200)this.versions.delete(this.versions.keys().next().value);
      if(found)delete found.advice;
      const word_evidence=Array.isArray(data.predicted_gloss)&&data.predicted_gloss.length<=128&&
        data.predicted_gloss.every(word=>typeof word==='string'&&word.trim()&&word.length<=128)?[...data.predicted_gloss]:[];
      const review_hint=word_evidence.length===1&&data.review_hint?.reason==='single_word_expansion'?
        {reason:'single_word_expansion'}:null;
      const entry={id,epoch:this.epoch,utterance_id:data.utterance_id,revision:data.revision,text:data.text,status:data.status,
        word_evidence,review_hint,confirmed:false,source:{...this.source}};
      this.draft=entry;this.advice=null;
      if(data.status==='rejected')this.message=data.message||'本句未能完成翻译，请核对实时词或分句重试。';
      if(data.status==='committed'&&data.text.trim()){
        if(!found){this.captions.push({...entry,original_text:data.text,edited:false});if(this.captions.length>200)this.captions.shift();}
        else if(!found.edited)Object.assign(found,entry,{original_text:data.text});
      }
    }else if(data.type==='assistance'){
      const id=`${data.session_id}:${data.utterance_id}`;
      if(this.versions.get(id)!==data.revision)return;
      if(this.captions.find(x=>x.id===id)?.revision!==data.revision)return;
      if(data.status==='pending'){
        if(!/^[0-9a-f]{32}$/.test(data.job_id||''))return;
        this.applyAdvice(data);
        if(!this.adviceJobs.has(data.job_id)&&this.adviceJobs.size<8){
          const job={epoch:this.epoch,session_id:this.sessionId,source_epoch:this.epoch,
            utterance_id:data.utterance_id,revision:data.revision,attempts:0};
          this.adviceJobs.set(data.job_id,job);this.pollAdvice(data.job_id,job);
        }
      }else this.applyAdvice(data);
    }else if(data.type==='utterance_reset'){
      this.words=[];this.draft=null;this.advice=null;this.message=data.message||'动作中断，正在重新积累输入。';
    }else if(data.type==='ack'){
      ++this.framesProcessed;
      const elapsed=this.now()-data.capture_ms;if(Number.isFinite(elapsed)&&elapsed>=0)this.lastFrameRoundTripMs=elapsed;
      if(Number.isFinite(data.queue_ms)&&data.queue_ms>=0)this.lastQueueMs=data.queue_ms;
      if(Number.isFinite(data.processing_ms)&&data.processing_ms>=0)this.lastProcessingMs=data.processing_ms;
      this.emitMetrics();return;
    }else if(data.type==='summary'){
      if(Number.isInteger(data.frames_processed)&&data.frames_processed>=0){
        this.framesProcessed=data.frames_processed;this.summaryReceived=true;this.emitMetrics(true);
      }
      return;
    }else if(['finished','stopped'].includes(data.type)){
      const empty=this.summaryReceived?this.framesProcessed===0:
        this.framesSent===0&&this.framesProcessed===0&&!this.currentCaption&&!this.words.length;
      this.status=data.type==='finished'&&empty?'error':'stopped';
      this.message=data.type!=='finished'?'识别已停止。':empty?'本次未采集到视频画面，请确认预览正在播放后重试。':
        this.currentCaption?.text?'本次识别已结束，译文请核对后使用。':
        this.words.length?'识别已结束，请核对本次实时词。':'本次未识别到可提交的手语，请检查画面或分句重试。';
      this.emitMetrics(true);
      this.emit();this.socket?.close();return;
    }
    this.emit();
  }
  noteCaptureStart(){this.captureStartedAt=this.now();}
  applyAdvice(data){
    const id=`${data.session_id}:${data.utterance_id}`;
    if(this.versions.get(id)!==data.revision)return;
    const item=this.captions.find(x=>x.id===id);if(!item||item.revision!==data.revision)return;
    const fallback=data.status==='pending'?'正在辅助核对…':data.status==='insufficient'?'信息不足，请人工核对原译文。':'本次辅助核对未完成，原译文已保留。';
    item.advice={utterance_id:data.utterance_id,revision:data.revision,status:data.status,text:typeof data.text==='string'&&data.text?data.text.slice(0,8000):fallback};
    if(this.draft?.id===id)this.advice=item.advice;
    this.emit();
  }
  async pollAdvice(id,job){
    if(job.epoch!==this.epoch||!this.adviceJobs.has(id))return;
    job.controller=new AbortController();const timeout=setTimeout(()=>job.controller.abort(),3000);
    try{
      const response=await this.fetcher('/v1/communication/realtime/assistance/'+id,{signal:job.controller.signal});
      if(!response.ok)throw new Error('Advice unavailable');
      const data=await response.json();
      if(job.epoch!==this.epoch||!this.adviceJobs.has(id))return;
      if(data.job_id!==id||data.session_id!==job.session_id||data.source_epoch!==job.source_epoch||
         data.utterance_id!==job.utterance_id||data.revision!==job.revision)throw new Error('Advice binding changed');
      if(['pending','running'].includes(data.status)&&++job.attempts<50){
        job.timer=setTimeout(()=>this.pollAdvice(id,job),300);return;
      }
      this.adviceJobs.delete(id);this.applyAdvice(['pending','running'].includes(data.status)?{...data,status:'unavailable',text:''}:data);
    }catch{
      if(job.epoch===this.epoch&&this.adviceJobs.has(id)){
        this.adviceJobs.delete(id);this.applyAdvice({...job,status:'unavailable',text:''});
      }
    }finally{clearTimeout(timeout);}
  }
  sendFrame(frame,epoch=this.epoch){
    if(epoch!==this.epoch||this.status!=='running'||this.socket?.readyState!==1)return false;
    if(this.socket.bufferedAmount>196608){this.fail('视频发送出现积压，识别暂时跟不上，请暂停后重试。');return false;}
    const text=JSON.stringify({...frame,type:'frame',source_epoch:epoch});
    if(text.length>262144){this.fail('当前画面数据过大，请缩小选区后重试。');return false;}
    try{this.socket.send(text);++this.framesSent;return true;}catch{this.fail('实时连接已结束。');return false;}
  }
  boundary(){if(this.status==='running')this.socket?.send(JSON.stringify({type:'boundary'}));}
  finish(){if(this.status==='running'){this.status='finishing';this.message='正在补全最后一句…';this.socket?.send(JSON.stringify({type:'finish'}));this.emit();}}
  stop(){
    ++this.epoch;clearTimeout(this.timer);
    for(const job of this.adviceJobs.values()){clearTimeout(job.timer);job.controller?.abort();}
    this.adviceJobs.clear();
    for(const item of this.captions)if(item.advice?.status==='pending')item.advice={...item.advice,status:'cancelled',text:'辅助核对已停止，原译文已保留。'};
    const socket=this.socket;this.socket=null;
    if(socket){if(socket.readyState===1)socket.send(JSON.stringify({type:'stop'}));socket.close();}
    this.pending?.reject(new Error('实时识别已取消'));this.pending=null;
    this.status='stopped';this.words=[];this.draft=null;this.advice=null;this.emit();this.emitMetrics(true);
  }
  fail(message){const pending=this.pending;this.pending=null;this.stop();this.status='error';this.message=message;pending?.reject(new Error(message));this.emit();this.emitMetrics(true);}
  confirm(id){
    const item=this.captions.find(x=>x.id===id);if(!item)return;
    item.confirmed=true;if(this.draft?.id===id)this.draft={...item};this.emit();
  }
  edit(id,text){
    const item=this.captions.find(x=>x.id===id);
    if(item&&typeof text==='string'&&text.trim()&&text.length<=8000){
      item.text=text.trim();item.edited=item.text!==item.original_text;item.confirmed=true;
      if(this.draft?.id===id)this.draft={...item};this.emit();
    }
  }
  clear(){this.stop();this.captions=[];this.versions.clear();this.message='记录已清空。';this.emit();}
}
