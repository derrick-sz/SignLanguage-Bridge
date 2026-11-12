import {RecognitionController} from './recognition-controller.mjs';

export const captionKind=kind=>kind==='text'?'句子草稿':kind==='gloss_sequence'?'手语词序列':kind==='classification'?'词语释义':'待核对文字';
export function clock(seconds){
  const total=Math.max(0,Math.floor(seconds||0));return Math.floor(total/60)+':'+String(total%60).padStart(2,'0');
}

export class CaptionSession {
  constructor({fetcher=globalThis.fetch,uid=()=>crypto.randomUUID(),now=()=>new Date().toISOString(),onUpdate=()=>{},maxQueued=2,limit=200}={}){
    if(!Number.isInteger(maxQueued)||maxQueued<1||maxQueued>4||!Number.isInteger(limit)||limit<1||limit>1000)throw new RangeError('Invalid caption budget');
    this.uid=uid;this.now=now;this.onUpdate=onUpdate;this.maxQueued=maxQueued;this.limit=limit;
    this.epoch=0;this.status='idle';this.inputFinished=false;this.language='csl';this.sourceLabel='共享画面';
    this.queue=[];this.current=null;this.items=[];this.cleared=null;this.notice='';this.skipped=0;this.empty=0;this.errors=0;
    this.recognizer=new RecognitionController({fetcher,uid,onState:state=>{
      if(this.current?.epoch===this.epoch&&this.status==='running'&&['ready','empty','error'].includes(state.status))this.current.outcome=state;
    }});
  }
  snapshot(){return {status:this.status,epoch:this.epoch,language:this.language,sourceLabel:this.sourceLabel,
    busy:!!this.current,pending:this.queue.length,skipped:this.skipped,empty:this.empty,errors:this.errors,
    notice:this.notice,captions:structuredClone(this.items),canUndo:!!this.cleared};}
  emit(){this.onUpdate(this.snapshot());}
  start({language=this.language,sourceLabel=this.sourceLabel}={}){
    if(!['csl','libras'].includes(language))throw new RangeError('Unsupported sign language');
    this.epoch++;this.queue=[];this.recognizer.invalidate();this.language=language;this.sourceLabel=sourceLabel;
    this.status='running';this.inputFinished=false;this.notice='';this.emit();return this.epoch;
  }
  pause(){this.epoch++;this.status='paused';this.inputFinished=false;this.queue=[];this.recognizer.invalidate();this.emit();}
  stop(){this.pause();this.status='stopped';this.emit();}
  enqueue(file,{startSeconds,endSeconds,epoch=this.epoch}={}){
    if(this.status!=='running'||this.inputFinished||epoch!==this.epoch)return false;
    if(!file?.size||file.size>128*1024*1024||!Number.isFinite(startSeconds)||!Number.isFinite(endSeconds)||startSeconds<0||endSeconds<=startSeconds){
      this.errors++;this.notice='这段画面无法提交，请重新开始采集。';this.emit();return false;
    }
    if(this.queue.length>=this.maxQueued){
      this.skipped++;this.notice='识别速度暂时跟不上，已跳过一个片段。请暂停画面，等待处理后继续。';this.emit();return false;
    }
    this.queue.push({id:'segment-'+this.uid(),file,epoch,language:this.language,sourceLabel:this.sourceLabel,startSeconds,endSeconds});
    this.pump();this.emit();return true;
  }
  finishInput(){
    if(this.status!=='running')return;
    this.inputFinished=true;if(!this.current&&!this.queue.length)this.status='stopped';this.emit();
  }
  async pump(){
    if(this.current||this.status!=='running'||!this.queue.length)return;
    const job=this.queue.shift();this.current=job;this.emit();
    try{
      await this.recognizer.recognize(job.file,job.language);
      if(job.epoch!==this.epoch||this.status!=='running')return;
      const outcome=job.outcome;
      if(outcome?.status==='ready'){
        const result=outcome.result;
        this.items.push({id:job.id,request_id:result.request_id,text:result.text,original_text:result.text,
          output_kind:result.output_kind,sign_language:result.sign_language,review_status:result.review_status,
          provenance:structuredClone(result.provenance||{}),source_label:job.sourceLabel,source_epoch:job.epoch,
          start_seconds:job.startSeconds,end_seconds:job.endSeconds,created_at:this.now(),edited:false});
        if(this.items.length>this.limit)this.items.splice(0,this.items.length-this.limit);
        this.cleared=null;
      }else if(outcome?.status==='empty'){
        this.empty++;this.notice='一段画面未返回清楚的识别文字。请检查选区是否包含脸部、双手和上半身。';
      }else{
        this.errors++;this.notice=outcome?.message||'这段字幕未能识别，请检查服务或重新选区。';
      }
    }finally{
      if(this.current===job)this.current=null;
      if(this.inputFinished&&!this.current&&!this.queue.length&&this.status==='running')this.status='stopped';
      this.emit();this.pump();
    }
  }
  edit(id,text){
    if(typeof text!=='string'||!text.trim()||text.length>8000)throw new RangeError('请填写有效的核对文字。');
    const item=this.items.find(x=>x.id===id);if(!item)return false;
    item.text=text;item.edited=text!==item.original_text;this.emit();return true;
  }
  clear(){this.pause();this.cleared=this.items;this.items=[];this.emit();}
  undoClear(){if(!this.cleared||this.items.length)return false;this.items=this.cleared;this.cleared=null;this.emit();return true;}
  transcript(){
    const lines=['手语桥 · 会议字幕','自动识别结果，请核对后使用。',''];
    for(const item of this.items){
      lines.push(`${clock(item.start_seconds)}–${clock(item.end_seconds)}  ${captionKind(item.output_kind)}${item.edited?' · 已人工修改':''}`,
        item.text);
      if(item.edited)lines.push('识别原文：'+item.original_text);
      lines.push('');
    }
    return lines.join('\n');
  }
}
