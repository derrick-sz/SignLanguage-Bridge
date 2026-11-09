const API='/v1/animation/composition';
const invalidText=text=>typeof text!=='string'||!text.trim()||[...text].length>200||/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f-\u009f]/u.test(text);
const validGlosses=values=>Array.isArray(values)&&values.length>0&&values.length<=32&&values.every(value=>typeof value==='string'&&value.trim());
const validOmissions=values=>Array.isArray(values)&&values.length>0&&values.every(value=>typeof value==='string'&&value.trim());

// The fast endpoint returns one complete composition, never a partial sentence.
// Every await is followed by a request guard so edits cannot replay old motion.
export class FastCompositionController {
  constructor({stage,fetcher=globalThis.fetch,onState=()=>{},now=()=>performance.now()}) {
    this.stage=stage;this.fetcher=fetcher.bind(globalThis);this.onState=onState;this.now=now;
    this.generation=0;this.abort=null;this.loaded=false;this.result=null;
  }
  invalidate(message='输入已改变，请点击组合并演示。',status='idle') {
    this.generation++;this.abort?.abort();this.abort=null;this.loaded=false;this.result=null;
    this.stage.stop();this.onState({status,message});
  }
  cancel(){this.invalidate('已取消本次演示。','cancelled');}
  stop() {
    if(!this.loaded){this.invalidate('已停止。可以重新组合演示。');return;}
    this.generation++;this.abort?.abort();this.abort=null;
    this.stage.pause();this.stage.seek(0);
    this.onState({...this.result,status:'ready',message:'已停止。点击播放可重新观看。'});
  }
  play(){if(this.loaded)this.stage.play();}
  pause(){if(this.loaded)this.stage.pause();}
  async generate(text) {
    this.invalidate('正在规划手语表达…');
    if(invalidText(text)){this.onState({status:'error',message:'请输入 1–200 个字符，支持换行。'});return;}
    const request=this.generation,abort=new AbortController(),started=this.now();this.abort=abort;
    const current=()=>request===this.generation&&!abort.signal.aborted;
    this.onState({status:'submitting',message:'正在规划并组合动作…',submitted_at_ms:started});
    try {
      const response=await this.fetcher(API,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text}),signal:abort.signal});
      if(!current())return;
      let data;
      try{data=await response.json();}catch{throw new Error('服务响应无法读取，请重新演示。');}
      if(!current())return;
      if(!response.ok)throw new Error(typeof data?.detail==='string'?data.detail:'快速组合暂时不可用，请稍后重试。');
      if(data?.input_text!==text||!validGlosses(data.glosses))throw new Error('组合结果与本次输入不一致，请重新演示。');
      const timings=data.timings_ms??{};
      if(typeof timings!=='object'||Array.isArray(timings)||Object.values(timings).some(value=>typeof value!=='number'||!Number.isFinite(value)||value<0))throw new Error('组合耗时记录无效，请重新演示。');
      const detail={glosses:[...data.glosses],timings_ms:{...timings},submitted_at_ms:started};
      if(data.status==='blocked') {
        if(data.motion!=null||!validGlosses(data.missing_glosses))throw new Error('缺词结果不完整，请重新演示。');
        this.onState({...detail,status:'blocked',missing_glosses:[...data.missing_glosses],message:'缺少动作：'+data.missing_glosses.join('、')+'。请修改文字或选择模型生成。'});
        return;
      }
      if(data.status==='needs_review') {
        if(data.motion!=null||!validOmissions(data.omitted_terms))throw new Error('文字规划核对结果不完整，请重新演示。');
        this.onState({...detail,status:'needs_review',omitted_terms:[...data.omitted_terms],message:'文字规划可能遗漏：'+data.omitted_terms.join('、')+'。请核对或缩短输入。'});
        return;
      }
      const motion=data.motion,clip=motion?.clip;
      if(data.status!=='ready'||typeof data.generation_text!=='string'||!data.generation_text||clip?.text!==data.generation_text||
          clip.source_kind!=='retrieved_word_composition'||typeof clip.id!=='string'||!clip.id||!/^[0-9a-f]{64}$/i.test(motion?.sha256??'')) {
        throw new Error('动作字幕或来源与本次组合不一致，请重新演示。');
      }
      this.onState({...detail,status:'loading',message:'组合完成，正在加载人物动作…'});
      const loaded=await this.stage.loadMotion(motion,clip.id,current,motion.sha256);
      if(!current()||loaded===false)return;
      this.loaded=true;
      this.result={...detail,subtitle:clip.text,client_play_request_ms:Math.max(0,this.now()-started)};
      // Ready must reach the view before play(), whose refresh emits a tick.
      this.onState({...this.result,status:'ready',message:'正在演示。请核对手语含义与动作衔接。'});
      if(current()&&this.loaded)this.stage.play();
    }catch(error) {
      if(current()) {
        this.loaded=false;this.result=null;this.stage.stop();
        this.onState({status:'error',message:error.message||'快速组合失败，请重试。'});
      }
    }
  }
}
