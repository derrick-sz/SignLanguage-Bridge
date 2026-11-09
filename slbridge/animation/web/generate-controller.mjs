const API = '/v1/animation/generation/jobs';
const ACTIVE = new Set(['queued','running']);
const PHASES = {preparing:'准备生成',body:'生成身体动作',hand:'生成双手动作',face:'生成面部动作',packaging:'整理演示动作'};

// An abortable delay avoids retaining a poll after an edit or page navigation.
export function pollDelay(ms, signal) {
  return new Promise(resolve => {
    const done = () => {clearTimeout(timer);signal?.removeEventListener('abort',done);resolve();};
    const timer = setTimeout(done,ms);
    if (signal?.aborted) done(); else signal?.addEventListener('abort',done,{once:true});
  });
}

export class GenerationController {
  constructor({stage,fetcher=globalThis.fetch,sleep=pollDelay,onState=()=>{}}) {
    this.stage=stage;this.fetcher=fetcher.bind(globalThis);this.sleep=sleep;this.onState=onState;
    this.generation=0;this.abort=null;this.job=null;this.loaded=false;this.subtitle='';
  }
  discardJob(job) {
    if (!job?.job_id || !ACTIVE.has(job.state)) return;
    // Best effort only: a slow cancellation must never block a new input.
    try {Promise.resolve(this.fetcher(API+'/'+encodeURIComponent(job.job_id),{method:'DELETE',keepalive:true})).catch(()=>{});} catch {}
  }
  invalidate(message='输入已改变，请点击生成演示。',status='idle') {
    this.generation++;this.abort?.abort();this.abort=null;
    this.discardJob(this.job);this.job=null;this.loaded=false;this.subtitle='';
    this.stage.stop();this.onState({status,message});
  }
  cancel() {this.invalidate('已取消生成。','cancelled');}
  stop() {
    if(!this.loaded){this.invalidate('已停止。可重新生成演示。');return;}
    this.generation++;this.abort?.abort();this.abort=null;
    this.stage.pause();this.stage.seek(0);
    this.onState({status:'ready',job:this.job,subtitle:this.subtitle,message:'已停止。点击播放可重新观看。'});
  }
  play() {if(this.loaded)this.stage.play();}
  pause() {if(this.loaded)this.stage.pause();}
  async generate(text,seconds=6) {
    this.invalidate('正在提交生成任务…');
    if(typeof text!=='string' || !text.trim() || [...text].length>200 || /[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f-\u009f]/u.test(text)) {
      this.onState({status:'error',message:'请输入 1–200 个字符，支持换行。'});return;
    }
    if(![4,6,8,10].includes(seconds)) {this.onState({status:'error',message:'请选择 4、6、8 或 10 秒。'});return;}
    const generation=this.generation,abort=new AbortController();this.abort=abort;
    const current=()=>generation===this.generation && !abort.signal.aborted;
    const request=async(url,options={})=>{
      // Keep the submit response readable after an edit so its assigned ID can
      // still be cancelled. Aborting POST can leave an unknown GPU job behind.
      const response=await this.fetcher(url,{...options,...(options.method==='POST'?{}:{signal:abort.signal})});
      let data;try {data=await response.json();}catch {throw new Error('服务响应无法读取，请重试。');}
      if(!response.ok)throw new Error(typeof data.detail==='string'?data.detail:typeof data.error==='string'?data.error:'生成请求失败，请重试。');
      return data;
    };
    this.onState({status:'submitting',message:'正在提交生成任务…'});
    try {
      let job=await request(API,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text,seconds})});
      if(!current()){this.discardJob(job);return;}
      if(!job.job_id)throw new Error('生成任务缺少标识，请重试。');
      const id=job.job_id;
      while(current()) {
        if(job.job_id!==id)throw new Error('任务标识不一致，请重新生成。');
        this.job=job;
        if(!ACTIVE.has(job.state))break;
        this.onState({status:job.state,job,message:job.state==='queued'?'正在排队，轮到后开始生成。':(PHASES[job.phase]||'正在生成动作')+'…'});
        await this.sleep(1500,abort.signal);
        if(!current())return;
        job=await request(API+'/'+encodeURIComponent(id));
        if(!current())return;
      }
      if(!current())return;
      if(job.state==='cancelled'){this.onState({status:'cancelled',job,message:'生成已取消。'});return;}
      if(job.state!=='complete')throw new Error(typeof job.error==='string'&&job.error?job.error:'生成未完成，请重试。');
      this.onState({status:'loading',job,message:'生成完成，正在加载人物动作…'});
      // Use the fixed task endpoint instead of following a server supplied URL.
      const data=await request(API+'/'+encodeURIComponent(id)+'/motion');
      if(!current())return;
      if(typeof data.clip?.text!=='string' || !data.clip.text || data.clip.text!==job.generation_text)throw new Error('动作字幕与本次生成文字不一致，请重新生成。');
      const loaded=await this.stage.loadMotion(data,data.clip.id,current,data.sha256);
      if(!current() || loaded===false)return;
      this.loaded=true;this.subtitle=data.clip.text;
      this.onState({status:'ready',job,subtitle:data.clip.text,message:'演示已准备，点击播放。生成演示仍需核对手语含义。'});
    } catch(error) {
      if(current()) {this.loaded=false;this.stage.stop();this.onState({status:'error',job:this.job,message:error.message||'生成失败，请重试。'});}
    }
  }
}
