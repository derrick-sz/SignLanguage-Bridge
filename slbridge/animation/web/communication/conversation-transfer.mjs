// Confirmed text only. Video, targets and advice never cross this channel.
export const TRANSFER_PROTOCOL='slbridge-conversation-v1';
const verified=new WeakSet(),duration=10000;
const kinds=['camera','screen','file','training_replay','diagnostic_replay','unknown'];
const invalid=()=>new Error('这条字幕无法加入对话，请重新核对后重试。');
const object=value=>value&&typeof value==='object'&&!Array.isArray(value);
function keys(value,allowed){if(!object(value)||Object.keys(value).some(key=>!allowed.includes(key)))throw invalid();}
function text(value,max){if(typeof value!=='string'||!value.trim()||value.length>max)throw invalid();return value;}
function fields(value){
  if(value.version!==1||value.confirmed!==true||!Number.isSafeInteger(value.revision)||value.revision<1||value.revision>1000000)throw invalid();
  if(typeof value.caption_id!=='string'||!/^[A-Za-z0-9_.:-]{1,160}$/.test(value.caption_id))throw invalid();
  keys(value.source,['kind','label','id']);
  if(!kinds.includes(value.source.kind))throw invalid();
  const source={kind:value.source.kind,label:text(value.source.label,160)};
  if(value.source.id!==undefined){if(typeof value.source.id!=='string'||!/^[a-z0-9][a-z0-9_-]{0,63}$/.test(value.source.id))throw invalid();source.id=value.source.id;}
  if(!Array.isArray(value.word_evidence)||value.word_evidence.length>128)throw invalid();
  const word_evidence=value.word_evidence.map(word=>text(word,128));
  let review_hint=null;
  if(value.review_hint!==null){keys(value.review_hint,['reason']);if(value.review_hint.reason!=='single_word_expansion'||word_evidence.length!==1)throw invalid();review_hint={reason:'single_word_expansion'};}
  return {version:1,caption_id:value.caption_id,revision:value.revision,text:text(value.text,8000),
    original_text:text(value.original_text,8000),confirmed:true,source,word_evidence,review_hint};
}
function fromCaption(caption){
  if(!caption||caption.status!=='committed'||caption.confirmed!==true)throw new Error('请先确认本句文字，再加入对话。');
  return fields({version:1,caption_id:caption.id,revision:caption.revision,text:caption.text,original_text:caption.original_text,
    confirmed:true,source:caption.source,word_evidence:caption.word_evidence||[],review_hint:caption.review_hint||null});
}
function bounded(payload){if(new TextEncoder().encode(JSON.stringify(payload)).byteLength>65536)throw new Error('本句信息过长，请分句后再加入对话。');}
async function fingerprint(value){
  if(!globalThis.crypto?.subtle)throw new Error('浏览器暂不支持文字交接，请复制文字后继续对话。');
  const bytes=new TextEncoder().encode(JSON.stringify(value));
  const result=await globalThis.crypto.subtle.digest('SHA-256',bytes);
  return 'rt_'+[...new Uint8Array(result)].map(x=>x.toString(16).padStart(2,'0')).join('');
}
async function wire(value){bounded({...value,id:'rt_'+'0'.repeat(64)});return {...value,id:await fingerprint(value)};}
export async function createRealtimeTransfer(caption){return wire(fromCaption(caption));}
export async function validateRealtimeTransfer(payload){
  keys(payload,['version','id','caption_id','revision','text','original_text','confirmed','source','word_evidence','review_hint']);
  if(typeof payload.id!=='string'||!/^rt_[0-9a-f]{64}$/.test(payload.id))throw invalid();
  const value=fields(payload);bounded({...value,id:payload.id});if(await fingerprint(value)!==payload.id)throw invalid();
  Object.freeze(value.source);Object.freeze(value.word_evidence);if(value.review_hint)Object.freeze(value.review_hint);
  const result=Object.freeze({...value,id:payload.id});verified.add(result);return result;
}
export function requireValidatedRealtimeTransfer(value){if(!verified.has(value))throw invalid();return value;}
const listener=callback=>{window.addEventListener('message',callback);return()=>window.removeEventListener('message',callback);};
const nonceValid=value=>typeof value==='string'&&/^[A-Za-z0-9_-]{8,80}$/.test(value);
function originOf(value){const url=new URL(value);if(!['http:','https:'].includes(url.protocol)||url.username||url.password)throw invalid();return url.origin;}

export class RealtimeHandoffSender{
  constructor({origin,openWindow=url=>window.open(url,'_blank'),uid=()=>crypto.randomUUID(),now=()=>Date.now(),
    setTimer=(fn,ms)=>setTimeout(fn,ms),clearTimer=id=>clearTimeout(id),listen=listener,onState=()=>{}}){
    this.origin=originOf(origin);Object.assign(this,{openWindow,uid,now,setTimer,clearTimer,onState});
    this.pending=null;this.target=null;this.disposed=false;this.unlisten=listen(event=>this.receive(event));
  }
  get busy(){return this.pending!==null;}
  send(caption){
    let value,target,nonce;
    try{
      if(this.disposed)throw new Error('交接已取消，请重新打开实时页。');
      if(this.pending)throw new Error('上一句正在加入对话，请稍候。');
      value=fromCaption(caption);
      nonce=this.uid();if(!nonceValid(nonce))throw invalid();
      target=this.target&&!this.target.closed?this.target:this.openWindow('/communicate/assist.html#realtime-dialogue');
      if(!target)throw new Error('无法打开对话窗口，请允许弹出窗口后重试。');
    }catch(error){this.onState({status:'error',message:error.message});return Promise.reject(error);}
    this.target=target;try{target.focus?.();}catch{}
    const job={target,nonce,expiresAt:this.now()+duration,payload:null,sent:false,timer:null,timeout:null};
    const result=new Promise((resolve,reject)=>Object.assign(job,{resolve,reject}));this.pending=job;
    this.onState({status:'waiting',message:'正在将已核对文字加入对话…'});
    job.timeout=this.setTimer(()=>this.finish(job,new Error('对话页响应超时，请回到实时页重试。')),duration);
    wire(value).then(payload=>{if(this.pending===job){job.payload=payload;this.poll(job);}}).catch(error=>this.finish(job,error));
    return result;
  }
  post(job,kind,extra={}){job.target.postMessage({protocol:TRANSFER_PROTOCOL,kind,nonce:job.nonce,expiresAt:job.expiresAt,...extra},this.origin);}
  poll(job){
    if(this.pending!==job)return;
    if(this.now()>=job.expiresAt){this.finish(job,new Error('对话页响应超时，请回到实时页重试。'));return;}
    if(job.target.closed){this.finish(job,new Error('对话窗口已关闭，请重新打开后重试。'),true);return;}
    try{this.post(job,'hello');}catch{this.finish(job,new Error('对话页无法接收文字，请重试。'));return;}
    job.timer=this.setTimer(()=>this.poll(job),250);
  }
  receive(event){
    const job=this.pending,data=event.data;
    if(!job||event.origin!==this.origin||event.source!==job.target||!object(data)||data.protocol!==TRANSFER_PROTOCOL||
      data.nonce!==job.nonce||data.expiresAt!==job.expiresAt)return;
    if(this.now()>=job.expiresAt){this.finish(job,new Error('对话页响应超时，请回到实时页重试。'));return;}
    if(data.kind==='ready'&&job.payload){try{this.post(job,'transfer',{payload:job.payload});job.sent=true;}catch{this.finish(job,new Error('对话页无法接收文字，请重试。'));}}
    else if(data.kind==='accepted'&&job.sent&&data.id===job.payload.id)this.finish(job,null,false,data.added);
    else if(data.kind==='rejected')this.finish(job,new Error('这条字幕未能加入对话，请重新核对后重试。'));
  }
  finish(job,error,forget=false,added=true){
    if(this.pending!==job)return;this.pending=null;this.clearTimer(job.timer);this.clearTimer(job.timeout);
    if(forget)this.target=null;
    if(error){try{this.post(job,'cancel');}catch{}this.onState({status:'error',message:error.message});job.reject(error);}
    else{this.onState({status:'sent',message:added?'已加入对话，请在对话页写下回复。':'这句已在对话中，可以继续回复。'});job.resolve({added:!!added});}
  }
  dispose(){this.disposed=true;this.unlisten?.();if(this.pending)this.finish(this.pending,new Error('交接已取消。'));}
}

export class RealtimeHandoffReceiver{
  constructor({origin,opener,now=()=>Date.now(),listen=listener,onTransfer,onError=()=>{}}){
    this.origin=originOf(origin);Object.assign(this,{opener,now,onTransfer,onError});this.disposed=false;this.nonce=null;
    this.unlisten=listen(event=>this.receive(event));
  }
  async receive(event){
    const data=event.data;
    if(this.disposed||!this.opener||event.origin!==this.origin||event.source!==this.opener||!object(data)||
      data.protocol!==TRANSFER_PROTOCOL||!nonceValid(data.nonce))return;
    const reply=(kind,extra={})=>{try{this.opener.postMessage({protocol:TRANSFER_PROTOCOL,kind,nonce:data.nonce,expiresAt:data.expiresAt,...extra},this.origin);}catch{}};
    if(data.kind==='cancel'){if(data.nonce===this.nonce)this.nonce=null;return;}
    if(!Number.isSafeInteger(data.expiresAt)||data.expiresAt<=this.now()||data.expiresAt-this.now()>duration)return;
    if(data.kind==='hello'){this.nonce=data.nonce;this.expiresAt=data.expiresAt;reply('ready');return;}
    if(data.kind!=='transfer'||data.nonce!==this.nonce||data.expiresAt!==this.expiresAt)return;
    try{
      const payload=await validateRealtimeTransfer(data.payload);
      if(this.disposed||data.nonce!==this.nonce||this.now()>=data.expiresAt)return;
      const added=this.onTransfer(payload);if(typeof added!=='boolean')throw invalid();reply('accepted',{id:payload.id,added});
    }catch{if(!this.disposed&&data.nonce===this.nonce){reply('rejected');this.onError('这条字幕未能加入对话，请回到实时页重新核对。');}}
  }
  dispose(){this.disposed=true;this.nonce=null;this.unlisten?.();}
}
