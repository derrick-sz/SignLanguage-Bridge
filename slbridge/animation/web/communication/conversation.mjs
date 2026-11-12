import {requireValidatedRealtimeTransfer} from './conversation-transfer.mjs';

export class Conversation {
  constructor({now=()=>new Date().toISOString(),limit=200}={}){this.now=now;this.limit=limit;this.items=[];this.cleared=null;}
  get messages(){return structuredClone(this.items);}
  append(message){
    if(typeof message.text!=='string'||!message.text.trim()||message.text.length>8000)throw new Error('请先核对要加入对话的文字。');
    if(this.items.some(x=>x.id===message.id))return false;
    this.items.push(structuredClone({...message,created_at:this.now()}));
    if(this.items.length>this.limit)this.items.splice(0,this.items.length-this.limit);
    this.cleared=null;return true;
  }
  confirmRecognition(result,text){
    if(!result?.recognized||typeof result.request_id!=='string')throw new Error('请先识别并核对文字。');
    return this.append({id:'recognition:'+result.request_id,direction:'received',text,original_text:result.text,
      sign_language:result.sign_language,output_kind:result.output_kind,review_status:result.review_status,
      provenance:result.provenance,text_confirmed_by_user:true});
  }
  confirmRealtime(payload){
    const value=requireValidatedRealtimeTransfer(payload);
    return this.append({id:value.id,direction:'received',text:value.text,original_text:value.original_text,
      sign_language:'csl',output_kind:'text',review_status:'unverified',text_confirmed_by_user:true,
      provenance:{kind:'realtime',caption_id:value.caption_id,revision:value.revision,source:value.source,
        word_evidence:value.word_evidence,review_hint:value.review_hint}});
  }
  addReply(text,id){return this.append({id:'reply:'+id,direction:'sent',text,original_text:text,sign_language:'csl',review_status:'unverified'});}
  clear(){this.cleared=this.items;this.items=[];}
  undoClear(){if(!this.cleared||this.items.length)return false;this.items=this.cleared;this.cleared=null;return true;}
  export(){return {schema_version:1,application:'SignLanguage Bridge',exported_at:this.now(),messages:this.messages};}
}

export function readableTranscript(data){
  const lines=['手语桥 · 本次对话',''];
  for(const message of data.messages){
    const source=message.provenance?.source?.kind;
    const who=message.direction==='sent'?'我':(source==='diagnostic_replay'?'问题回放的手语（':source==='training_replay'?'训练示范的手语（':'对方的手语（')+(message.sign_language==='libras'?'Libras':'中国手语')+'）';
    lines.push(who+'  '+new Date(message.created_at).toLocaleTimeString('zh-CN',{hour:'2-digit',minute:'2-digit'}));
    lines.push(...message.text.split(/\r?\n/).map(line=>'  '+line));
    if(message.provenance?.kind==='realtime')lines.push('  来源：'+message.provenance.source.label,'  文字已由使用者核对。');
    if(message.direction==='received'&&message.original_text!==message.text)lines.push('  识别原文：'+message.original_text,'  上方文字经过使用者核对修改。');
    lines.push('');
  }
  return lines.join('\n');
}
