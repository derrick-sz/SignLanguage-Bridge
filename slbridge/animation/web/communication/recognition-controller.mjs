export class RecognitionController {
  constructor({fetcher=globalThis.fetch,onState=()=>{},uid=()=>crypto.randomUUID(),maxBytes=128*1024*1024}={}){
    this.fetcher=fetcher.bind(globalThis);this.onState=onState;this.uid=uid;this.maxBytes=maxBytes;this.generation=0;this.abort=null;
  }
  invalidate(message='输入已改变，请重新识别。'){
    this.generation++;this.abort?.abort();this.abort=null;this.onState({status:'idle',message});
  }
  cancel(){this.invalidate('已取消本次识别。');}
  async recognize(file,language){
    this.invalidate();
    if(!file?.size||file.size>this.maxBytes||!['csl','libras'].includes(language)){
      this.onState({status:'error',message:'请选择128MB以内的手语短视频。'});return;
    }
    const generation=this.generation,abort=new AbortController(),requestId=this.uid();this.abort=abort;
    const current=()=>generation===this.generation&&!abort.signal.aborted;
    this.onState({status:'working',message:'正在识别这段手语，请稍候…'});
    try{
      const params=new URLSearchParams({sign_language:language,request_id:requestId,filename:file.name});
      const response=await this.fetcher('/v1/communication/recognitions?'+params,{method:'POST',body:file,headers:{'Content-Type':file.type||'application/octet-stream'},signal:abort.signal});
      if(!current())return;
      let data;try{data=await response.json();}catch{throw new Error('无法读取识别结果，请稍后重试。');}if(!current())return;
      if(!response.ok)throw new Error(typeof data.detail==='string'?data.detail:'这段视频未能识别，请重新录制或更换视频。');
      if(data.request_id!==requestId||data.sign_language!==language||typeof data.recognized!=='boolean'||typeof data.text!=='string')throw new Error('识别结果与本次输入不一致，请重试。');
      this.onState(data.recognized?{status:'ready',message:'识别完成，请核对文字。',result:data}:{status:'empty',message:'未识别到清楚的手语。请让双手保持可见，再试一次。'});
    }catch(error){if(current())this.onState({status:'error',message:error.message||'识别暂不可用，请稍后重试。'});}
    finally{if(current())this.abort=null;}
  }
}
