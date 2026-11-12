import {MotionStage} from '/animation/static/stage.js';
import {FastCompositionController} from '/generate/fast-composition-controller.mjs';
import {GenerationController} from '/generate/generate-controller.mjs';
import {Conversation,readableTranscript} from './conversation.mjs';
import {RecognitionController} from './recognition-controller.mjs';
import {CameraCapture} from './capture.mjs';
import {RealtimeHandoffReceiver} from './conversation-transfer.mjs';

const $=id=>document.getElementById(id);
const conversation=new Conversation();
let realtimeReceiver=null;
let recognitionCapabilities={ready:false,languages:[]},recognitionBusy=false,lastRecognized=null;
let sourceFile=null,sourceUrl=null,sourceEpoch=0,sourceTooLong=false,isLive=false,isRecording=false;
let stageReady=false,plannerReady=false,plannerState='starting',generationEnabled=false,replyBusy=false,replyLoaded=false,replyMode='fast';
let pendingReply=null,shownReplyText='',toastTimer=null,recordTimer=null,readinessTimer=null,readinessEpoch=0,pollCount=0,disposed=false;
const time=value=>Math.floor((value||0)/60)+':'+String(Math.floor((value||0)%60)).padStart(2,'0');
const languageName=value=>value==='libras'?'巴西手语 Libras':'中国手语';
const activeReply=()=>replyMode==='fast'?fast:generator;

function notice(message){$('notice').textContent=message;$('notice').hidden=false;clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('notice').hidden=true,4000);}
function setStatus(id,message,error=false){$(id).textContent=message;$(id).dataset.tone=error?'error':'';}
function mobileView(value){
  $('communication-grid').dataset.mobile=value;
  $('tab-understand').setAttribute('aria-pressed',String(value==='understand'));
  $('tab-express').setAttribute('aria-pressed',String(value==='express'));
}
function updateControls(){
  const selected=recognitionCapabilities.languages.find(x=>x.id===$('sign-language').value);
  $('recognize').disabled=!selected?.ready||!sourceFile||sourceTooLong||recognitionBusy||isRecording;
  $('recognition-cancel').hidden=!recognitionBusy;
  $('record-start').hidden=!isLive||isRecording;
  $('record-finish').hidden=!isRecording;
  $('camera-close').hidden=!isLive;
  $('camera-switch').hidden=isLive;
  $('upload-video').disabled=isRecording;
  $('sign-language').disabled=isRecording;
  const length=[...$('reply-text').value].length;
  $('reply-count').textContent=length+' / 200';
  const canGenerate=generationEnabled&&!['checking','starting'].includes(plannerState);
  $('send-reply').disabled=!stageReady||!(plannerReady||canGenerate)||replyBusy||!$('reply-text').value.trim()||length>200;
  $('send-reply').textContent=!plannerReady&&canGenerate?'生成手语':'转成手语';
  $('reply-cancel').hidden=!replyBusy;
  $('try-generation').disabled=!generationEnabled||replyBusy||!stageReady;
  $('motion-play').disabled=!replyLoaded;
  $('motion-stop').disabled=!replyLoaded&&!replyBusy;
  document.querySelectorAll('[data-phrase]').forEach(button=>button.disabled=!stageReady||!plannerReady||replyBusy);
  const ready=recognitionCapabilities.ready&&stageReady&&plannerReady;
  $('connection').dataset.ready=String(ready);
  $('connection').lastChild.textContent=ready?'已准备就绪':recognitionCapabilities.ready?'识别已连接':stageReady&&plannerReady?'文字表达可用':'正在准备连接';
}
function renderConversation(){
  const messages=conversation.messages;
  $('conversation-empty').hidden=messages.length>0;
  $('clear-conversation').disabled=!messages.length;$('export-conversation').disabled=!messages.length;
  $('conversation-list').replaceChildren(...messages.map(message=>{
    const item=document.createElement('li');item.className='message'+(message.direction==='sent'?' is-mine':'');
    const meta=document.createElement('div');meta.className='message-meta';
    const replay=message.provenance?.source?.kind;
    const label=document.createElement('span');label.textContent=message.direction==='sent'?'我的文字':
      (replay==='training_replay'?'训练示范手语':replay==='diagnostic_replay'?'问题回放手语':'对方的手语')+' · '+languageName(message.sign_language);
    const clock=document.createElement('time');clock.dateTime=message.created_at;clock.textContent=new Date(message.created_at).toLocaleTimeString('zh-CN',{hour:'2-digit',minute:'2-digit'});
    meta.append(label,clock);
    const body=document.createElement('p');body.className='message-body';body.textContent=message.text;
    const actions=document.createElement('div');actions.className='message-actions';
    const copy=document.createElement('button');copy.className='quiet';copy.textContent='复制文字';copy.addEventListener('click',()=>copyText(message.text));
    const express=document.createElement('button');express.className='quiet';express.textContent='用手语展示';
    express.addEventListener('click',()=>{if(fillReply(message.text))beginReply('fast',true);});
    actions.append(copy,express);item.append(meta,body,actions);
    if(message.provenance?.kind==='realtime'){
      const source=document.createElement('p');source.className='small-note';source.textContent='来源：'+message.provenance.source.label+' · 文字已核对';item.append(source);
    }
    if(message.direction==='received'&&message.text!==message.original_text){
      const details=document.createElement('details'),summary=document.createElement('summary'),raw=document.createElement('p');
      summary.textContent='查看识别原文';raw.textContent=message.original_text;details.append(summary,raw);item.append(details);
    }
    return item;
  }));
  if(messages.length)$('conversation-list').scrollTop=$('conversation-list').scrollHeight;
}
async function copyText(text){try{await navigator.clipboard.writeText(text);notice('文字已复制。');}catch{notice('复制暂不可用，可以选择文字手动复制。');}}
function resetRecognition(){lastRecognized=null;$('recognition-result').hidden=true;$('recognized-text').value='';}
const recognition=new RecognitionController({onState:state=>{
  recognitionBusy=state.status==='working';
  setStatus('recognition-status',state.message||'',state.status==='error');
  if(['working','idle','empty','error'].includes(state.status))resetRecognition();
  if(state.status==='ready'){
    lastRecognized=state.result;$('recognition-result').hidden=false;
    $('recognized-text').value=state.result.text;$('original-text').textContent=state.result.text;
    $('result-language').textContent=languageName(state.result.sign_language);
    $('recognition-kind').textContent=state.result.output_kind==='gloss_sequence'?'这次识别得到手语词。请核对、整理成你要表达的文字。':state.result.review_status==='draft'?'当前释义为待核对草案，你可以修改后加入对话。':'请核对文字，你可以直接修改。';
    const portuguese=state.result.translations?.['pt-BR'];$('portuguese-caption').hidden=!portuguese;
    $('portuguese-caption').textContent=portuguese?'葡语原词：'+portuguese:'';
    $('confirm-recognition').disabled=false;
    if(document.querySelector('.understand-panel').contains(document.activeElement))$('recognition-result').scrollIntoView({block:'nearest'});
  }
  updateControls();
}});
function clearSourceUrl(){if(sourceUrl)URL.revokeObjectURL(sourceUrl);sourceUrl=null;}
function showFile(file,autoRecognize=false){
  capture.close();recognition.invalidate('视频已选好，点击识别这段手语。');resetRecognition();
  sourceEpoch++;const epoch=sourceEpoch;sourceFile=file;sourceTooLong=false;isLive=false;clearSourceUrl();
  sourceUrl=URL.createObjectURL(file);const video=$('source-video');video.srcObject=null;video.src=sourceUrl;
  video.controls=true;video.muted=false;video.classList.remove('is-mirrored');video.hidden=false;$('video-empty').hidden=true;
  $('source-info').textContent=file.name+' · 请使用15秒以内的手语片段。';
  video.onloadedmetadata=()=>{
    if(epoch!==sourceEpoch)return;
    sourceTooLong=Number.isFinite(video.duration)&&video.duration>15.05;
    if(sourceTooLong)setStatus('recognition-status','这段视频超过15秒，请换一段更短的视频。',true);
    updateControls();
  };
  updateControls();if(autoRecognize)recognizeCurrent();
}
function recognizeCurrent(){
  const language=$('sign-language').value;
  if(!sourceFile||sourceTooLong)return;
  if(!recognitionCapabilities.languages.some(x=>x.id===language&&x.ready)){setStatus('recognition-status','这类手语的识别服务尚未就绪，请在设置中重新连接。',true);return;}
  $('source-video').pause();recognition.recognize(sourceFile,language);
}
const capture=new CameraCapture({onState:state=>{
  if(state.status!=='recording'){clearInterval(recordTimer);recordTimer=null;isRecording=false;$('record-indicator').hidden=true;}
  if(state.status==='opening')setStatus('recognition-status','正在开启摄像头，请允许浏览器访问摄像头。');
  if(state.status==='preview'){
    sourceEpoch++;sourceFile=null;sourceTooLong=false;clearSourceUrl();isLive=true;resetRecognition();
    const video=$('source-video');video.removeAttribute('src');video.srcObject=state.stream;video.controls=false;video.muted=true;video.hidden=false;
    video.classList.toggle('is-mirrored',$('mirror-camera').checked);video.play().catch(()=>{});$('video-empty').hidden=true;
    setStatus('recognition-status','摄像头已开启。准备好后开始录制。');
    $('source-info').textContent='单次录制最长12秒，结束后识别。请让双手和上半身清楚可见。';
  }
  if(state.status==='recording'){
    isRecording=true;$('record-indicator').hidden=false;const started=performance.now();$('record-time').textContent='0:00';
    recordTimer=setInterval(()=>$('record-time').textContent=time((performance.now()-started)/1000),200);
    setStatus('recognition-status','正在录制。动作完成后，点击结束并识别。');
  }
  if(state.status==='closed'){
    if(isLive){$('source-video').srcObject=null;$('source-video').hidden=true;$('video-empty').hidden=false;}
    isLive=false;
  }
  if(state.status==='recorded')showFile(state.file,true);
  if(state.status==='error')setStatus('recognition-status',state.message,true);
  updateControls();
}});
async function openCamera(){
  if(!navigator.mediaDevices?.getUserMedia){setStatus('recognition-status','当前浏览器无法开启摄像头，请上传视频。',true);return;}
  recognition.invalidate('正在切换到摄像头。');resetRecognition();await capture.open();
}

function replyState(mode,state){
  if(replyMode!==mode)return;
  replyBusy=['submitting','queued','running','loading'].includes(state.status);replyLoaded=state.status==='ready';
  const messages={idle:'写下回复，转成手语说给对方。',submitting:'正在准备手语表达…',queued:'正在等待生成，你可以取消。',running:'正在生成完整动作，需要一些时间…',loading:'正在准备人物动作…',ready:mode==='fast'?'手语已准备，请与对方一起核对。':'完整动作已准备，可以播放。',cancelled:'已取消本次表达。',blocked:'这句话暂时缺少完整的手语动作。请调整表达，或尝试完整生成。',needs_review:'这句话有内容未能完整表达。请调整或分句后重试。',error:state.message||'这次表达未能完成，请稍后再试。'};
  setStatus('reply-status',messages[state.status]||'正在准备手语表达…',['error','blocked','needs_review'].includes(state.status));
  $('reply-recovery').hidden=!['blocked','needs_review'].includes(state.status);
  $('avatar-caption').textContent=state.subtitle||'你的文字会显示在这里，方便一起核对。';
  $('avatar-status').textContent=replyBusy?'正在准备表达':replyLoaded?'可以播放':stageReady?'人物已准备':'正在准备人物';
  $('motion-quality').hidden=true;
  if(replyLoaded){
    shownReplyText=state.subtitle;
    const clearance=stage.tracks?.diagnostics?.body_clearance;
    $('motion-quality').hidden=!(clearance?.remaining_core_hits||clearance?.max_offset_step>.015||state.job?.quality_warning);
    if(pendingReply){
      if(!pendingReply.replay)conversation.addReply(state.subtitle,pendingReply.id);
      pendingReply=null;renderConversation();$('avatar-stage').scrollIntoView({block:'nearest'});
    }
  }
  if(['idle','submitting','cancelled','error','blocked','needs_review'].includes(state.status))shownReplyText='';
  updateControls();
}
const stage=new MotionStage($('avatar-stage'),state=>{
  $('avatar-time').textContent=state.duration>0&&state.duration<1?state.time.toFixed(1)+' / '+state.duration.toFixed(1)+' 秒':time(state.time)+' / '+time(state.duration);
  if(replyLoaded){$('avatar-status').textContent=state.playing?'正在表达':state.time>=state.duration?'表达结束':'已暂停';$('motion-play').textContent=state.playing?'暂停':state.time>=state.duration?'重播':'继续';}
});
const fast=new FastCompositionController({stage,onState:state=>replyState('fast',state)});
const generator=new GenerationController({stage,onState:state=>replyState('generate',state)});
function invalidateReply(message='文字已修改，点击转成手语。'){
  pendingReply=null;fast.invalidate(message);generator.invalidate(message);
  if(message)setStatus('reply-status',message);updateControls();
}
function fillReply(text){
  if([...text].length>200){notice('请先将这段文字缩短到200字以内。');return false;}
  invalidateReply();$('reply-text').value=text;mobileView('express');updateControls();return true;
}
function beginReply(mode=plannerReady?'fast':'generate',replay=false){
  const text=$('reply-text').value;
  if(!text.trim()||[...text].length>200){notice('请填写200字以内的中文。');return;}
  if(!stageReady||(mode==='fast'&&!plannerReady)){notice('手语表达服务还在准备，请稍后重试。');return;}
  invalidateReply('');replyMode=mode;pendingReply={id:crypto.randomUUID(),replay};
  if(mode==='fast')fast.generate(text);else generator.generate(text,[...text].length>30?10:6);
}
function updateCapabilityList(){
  const labels=recognitionCapabilities.languages.map(item=>item.label+'：'+(item.ready?item.scope:'尚未就绪'));
  labels.push('中文转中国手语：'+(stageReady&&plannerReady?'已连接':'正在准备'));
  $('capability-list').replaceChildren(...labels.map(label=>{const li=document.createElement('li');li.textContent=label;return li;}));
}
async function connectServices(reset=false){
  if(disposed)return;clearTimeout(readinessTimer);if(reset)pollCount=0;const epoch=++readinessEpoch;pollCount++;
  const request=async url=>{const response=await fetch(url,{cache:'no-store',signal:AbortSignal.timeout(5000)});if(!response.ok)throw new Error('unavailable');return response.json();};
  const results=await Promise.allSettled([request('/v1/communication/recognition'),request('/v1/animation/composition'),request('/v1/animation/generation')]);
  if(disposed||epoch!==readinessEpoch)return;
  recognitionCapabilities=results[0].status==='fulfilled'?results[0].value:{ready:false,languages:[]};
  const composition=results[1].status==='fulfilled'?results[1].value:null;
  plannerState=composition?.planner?.state||'unavailable';
  plannerReady=composition?.enabled===true&&plannerState==='ready';generationEnabled=results[2].status==='fulfilled'&&results[2].value.enabled===true;
  if(!recognitionBusy&&!lastRecognized&&!isLive&&!sourceFile)setStatus('recognition-status',recognitionCapabilities.ready?'识别服务已连接。可以录制一段手语，或上传视频。':'识别暂未连接，文字回复仍可使用。可在设置中重新连接。',!recognitionCapabilities.ready);
  if(!replyBusy&&!replyLoaded)setStatus('reply-status',plannerReady&&stageReady?'写下回复，转成手语说给对方。':generationEnabled&&!['checking','starting'].includes(plannerState)?'当前需要生成完整动作，可能需要几分钟，可以取消。':'正在准备文字表达，请稍候…');
  updateCapabilityList();updateControls();
  if(pollCount<60&&(composition?.planner?.state==='starting'||!recognitionCapabilities.ready&&pollCount<8))readinessTimer=setTimeout(()=>connectServices(),1500);
}

$('upload-video').addEventListener('click',()=>$('video-file').click());
$('video-file').addEventListener('change',()=>{const file=$('video-file').files[0];if(file)showFile(file);});
$('camera-open').addEventListener('click',openCamera);$('camera-switch').addEventListener('click',openCamera);
$('camera-close').addEventListener('click',()=>{capture.close();recognition.invalidate('摄像头已关闭，可以上传视频或重新开启。');});
$('record-start').addEventListener('click',()=>{if(!capture.start())setStatus('recognition-status','当前浏览器不支持录制，请上传视频。',true);});
$('record-finish').addEventListener('click',()=>capture.finish());$('recognize').addEventListener('click',recognizeCurrent);
$('recognition-cancel').addEventListener('click',()=>recognition.cancel());
$('sign-language').addEventListener('change',()=>{recognition.invalidate('手语已切换，请重新识别当前视频。');updateControls();});
$('mirror-camera').addEventListener('change',()=>$('source-video').classList.toggle('is-mirrored',isLive&&$('mirror-camera').checked));
$('confirm-recognition').addEventListener('click',()=>{try{conversation.confirmRecognition(lastRecognized,$('recognized-text').value);$('confirm-recognition').disabled=true;renderConversation();notice('已加入本次对话。');}catch(error){notice(error.message);}});
$('use-recognition').addEventListener('click',()=>{if(fillReply($('recognized-text').value)){$('reply-text').focus();notice('已带入文字，可以编辑后转成手语。');}});
$('reply-form').addEventListener('submit',event=>{event.preventDefault();if(!$('send-reply').disabled)beginReply();});
$('reply-text').addEventListener('input',()=>{invalidateReply();updateControls();});
$('reply-cancel').addEventListener('click',()=>{pendingReply=null;activeReply().cancel();});
$('try-generation').addEventListener('click',()=>beginReply('generate'));
document.querySelectorAll('[data-phrase]').forEach(button=>button.addEventListener('click',()=>{fillReply(button.dataset.phrase);beginReply();}));
$('motion-play').addEventListener('click',()=>{if(stage.clock.playing)activeReply().pause();else activeReply().play();});
$('motion-stop').addEventListener('click',()=>{if(replyBusy){pendingReply=null;activeReply().cancel();}else activeReply().stop();});
$('motion-slow').addEventListener('click',()=>{const enabled=$('motion-slow').getAttribute('aria-pressed')!=='true';$('motion-slow').setAttribute('aria-pressed',String(enabled));stage.setSpeed(enabled?0.5:1);});
$('motion-closeup').addEventListener('click',()=>{const enabled=$('motion-closeup').getAttribute('aria-pressed')!=='true';$('motion-closeup').setAttribute('aria-pressed',String(enabled));stage.setView(enabled?'hands':'front');});
$('settings-open').addEventListener('click',()=>{$('settings').showModal();updateCapabilityList();});
$('retry-connections').addEventListener('click',()=>connectServices(true));
$('tab-understand').addEventListener('click',()=>mobileView('understand'));$('tab-express').addEventListener('click',()=>mobileView('express'));
$('clear-conversation').addEventListener('click',()=>{recognition.invalidate('记录已清空。');invalidateReply('记录已清空，可以开始新的表达。');conversation.clear();renderConversation();$('undo-clear').hidden=false;});
$('undo-clear').addEventListener('click',()=>{
  if(conversation.undoClear()){
    renderConversation();notice('对话记录已恢复。');
    setStatus('recognition-status',sourceFile?'对话已恢复，可以继续识别当前视频。':'对话已恢复，可以上传下一段视频。');
    setStatus('reply-status','对话已恢复，可以继续表达。');
  }
  $('undo-clear').hidden=true;
});
$('export-conversation').addEventListener('click',()=>{
  const blob=new Blob(['\ufeff'+readableTranscript(conversation.export())],{type:'text/plain;charset=utf-8'}),url=URL.createObjectURL(blob);
  const link=document.createElement('a');link.href=url;link.download='手语桥-本次对话.txt';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
window.addEventListener('pagehide',()=>{disposed=true;realtimeReceiver?.dispose();clearTimeout(readinessTimer);clearTimeout(toastTimer);clearInterval(recordTimer);capture.close();clearSourceUrl();recognition.invalidate();fast.invalidate();generator.invalidate();stage.dispose();});
if(window.opener&&window.location.hash==='#realtime-dialogue'){
  realtimeReceiver=new RealtimeHandoffReceiver({origin:window.location.origin,opener:window.opener,onTransfer:payload=>{
    const added=conversation.confirmRealtime(payload);renderConversation();mobileView('express');$('reply-text').focus();
    notice(added?'已加入已核对的实时字幕，请写下回复。':'这句已在本次对话中。');return added;
  },onError:notice});
}
stage.ready.then(()=>{stageReady=true;stage.renderer.domElement.setAttribute('aria-label','中国手语数字人');$('avatar-status').textContent='人物已准备';if(plannerReady)setStatus('reply-status','写下回复，转成手语说给对方。');updateCapabilityList();updateControls();}).catch(()=>{$('avatar-status').textContent='人物暂未连接';setStatus('reply-status','人物未能加载，请刷新页面重试。',true);});
renderConversation();updateControls();connectServices();
