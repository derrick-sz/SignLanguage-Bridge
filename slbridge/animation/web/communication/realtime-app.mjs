import {RealtimeClient} from './realtime-client.mjs';
import {LiveFrameCapture} from './realtime-capture.mjs';
import {ScreenSource} from './screen-capture.mjs';
import {fullRegion,videoRect,selectRegion} from './selection.mjs';
import {demoItems} from './realtime-demo.mjs';
import {RealtimeHandoffSender} from './conversation-transfer.mjs';

const $=id=>document.getElementById(id),video=$('video'),preview=$('preview');
let stream=null,url=null,sourceType=null,sourceVersion=0,intent=0,region=null,selecting=false,drag=null;
let capabilities={enabled:false,ready:false},starting=false,disposed=false,editId=null,timer=null,rendered='';
let transportMetrics={},captureMetrics={};
let catalogItems=[],selectedDemo=null,sourceLabel='尚未选择画面';
let handoff=null;
const handoffHint='完成本句并暂停，确认文字后可加入对话并回复。';
const replaySource=()=>['file','training_replay','diagnostic_replay'].includes(sourceType);
const status=(text,error=false)=>{$('status').textContent=text;$('status').dataset.error=String(error);};
const adviceText=advice=>advice.text||({pending:'正在辅助核对，原译文已交付。',running:'正在辅助核对，原译文已交付。',
  insufficient:'词证据不足，暂未给出建议。',timeout:'辅助核对超时，原译文已保留。',
  busy:'辅助核对繁忙，原译文已保留。',cancelled:'辅助核对已取消。'}[advice.status]||'本次没有可用建议，原译文已保留。');
const reviewText='只有一个实时词，中文扩写较多，请结合动作重点核对。';
const evidenceText=item=>item.word_evidence?.length?'本句实时词：'+item.word_evidence.join(' · '):'本句未收到独立的词证据。';
const client=new RealtimeClient({onUpdate:render,onMetrics:value=>{transportMetrics=value;renderMetrics();}});
const capture=new LiveFrameCapture({video,onFrame:frame=>client.sendFrame(frame),onError:message=>stop(message,true),
  onMetrics:value=>{captureMetrics=value;renderMetrics();}});
const screen=new ScreenSource({onState:state=>{if(state.status==='error')status(state.message,true);if(state.status==='ended')resetSource('共享画面已结束。');}});

function controls(){
  const running=['running','finishing'].includes(client.status);
  $('start').disabled=starting||running||handoff?.busy||!region||!video.videoWidth||!capabilities.ready||capabilities.busy;
  $('stop').disabled=!running&&!starting;$('boundary').disabled=$('finish').disabled=client.status!=='running';
  $('select').disabled=$('full').disabled=!video.videoWidth;
  $('camera').disabled=!navigator.mediaDevices?.getUserMedia;$('screen').disabled=!navigator.mediaDevices?.getDisplayMedia;
  $('assistance-choice').hidden=!capabilities.features?.llm_assistance;
  $('assistance').disabled=running||starting;
  $('connection').textContent=capabilities.ready?(capabilities.busy&&client.status!=='running'?'实时服务正在使用':'实时服务已就绪'):'实时服务未就绪';
  $('source-label').textContent='来源：'+sourceLabel;
  $('start').textContent=selectedDemo?'识别此示范':'开始实时识别';
}
function renderMetrics(){
  const ms=value=>Number.isFinite(value)?`${Math.round(value)} 毫秒`:'—';
  $('metric-ready').textContent=ms(transportMetrics.readyMs);
  $('metric-roundtrip').textContent=ms(transportMetrics.frameRoundTripMs);
  $('metric-queue').textContent=ms(transportMetrics.queueMs);
  $('metric-processing').textContent=ms(transportMetrics.processingMs);
  $('metric-encode').textContent=ms(captureMetrics.encodeMs);
  $('metric-skipped').textContent=String(captureMetrics.missedSlots??0);
  $('metric-captured').textContent=String(captureMetrics.capturedFrames??0);
  $('metric-sent').textContent=String(transportMetrics.framesSent??0);
  $('metric-processed').textContent=String(transportMetrics.framesProcessed??0);
}
function openReview(item){
  editId=item.id;$('edit-text').value=item.text;$('edit-evidence').textContent=evidenceText(item);
  $('edit-warning').hidden=!item.review_hint;$('edit-warning').textContent=item.review_hint?reviewText:'';
  $('edit-use-words').hidden=item.word_evidence?.length!==1;
  $('edit-dialog').showModal();
}
const handoffBusy=state=>starting||['connecting','running','finishing'].includes(state.status)||handoff?.busy;
async function joinConversation(id){
  if(handoffBusy(client.snapshot()))return;
  const item=client.snapshot().captions.find(entry=>entry.id===id);if(!item?.confirmed)return;
  try{
    handoff??=new RealtimeHandoffSender({origin:window.location.origin,onState:state=>{
      $('handoff-status').textContent=state.message;$('handoff-status').dataset.error=String(state.status==='error');render(client.snapshot());
    }});
    await handoff.send(item);
  }catch(error){$('handoff-status').textContent=error.message||'暂时无法加入对话，请重试。';$('handoff-status').dataset.error='true';render(client.snapshot());}
}
function render(state){
  if(['error','stopped'].includes(state.status)&&(capture?.active||capture?.draining))capture.stop();
  if(state.message)status(state.message,state.status==='error');
  const nodes=state.words.map(word=>{const span=document.createElement('span');span.className='word '+word.status;span.textContent=word.text;return span;});
  if(!nodes.length){const span=document.createElement('span');span.className='placeholder';span.textContent='动作中的词会先显示在这里';nodes.push(span);}
  $('words').replaceChildren(...nodes);
  const sentence=state.currentCaption;
  $('sentence').textContent=sentence?.text||'完整表达会在这里出现';
  $('sentence-state').textContent=sentence?.confirmed?(sentence.edited?'已人工修改并确认':'已人工确认'):sentence?.status==='committed'?(sentence.review_hint?'词证据较少 · 待重点核对':'已提交机器译文 · 请核对'):sentence?.status==='pending'?'正在补全这句中文…':sentence?.status==='rejected'?'本句翻译未完成，请核对实时词或分句重试。':sentence?.text?'中文草稿 · 仍可能修订':'等待有效动作与句子边界';
  $('sentence-evidence').hidden=!sentence?.word_evidence?.length;
  $('sentence-evidence').textContent=sentence?evidenceText(sentence):'';
  $('review-warning').hidden=!sentence?.review_hint||sentence.confirmed;
  if($('review-warning').textContent!==reviewText)$('review-warning').textContent=reviewText;
  const currentAdvice=state.advice?.revision===sentence?.revision&&state.advice?.utterance_id===sentence?.utterance_id?state.advice:null;
  $('advice').hidden=!currentAdvice;$('advice').textContent=currentAdvice?`${sentence.edited?'原译文辅助核对':'辅助核对'}：${adviceText(currentAdvice)}`:'';
  $('timing').textContent=state.firstWordTotalMs!==null?`首词 ${(state.firstWordTotalMs/1000).toFixed(2)} 秒（含准备）`:'';
  $('count').textContent=state.captions.length;$('history-empty').hidden=state.captions.length>0;
  const signature=JSON.stringify([state.captions,!!handoffBusy(state)]);
  if(signature!==rendered){
    rendered=signature;$('history').replaceChildren(...state.captions.map((item,index)=>{
      const li=document.createElement('li'),meta=document.createElement('small'),text=document.createElement('p'),edit=document.createElement('button'),actions=document.createElement('div');
      meta.textContent=`第 ${index+1} 句 · ${item.confirmed?(item.edited?'已人工修改并确认':'已人工确认'):item.review_hint?'待重点核对':'机器译文待核对'}`;text.textContent=item.text;edit.textContent='核对';
      edit.onclick=()=>openReview(item);actions.className='caption-actions';
      if(!item.confirmed){const confirm=document.createElement('button');confirm.textContent='确认';confirm.onclick=()=>client.confirm(item.id);actions.append(confirm);}
      const join=document.createElement('button');join.textContent='加入对话并回复';
      join.disabled=!item.confirmed||!!handoffBusy(state);join.title=!item.confirmed?'请先确认本句文字':'采集暂停后可加入对话';
      join.onclick=()=>joinConversation(item.id);
      actions.append(edit,join);li.append(actions,meta,text);
      if(item.source){const origin=document.createElement('p');origin.className='note';origin.textContent='来源：'+item.source.label;li.append(origin);}
      if(item.word_evidence?.length){const evidence=document.createElement('p');evidence.className='note';evidence.textContent=evidenceText(item);li.append(evidence);}
      if(item.review_hint&&!item.confirmed){const warning=document.createElement('p');warning.className='review-warning';warning.textContent=reviewText;li.append(warning);}
      if(item.edited){const details=document.createElement('details'),summary=document.createElement('summary'),raw=document.createElement('p');summary.textContent='识别原文';raw.textContent=item.original_text;details.append(summary,raw);
        if(item.review_hint){const hint=document.createElement('p');hint.textContent='原译文提示：'+reviewText;details.append(hint);}li.append(details);}
      if(item.advice?.revision===item.revision){const advice=document.createElement('p');advice.className='note';advice.textContent=(item.edited?'原译文辅助核对：':'辅助核对：')+adviceText(item.advice);li.append(advice);}return li;
    }));
  }
  controls();
}
function stop(message='已暂停，已提交的中文记录会保留。',error=false){
  ++intent;starting=false;capture.stop();client.stop();if(replaySource())video.pause();status(message,error);controls();
}
async function finishCurrent(){
  if(client.status!=='running')return;
  const action=intent,version=sourceVersion;
  if(!await capture.finish()||action!==intent||version!==sourceVersion||client.status!=='running')return;
  client.finish();if(replaySource())video.pause();controls();
  status('正在补全本句，完成后保留译文并暂停采集。');
}
function resetSource(message='请选择新的画面。'){
  stop(message);++sourceVersion;stream?.getTracks().forEach(track=>track.stop());stream=null;screen.close();
  transportMetrics={};captureMetrics={};renderMetrics();
  video.pause();video.srcObject=null;video.removeAttribute('src');video.load();if(url)URL.revokeObjectURL(url);url=null;
  region=null;sourceType=null;selectedDemo=null;sourceLabel='尚未选择画面';$('demo-description').hidden=true;
  selecting=false;paint();$('source-empty').hidden=false;controls();
}
function loaded(){region=fullRegion();$('source-empty').hidden=true;paint();controls();status('画面已就绪，可拖框选择手语者或使用整个画面。');}
function paint(){
  const dimensions=[video.videoWidth,video.videoHeight,preview.clientWidth,preview.clientHeight];
  const ready=dimensions.every(value=>Number.isFinite(value)&&value>0);
  $('roi').classList.toggle('selecting',selecting);$('selection').hidden=!region||!ready;
  if(region&&ready){const frame=videoRect(...dimensions);
    Object.assign($('selection').style,{left:frame.x+region.x*frame.width+'px',top:frame.y+region.y*frame.height+'px',width:region.width*frame.width+'px',height:region.height*frame.height+'px'});}
  $('region').textContent=selecting?'请拖框包含脸部、双手与上半身':region?'已确认识别选区':'尚未选择画面';
}
async function health(){
  try{const response=await fetch('/v1/communication/realtime');capabilities=response.ok?await response.json():{ready:false};}
  catch{capabilities={ready:false};}controls();
  if(!disposed)timer=setTimeout(health,5000);
}
$('retry').onclick=()=>{clearTimeout(timer);health();loadDemos();};
async function loadDemos(){
  try{const response=await fetch('/v1/communication/demos');if(!response.ok)return;
    catalogItems=demoItems(await response.json());if(disposed||!catalogItems.length)return;
    const options=catalogItems.map(item=>{const option=document.createElement('option');option.value=item.id;option.textContent=item.title+' · '+item.annotation;return option;});
    $('demo-choice').replaceChildren(...options);$('demo-panel').hidden=false;
  }catch{$('demo-panel').hidden=true;}
}
$('demo-load').onclick=()=>{
  const item=catalogItems.find(entry=>entry.id===$('demo-choice').value);if(!item)return;
  resetSource();selectedDemo=item;sourceType=item.kind;
  sourceLabel=item.kind==='diagnostic_replay'?'已知问题回放 · CSL-Daily train':'训练示范回放 · CSL-Daily train';
  $('demo-description').textContent='示范标注：'+item.annotation+'　'+item.note;$('demo-description').hidden=false;
  const version=sourceVersion;video.onloadedmetadata=()=>{if(version===sourceVersion)loaded();};video.src=item.media_url;video.load();controls();
};
$('camera').onclick=async()=>{
  resetSource();const version=sourceVersion;
  try{const next=await navigator.mediaDevices.getUserMedia({audio:false,video:{width:{ideal:640},height:{ideal:480},frameRate:{ideal:30,max:30}}});
    if(version!==sourceVersion||disposed){next.getTracks().forEach(track=>track.stop());return;}
    stream=next;sourceType='camera';sourceLabel='摄像头实时采集';video.srcObject=next;video.onloadedmetadata=()=>{if(version===sourceVersion)loaded();};await video.play();
    next.getVideoTracks()[0]?.addEventListener('ended',()=>{if(stream===next)resetSource('摄像头已断开，识别已暂停。');});
  }catch{status('摄像头未能开启，可检查浏览器权限或选择视频。',true);}
};
$('screen').onclick=async()=>{resetSource();const version=sourceVersion,next=await screen.open();
  if(!next)return;if(version!==sourceVersion||disposed){next.getTracks().forEach(track=>track.stop());return;}
  stream=next;sourceType='screen';sourceLabel='共享画面实时采集';video.srcObject=next;video.onloadedmetadata=()=>{if(version===sourceVersion)loaded();};try{await video.play();}catch{status('请点击预览播放共享画面。',true);}
};
$('file').onchange=()=>{const file=$('file').files[0];if(!file)return;resetSource();if(!file.size||file.size>128*1024*1024){status('请选择128MB以内的视频。',true);return;}
  sourceType='file';sourceLabel='本地视频 · '+file.name;url=URL.createObjectURL(file);const version=sourceVersion;video.onloadedmetadata=()=>{if(version===sourceVersion)loaded();};video.src=url;video.load();
};
$('select').onclick=()=>{stop('拖框选择手语者。');selecting=true;paint();};
$('full').onclick=()=>{stop();selecting=false;region=fullRegion();paint();controls();};
const point=event=>{const box=preview.getBoundingClientRect();return{x:event.clientX-box.left,y:event.clientY-box.top};};
$('roi').onpointerdown=event=>{if(selecting){drag=point(event);$('roi').setPointerCapture(event.pointerId);}};
$('roi').onpointerup=event=>{if(!drag)return;const selected=selectRegion(drag,point(event),video.videoWidth,video.videoHeight,preview.clientWidth,preview.clientHeight);drag=null;
  if(selected){region=selected;selecting=false;status('选区已确认，可以开始实时识别。');}paint();controls();};
$('start').onclick=async()=>{
  if(!region||document.hidden||starting||handoff?.busy)return;
  const action=++intent,version=sourceVersion;starting=true;selecting=false;paint();controls();
  captureMetrics={};renderMetrics();
  $('handoff-status').textContent=handoffHint;$('handoff-status').dataset.error='false';
  if(selectedDemo){video.pause();video.currentTime=0;}else if(sourceType==='file'&&video.ended)video.currentTime=0;
  const actual=stream?.getVideoTracks()[0]?.getSettings?.().frameRate;
  const fps=selectedDemo?.fps??(Number.isFinite(actual)?Math.max(10,Math.min(30,actual)):30);
  try{await client.start({fps,assistance:$('assistance').checked,source:{kind:sourceType,label:sourceLabel,...(selectedDemo?{id:selectedDemo.id}:{})}});if(action!==intent||version!==sourceVersion||document.hidden){if(action===intent)client.stop();return;}
    await video.play();if(action!==intent||version!==sourceVersion||document.hidden){if(action===intent)client.stop();return;}
    client.noteCaptureStart();capture.start({fps,region});
    if(action!==intent||version!==sourceVersion||client.status!=='running')return;
    status('实时词正在识别，停顿后补全中文。');
  }catch(error){if(action===intent)stop(error.message||'实时识别未能开始。',true);}
  finally{if(action===intent){starting=false;controls();}}
};
$('stop').onclick=()=>stop('已中止识别，未完成句已丢弃；已提交的中文记录会保留。');
$('boundary').onclick=()=>client.boundary();$('finish').onclick=finishCurrent;
video.addEventListener('ended',finishCurrent);
video.addEventListener('pause',()=>{if(capture.active&&replaySource()&&!video.ended)stop('视频已暂停，点击开始可继续。');});
video.addEventListener('seeking',()=>{if(capture.active)stop('视频位置已改变，请重新开始识别。');});
video.addEventListener('error',()=>stop('视频无法读取，请换一个视频。',true));
$('clear').onclick=()=>{stop();client.clear();};$('edit-cancel').onclick=()=>$('edit-dialog').close();
$('edit-use-words').onclick=()=>{const item=client.snapshot().captions.find(x=>x.id===editId);if(item?.word_evidence?.length===1){$('edit-text').value=item.word_evidence[0];$('edit-warning').hidden=false;$('edit-warning').textContent='已填入实时词，请核对后保存；识别原文将保留。';}};
$('edit-form').onsubmit=event=>{event.preventDefault();client.edit(editId,$('edit-text').value);$('edit-dialog').close();};
document.addEventListener('visibilitychange',()=>{if(document.hidden)stop('页面已切到后台，实时识别已暂停。');});
window.addEventListener('pagehide',()=>{disposed=true;clearTimeout(timer);handoff?.dispose();resetSource();});
new ResizeObserver(paint).observe(preview);render(client.snapshot());health();loadDemos();
