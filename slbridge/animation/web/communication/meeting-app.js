import {fullRegion,videoRect,selectRegion,cropRect} from './selection.mjs';
import {ScreenSource,RegionRecorder} from './screen-capture.mjs';
import {CaptionSession,captionKind,clock} from './caption-session.mjs';

const $=id=>document.getElementById(id),video=$('meeting-video'),box=$('preview-box');
let source=null,sourceVersion=0,sourceUrl=null,ready=false,region=null,running=false,sourceEnded=false,disposed=false;
let captureIntent=0,starting=false;
let capabilities={ready:false,languages:[]},connectionVersion=0,connectionTimer=null,toastTimer=null,editId=null,drag=null,rendered='';
const status=(text,error=false)=>{$('capture-status').textContent=text;$('capture-status').dataset.error=String(error);};
const notify=text=>{clearTimeout(toastTimer);$('notice').textContent=text;$('notice').hidden=false;toastTimer=setTimeout(()=>$('notice').hidden=true,4000);};
const session=new CaptionSession({onUpdate:renderSession});
const recorder=new RegionRecorder({video,onSegment:(file,meta)=>session.enqueue(file,meta),onState:state=>{
  if(state.status==='recording'){
    $('capture-badge').hidden=false;$('segment-progress').textContent=state.seconds===undefined?'采集已开始':`当前片段 ${state.seconds.toFixed(1)} 秒`;
    if(recorder.canvas)drawCrop(recorder.canvas);
  }else if(['error','reselect'].includes(state.status)){
    pauseCapture(state.message);if(state.status==='reselect'){region=null;paintSelection();}
    status(state.message,true);
  }else if(state.status==='short')notify(state.message);
}});
const screen=new ScreenSource({onState:state=>{
  if(state.status==='opening')status('在浏览器中选择要共享的屏幕或会议窗口。');
  if(state.status==='error')status(state.message,true);
  if(state.status==='ended'){resetSource();status('屏幕共享已结束。字幕记录已保留。');}
}});

function updateControls(){
  const language=capabilities.languages.find(x=>x.id===$('sign-language').value);
  $('start-captions').disabled=!ready||!region||!language?.ready||running||starting;
  $('start-captions').hidden=running;$('pause-captions').hidden=!running;$('submit-segment').hidden=!running;
  $('start-captions').textContent=sourceEnded?'重新播放并识别':session.status==='paused'?'继续字幕':'开始字幕';
  $('full-region').disabled=!ready;$('reselect').disabled=!ready;$('end-source').disabled=!source;
  $('selection-overlay').dataset.running=String(running);$('capture-badge').hidden=!running;
  $('connection').dataset.ready=String(!!language?.ready);
  $('connection').textContent=language?.ready?'识别已连接':language?'模型准备中':'识别未连接';
  const sharingSupported=!!navigator.mediaDevices?.getDisplayMedia;
  $('share-screen').disabled=!sharingSupported;
  $('share-screen').title=sharingSupported?'选择屏幕、窗口或浏览器标签页':'当前浏览器不支持屏幕共享，可以使用视频演示';
}

function renderSession(state){
  $('caption-count').textContent=state.captions.length;
  $('history-empty').hidden=state.captions.length>0;
  $('export-captions').disabled=!state.captions.length;$('clear-captions').disabled=!state.captions.length;
  $('undo-clear').hidden=!state.canUndo;
  $('session-notice').hidden=!state.notice;$('session-notice').textContent=state.notice;
  const phase=state.busy?'正在识别片段':running?'正在采集画面':state.status==='paused'?'已暂停':sourceEnded?'回放已结束':'等待开始';
  $('caption-state').textContent=phase;$('caption-state').dataset.busy=String(state.busy);
  $('queue-status').textContent=state.busy?`正在处理一段${state.pending?`，另有 ${state.pending} 段等待`:''}。`:
    `字幕在每段识别完成后追加。${state.skipped?` 已跳过 ${state.skipped} 段。`:''}${state.empty?` ${state.empty} 段无结果。`:''}`;
  const latest=state.captions.at(-1);document.querySelector('.current-caption').dataset.empty=String(!latest);
  $('live-caption').textContent=latest?.text||'字幕将在这里出现';
  $('caption-kind').textContent=latest?`${captionKind(latest.output_kind)} · ${latest.edited?'已人工修改':'请结合上下文核对'}`:'选择画面后，开始一次手语交流。';
  const signature=JSON.stringify(state.captions);
  if(signature!==rendered){
    const oldCount=$('caption-history').children.length;rendered=signature;
    $('caption-history').replaceChildren(...state.captions.map((item,index)=>{
      const row=document.createElement('li');row.dataset.segmentId=item.id;
      const meta=document.createElement('div');meta.className='caption-meta';
      const label=document.createElement('span');label.textContent=`${clock(item.start_seconds)}–${clock(item.end_seconds)} · ${captionKind(item.output_kind)}${item.edited?' · 已修改':''}`;
      const edit=document.createElement('button');edit.textContent='核对';edit.setAttribute('aria-label',`修改第 ${index+1} 段字幕`);
      edit.addEventListener('click',()=>{editId=item.id;$('edit-text').value=item.text;$('edit-dialog').showModal();$('edit-text').focus();});
      meta.append(label,edit);const text=document.createElement('p');text.className='caption-text';text.textContent=item.text;row.append(meta,text);
      if(item.edited){const details=document.createElement('details'),summary=document.createElement('summary'),raw=document.createElement('p');summary.textContent='识别原文';raw.textContent=item.original_text;details.append(summary,raw);row.append(details);}
      return row;
    }));
    if(state.captions.length>oldCount)$('caption-history').scrollTop=$('caption-history').scrollHeight;
  }
  if(sourceEnded&&!running&&!state.busy&&!state.pending&&state.status==='stopped')status('视频回放已结束。你可以核对字幕，或重新播放识别。');
  updateControls();
}

function drawCrop(canvas){
  const preview=$('crop-preview'),context=preview.getContext('2d');context.clearRect(0,0,preview.width,preview.height);
  const scale=Math.min(preview.width/canvas.width,preview.height/canvas.height),w=canvas.width*scale,h=canvas.height*scale;
  context.drawImage(canvas,(preview.width-w)/2,(preview.height-h)/2,w,h);$('crop-empty').hidden=true;
}
function paintSelection(){
  $('selection-rect').hidden=!ready||!region;
  if(!ready||!region){$('region-label').textContent=ready?'请拖框选中手语者':'先选择画面，再框选手语者';return;}
  const frame=videoRect(video.videoWidth,video.videoHeight,box.clientWidth,box.clientHeight);
  Object.assign($('selection-rect').style,{left:frame.x+region.x*frame.width+'px',top:frame.y+region.y*frame.height+'px',width:region.width*frame.width+'px',height:region.height*frame.height+'px'});
  const crop=cropRect(region,video.videoWidth,video.videoHeight);
  $('region-label').textContent=region.width===1&&region.height===1?'当前识别整个画面':`选区已确认 · ${crop.width} × ${crop.height}`;
  const canvas=document.createElement('canvas');canvas.width=crop.width;canvas.height=crop.height;
  canvas.getContext('2d').drawImage(video,crop.x,crop.y,crop.width,crop.height,0,0,crop.width,crop.height);drawCrop(canvas);
}
function pauseCapture(message='已暂停。继续时将从新片段开始，已完成的字幕保留。'){
  captureIntent++;starting=false;running=false;recorder.pause();if(source?.type==='file')video.pause();session.pause();
  $('segment-progress').textContent='采集已暂停';status(message);updateControls();
}
function resetSource(){
  const version=++sourceVersion;captureIntent++;starting=false;running=false;ready=false;region=null;sourceEnded=false;source=null;
  recorder.pause();session.stop();screen.close();video.onloadeddata=null;video.onerror=null;video.pause();
  video.srcObject=null;video.removeAttribute('src');video.load();video.hidden=true;
  if(sourceUrl)URL.revokeObjectURL(sourceUrl);sourceUrl=null;
  $('preview-empty').hidden=false;$('selection-overlay').hidden=true;$('selection-rect').hidden=true;$('crop-empty').hidden=false;
  $('crop-preview').getContext('2d').clearRect(0,0,160,100);
  $('source-label').textContent='尚未选择来源';$('segment-progress').textContent='准备就绪后开始';paintSelection();updateControls();return version;
}
function configureVideo(version){
  video.hidden=false;$('preview-empty').hidden=true;
  video.onloadeddata=()=>{
    if(version!==sourceVersion)return;
    if(video.videoWidth<32||video.videoHeight<32){status('画面太小，请选择更清楚的视频。',true);return;}
    ready=true;region=fullRegion();$('selection-overlay').hidden=false;paintSelection();updateControls();
    status('画面已就绪。可拖框选中手语者，再开始字幕。');
  };
  video.onerror=()=>{if(version===sourceVersion){ready=false;status('这个视频无法播放，请换用MP4或WebM。',true);updateControls();}};
}
async function chooseScreen(){
  const version=resetSource();const stream=await screen.open();
  if(!stream||version!==sourceVersion)return;
  source={type:'screen',name:stream.getVideoTracks()[0]?.label||'共享画面'};$('source-label').textContent=source.name;
  configureVideo(version);video.srcObject=stream;
  try{await video.play();}catch{status('共享已连接，但预览未能播放，请重新选择来源。',true);}updateControls();
}
function chooseFile(file){
  if(!file?.size||file.size>128*1024*1024){notify('请选择128MB以内的视频。');return;}
  const version=resetSource();source={type:'file',name:file.name};$('source-label').textContent='视频演示 · '+file.name;
  configureVideo(version);sourceUrl=URL.createObjectURL(file);video.src=sourceUrl;video.load();updateControls();
}
async function startCaptions(){
  if(!ready||!region||running||starting||document.hidden)return;
  const version=sourceVersion,intent=++captureIntent;starting=true;updateControls();
  if(sourceEnded||video.ended){video.currentTime=0;sourceEnded=false;}
  try{
    await video.play();if(version!==sourceVersion||intent!==captureIntent||disposed||document.hidden)return;
    const sourceEpoch=session.start({language:$('sign-language').value,sourceLabel:source.name});running=true;
    recorder.start({region,sourceEpoch,mode:$('segment-mode').value,maxSeconds:$('segment-mode').value==='interval'?6:8});
    status($('segment-mode').value==='interval'?'正在采集，每6秒提交一段。可以随时暂停或手动提交。':'正在采集。画面停顿或达到8秒时提交一段，也可手动提交。');
    renderSession(session.snapshot());
  }catch(error){if(intent===captureIntent){pauseCapture('采集未能启动，请重新选择来源。');status(error.message||'采集未能启动。',true);}}
  finally{if(intent===captureIntent){starting=false;updateControls();}}
}
video.addEventListener('ended',async()=>{
  if(source?.type!=='file'||!running)return;
  const version=sourceVersion,epoch=session.epoch;running=false;sourceEnded=true;updateControls();status('回放已结束，正在提交最后一个片段…');
  await recorder.finish();
  if(version!==sourceVersion||epoch!==session.epoch)return;
  session.finishInput();
  const state=session.snapshot();renderSession(state);
  if(state.busy||state.pending)status('回放已结束，正在识别最后的片段。');
});

function point(event){const bounds=box.getBoundingClientRect();return {x:event.clientX-bounds.left,y:event.clientY-bounds.top};}
$('selection-overlay').addEventListener('pointerdown',event=>{
  if(!ready||running||event.button!==0)return;
  drag={start:point(event),previous:region};event.currentTarget.setPointerCapture(event.pointerId);event.preventDefault();
});
$('selection-overlay').addEventListener('pointermove',event=>{
  if(!drag)return;
  const value=selectRegion(drag.start,point(event),video.videoWidth,video.videoHeight,box.clientWidth,box.clientHeight);
  if(value){region=value;paintSelection();}
});
$('selection-overlay').addEventListener('pointerup',event=>{
  if(!drag)return;
  const value=selectRegion(drag.start,point(event),video.videoWidth,video.videoHeight,box.clientWidth,box.clientHeight);
  region=value||drag.previous;drag=null;paintSelection();updateControls();
  status(value?'选区已确认。请让脸部、双手和上半身都在框内。':'选区太小或从画面外开始，已保留原选区。');
});
$('selection-overlay').addEventListener('pointercancel',()=>{if(drag){region=drag.previous;drag=null;paintSelection();}});
new ResizeObserver(()=>{if(ready)paintSelection();}).observe(box);

async function connectServices(){
  if(disposed)return;clearTimeout(connectionTimer);const version=++connectionVersion;
  try{
    const response=await fetch('/v1/communication/recognition',{cache:'no-store',signal:AbortSignal.timeout(6000)});
    if(!response.ok)throw new Error('unavailable');const value=await response.json();
    if(version!==connectionVersion||disposed)return;
    capabilities={...value,languages:Array.isArray(value.languages)?value.languages:[]};
  }catch{if(version!==connectionVersion||disposed)return;capabilities={ready:false,languages:[]};}
  const lines=capabilities.languages.map(item=>item.label+'：'+(item.ready?item.scope:'模型正在准备')+(item.model_id?' · '+item.model_id:''));
  $('capability-list').replaceChildren(...(lines.length?lines:['识别服务暂未连接，请重新连接。']).map(text=>{const li=document.createElement('li');li.textContent=text;return li;}));
  if(!source&&!session.snapshot().captions.length)status(capabilities.ready?'选择共享画面或视频，开始会议字幕。':'识别服务正在准备。可以先选择画面，在设置中重新连接。',!capabilities.ready);
  updateControls();connectionTimer=setTimeout(connectServices,capabilities.ready?12000:2500);
}
$('share-screen').addEventListener('click',chooseScreen);
$('upload-demo').addEventListener('click',()=>$('source-file').click());
$('source-file').addEventListener('change',()=>{const file=$('source-file').files[0];if(file)chooseFile(file);$('source-file').value='';});
$('start-captions').addEventListener('click',startCaptions);
$('pause-captions').addEventListener('click',()=>pauseCapture());
$('submit-segment').addEventListener('click',()=>recorder.submit());
$('end-source').addEventListener('click',()=>{resetSource();status('本次采集已结束。字幕记录已保留。');});
$('reselect').addEventListener('click',()=>{pauseCapture('已暂停。请在预览中拖框重新选择手语者。');});
$('full-region').addEventListener('click',()=>{pauseCapture('已选择全画面，点击开始继续识别。');region=fullRegion();paintSelection();updateControls();});
$('sign-language').addEventListener('change',()=>{pauseCapture('手语已切换。点击开始，使用新语言识别后续片段。');});
$('segment-mode').addEventListener('change',()=>pauseCapture('分段方式已改变。关闭设置后点击开始继续。'));
$('settings-open').addEventListener('click',()=>$('settings').showModal());
$('retry-connections').addEventListener('click',connectServices);
$('edit-close').addEventListener('click',()=>$('edit-dialog').close());
$('edit-form').addEventListener('submit',event=>{event.preventDefault();try{session.edit(editId,$('edit-text').value);$('edit-dialog').close();}catch(error){notify(error.message);}});
$('clear-captions').addEventListener('click',()=>{pauseCapture('已暂停并清空字幕，可以撤销清空。');session.clear();});
$('undo-clear').addEventListener('click',()=>session.undoClear());
$('export-captions').addEventListener('click',()=>{
  const url=URL.createObjectURL(new Blob([session.transcript()],{type:'text/plain;charset=utf-8'}));
  const link=document.createElement('a');link.href=url;link.download='手语桥-会议字幕.txt';link.click();setTimeout(()=>URL.revokeObjectURL(url),4000);
});
document.addEventListener('visibilitychange',()=>{if(document.hidden&&(running||starting))pauseCapture('页面进入后台，采集已暂停。回到本页后可继续字幕。');});
window.addEventListener('pagehide',()=>{disposed=true;connectionVersion++;clearTimeout(connectionTimer);clearTimeout(toastTimer);resetSource();});
renderSession(session.snapshot());connectServices();
