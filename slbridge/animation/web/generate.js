import { MotionStage } from '/animation/static/stage.js';
import { GenerationController } from './generate-controller.mjs';
import { FastCompositionController } from './fast-composition-controller.mjs';

const $=id=>document.getElementById(id);
let stageReady=false,busy=false,loaded=false,mode='fast';
let submittedAt=null,firstPlaybackMs=null;
let plannerReady=false,readinessState='checking',readinessAttempts=0;
let readinessTimer=null,readinessRequest=null,pageClosed=false,stageFailed=false,uiStatus='idle';
const MAX_READINESS_ATTEMPTS=90;
const clock=value=>Math.floor((value||0)/60)+':'+String(Math.floor((value||0)%60)).padStart(2,'0');
const seconds=value=>(value/1000).toFixed(2)+' 秒';
const activeController=()=>mode==='fast'?fast:generator;

function updateButton() {
  const size=[...$('expression').value].length;
  $('count').textContent=size+' / 200';
  $('generate').disabled=!stageReady||(mode==='fast'&&!plannerReady)||busy||!$('expression').value.trim()||size>200;
}
function showReadinessStatus() {
  if(mode!=='fast'||stageFailed||busy||loaded||(uiStatus!=='idle'&&plannerReady))return;
  const messages={checking:'正在检查文字规划器…',starting:'文字规划器正在准备，首次启动需要一些时间…',
    unconfigured:'快速组合尚未配置。可选择“模型生成（较慢）”，或检查本机服务配置。',
    failed:'文字规划器准备失败。可选择“模型生成（较慢）”，或检查本机服务后刷新。',
    timeout:'文字规划器仍未准备完成，请稍后刷新，或选择“模型生成（较慢）”。',
    unavailable:'暂时无法读取文字规划器状态。请刷新重试，或选择“模型生成（较慢）”。',
    ready:'人物与文字规划器已准备。输入文字后点击组合并演示。'};
  $('status').textContent=stageReady?messages[readinessState]:'正在准备人物…';
  $('status').dataset.tone=['failed','unconfigured','unavailable','timeout'].includes(readinessState)?'error':'';
  $('playback-state').textContent=plannerReady?'等待输入':['checking','starting'].includes(readinessState)?'准备文字规划器':'快速组合未就绪';
}
async function checkReadiness(reset=false) {
  if(reset){clearTimeout(readinessTimer);readinessTimer=null;readinessAttempts=0;}
  if(pageClosed||readinessRequest)return;
  if(reset){plannerReady=false;readinessState='checking';updateButton();showReadinessStatus();}
  const request=new AbortController();readinessRequest=request;
  const timeout=setTimeout(()=>request.abort(),5000);
  let pollAgain=false;
  try {
    readinessAttempts++;
    const response=await fetch('/v1/animation/composition',{method:'GET',cache:'no-store',signal:request.signal});
    if(!response.ok)throw new Error('Readiness unavailable');
    const data=await response.json();
    if(pageClosed)return;
    plannerReady=data.enabled===true&&data.planner?.state==='ready';
    if(plannerReady)readinessState='ready';
    else if(data.enabled!==true||data.planner?.state==='unconfigured')readinessState='unconfigured';
    else if(data.planner?.state==='starting'){readinessState='starting';pollAgain=true;}
    else readinessState='failed';
  }catch{
    if(pageClosed)return;
    plannerReady=false;readinessState='unavailable';
  }finally{
    clearTimeout(timeout);readinessRequest=null;
    if(!pageClosed) {
      if(pollAgain&&readinessAttempts>=MAX_READINESS_ATTEMPTS){readinessState='timeout';pollAgain=false;}
      updateButton();showReadinessStatus();
      if(pollAgain)readinessTimer=setTimeout(()=>checkReadiness(),1000);
    }
  }
}
function showGlosses(glosses=[],missing=[],omitted=[]) {
  $('gloss-panel').hidden=!glosses.length;
  $('glosses').replaceChildren(...glosses.map(word=>{
    const span=document.createElement('span');span.textContent=word;
    if(missing.includes(word))span.className='missing';
    return span;
  }));
  $('missing-words').hidden=!missing.length&&!omitted.length;
  $('missing-words').textContent=omitted.length?'文字规划可能遗漏：'+omitted.join('、')+'。请核对或缩短输入，本次未播放。':missing.length?'缺少动作：'+missing.join('、')+'。本次未播放。':'';
}
function showTiming(timings={}) {
  const entries=[['文字规划',timings.planning_ms??timings.planning],['动作组合',timings.composition_ms??timings.composition],['服务处理',timings.total_ms??timings.total]];
  const details=entries.filter(([,value])=>typeof value==='number'&&Number.isFinite(value)).map(([label,value])=>label+' '+seconds(value));
  $('processing-time').textContent=details.join('；');
  $('playback-latency').textContent=firstPlaybackMs===null?'':'提交至播放状态 '+seconds(firstPlaybackMs)+'（人物已准备后）';
  $('timing-panel').hidden=!details.length&&firstPlaybackMs===null;
}
function renderState(controllerMode,state) {
  if(controllerMode!==mode)return;
  uiStatus=state.status;
  busy=['submitting','queued','running','loading'].includes(state.status);
  loaded=state.status==='ready';
  if(['idle','submitting','cancelled','error','blocked','needs_review'].includes(state.status)) {
    submittedAt=null;firstPlaybackMs=null;
  }
  if(mode==='fast'&&typeof state.submitted_at_ms==='number')submittedAt=state.submitted_at_ms;
  $('status').textContent=state.message||'';
  $('status').dataset.tone=['error','blocked','needs_review'].includes(state.status)?'error':'';
  $('wait-hint').hidden=!busy;
  $('wait-hint').textContent=mode==='fast'?'正在规划词序、选择动作并加载人物演示。':'模型生成可能超过两分钟，进度会随实际任务更新。';
  $('cancel').disabled=!busy;
  $('play').disabled=!loaded;$('pause').disabled=true;$('stop').disabled=!busy&&!loaded;
  $('subtitle').textContent=state.subtitle||'演示准备完成后显示本次文字。';
  $('playback-state').textContent=loaded?'可以播放':state.status==='needs_review'?'待核对':state.status==='blocked'?'缺少动作':state.status==='error'?'演示失败':busy?(mode==='fast'?'正在组合':'正在生成'):state.status==='cancelled'?'已取消':'等待输入';
  $('quality-warning').textContent=mode==='fast'?'词语含义、动作衔接和表情仍需核对。':state.job?.quality_warning?'检测到动作跳变线索，请核对手形与衔接。':'手形、表情和动作自然度尚需核对。';
  const clearance=loaded?stage.tracks?.diagnostics?.body_clearance:null;
  if(clearance?.remaining_core_hits)$('quality-warning').textContent+=' 检测到穿模线索，请重点核对手部与身体接触。';
  if(clearance?.max_offset_step>.015)$('quality-warning').textContent+=' 避让过渡变化较大，请重点核对手腕衔接。';
  $('language-review').textContent='手语含义：尚未人工核对';
  showGlosses(mode==='fast'?state.glosses:[],state.missing_glosses,state.omitted_terms);
  showTiming(mode==='fast'?state.timings_ms:{});
  updateButton();
  showReadinessStatus();
}
const stage=new MotionStage($('stage'),state=>{
  $('time').textContent=clock(state.time)+' / '+clock(state.duration);
  if(loaded) {
    $('playback-state').textContent=state.playing?'播放中':state.time>=state.duration?'播放结束':'已暂停';
    $('play').disabled=state.playing;$('pause').disabled=!state.playing;
    // This is the first observed playing callback, not a canvas-pixel metric.
    if(mode==='fast'&&state.playing&&submittedAt!==null&&firstPlaybackMs===null) {
      firstPlaybackMs=Math.max(0,performance.now()-submittedAt);
      $('playback-latency').textContent='提交至播放状态 '+seconds(firstPlaybackMs)+'（人物已准备后）';
      $('timing-panel').hidden=false;
    }
  }
});
const fast=new FastCompositionController({stage,onState:state=>renderState('fast',state)});
const generator=new GenerationController({stage,onState:state=>renderState('generate',state)});
function invalidateAll(message) {
  fast.invalidate(message);generator.invalidate(message);
}
$('generate-form').addEventListener('submit',event=>{
  event.preventDefault();if($('generate').disabled)return;
  if(mode==='fast')fast.generate($('expression').value);
  else generator.generate($('expression').value,Number($('seconds').value));
});
$('expression').addEventListener('input',()=>invalidateAll(mode==='fast'?'输入已改变，请点击组合并演示。':'输入已改变，请点击生成演示。'));
$('mode').addEventListener('change',()=>{
  mode=$('mode').value;
  $('duration-field').hidden=mode==='fast';$('duration-help').hidden=mode!=='fast';
  $('generate').textContent=mode==='fast'?'组合并演示':'生成演示';
  $('mode-help').textContent=mode==='fast'?'复用词语动作，组合完成后自动播放。缺少动作会明确提示。':'根据文字从头生成新动作，完成后点击播放；本机实测可能需要数分钟。';
  invalidateAll(mode==='fast'?'已选择快速组合，输入文字后点击组合并演示。':'已选择模型生成，输入文字后点击生成演示。');
  if(mode==='fast')checkReadiness(true);
});
$('seconds').addEventListener('change',()=>generator.invalidate('时长已改变，请重新生成演示。'));
$('cancel').addEventListener('click',()=>activeController().cancel());
$('stop').addEventListener('click',()=>activeController().stop());
$('play').addEventListener('click',()=>activeController().play());
$('pause').addEventListener('click',()=>activeController().pause());
$('view').addEventListener('change',()=>stage.setView($('view').value));
window.addEventListener('pagehide',()=>{pageClosed=true;clearTimeout(readinessTimer);readinessRequest?.abort();invalidateAll();stage.dispose?.();});
stage.ready.then(()=>{stageReady=true;updateButton();if(mode==='fast')showReadinessStatus();else $('status').textContent='人物已准备。输入文字后点击生成演示。';}).catch(error=>{stageFailed=true;$('status').textContent='人物初始化失败：'+error.message;$('status').dataset.tone='error';});
checkReadiness();
