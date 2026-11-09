import { MotionStage } from '/animation/static/stage.js';
import { ScenePreview, appendExpression, renderSource } from './scenes-controller.mjs';

const $ = id => document.getElementById(id);
let ready = false, loaded = false, playing = false;
let pageOffset = 0, catalogGeneration = 0, catalogAbort = null, selected = null;
const clock = value => Math.floor((value || 0) / 60) + ':' + String(Math.floor((value || 0) % 60)).padStart(2,'0');
function status(message, error=false) {
  $('status').textContent = message; $('status').dataset.tone = error ? 'error' : '';
}
function updateButton() {
  $('preview').disabled = !ready || !$('expression').value.trim() || !$('confirmed').checked;
}
function renderPlan(plan) {
  $('plan-details').hidden = !plan;
  $('download').disabled = !plan;
  $('sources').replaceChildren(); $('missing').replaceChildren();
  if (!plan) return;
  $('plan-review').textContent = '仅匹配已有完整原句。语言审核：' + (plan.language_review || 'NOT_RUN') + '；组合审核：' + (plan.composition_review || 'NOT_RUN');
  for (const item of plan.items || []) {
    const li = document.createElement('li');
    li.textContent = item.text + ' · ' + item.sample_id + ' · ' + item.frames + ' 帧';
    $('sources').appendChild(li);
  }
  $('missing-box').hidden = !(plan.unmatched || []).length;
  for (const text of plan.unmatched || []) {
    const li = document.createElement('li'); li.textContent = text; $('missing').appendChild(li);
  }
}
const stage = new MotionStage($('stage'), state => {
  playing = state.playing;
  $('time').textContent = clock(state.time) + ' / ' + clock(state.duration);
  if (loaded) {
    $('pause').textContent = state.playing ? '暂停' : '继续';
    $('playback-state').textContent = state.playing ? '研究预览播放中' : state.time >= state.duration ? '播放结束' : '已暂停';
  }
});
const preview = new ScenePreview({stage,onState:state => {
  if (state.status === 'idle') {
    loaded = false; $('pause').disabled = true;
    $('playback-state').textContent = '待确认';
    $('subtitle').textContent = '选择原句只填入文本，确认后再预览。';
  }
  if (state.status === 'playing') {
    loaded = true; $('pause').disabled = false;
    $('playback-state').textContent = '研究预览播放中';
    // Captions always come from matched source records, never from the search query.
    $('subtitle').textContent = state.plan.items.map(item => item.text).join(' / ');
  }
  if (state.status === 'blocked') $('playback-state').textContent = '缺项，未播放';
  if (state.status === 'error') $('playback-state').textContent = '未能预览';
  renderPlan(state.plan || null);
  if (state.message) status(state.message,['blocked','error'].includes(state.status));
}});

function edit(message='表达已改变，请重新核对并确认。') {
  preview.invalidate(); $('confirmed').checked = false; updateButton(); status(message);
}
function choose(asset, append=false) {
  // Even a refused append cancels an older preview immediately.
  edit();
  try {
    $('expression').value = append ? appendExpression($('expression').value,asset.text) : asset.text;
    selected = asset.sample_id;
    for (const card of $('asset-list').children) card.classList.toggle('selected',card.dataset.sampleId === selected);
    updateButton();
    status(append ? '已换行加入表达。核对所有句子后可研究预览。' : '已填入真实原句。核对后可研究预览。');
  } catch (error) { status(error.message,true); }
}
function renderAssets(assets) {
  $('asset-list').replaceChildren();
  for (const asset of assets) {
    const card = document.createElement('article'); card.className = 'asset-card';
    card.dataset.sampleId = asset.sample_id;
    card.classList.toggle('selected',asset.sample_id === selected);
    const text = document.createElement('p'); text.className = 'source-text'; renderSource(text,asset);
    const meta = document.createElement('p'); meta.className = 'asset-meta';
    meta.textContent = asset.sample_id + ' · ' + asset.split + ' · ' + Number(asset.duration).toFixed(1) + ' 秒 · ' + asset.frames + ' 帧';
    const actions = document.createElement('div'); actions.className = 'asset-actions';
    const select = document.createElement('button'); select.type = 'button'; select.textContent = '选用原句';
    select.setAttribute('aria-label','选用原句：' + asset.text); select.addEventListener('click',()=>choose(asset));
    const add = document.createElement('button'); add.type = 'button'; add.textContent = '＋ 加入表达';
    add.setAttribute('aria-label','加入表达：' + asset.text); add.addEventListener('click',()=>choose(asset,true));
    actions.append(select,add); card.append(text,meta,actions); $('asset-list').appendChild(card);
  }
  if (!assets.length) {
    const empty = document.createElement('p'); empty.className = 'empty';
    empty.textContent = '没有匹配的真实原句。请换一个关键词，或切换场景查看候选。';
    $('asset-list').appendChild(empty);
  }
}
async function catalog(offset=0) {
  const generation = ++catalogGeneration;
  catalogAbort?.abort(); const controller = new AbortController(); catalogAbort = controller;
  const scene = $('scene').value;
  const params = new URLSearchParams({q:$('search').value.trim(),limit:'30',offset:String(offset)});
  if (scene) params.set('scene',scene);
  $('catalog-state').textContent = '正在检索…';
  $('prev').disabled = true; $('next').disabled = true;
  try {
    const response = await fetch('/v1/animation/scenes?' + params,{signal:controller.signal});
    const page = await response.json();
    if (generation !== catalogGeneration) return;
    if (!response.ok) throw new Error(typeof page.detail === 'string' ? page.detail : '原句库暂时无法读取。');
    // Initial catalogue supplies all scenes. Later filtered responses may contain only one.
    for (const item of page.scenes) {
      let option = [...$('scene').options].find(option=>option.value === item.id);
      if (!option) {option = document.createElement('option'); option.value = item.id; $('scene').appendChild(option);}
      option.textContent = item.title;
    }
    const current = page.scenes.find(item=>item.id === scene);
    $('scene-summary').textContent = current ? '候选文本 ' + current.candidate_texts + ' 条 · 固定目标精确覆盖 ' + current.target_exact + ' / ' + current.target_total + ' 条（非语言正确率）' : '完整原句按文本去重；固定目标覆盖可在各场景查看。';
    pageOffset = page.offset; renderAssets(page.assets);
    $('catalog-state').textContent = page.total + ' 条原句';
    $('page-label').textContent = page.total ? (page.offset + 1) + '–' + (page.offset + page.assets.length) + ' / ' + page.total : '0 条';
    $('prev').disabled = page.offset === 0; $('next').disabled = !page.has_more;
  } catch (error) {
    if (generation !== catalogGeneration || controller.signal.aborted) return;
    $('catalog-state').textContent = '读取失败';
    $('asset-list').replaceChildren(); status(error.message,true);
  }
}

$('expression').addEventListener('input',()=>edit());
$('confirmed').addEventListener('change',()=>{preview.invalidate();updateButton();status('确认状态已更新；点击研究预览开始。');});
$('scene').addEventListener('change',()=>{edit('场景已切换，旧播放已停止。');selected=null;catalog();});
$('search-form').addEventListener('submit',event=>{event.preventDefault();catalog();});
$('prev').addEventListener('click',()=>catalog(Math.max(0,pageOffset-30)));
$('next').addEventListener('click',()=>catalog(pageOffset+30));
$('preview').addEventListener('click',()=>preview.preview($('expression').value,$('confirmed').checked));
$('stop').addEventListener('click',()=>{preview.invalidate();status('已停止。重新预览需点击研究预览。');});
$('pause').addEventListener('click',()=>{if (loaded) playing ? stage.pause() : stage.play();});
$('view').addEventListener('change',()=>stage.setView($('view').value));
$('download').addEventListener('click',()=>{
  if (!preview.plan) return;
  const blob = new Blob([JSON.stringify(preview.plan,null,2)],{type:'application/json'});
  const url = URL.createObjectURL(blob); const anchor = document.createElement('a');
  anchor.href = url; anchor.download = 'scene-expression-plan.json'; anchor.click();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
});
window.addEventListener('pagehide',()=>{preview.invalidate();catalogAbort?.abort();stage.dispose?.();});
stage.ready.then(()=>{ready=true;updateButton();status('人物已准备。选择或加入原句，确认后可研究预览。');}).catch(error=>status('人物初始化失败：' + error.message,true));
catalog();
