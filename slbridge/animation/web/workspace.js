import { MotionStage } from "./stage.js";
import { PlaybackGeneration } from "./player-core.mjs";

const $ = id => document.getElementById(id);
const gate = new PlaybackGeneration();
const groups = { reference: "参考重建", fitted: "SMPL-X 拟合", candidate: "研究候选", corpus: "全量语料候选", generated: "自定义生成候选" };
let assets = [], selected = null, filter = "all", origin = { kind: "typed" };
let library = "demo", corpusSelected = null, corpusAssets = [], corpusOffset = 0;
let corpusController = null, corpusGeneration = 0, corpusTimer = null;
let controller = null, loaded = false, activePlan = null, recognizing = false;
let loadedModel = null;
const timeLabel = seconds => {
  const s = Math.max(0, Math.floor(seconds || 0));
  return Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0");
};
function status(message, error = false) {
  $("status").textContent = message;
  $("status").dataset.tone = error ? "error" : "";
}
function controls(enabled) {
  loaded = enabled;
  for (const id of ["pause", "stop", "replay", "timeline"]) $(id).disabled = !enabled;
  $("export-review").disabled = !enabled;
  updateJawControls();
}
function updateJawControls(){
  $("nonmanual-jaw").disabled=!loaded||!stage.nonmanualAvailable;
  $("nonmanual-jaw").checked=Boolean(stage.nonmanualEnabled);
  const manual=stage.expression&&Object.hasOwn(stage.expression,"jawOpen");
  const automatic=stage.clip?.source_kind==="pretrained_text_conditioned_generation"?
    "嘴部使用生成的下颌动作；含义与完整嘴形待审核，暂不含 GLB 导出。":
    "嘴部来自原始下颌参数的有界候选；含义与原片时间对齐未审核，暂不含 GLB 导出。";
  $("nonmanual-state").textContent=manual?"手动下颌覆盖中；恢复按钮回到自动／中立。当前面部检查不含在烘焙下载中。":
    stage.nonmanualAvailable?(stage.nonmanualEnabled?automatic:"嘴部候选已关闭；可在同一时刻开启对照。"):
    "当前片段无已校准嘴部候选。手动试调仅检查变形，尚未审核为手语表情。";
}
const stage = new MotionStage($("stage"), state => {
  $("time").textContent = timeLabel(state.time) + " / " + timeLabel(state.duration);
  if (loaded) {
    $("timeline").max = String(state.duration);
    $("timeline").value = String(state.time);
    $("pause").textContent = state.playing ? "暂停" : "继续";
    $("playback-state").textContent = state.playing ? "播放中" : state.time >= state.duration ? "已结束" : "已暂停";
  }
});

async function request(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error?.message || data.detail || "请求未完成");
  return data;
}
function invalidate() {
  gate.cancel(); controller?.abort(); controller = null;
  stage.stop(); controls(false); activePlan = null;
  $("timeline").value = "0"; $("timeline").max = "0";
  $("time").textContent = "0:00 / 0:00";
  $("motion-title").textContent = "完整表达预览";
  $("source-badge").textContent = "待选择动作";
  $("provenance").textContent = "选择动作并确认后显示来源。";
  $("playback-state").textContent = "待确认";
  $("subtitle").textContent = "确认输入后预览";
  $("download-motion").hidden = true;
  resetFaceInputs();
}
function resetFaceInputs() {
  for (const id of ["face-brow","face-blink","face-jaw"]) $(id).value = "0";
}
function updateInput() {
  $("character-count").textContent = $("input").value.length + " / 500";
  const labels = { typed: "手动输入", catalog: "动作库选择", corpus: "全量语料选择 · 待审核", recognition: "识别候选 · 请核对",
    edited_recognition: "已编辑 · 原始输出保留", desktop_handoff: "桌面候选 · 请核对" };
  $("origin-label").textContent = labels[origin.kind] || "输入候选 · 请核对";
  $("raw-result").hidden = typeof origin.raw_text !== "string";
  $("raw-result").textContent = typeof origin.raw_text === "string" ? "原始输出：" + origin.raw_text : "";
}
function renderLibrary() {
  const search = $("search").value.trim().toLocaleLowerCase();
  const visible = assets.filter(a => (filter === "all" || a.group === filter) &&
    [a.text, ...a.aliases].some(s => s.toLocaleLowerCase().includes(search)));
  $("visible-count").textContent = visible.length + " 条";
  $("asset-list").replaceChildren();
  for (const asset of visible) {
    const button = document.createElement("button");
    button.className = "asset-option" + (selected === asset.id ? " selected" : "");
    button.textContent = asset.text;
    button.dataset.assetId = asset.id;
    const meta = document.createElement("small");
    meta.textContent = groups[asset.group] + " · " + asset.duration.toFixed(1) + " 秒";
    if (asset.quality_warning) {
      const warning = document.createElement("span"); warning.className = "quality-dot";
      warning.textContent = "有跳变线索"; meta.appendChild(warning);
    }
    button.appendChild(meta);
    button.addEventListener("click", () => {
      invalidate(); corpusSelected = null; selected = asset.id; $("input").value = asset.text;
      origin = { kind: "catalog", asset_id: asset.id };
      $("origin-label").textContent = "动作库选择"; updateInput(); renderLibrary();
      status("已选择完整表达。确认输入后可研究预览。");
    });
    $("asset-list").appendChild(button);
  }
  if (!visible.length) {
    const empty = document.createElement("p"); empty.className = "empty";
    empty.textContent = "没有匹配的现成动作"; $("asset-list").appendChild(empty);
  }
}

function renderCorpus() {
  $("corpus-list").replaceChildren();
  for (const asset of corpusAssets) {
    const button = document.createElement("button");
    button.className = "asset-option" + (corpusSelected?.sample_id === asset.sample_id ? " selected" : "");
    button.textContent = asset.text; button.disabled = !asset.playable;
    const meta = document.createElement("small");
    meta.textContent = asset.sample_id + " · 签者 " + asset.signer + " · " + asset.split + " · " + asset.duration.toFixed(1) + " 秒";
    const state = document.createElement("small");
    state.textContent = !asset.playable ? "尚不可播放 · " + asset.state :
      (asset.quality_warning ? "有跳变线索 · " : "") + "动作与语言未审核";
    button.appendChild(meta); button.appendChild(state);
    button.addEventListener("click", () => selectCorpusAsset(asset));
    $("corpus-list").appendChild(button);
  }
  if (!corpusAssets.length) {
    const empty = document.createElement("p"); empty.className = "empty";
    empty.textContent = "没有匹配的语料，可更换关键词或筛选条件。"; $("corpus-list").appendChild(empty);
  }
}
function selectCorpusAsset(asset) {
  if (!asset.playable) return;
  invalidate(); corpusSelected = asset; selected = asset.id;
  $("input").value = asset.text;
  origin = { kind: "corpus", sample_id: asset.sample_id, asset_id: asset.id };
  updateInput(); renderCorpus(); status("已选择完整语料句。确认后按需加载；动作、表情及时间对齐均未审核。");
}
async function loadCorpusPage(offset = 0) {
  const generation = ++corpusGeneration;
  corpusController?.abort(); corpusController = new AbortController();
  const localController = corpusController;
  const params = new URLSearchParams({ q: $("corpus-search").value.trim(),
    quality: $("corpus-quality").value || "all", state: $("corpus-state").checked ? "all" : "converted_candidate",
    limit: "30", offset: String(offset) });
  if ($("corpus-split").value) params.set("split", $("corpus-split").value);
  $("corpus-page-status").textContent = "正在检索…";
  $("corpus-prev").disabled = true; $("corpus-next").disabled = true;
  try {
    const page = await request("/v1/animation/corpus/assets?" + params, { signal: localController.signal });
    if (generation !== corpusGeneration || library !== "corpus") return;
    corpusAssets = page.assets; corpusOffset = page.offset;
    $("corpus-page-status").textContent = "匹配 " + page.total + " 条 · 第 " + (Math.floor(page.offset / page.limit) + 1) + " 页";
    $("corpus-prev").disabled = page.offset === 0; $("corpus-next").disabled = !page.has_more;
    renderCorpus();
  } catch (error) {
    if (error.name !== "AbortError" && generation === corpusGeneration)
      $("corpus-page-status").textContent = "检索失败：" + error.message;
  } finally { if (corpusController === localController) corpusController = null; }
}
function switchLibrary(next) {
  invalidate(); library = next; selected = null; corpusSelected = null;
  corpusGeneration++; corpusController?.abort(); clearTimeout(corpusTimer);
  $("demo-library").hidden = next !== "demo"; $("corpus-library").hidden = next !== "corpus";
  $("visible-count").hidden = next !== "demo";
  $("library-demo").classList.toggle("active", next === "demo");
  $("library-corpus").classList.toggle("active", next === "corpus");
  if (next === "corpus") loadCorpusPage(0);
  else renderLibrary();
  status(next === "corpus" ? "搜索并选择完整语料句，再确认预览。" : "已返回原有动作库。");
}

async function preview() {
  const isCorpus = library === "corpus";
  const chosenCorpus = corpusSelected;
  if (isCorpus && !chosenCorpus) { status("请先从全量语料中选择完整句。", true); return; }
  invalidate();
  const token = gate.begin();
  controller = new AbortController();
  const currentController = controller;
  $("play").disabled = true;
  $("stop").disabled = false;
  status("正在匹配完整表达…");
  try {
    const plan = await request(isCorpus ? "/v1/animation/corpus/plans" : "/v1/animation/plans", {
      method: "POST", headers: { "Content-Type": "application/json" }, signal: currentController.signal,
      body: JSON.stringify({ text: $("input").value, ...(isCorpus ? { sample_id: chosenCorpus.sample_id } : { asset_id: selected }),
        confirmed: true, research_preview: true, origin }),
    });
    if (!gate.current(token)) return;
    if (!plan.can_play) {
      status(plan.reason, true);
      $("source-badge").textContent = plan.status === "ambiguous" ? "请选择动作版本" : "缺少完整表达资源";
      return;
    }
    const item = plan.items[0];
    status("正在准备动作…");
    const motionUrl = isCorpus ? "/v1/animation/corpus/motions/" + encodeURIComponent(item.sample_id) +
      "?expected_sha256=" + encodeURIComponent(item.sha256) : "/v1/animation/motions/" + encodeURIComponent(item.asset_id);
    const motion = await request(motionUrl,
      { signal: currentController.signal });
    if (!gate.current(token)) return;
    if (isCorpus && (item.sample_id !== chosenCorpus.sample_id || motion.clip?.id !== item.asset_id ||
        motion.clip?.source_id !== item.sample_id || motion.sha256 !== item.sha256))
      throw new Error("动作来源或版本已改变，请重新选择并确认。");
    const ok = await stage.loadMotion(motion, item.asset_id, () => gate.current(token), item.sha256);
    if (!ok || !gate.current(token)) return;
    activePlan = plan; selected = item.asset_id;
    const asset = isCorpus ? { ...item.asset, sha256: item.sha256 } : assets.find(a => a.id === selected);
    $("motion-title").textContent = asset.text;
    $("subtitle").textContent = asset.text;
    $("source-badge").textContent = groups[asset.group] + " · " + asset.fps + " FPS";
    const diagnostics = stage.tracks.diagnostics;
    $("provenance").textContent = "动作 " + asset.id + " · 资源 SHA256 " + asset.sha256 +
      " · 重定向 " + stage.tracks.mappedBones + " 骨骼 · 派生处理 " + stage.tracks.corrected + " 帧。" +
      (diagnostics ? "手腕目标平均偏差：" + diagnostics.wrist_error_before_mean.toFixed(5) + " → " +
        diagnostics.wrist_error_after_mean.toFixed(5) + "（模型坐标单位）；不是语言准确率。" : "") +
      "原始姿态保留；非手控语法未认证。";
    if (isCorpus) $("provenance").textContent += " 参数 SHA256 " + asset.parameter_sha256 +
      "；按需派生手势候选，原视频时间对齐未验证。";
    const pilot = stage.presentation?.pilots?.find(p => p.asset_id === asset.id);
    if(pilot?.temporal_profile) $("source-badge").textContent += " · 时序修复候选";
    $("download-motion").hidden = !pilot?.export_file;
    if (pilot?.export_file) $("download-motion").href = "/animation/presentation/" + encodeURIComponent(pilot.export_file);
    controls(true); stage.setSpeed(Number($("speed").value)); stage.play();
    status(isCorpus ? "正在预览全量语料候选。原始／修订版保持同一时刻，含义与表情待审核。" :
      pilot?.temporal_profile?"正在预览时序修复候选，请对照原始版复核手形。":"正在预览完整片段。可切换原始显示，对照同一时刻。");
    if (isCorpus) renderCorpus(); else renderLibrary();
  } catch (error) {
    if (error.name !== "AbortError" && gate.current(token)) status(String(error.message), true);
  } finally {
    if (gate.current(token)) { $("play").disabled = false; $("stop").disabled = !loaded; controller = null; }
    else if (controller === currentController || controller === null) $("play").disabled = false;
  }
}
$("input").addEventListener("input", () => {
  invalidate(); selected = null; corpusSelected = null; origin = { ...origin, kind: origin.request_id ? "edited_recognition" : "typed" };
  updateInput(); status("输入已更改，请重新确认。"); renderLibrary(); $("play").disabled = false;
});
$("comparison").addEventListener("click", () => {
  const enabled = !stage.comparison;
  stage.setComparison(enabled);
  $("comparison").setAttribute("aria-pressed", String(enabled));
  $("comparison").textContent = enabled ? "返回单画面" : "同时对照";
  document.querySelectorAll(".comparison-labels").forEach(node => { node.hidden = !enabled; });
  $("original").disabled = enabled; $("improved").disabled = enabled;
});
$("skeleton").addEventListener("change", () => stage.setSkeletonVisible($("skeleton").checked));
$("hide-caption").addEventListener("change", () => {
  document.querySelectorAll(".stage-caption").forEach(node => { node.hidden = $("hide-caption").checked; });
});
for (const id of ["face-brow","face-blink"]) $(id).addEventListener("input", () => {
  stage.setExpression({browRaise:Number($("face-brow").value),blink:Number($("face-blink").value)});updateJawControls();
});
$("face-jaw").addEventListener("input",()=>{stage.setExpression({jawOpen:Number($("face-jaw").value)});updateJawControls();});
$("face-reset").addEventListener("click", () => {resetFaceInputs();stage.setExpression({});updateJawControls();});
$("nonmanual-jaw").addEventListener("change",()=>{stage.setNonmanualEnabled($("nonmanual-jaw").checked);updateJawControls();});
$("pilot").addEventListener("change", () => {
  const asset=assets.find(a=>a.id===$("pilot").value);if(!asset)return;
  if(library!=="demo")switchLibrary("demo");
  invalidate();selected=asset.id;origin={kind:"catalog",asset_id:asset.id};$("input").value=asset.text;
  updateInput();renderLibrary();status("已选择试点完整短句，核对后确认预览。");
});
$("export-review").addEventListener("click", () => {
  if(!loaded)return;
  const blob=new Blob([JSON.stringify(stage.exportReview(),null,2)],{type:"application/json"});
  const url=URL.createObjectURL(blob),link=document.createElement("a");link.href=url;
  link.download="animation-review-"+selected+".json";link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
$("input").addEventListener("keydown", event => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") { event.preventDefault(); preview(); }
});
$("play").addEventListener("click", preview);
$("pause").addEventListener("click", () => {
  if (!loaded) return;
  if (stage.clock.playing) { stage.pause(); status("已暂停，可拖动时间轴查看手形。"); }
  else { stage.play(); status("继续预览。"); }
});
$("stop").addEventListener("click", () => { invalidate(); $("play").disabled = false; status("已停止。旧请求不会重新启动动作。"); });
$("replay").addEventListener("click", () => { if (loaded) { stage.seek(0); stage.play(); status("重新预览当前片段。"); } });
$("timeline").addEventListener("input", () => {
  if (!loaded) return;
  const requestedTime = Number($("timeline").value);
  stage.pause(); stage.seek(requestedTime); status("已定位，可继续播放。");
});
$("speed").addEventListener("change", () => stage.setSpeed(Number($("speed").value)));
for (const profile of ["original", "improved"]) $(profile).addEventListener("click", () => {
  stage.setProfile(profile);
  $("original").classList.toggle("active", profile === "original");
  $("improved").classList.toggle("active", profile === "improved");
});
document.querySelectorAll("[data-view]").forEach(button => button.addEventListener("click", () => {
  stage.setView(button.dataset.view);
  document.querySelectorAll("[data-view]").forEach(b => b.classList.toggle("active", b === button));
}));
document.querySelectorAll("[data-filter]").forEach(button => button.addEventListener("click", () => {
  filter = button.dataset.filter;
  document.querySelectorAll("[data-filter]").forEach(b => b.classList.toggle("active", b === button)); renderLibrary();
}));
$("search").addEventListener("input", renderLibrary);
$("library-demo").addEventListener("click", () => switchLibrary("demo"));
$("library-corpus").addEventListener("click", () => switchLibrary("corpus"));
$("corpus-search").addEventListener("input", () => {
  clearTimeout(corpusTimer); corpusGeneration++; corpusController?.abort();
  corpusTimer = setTimeout(() => loadCorpusPage(0), 250);
});
for (const id of ["corpus-split", "corpus-quality", "corpus-state"])
  $(id).addEventListener("change", () => loadCorpusPage(0));
$("corpus-prev").addEventListener("click", () => loadCorpusPage(Math.max(0, corpusOffset - 30)));
$("corpus-next").addEventListener("click", () => loadCorpusPage(corpusOffset + 30));
$("recognition-toggle").addEventListener("click", () => {
  $("recognition-panel").hidden = !$("recognition-panel").hidden;
  $("recognition-toggle").textContent = $("recognition-panel").hidden ? "接入识别结果" : "收起识别输入";
});
$("video").addEventListener("change", () => { invalidate(); $("recognition-status").textContent = "文件已更换，点击识别后确认结果。"; });
$("model").addEventListener("change", () => {
  invalidate(); loadedModel = null; $("recognize").disabled = true;
  $("recognition-status").textContent = "请加载当前选择的模型后识别。";
});
$("load-model").addEventListener("click", async () => {
  invalidate();
  const model = $("model").value;
  if (!model) return;
  $("load-model").disabled = true; $("model").disabled = true; $("recognize").disabled = true;
  $("recognition-status").textContent = "正在加载现有模型…";
  try {
    const response = await request("/v1/models/switch", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_id: model }) });
    loadedModel = { alias: model, generation: response.status.generation };
    $("recognize").disabled = false;
    $("recognition-status").textContent = "模型已加载，可识别文件。";
  } catch (error) { $("recognition-status").textContent = error.message; }
  finally { $("load-model").disabled = false; $("model").disabled = false; }
});
$("recognize").addEventListener("click", async () => {
  const file = $("video").files[0];
  if (!file || recognizing || !loadedModel) { $("recognition-status").textContent = "请先选择文件并加载模型。"; return; }
  invalidate(); const token = gate.begin(); recognizing = true;
  controller = new AbortController(); const localController = controller;
  $("recognize").disabled = true; $("recognition-status").textContent = "正在识别，结果不会自动播放…";
  const form = new FormData(); form.append("file", file);
  form.append("request_id", "animation-" + crypto.randomUUID());
  if (loadedModel.generation) form.append("expected_generation", loadedModel.generation);
  try {
    const body = await request("/v3/recognitions", { method: "POST", body: form, signal: localController.signal });
    if (!gate.current(token)) return;
    const result = body.result, prediction = result?.prediction;
    const text = prediction?.text || prediction?.tokens?.map(t => t.gloss).join(" ");
    if (result?.status !== "recognized" || !text) throw new Error("本次没有可确认的文本或 Gloss 输出。");
    if (String(result.sign_language).toLowerCase() !== "csl") throw new Error("当前动画库为 CSL；其他手语结果需要独立规划与动作库。");
    origin = { kind: "recognition", raw_text: text, request_id: body.request_id,
      model_id: body.model_id || result.model_id, model_generation: body.model_generation,
      source_epoch: body.source_epoch, session_id: body.session_id, task_type: result.task_type };
    $("input").value = text; selected = null; corpusSelected = null; updateInput();
    $("recognition-status").textContent = "识别完成。请核对或编辑输入，再确认预览。";
    status("识别结果尚未确认，没有自动播放。");
  } catch (error) {
    if (error.name !== "AbortError" && gate.current(token)) $("recognition-status").textContent = error.message;
  } finally {
    recognizing = false; $("recognize").disabled = !loadedModel;
    if (controller === localController) controller = null;
  }
});

async function init() {
  try {
    const catalog = await request("/v1/animation/assets");
    assets = catalog.assets; $("asset-count").textContent = assets.length + " 段资源";
    $("input").value = assets.find(a => a.text === "认识你很高兴")?.text || assets[0]?.text || "";
    // A URL fragment is a draft handoff, never an instruction to autoplay.
    if (location.hash) {
      try {
        const handoff = JSON.parse(decodeURIComponent(location.hash.slice(1)));
        if (typeof handoff.text === "string" && handoff.text.length <= 500) {
          $("input").value = handoff.text; origin = handoff.origin || { kind: "desktop_handoff" };
          $("origin-label").textContent = "桌面结果 · 待确认";
        }
      } catch { status("无法读取桌面传入内容，请手动输入。", true); }
      history.replaceState(null, "", location.pathname);
    }
    updateInput(); renderLibrary(); await stage.ready;
    if(stage.presentation?.pilots?.length){
      $("pilot-panel").hidden=false;
      for(const pilot of stage.presentation.pilots){
        const asset=assets.find(a=>a.id===pilot.asset_id);if(!asset)continue;
        const option=document.createElement("option");option.value=asset.id;option.textContent=asset.text;$("pilot").appendChild(option);
      }
    }
    status("角色已就绪。选择完整表达，确认后预览。");
    try {
      const corpus = await request("/v1/animation/corpus");
      if (corpus.enabled) {
        $("library-corpus").disabled = false;
        $("corpus-summary").textContent = "已转换 " + (corpus.counts.converted_candidate || 0) + " / " + corpus.total +
          " 段索引；仅为数值候选，动作与语言未审核。";
      }
    } catch { $("library-corpus").disabled = true; }
    try {
      const models = await request("/v1/models");
      for (const model of models.models) {
        const option = document.createElement("option"); option.value = model.model_id; option.textContent = model.model_id;
        $("model").appendChild(option);
      }
      if (models.models.length === 1 && models.active) {
        $("model").value = models.models[0].model_id;
        loadedModel = { alias: models.models[0].model_id, generation: models.active.generation };
        $("recognize").disabled = false;
        $("recognition-status").textContent = "模型已加载；选择文件后识别，结果需确认。";
      }
      if (!models.models.length) $("recognition-status").textContent = "服务未登记识别模型；可继续预览现成动作。";
    } catch { $("recognition-status").textContent = "当前服务使用独立语言路由，请从桌面确认结果后传入。"; }
  } catch (error) { status("资源加载失败：" + error.message, true); $("play").disabled = true; }
}
window.addEventListener("pagehide", () => { corpusGeneration++; corpusController?.abort(); clearTimeout(corpusTimer); invalidate(); stage.dispose(); });
init();
