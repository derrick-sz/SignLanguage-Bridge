export function renderSource(node, asset) { node.textContent = asset.text; }

export function appendExpression(input, text) {
  const next = input.trim() ? input.trim() + '\n' + text : text;
  if (next.length > 500) throw new Error('表达最多 500 个字符，请先删减。');
  // Full matching and authoritative boundary validation remain on the server.
  if (next.split(/[。！？!?；;\n]+/u).filter(s => s.trim()).length > 6) {
    throw new Error('最多加入 6 句，请先删减表达。');
  }
  return next;
}

export class ScenePreview {
  constructor({stage, fetcher = globalThis.fetch, onState = () => {}}) {
    this.stage = stage; this.fetcher = fetcher.bind(globalThis); this.onState = onState;
    this.generation = 0; this.abort = null; this.plan = null;
  }
  invalidate() {
    this.generation++; this.abort?.abort(); this.abort = null; this.plan = null;
    this.stage.stop(); this.onState({status:'idle'});
  }
  async preview(text, confirmed) {
    this.invalidate();
    if (text.length > 500) {
      this.onState({status:'error',message:'表达最多 500 个字符，请先删减。'}); return;
    }
    if (!text.trim() || !confirmed) {
      this.onState({status:'error',message:'请输入表达，并勾选研究预览确认。'}); return;
    }
    const generation = this.generation;
    const controller = new AbortController(); this.abort = controller;
    const current = () => generation === this.generation && !controller.signal.aborted;
    const request = async (url, options={}) => {
      const response = await this.fetcher(url,{...options,signal:controller.signal});
      const data = await response.json();
      if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : data.error?.message || '请求失败，请重试。');
      return data;
    };
    this.onState({status:'loading',message:'正在核对完整原句…'});
    try {
      const plan = await request('/v1/animation/scenes/plans', {
        method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({text,confirmed:true,research_preview:true})
      });
      if (!current()) return;
      this.plan = plan;
      if (!plan.can_play || plan.unmatched?.length) {
        this.onState({status:'blocked',plan,message:plan.reason || '表达尚未全部覆盖，无法预览。'}); return;
      }
      this.onState({status:'loading',plan,message:'原句匹配完成，正在加载动作…'});
      const data = await request('/v1/animation/scenes/plans/' + encodeURIComponent(plan.plan_id) + '/motion');
      if (!current()) return;
      const loaded = await this.stage.loadMotion(data,data.clip.id,current,data.sha256);
      if (!current() || loaded === false) return;
      this.stage.play(); this.onState({status:'playing',plan,message:'研究预览播放中；语言与衔接自然度尚未验收。'});
    } catch (error) {
      if (current()) this.onState({status:'error',plan:this.plan,message:error.message});
    }
  }
}
