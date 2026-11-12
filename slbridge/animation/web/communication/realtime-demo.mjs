// Display-only catalog metadata. This module never contacts recognition/LLM.
export function demoItems(data){
  if(data?.enabled===false)return [];
  if(!data||data.schema_version!==1||data.enabled!==true||data.sign_language!=='csl'||
      typeof data.catalog_sha256!=='string'||!/^[a-f0-9]{64}$/.test(data.catalog_sha256)||!Array.isArray(data.items)||data.items.length<1||data.items.length>16)
    throw new Error('示范目录暂不可用。');
  const seen=new Set();
  return data.items.map(item=>{
    if(!item||typeof item.id!=='string'||!(/^[a-z0-9][a-z0-9_-]{0,63}$/).test(item.id)||seen.has(item.id)||
        typeof item.sha256!=='string'||!/^[a-f0-9]{64}$/.test(item.sha256)||item.source_split!=='train'||
        !['training_replay','diagnostic_replay'].includes(item.kind)||
        !Number.isFinite(item.fps)||item.fps<10||item.fps>30||!Number.isInteger(item.frames)||item.frames<1||item.frames>450||
        !Number.isFinite(item.duration_seconds)||item.duration_seconds<=0||item.duration_seconds>15||Math.abs(item.duration_seconds-item.frames/item.fps)>.001||
        ![item.width,item.height].every(n=>Number.isInteger(n)&&n>=16&&n<=512)||
        ![['title',80],['annotation',200],['note',300]].every(([k,limit])=>typeof item[k]==='string'&&item[k].trim()&&item[k].length<=limit)||
        item.media_url!==`/v1/communication/demos/${item.id}/video?sha256=${item.sha256}`)
      throw new Error('示范目录内容无效，请重新加载服务。');
    seen.add(item.id);
    return {id:item.id,title:item.title,annotation:item.annotation,note:item.note,kind:item.kind,
      media_url:item.media_url,fps:item.fps,duration_seconds:item.duration_seconds};
  });
}
