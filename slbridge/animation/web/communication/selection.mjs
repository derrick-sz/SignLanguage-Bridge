const positive=value=>Number.isFinite(value)&&value>0;
export const fullRegion=()=>({x:0,y:0,width:1,height:1});

export function videoRect(sourceWidth,sourceHeight,boxWidth,boxHeight){
  if(![sourceWidth,sourceHeight,boxWidth,boxHeight].every(positive))throw new RangeError('视频尺寸尚未就绪。');
  const scale=Math.min(boxWidth/sourceWidth,boxHeight/sourceHeight);
  const width=sourceWidth*scale,height=sourceHeight*scale;
  return {x:(boxWidth-width)/2,y:(boxHeight-height)/2,width,height};
}

export function selectRegion(start,end,sourceWidth,sourceHeight,boxWidth,boxHeight){
  const frame=videoRect(sourceWidth,sourceHeight,boxWidth,boxHeight);
  if(![start?.x,start?.y,end?.x,end?.y].every(Number.isFinite))return null;
  if(start.x<frame.x||start.x>frame.x+frame.width||start.y<frame.y||start.y>frame.y+frame.height)return null;
  const clamp=(v,low,high)=>Math.max(low,Math.min(high,v));
  const ex=clamp(end.x,frame.x,frame.x+frame.width),ey=clamp(end.y,frame.y,frame.y+frame.height);
  if(Math.abs(ex-start.x)<12||Math.abs(ey-start.y)<12)return null;
  const region={x:(Math.min(start.x,ex)-frame.x)/frame.width,y:(Math.min(start.y,ey)-frame.y)/frame.height,
    width:Math.abs(ex-start.x)/frame.width,height:Math.abs(ey-start.y)/frame.height};
  const crop=cropRect(region,sourceWidth,sourceHeight);
  return crop.width<32||crop.height<32?null:region;
}

export function cropRect(region,width,height){
  if(!region||![width,height].every(positive)||![region.x,region.y,region.width,region.height].every(Number.isFinite)
      ||region.x<0||region.y<0||region.width<=0||region.height<=0||region.x+region.width>1.0000001||region.y+region.height>1.0000001){
    throw new RangeError('请重新框选有效的视频区域。');
  }
  const x=Math.floor(region.x*width+1e-6),y=Math.floor(region.y*height+1e-6);
  const right=Math.min(width,Math.ceil((region.x+region.width)*width-1e-6));
  const bottom=Math.min(height,Math.ceil((region.y+region.height)*height-1e-6));
  if(right-x<2||bottom-y<2)throw new RangeError('选区太小，请包含双手和上半身。');
  return {x,y,width:right-x,height:bottom-y};
}

export function encodeSize(crop){
  if(![crop?.width,crop?.height].every(positive))throw new RangeError('无效选区尺寸。');
  const scale=Math.min(1,960/crop.width,720/crop.height);
  return {width:Math.max(2,Math.floor(crop.width*scale/2)*2),height:Math.max(2,Math.floor(crop.height*scale/2)*2)};
}
