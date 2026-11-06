// Geometrically calibrated display candidate, not linguistic mouthing or lip sync.
export const JAW_PROFILE = 'smplx-jaw-godette-calibrated-v1';
export const JAW_TEMPLATE_SHA256 = '376021446ddc86e99acacd795182bbef903e61d33b76b9d8b359c2b0865bd992';
export const JAW_AVATAR_SHA256 = 'b42c21dacb5d07617790c40c69f68a074aca58d74d76b898f2cb98bbbf96862c';
export const MAX_JAW_ANGLE = 0.16;

export function jawCandidateForClip(clip,avatarSha,baked=false){
  const p=clip?.nonmanual_candidate;
  if(baked || !p || p.profile!==JAW_PROFILE || p.source_joint!==22 ||
      p.source_template_sha256!==JAW_TEMPLATE_SHA256 || clip.template_sha256!==JAW_TEMPLATE_SHA256 ||
      p.target_avatar_sha256!==JAW_AVATAR_SHA256 || avatarSha!==JAW_AVATAR_SHA256 ||
      p.status!=='research_candidate' || p.export_supported!==false)return null;
  return compileJawCandidate(clip.poses,clip.fps,p.profile);
}

export function compileJawCandidate(poses, fps, profile) {
  if(profile!==JAW_PROFILE || !Number.isFinite(fps) || fps<=0 || fps>120 ||
      !Array.isArray(poses) || !poses.length || poses.length>3000)throw new Error('Invalid jaw candidate input');
  const diagnostics={rejected_frames:0,clamped_frames:0,source_joint:22,max_angle_radians:MAX_JAW_ANGLE,max_speed_radians_per_second:0.6};
  const bounded=poses.map(frame=>{
    if(!Array.isArray(frame?.[22]) || frame[22].length!==3 || !frame[22].every(Number.isFinite))throw new Error('Invalid jaw rotation');
    const [x,y,z]=frame[22],angle=Math.hypot(x,y,z);
    if(angle>0.35){diagnostics.rejected_frames++;return 0;}
    const twist=angle<1e-12?0:2*Math.atan2(Math.sin(angle/2)*x/angle,Math.cos(angle/2));
    if(twist<0 || twist>MAX_JAW_ANGLE)diagnostics.clamped_frames++;
    return Math.max(0,Math.min(MAX_JAW_ANGLE,twist));
  });
  const median=bounded.map((value,i)=>[bounded[Math.max(0,i-1)],value,bounded[Math.min(bounded.length-1,i+1)]].sort((a,b)=>a-b)[1]);
  const angles=[];
  for(const value of median){
    const previous=angles.at(-1);
    angles.push(previous===undefined?value:Math.max(previous-0.6/fps,Math.min(previous+0.6/fps,value)));
  }
  return {profile,fps,angles,diagnostics};
}

export function sampleJawCandidate(track,time){
  if(!Number.isFinite(time))throw new Error('Invalid jaw sample time');
  if(!track)return 0;
  const at=Math.max(0,Math.min(track.angles.length-1,time*track.fps)),from=Math.floor(at),to=Math.min(from+1,track.angles.length-1);
  return track.angles[from]+(track.angles[to]-track.angles[from])*(at-from);
}
