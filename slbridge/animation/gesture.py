"""Reversible gesture candidates: short isolated SO(3) islands + bounded filtering."""
import numpy as np

GESTURE_PROFILE='soke-gesture-refinement-v1'
MAPPED_JOINTS=np.r_[np.arange(12,22),np.arange(25,55)]


def _validated(poses,fps):
    source=np.asarray(poses,dtype=np.float64)
    if source.ndim!=3 or source.shape[1:]!=(55,3) or not len(source) or not np.isfinite(source).all() or not np.isfinite(fps) or not 0<fps<=120:
        raise ValueError('Expected finite T x 55 x 3 poses and valid FPS')
    return source


def _quaternions(poses):
    angles=np.linalg.norm(poses,axis=-1,keepdims=True)
    axes=np.divide(poses,angles,out=np.zeros_like(poses),where=angles>1e-12)
    return np.concatenate([axes*np.sin(angles/2),np.cos(angles/2)],axis=-1)


def _angles(a,b):
    return np.degrees(2*np.arccos(np.clip(np.abs(np.sum(a*b,axis=-1)),0,1)))


def _slerp(a,b,amount):
    cosine=np.sum(a*b,axis=-1,keepdims=True)
    target=np.where(cosine<0,-b,b)
    theta=np.arccos(np.clip(np.abs(cosine),0,1));sine=np.sin(theta)
    t=np.asarray(amount,dtype=np.float64)
    if t.shape==a.shape[:-1]:t=t[...,None]
    safe=np.where(sine>1e-9,sine,1)
    curved=a*np.sin((1-t)*theta)/safe+target*np.sin(t*theta)/safe
    result=np.where(sine>1e-9,curved,a*(1-t)+target*t)
    return result/np.linalg.norm(result,axis=-1,keepdims=True)


def _rotvec(q):
    q=np.where(q[...,3:4]<0,-q,q)
    length=np.linalg.norm(q[...,:3],axis=-1,keepdims=True)
    angle=2*np.arctan2(length,q[...,3:4])
    return np.divide(q[...,:3]*angle,length,out=np.zeros_like(q[...,:3]),where=length>1e-12)


def motion_metrics(poses,fps):
    source=_validated(poses,fps);q=_quaternions(source)
    steps=_angles(q[1:],q[:-1])
    selected=steps[:,MAPPED_JOINTS]
    return {'mapped_max_step_degrees':float(selected.max()) if selected.size else 0.0,
            'body_max_step_degrees':float(steps[:,12:22].max()) if steps.size else 0.0,
            'hand_max_step_degrees':float(steps[:,25:].max()) if steps.size else 0.0,
            'mapped_step_p95_degrees':float(np.percentile(selected,95)) if selected.size else 0.0,
            'mapped_step_events_over30':int((selected>30).sum()),
            'mapped_step_events_over60':int((selected>60).sum())}


def refine_gestures(poses,fps):
    source=_validated(poses,fps);original=_quaternions(source);repaired=original.copy()
    count=len(source);mask=np.zeros((count,55),dtype=bool);events=[]
    bridge_limit=np.full(55,30.0);bridge_limit[12:22]=60.0
    threshold=np.full(55,45.0);threshold[12:22]=60.0
    allowed=np.zeros(55,dtype=bool);allowed[MAPPED_JOINTS]=True
    for span in (1,2):
        if span/fps>0.070 or count<span+4:continue
        left=original[1:count-span-2];right=original[span+2:count-1]
        bridge=_angles(left,right)
        entry=_angles(left,original[2:count-span-1])
        exit=_angles(original[span+1:count-2],right)
        stable=(_angles(original[:count-span-3],left)<20)&(_angles(right,original[span+3:])<20)
        residual=np.full(bridge.shape,np.inf)
        for offset in range(span):
            prediction=_slerp(left,right,(offset+1)/(span+1))
            residual=np.minimum(residual,_angles(original[2+offset:count-span-1+offset],prediction))
        candidates=(bridge<=bridge_limit)&(entry>threshold)&(exit>threshold)&stable&(residual>threshold)&allowed
        for index,joint in np.argwhere(candidates):
            frame=int(index)+2;joint=int(joint)
            if mask[frame-1:frame+span+1,joint].any():continue
            maximum=0.0
            for offset in range(span):
                replacement=_slerp(original[frame-1,joint],original[frame+span,joint],(offset+1)/(span+1))
                maximum=max(maximum,float(_angles(original[frame+offset,joint],replacement)))
                repaired[frame+offset,joint]=replacement;mask[frame+offset,joint]=True
            events.append({'frame_start':frame,'frames':span,'joint':joint,'time_seconds':frame/fps,
                           'maximum_correction_degrees':maximum,'anchor_bridge_degrees':float(bridge[index,joint]),
                           'minimum_residual_degrees':float(residual[index,joint])})
    filtered=repaired.copy()
    if count>2:
        middle=_slerp(repaired[:-2],repaired[2:],0.5)
        candidate=_slerp(repaired[1:-1],middle,0.25)
        correction=_angles(repaired[1:-1],candidate)
        limits=np.full(55,3.0);limits[12:22]=5.0
        amount=np.minimum(1,np.divide(limits,correction,out=np.ones_like(correction),where=correction>1e-10))
        filtered[1:-1]=_slerp(repaired[1:-1],candidate,amount)
        # Unclassified large transitions need source review, not an invented bridge.
        # Freeze both endpoints so smoothing cannot enlarge a near-antipodal step.
        large=_angles(repaired[1:],repaired[:-1])>30
        frozen=np.zeros((count,55),dtype=bool);frozen[:-1]|=large;frozen[1:]|=large
        filtered[frozen]=repaired[frozen]
    result=source.copy()
    result[1:-1,MAPPED_JOINTS]=_rotvec(filtered[1:-1,MAPPED_JOINTS])
    differences=_angles(original,_quaternions(result))
    def regular_max(joints):
        values=np.where(~mask[:,joints],differences[:,joints],0)
        return float(values.max()) if values.size else 0.0
    finger=regular_max(np.arange(25,55));body=regular_max(np.arange(12,22))
    if finger>3.00001 or body>5.00001:raise ValueError('Unmarked correction exceeded gesture limits')
    audit={'profile':GESTURE_PROFILE,'isolated_repair_events':events,'isolated_repaired_joint_frames':int(mask.sum()),
           'regular_correction_max_degrees':max(finger,body),'regular_finger_correction_max_degrees':finger,
           'regular_body_correction_max_degrees':body,'maximum_correction_degrees':float(differences[:,MAPPED_JOINTS].max()),
           'before':motion_metrics(source,fps),'after':motion_metrics(result,fps),
           'endpoints_preserved':True,'language_review':'NOT_RUN','motion_review':'NOT_RUN'}
    return result,audit,mask
