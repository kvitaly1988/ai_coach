from __future__ import annotations
import numpy as np
from scipy.signal import savgol_filter
from typing import List
LS=11; RS=12; LE=13; RE=14; LW=15; RW=16; LH=23; RH=24; LK=25; RK=26; LA=27; RA=28; LFI=31; RFI=32
def _angle(a,b,c):
    ba=a-b; bc=c-b
    if np.linalg.norm(ba)<1e-6 or np.linalg.norm(bc)<1e-6: return np.nan
    ba=ba/(np.linalg.norm(ba)+1e-9); bc=bc/(np.linalg.norm(bc)+1e-9)
    return np.degrees(np.arccos(np.clip(np.dot(ba,bc),-1.0,1.0)))
def normalize_landmarks(seq: List[np.ndarray]) -> List[np.ndarray]:
    out=[]
    for arr in seq:
        if np.isnan(arr).all(): out.append(arr.copy()); continue
        pts=arr[:,:2]
        shoulders=[pts[LS],pts[RS]]; hips=[pts[LH],pts[RH]]
        shoulders=[p for p in shoulders if not np.isnan(p).any()]
        hips=[p for p in hips if not np.isnan(p).any()]
        if not shoulders or not hips: out.append(arr.copy()); continue
        sh_c=np.mean(np.vstack(shoulders),0); hip_c=np.mean(np.vstack(hips),0)
        torso_len=np.linalg.norm(sh_c-hip_c) or 1.0
        tmp=arr.copy(); tmp[:,:2]=(tmp[:,:2]-hip_c)/torso_len; out.append(tmp)
    return out
def compute_angles_per_frame(lm: np.ndarray)->dict:
    a={}; pts=lm[:,:2]; ok=lambda p: not np.isnan(p).any()
    a['elbow_left']  = _angle(pts[LS],pts[LE],pts[LW])   if ok(pts[LS]) and ok(pts[LE]) and ok(pts[LW]) else np.nan
    a['elbow_right'] = _angle(pts[RS],pts[RE],pts[RW])   if ok(pts[RS]) and ok(pts[RE]) and ok(pts[RW]) else np.nan
    a['shoulder_left']  = _angle(pts[LE],pts[LS],pts[LH]) if ok(pts[LE]) and ok(pts[LS]) and ok(pts[LH]) else np.nan
    a['shoulder_right'] = _angle(pts[RE],pts[RS],pts[RH]) if ok(pts[RE]) and ok(pts[RS]) and ok(pts[RH]) else np.nan
    a['hip_left']  = _angle(pts[LS],pts[LH],pts[LK]) if ok(pts[LS]) and ok(pts[LH]) and ok(pts[LK]) else np.nan
    a['hip_right'] = _angle(pts[RS],pts[RH],pts[RK]) if ok(pts[RS]) and ok(pts[RH]) and ok(pts[RK]) else np.nan
    a['knee_left']  = _angle(pts[LH],pts[LK],pts[LA]) if ok(pts[LH]) and ok(pts[LK]) and ok(pts[LA]) else np.nan
    a['knee_right'] = _angle(pts[RH],pts[RK],pts[RA]) if ok(pts[RH]) and ok(pts[RK]) and ok(pts[RA]) else np.nan
    a['ankle_left']  = _angle(pts[LK],pts[LA],pts[LFI]) if ok(pts[LK]) and ok(pts[LA]) and ok(pts[LFI]) else np.nan
    a['ankle_right'] = _angle(pts[RK],pts[RA],pts[RFI]) if ok(pts[RK]) and ok(pts[RA]) and ok(pts[RFI]) else np.nan
    shoulders=[pts[LS],pts[RS]]; shoulders=[p for p in shoulders if ok(p)]
    hips=[pts[LH],pts[RH]]; hips=[p for p in hips if ok(p)]
    if shoulders and hips:
        sh_c=np.mean(np.vstack(shoulders),0); hip_c=np.mean(np.vstack(hips),0)
        v=sh_c-hip_c; vert=np.array([0.0,-1.0])
        if np.linalg.norm(v)<1e-6: a['torso']=np.nan
        else: a['torso']=np.degrees(np.arccos(np.clip(np.dot(v/ (np.linalg.norm(v)+1e-9),vert),-1.0,1.0)))
    else: a['torso']=np.nan
    return a
def compute_angles_sequence(norm_landmarks: List[np.ndarray]) -> dict:
    s={}
    for lm in norm_landmarks:
        ang=compute_angles_per_frame(lm)
        for k,v in ang.items(): s.setdefault(k,[]).append(v)
    for k in s.keys(): s[k]=np.array(s[k],np.float32)
    return s
def smooth_series(series: dict, window:int=11, poly:int=3)->dict:
    out={}
    for k,v in series.items():
        x=v.copy(); nans=np.isnan(x)
        if nans.any():
            idx=np.arange(len(x)); x[nans]=np.interp(idx[nans], idx[~nans], x[~nans])
        if len(x)>=max(window, poly+2):
            try: x=savgol_filter(x, window_length=window, polyorder=poly, mode='interp')
            except Exception: pass
        out[k]=x.astype(np.float32)
    return out
