from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
def stack_features(series: Dict[str, np.ndarray], order: List[str]) -> np.ndarray:
    mats=[series[k].reshape(-1,1) for k in order if k in series]
    if not mats:
        T=len(next(iter(series.values()))); return np.zeros((T,0))
    return np.concatenate(mats,1)
def align_by_dtw(user_feats: np.ndarray, ref_feats: np.ndarray):
    _, path = fastdtw(user_feats, ref_feats, dist=lambda a,b: euclidean(a,b))
    user_aligned=[user_feats[i] for (i,_) in path]
    ref_aligned=[ref_feats[j] for (_,j) in path]
    return np.vstack(user_aligned), np.vstack(ref_aligned), path
