from __future__ import annotations
import numpy as np
def compute_score(angle_mae_deg, tempo_error_s, smoothness_val, joint_weights=None, alpha=0.8,beta=10.0,gamma=0.02):
    if joint_weights is None: joint_weights={k:1.0 for k in angle_mae_deg.keys()}
    wsum=sum(joint_weights.get(k,1.0) for k in angle_mae_deg.keys()) or 1.0
    E=sum(angle_mae_deg[k]*joint_weights.get(k,1.0) for k in angle_mae_deg.keys())/wsum
    raw=100.0 - alpha*E - beta*tempo_error_s - gamma*smoothness_val
    return float(np.clip(raw,0.0,100.0))
