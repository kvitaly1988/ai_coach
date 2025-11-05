from __future__ import annotations
import numpy as np
def tempo_error_from_path(path, fps_user: float, fps_ref: float) -> float:
    if not path: return 0.0
    diffs=[abs(i/fps_user - j/fps_ref) for (i,j) in path]
    return float(np.mean(diffs))
