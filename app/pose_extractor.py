from __future__ import annotations
import cv2, numpy as np
from dataclasses import dataclass
from typing import List, Tuple
import mediapipe as mp
mp_pose = mp.solutions.pose
@dataclass
class PoseSequence:
    fps: float
    frame_times: np.ndarray
    landmarks: List[np.ndarray]
def get_video_frames(path: str, max_width: int = 960) -> Tuple[float, list]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened(): raise FileNotFoundError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok: break
        if frame.shape[1] > max_width:
            scale = max_width / frame.shape[1]
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release(); return fps, frames
def extract_pose_from_video(path: str, model_complexity: int = 1, smooth_landmarks: bool = True) -> PoseSequence:
    fps, frames = get_video_frames(path)
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=model_complexity,
                        enable_segmentation=False, smooth_landmarks=smooth_landmarks)
    results = []
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            arr = np.zeros((33,3), np.float32)
            for i,p in enumerate(lm): arr[i]=[p.x,p.y,getattr(p,'visibility',1.0)]
        else:
            arr = np.full((33,3), np.nan, np.float32)
        results.append(arr)
    pose.close()
    T=len(results); frame_times=np.arange(T)/float(fps)
    return PoseSequence(fps=fps, frame_times=frame_times, landmarks=results)
