# -*- coding: utf-8 -*-
import os
import json
import tempfile
from pathlib import Path

import numpy as np
import cv2
import streamlit as st

from app.pose_extractor import extract_pose_from_video, get_video_frames
from app.preprocessing import normalize_landmarks, compute_angles_sequence, smooth_series
from app.dtw_utils import stack_features, align_by_dtw
from app.metrics import tempo_error_from_path
from app.scoring import compute_score
from app.coach import generate_tips
from app.visualization import (
    plot_angle_series,
    draw_skeleton,
    make_side_by_side,
    draw_joint_overlay,
    JOINT_NAMES_RU,
)

# ---------------------------- НАСТРОЙКИ UI ----------------------------
st.set_page_config(page_title="AI-коуч: оценка техники", layout="wide")
st.title("🤸 AI-коуч: оценка качества выполнения по видео")

@st.cache_resource
def load_config():
    with open("app/elements_config.json", "r", encoding="utf-8") as f:
        return json.load(f)

config = load_config()
elements = list(config.keys())
el = st.selectbox("Выберите элемент", elements, format_func=lambda k: config[k]["title"])

st.markdown("Загрузите короткое видео (5–15 сек, 25–30 FPS) и нажмите **Анализировать**.")

# ---------------------------- СЛУЖЕБНЫЕ ФУНКЦИИ ----------------------------
def _md5_bytes(b: bytes) -> str:
    import hashlib
    h = hashlib.md5()
    h.update(b)
    return h.hexdigest()

if "user_video_path" not in st.session_state:
    st.session_state.user_video_path = None
if "user_video_hash" not in st.session_state:
    st.session_state.user_video_hash = None
if "analysis" not in st.session_state:
    st.session_state.analysis = None

def _save_upload_to_tmp(uploaded_file):
    data = uploaded_file.read()
    h = _md5_bytes(data)
    tmp_dir = Path(tempfile.gettempdir()) / "ai_coach_uploads"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_path = tmp_dir / f"{h}.mp4"
    if not out_path.exists():
        with open(out_path, "wb") as f:
            f.write(data)
    st.session_state.user_video_path = str(out_path)
    st.session_state.user_video_hash = h
    return st.session_state.user_video_path, h

@st.cache_data(show_spinner=False)
def run_analysis_cached(user_video_hash, user_video_path, cfg_json, ref_mtime):
    cfg = json.loads(cfg_json)
    ref_path = cfg["reference_video"]

    user_seq = extract_pose_from_video(user_video_path)
    ref_seq = extract_pose_from_video(ref_path)

    user_norm = normalize_landmarks(user_seq.landmarks)
    ref_norm = normalize_landmarks(ref_seq.landmarks)
    user_angles = smooth_series(compute_angles_sequence(user_norm), 11, 3)
    ref_angles = smooth_series(compute_angles_sequence(ref_norm), 11, 3)

    feat_keys = [k for k in user_angles.keys() if any(s in k for s in ["torso", "hip", "knee"])] or list(user_angles.keys())[:6]
    user_feats = stack_features(user_angles, feat_keys)
    ref_feats = stack_features(ref_angles, feat_keys)

    _, _, path = align_by_dtw(user_feats, ref_feats)
    idx_user = [p[0] for p in path]
    idx_ref = [p[1] for p in path]

    angle_mae = {}
    for k in set(user_angles.keys()).intersection(ref_angles.keys()):
        angle_mae[k] = float(np.nanmean(np.abs(user_angles[k][idx_user] - ref_angles[k][idx_ref])))

    tempo_err = tempo_error_from_path(path, user_seq.fps, ref_seq.fps)
    smooth_val = float(sum(np.sum(np.abs(np.diff(user_angles[k]))) for k in user_angles.keys()))

    return {
        "user_fps": user_seq.fps,
        "ref_fps": ref_seq.fps,
        "user_landmarks_raw": user_seq.landmarks,
        "ref_landmarks_raw": ref_seq.landmarks,
        "user_angles": user_angles,
        "ref_angles": ref_angles,
        "idx_user": idx_user,
        "idx_ref": idx_ref,
        "path": path,
        "angle_mae": angle_mae,
        "tempo_err": tempo_err,
        "smooth_val": smooth_val,
    }

@st.cache_data(show_spinner=False)
def load_frames_cached(video_path: str):
    return get_video_frames(video_path)

# ---------------------------- ЗАГРУЗКА ВИДЕО ----------------------------
user_file = st.file_uploader("Видео (mp4/avi/mov/mkv)", type=["mp4", "avi", "mov", "mkv"])
analyze_clicked = st.button("Анализировать", type="primary")
colL, colR = st.columns([1, 1])

if analyze_clicked:
    if not user_file:
        st.error("Пожалуйста, загрузите видео.")
        st.stop()
    user_path, user_hash = _save_upload_to_tmp(user_file)
    ref_path = config[el]["reference_video"]
    if not os.path.exists(ref_path):
        st.error(f"Не найден эталон: {ref_path}")
        st.stop()
    ref_mtime = os.path.getmtime(ref_path)
    with st.spinner("Выполняется анализ..."):
        st.session_state.analysis = run_analysis_cached(
            user_hash, user_path, json.dumps(config[el], ensure_ascii=False), ref_mtime
        )

if st.session_state.analysis is None:
    st.info("Загрузите видео и нажмите «Анализировать».")
    st.stop()

A = st.session_state.analysis

# ---------------------------- РЕЗУЛЬТАТЫ ----------------------------
important = config[el].get("important_joints", {})
score = compute_score(A["angle_mae"], A["tempo_err"], A["smooth_val"], joint_weights=important)

with colL:
    st.subheader("Итоговая оценка")
    st.metric("Качество техники", f"{score:.1f} / 100")
    st.caption("Составляющие:")
    st.write(f"Средняя ошибка углов: **{np.mean(list(A['angle_mae'].values())):.1f}°**")
    st.write(f"Рассинхрон темпа: **{A['tempo_err']:.2f} с**")

with colR:
    st.subheader("Графики углов (выравненные)")
    idx_user = A["idx_user"]
    idx_ref = A["idx_ref"]
    figs = plot_angle_series(
        {k: np.array(v)[idx_user] for k, v in A["user_angles"].items()},
        {k: np.array(v)[idx_ref] for k, v in A["ref_angles"].items()},
        sorted([k for k in A["angle_mae"].keys() if any(s in k for s in ["torso", "hip", "knee", "shoulder"])])[:6],
    )
    for fig in figs:
        st.pyplot(fig)

# ---------------------------- СОВЕТЫ + СТОП-КАДРЫ ----------------------------
st.markdown("---")
st.subheader("Советы по улучшению и стоп-кадры")

thresholds = config[el].get("tips_thresholds_deg", None)
tips = generate_tips(A["angle_mae"], thresholds_deg=thresholds)

aligned_user = {k: np.array(v)[A["idx_user"]] for k, v in A["user_angles"].items()}
aligned_ref = {k: np.array(v)[A["idx_ref"]] for k, v in A["ref_angles"].items()}

user_path = st.session_state.user_video_path
ref_path = config[el]["reference_video"]
_, user_frames = load_frames_cached(user_path)
_, ref_frames = load_frames_cached(ref_path)

worst_idx_per_joint = {}
for k in aligned_user.keys():
    if k in aligned_ref:
        err = np.abs(aligned_user[k] - aligned_ref[k]).astype(np.float32)
        err = np.nan_to_num(err, nan=0.0, posinf=0.0, neginf=0.0)
        worst_idx_per_joint[k] = int(np.argmax(err))

ru2key = {
    "левый локоть": "elbow_left",
    "правый локоть": "elbow_right",
    "левое плечо": "shoulder_left",
    "правое плечо": "shoulder_right",
    "левое бедро": "hip_left",
    "правое бедро": "hip_right",
    "левое колено": "knee_left",
    "правое колено": "knee_right",
    "левая лодыжка": "ankle_left",
    "правая лодыжка": "ankle_right",
    "корпус (наклон)": "torso",
    "корпус": "torso",
    "наклон": "torso",
}

default_joint = max(A["angle_mae"].items(), key=lambda kv: kv[1])[0] if A["angle_mae"] else None

for t in tips:
    st.write("• " + t)
    jkey = None
    low = t.lower()
    for phrase, key in ru2key.items():
        if phrase in low:
            jkey = key
            break
    if jkey is None:
        jkey = default_joint

    if jkey in worst_idx_per_joint and len(A["idx_user"]) > 0:
        idx = worst_idx_per_joint[jkey]
        fu = A["idx_user"][idx]
        fr = A["idx_ref"][idx]
        fu = max(0, min(fu, len(user_frames) - 1))
        fr = max(0, min(fr, len(ref_frames) - 1))
        lm_u = A["user_landmarks_raw"][fu]
        lm_r = A["ref_landmarks_raw"][fr]
        uf = draw_joint_overlay(user_frames[fu].copy(), lm_u, lm_r, jkey)
        st.image(
            cv2.cvtColor(uf, cv2.COLOR_BGR2RGB),
            caption=f"Стоп-кадр: {JOINT_NAMES_RU.get(jkey, jkey)} — красная точка: вы; зелёный крест: эталон",
        )

# ---------------------------- ПОКАДРОВОЕ СРАВНЕНИЕ ----------------------------
st.markdown("---")
st.subheader("Покадровое сравнение (с выравниванием)")
error_thresh = st.slider("Порог несоответствия, °", 0.0, 30.0, 12.0, 0.5)
show_only_bad = st.checkbox("Показывать только проблемные кадры", value=False)

aligned_len = len(A["idx_user"])
per_frame_err = np.zeros(aligned_len, float)
keys = list(A["angle_mae"].keys())
for i in range(aligned_len):
    s = 0.0
    c = 0
    for k in keys:
        u = A["user_angles"][k][A["idx_user"]][i]
        r = A["ref_angles"][k][A["idx_ref"]][i]
        if not (np.isnan(u) or np.isnan(r)):
            s += abs(u - r)
            c += 1
    per_frame_err[i] = (s / c) if c else np.nan
per_frame_err = np.nan_to_num(per_frame_err, nan=0.0)

frame_candidates = [i for i in range(aligned_len) if (not show_only_bad) or (per_frame_err[i] >= error_thresh)]
if not frame_candidates:
    st.success("Нет кадров, превышающих порог — техника близка к эталону.")
else:
    i = st.slider("Кадр (выравнивание)", 0, len(frame_candidates) - 1, 0, 1)
    idx = frame_candidates[i]
    st.write(f"Средняя ошибка: **{per_frame_err[idx]:.1f}°** (порог {error_thresh:.1f}°)")
    fu = A["idx_user"][idx]
    fr = A["idx_ref"][idx]
    fu = max(0, min(fu, len(user_frames) - 1))
    fr = max(0, min(fr, len(ref_frames) - 1))
    uf = draw_skeleton(user_frames[fu].copy(), A["user_landmarks_raw"][fu])
    rf = draw_skeleton(ref_frames[fr].copy(), A["ref_landmarks_raw"][fr])
    combo = make_side_by_side(uf, rf, per_frame_err[idx] >= error_thresh, per_frame_err[idx])
    st.image(cv2.cvtColor(combo, cv2.COLOR_BGR2RGB), caption="Пользователь (слева) vs Эталон (справа)")

# ---------------------------- ЭКСПОРТ ВИДЕО-ОТЧЁТА ----------------------------
st.markdown("---")
st.subheader("Экспорт видео-отчёта")

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    report_thresh = st.slider("Порог включения кадра, °", 5.0, 30.0, float(max(12.0, error_thresh)), 0.5)
with col2:
    max_frames = st.number_input("Макс. число кадров", min_value=20, max_value=3000, value=300, step=10)
with col3:
    fps_out = st.number_input("FPS отчёта", min_value=5, max_value=60, value=25, step=1)

def _build_report_frames(selected_idx, per_frame_err, border_thresh):
    frames_out = []
    if not selected_idx:
        return frames_out
    idx0 = selected_idx[0]
    fu0 = A["idx_user"][idx0]
    fr0 = A["idx_ref"][idx0]
    fu0 = max(0, min(fu0, len(user_frames) - 1))
    fr0 = max(0, min(fr0, len(ref_frames) - 1))
    uf0 = draw_skeleton(user_frames[fu0].copy(), A["user_landmarks_raw"][fu0])
    rf0 = draw_skeleton(ref_frames[fr0].copy(), A["ref_landmarks_raw"][fr0])
    combo0 = make_side_by_side(uf0, rf0, per_frame_err[idx0] >= border_thresh, per_frame_err[idx0])
    H, W = combo0.shape[:2]
    frames_out.append(combo0)
    for idx in selected_idx[1:]:
        fu = A["idx_user"][idx]
        fr = A["idx_ref"][idx]
        fu = max(0, min(fu, len(user_frames) - 1))
        fr = max(0, min(fr, len(ref_frames) - 1))
        uf = draw_skeleton(user_frames[fu].copy(), A["user_landmarks_raw"][fu])
        rf = draw_skeleton(ref_frames[fr].copy(), A["ref_landmarks_raw"][fr])
        combo = make_side_by_side(uf, rf, per_frame_err[idx] >= border_thresh, per_frame_err[idx])
        if combo.shape[0] != H or combo.shape[1] != W:
            combo = cv2.resize(combo, (W, H))
        frames_out.append(combo)
    return frames_out

def _export_with_opencv(frames, fps, try_variants):
    if not frames:
        return None
    H, W = frames[0].shape[:2]
    tmpdir = tempfile.gettempdir()
    for fourcc_str, ext in try_variants:
        out_path = os.path.join(tmpdir, f"ai_coach_report{ext}")
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        vw = cv2.VideoWriter(out_path, fourcc, float(fps), (W, H))
        if not vw.isOpened():
            continue
        for f in frames:
            vw.write(f)
        vw.release()
        try:
            if os.path.getsize(out_path) > 1024:
                return out_path
        except Exception:
            pass
    return None

def _export_with_imageio(frames, fps):
    try:
        import imageio.v3 as iio
    except Exception:
        return None
    if not frames:
        return None
    tmpdir = tempfile.gettempdir()
    out_path = os.path.join(tmpdir, "ai_coach_report.mp4")
    rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    try:
        iio.imwrite(out_path, rgb_frames, fps=float(fps), codec="libx264", quality=7)
        if os.path.getsize(out_path) > 1024:
            return out_path
    except Exception:
        return None
    return None

def export_report_video():
    variants = [("mp4v", ".mp4"), ("avc1", ".mp4"), ("XVID", ".avi"), ("MJPG", ".avi")]
    selected = [i for i in range(len(per_frame_err)) if per_frame_err[i] >= report_thresh] or list(range(len(per_frame_err)))
    if len(selected) > max_frames:
        step = max(1, len(selected) // max_frames)
        selected = selected[::step][:max_frames]
    frames_combo = _build_report_frames(selected, per_frame_err, report_thresh)
    if not frames_combo:
        return None, "Нет кадров для отчёта."
    out_path = _export_with_opencv(frames_combo, fps_out, variants)
    if out_path:
        return out_path, None
    out_path = _export_with_imageio(frames_combo, fps_out)
    if out_path:
        return out_path, None
    return None, "Не удалось открыть видеокодек. Установите `imageio[ffmpeg]` или кодеки для OpenCV."

if st.button("Сформировать видео-отчёт (.mp4/.avi)"):
    with st.spinner("Генерирую видео-отчёт..."):
        out_path, err = export_report_video()
    if err:
        st.error(err)
    else:
        st.success("Отчёт готов.")
        try:
            st.video(out_path)
        except Exception:
            pass
        ext = os.path
