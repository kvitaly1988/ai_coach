# app/visualization.py
from __future__ import annotations
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Tuple, Optional

# ---- Загрузка шрифта с поддержкой кириллицы ----
def _load_font(size: int = 22):
    candidates = [
        "app/assets/DejaVuSans.ttf",                          # локально в проекте
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",    # Linux
        "C:/Windows/Fonts/arial.ttf",                         # Windows
        "/System/Library/Fonts/Supplemental/Arial.ttf",       # macOS
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size=size)
        except Exception:
            continue
    return ImageFont.load_default()

# ---- Рёбра скелета (индексы MediaPipe) ----
CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (27, 31), (28, 32),
]

# ---- Русские имена ключей углов ----
JOINT_NAMES_RU = {
    'elbow_left': 'Левый локоть',
    'elbow_right': 'Правый локоть',
    'shoulder_left': 'Левое плечо',
    'shoulder_right': 'Правое плечо',
    'hip_left': 'Левое бедро',
    'hip_right': 'Правое бедро',
    'knee_left': 'Левое колено',
    'knee_right': 'Правое колено',
    'ankle_left': 'Левая лодыжка',
    'ankle_right': 'Правая лодыжка',
    'torso': 'Корпус (наклон)',
}

# ---- Вспомогательные функции ----
def _valid_norm_pt(xy, vis=None, vis_thresh=0.5):
    return (
        np.isfinite(xy[0]) and np.isfinite(xy[1]) and
        0.0 <= xy[0] <= 1.0 and 0.0 <= xy[1] <= 1.0 and
        (vis is None or (np.isfinite(vis) and vis >= vis_thresh))
    )

def _to_px(xy01, w, h):
    x = int(round(float(xy01[0]) * w))
    y = int(round(float(xy01[1]) * h))
    x = 0 if x < 0 else (w - 1 if x >= w else x)
    y = 0 if y < 0 else (h - 1 if y >= h else y)
    return (x, y)

# ---- Отрисовка скелета ----
def draw_skeleton(frame, lm_norm01, vis_thresh: float = 0.5):
    img = frame.copy()
    h, w = img.shape[:2]
    coords = np.asarray(lm_norm01, np.float32)
    if coords.ndim != 2 or coords.shape[0] < 33 or coords.shape[1] < 2:
        return img
    xy = coords[:, :2]
    vis = coords[:, 2] if coords.shape[1] >= 3 else None
    for a, b in CONNECTIONS:
        if _valid_norm_pt(xy[a], None if vis is None else vis[a], vis_thresh) and \
           _valid_norm_pt(xy[b], None if vis is None else vis[b], vis_thresh):
            cv2.line(img, _to_px(xy[a], w, h), _to_px(xy[b], w, h), (0, 255, 0), 2)
    for i in range(min(33, xy.shape[0])):
        if _valid_norm_pt(xy[i], None if vis is None else vis[i], vis_thresh):
            cv2.circle(img, _to_px(xy[i], w, h), 3, (0, 0, 255), -1)
    return img

# ---- Текст по-русски (через Pillow) ----
def _put_text_ru(img_bgr, text, org, color_bgr=(255, 255, 255), size=22):
    font = _load_font(size)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil)
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    draw.text(org, text, font=font, fill=color_rgb)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

# ---- Сборка «бок-о-бок» ----
def make_side_by_side(user_img, ref_img, is_bad, err_value):
    h1, w1 = user_img.shape[:2]
    h2, w2 = ref_img.shape[:2]
    H = max(h1, h2)
    if h1 != H:
        user_img = cv2.resize(user_img, (int(w1 * H / h1), H))
    if h2 != H:
        ref_img = cv2.resize(ref_img, (int(w2 * H / h2), H))
    combo = np.concatenate([user_img, ref_img], 1)
    color = (0, 0, 255) if is_bad else (0, 200, 0)
    cv2.rectangle(combo, (0, 0), (combo.shape[1] - 1, combo.shape[0] - 1), color, 4)
    combo = _put_text_ru(combo, f"Средняя ошибка: {err_value:.1f}°", (16, 26), color_bgr=color, size=24)
    combo = _put_text_ru(combo, "Пользователь", (16, H - 28), color_bgr=(255, 255, 255), size=22)
    combo = _put_text_ru(combo, "Эталон", (user_img.shape[1] + 16, H - 28), color_bgr=(255, 255, 255), size=22)
    return combo

# ---- Подсветка сустава на стоп-кадре совета ----
def _center_point(lm, ids):
    pts = [lm[i, :2] for i in ids if i < lm.shape[0] and np.all(np.isfinite(lm[i, :2]))]
    if not pts:
        return None
    return np.mean(np.stack(pts, 0), 0)

def draw_joint_overlay(user_frame, lm_user, lm_ref, joint_key):
    img = user_frame.copy()
    h, w = img.shape[:2]
    if joint_key == 'torso':
        pu = _center_point(lm_user, [11, 12]); hu = _center_point(lm_user, [23, 24])
        pr = _center_point(lm_ref,  [11, 12]); hr = _center_point(lm_ref,  [23, 24])
        if pu is None or hu is None or pr is None or hr is None:
            return img
        cu = (pu + hu) / 2.0; cr = (pr + hr) / 2.0
        pt_u = _to_px(cu, w, h); pt_r = _to_px(cr, w, h)
    else:
        idx_map = {
            'elbow_left': 13, 'elbow_right': 14,
            'shoulder_left': 11, 'shoulder_right': 12,
            'hip_left': 23, 'hip_right': 24,
            'knee_left': 25, 'knee_right': 26,
            'ankle_left': 27, 'ankle_right': 28,
        }
        idx = idx_map.get(joint_key)
        if idx is None or idx >= lm_user.shape[0] or idx >= lm_ref.shape[0]:
            return img
        if not (np.all(np.isfinite(lm_user[idx, :2])) and np.all(np.isfinite(lm_ref[idx, :2]))):
            return img
        pt_u = _to_px(lm_user[idx, :2], w, h); pt_r = _to_px(lm_ref[idx, :2], w, h)

    cv2.circle(img, pt_u, 6, (0, 0, 255), -1)  # пользователь
    cv2.drawMarker(img, pt_r, (0, 200, 0), markerType=cv2.MARKER_TILTED_CROSS, markerSize=14, thickness=2)  # эталон
    cv2.line(img, pt_u, pt_r, (0, 200, 255), 2)
    name = JOINT_NAMES_RU.get(joint_key, joint_key)
    img = _put_text_ru(img, name, (pt_u[0] + 10, max(10, pt_u[1] - 10)), color_bgr=(255, 255, 255), size=22)
    return img

# ---- Графики углов (на русском) ----
def plot_angle_series(series_user: Dict[str, np.ndarray],
                      series_ref: Dict[str, np.ndarray],
                      keys: List[str]):
    import matplotlib.pyplot as plt
    figs = []
    # Настройки шрифта (чтобы подписи были на русском)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.titlesize': 11})
    for k in keys:
        fig = plt.figure(figsize=(5, 3), dpi=120)
        u = series_user.get(k, None)
        r = series_ref.get(k, None)
        label_ru = JOINT_NAMES_RU.get(k, k)
        if u is not None:
            plt.plot(u, label='Пользователь')
        if r is not None:
            plt.plot(r, label='Эталон')
        plt.title(label_ru)
        plt.xlabel('Кадр (выравнено)')
        plt.ylabel('Угол (°)')
        plt.legend()
        plt.tight_layout()
        figs.append(fig)
    return figs
