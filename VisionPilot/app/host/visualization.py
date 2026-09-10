from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from wire import VisionPilotOutput
from occupancy_view import OccupancyRenderer, OccupancyScene, SceneDetection, PANEL_H, PANEL_W


NET_W = 1024
NET_H = 512
D_MAX = 150.0
CURV_SCALE = 0.21
FLAG_THRESHOLD = 0.65
WHEELBASE_M = 2.984
STEER_RATIO = 16.8

# Same warped-pixel -> world fallback matrix used by the C++ debug/production view.
H_WARP_TO_WORLD = np.array(
    [
        [0.00209514907, -0.000941721466, -9.24906396],
        [0.00662758637, -0.000352940531, -3.33396502],
        [0.000120077371, -0.00411343505, 1.0],
    ],
    dtype=np.float64,
)

DET_COLORS = {
    1: (0, 0, 220),
    2: (0, 210, 255),
    3: (220, 200, 0),
}
CLR_OTHER = (80, 200, 80)
CLR_GREEN = (0, 255, 0)
CLR_FUSED = (0, 220, 255)
CLR_TEXT = (220, 220, 220)
CLR_MUTED = (155, 160, 166)
CLR_PANEL = (15, 18, 22)
CLR_ACCENT = (180, 220, 255)
CLR_LINE = (55, 60, 68)


@dataclass
class RenderConfig:
    # Existing knobs retained for drop-in compatibility with the previous host.
    display_width: int = 1920
    side_panel_width: int = 520
    show_autosteer: bool = True
    show_fused_path: bool = True
    show_detections: bool = True
    path_corridor: bool = True
    homography_path: Optional[str] = None
    homography_key: str = "H"
    homography_space: str = "native"  # H.yaml matches raw/native camera pixels by default

    # New visualization controls.
    debug_level: int = 0             # 0=clean, 1=compact, 2=model detail, 3=raw transmitted outputs
    camera_view: str = "production" # "production" or "debug"
    camera_space: str = "native"    # "native" overlays on original frame; "net" mirrors C++ resized view
    show_occupancy: bool = True
    speed_limit_ms: float = 0.0
    icons_dir: Optional[str] = None
    wheel_dir: Optional[str] = None
    source_label: str = "remote/V4M"


@dataclass
class HostMetrics:
    net_rtt_ms: float = 0.0
    tx_mbps: float = 0.0
    tx_bytes: int = 0
    pending_count: int = 0
    result_fps: float = 0.0
    result_fps_ema: float = 0.0
    rtt_ema_ms: float = 0.0
    rtt_jitter_ms: float = 0.0
    raw_w: int = 0
    raw_h: int = 0
    crop_top: int = 0
    crop_h: int = 0


def _clip_u8(v: float) -> int:
    return int(np.clip(v, 0.0, 255.0))


def _alpha_rect(img: np.ndarray, p1, p2, color, alpha: float) -> None:
    x1, y1 = max(0, p1[0]), max(0, p1[1])
    x2, y2 = min(img.shape[1], p2[0]), min(img.shape[0], p2[1])
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    block = np.full_like(roi, color)
    cv2.addWeighted(block, alpha, roi, 1.0 - alpha, 0.0, roi)


def _text(img, s, xy, scale=0.48, color=CLR_TEXT, thick=1, font=cv2.FONT_HERSHEY_SIMPLEX):
    x, y = int(xy[0]), int(xy[1])
    cv2.putText(img, str(s), (x + 1, y + 1), font, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, str(s), (x, y), font, scale, color, thick, cv2.LINE_AA)


def _text_centered(img, s, cx, baseline_y, scale, color, thick=1, font=cv2.FONT_HERSHEY_DUPLEX):
    (tw, _), _ = cv2.getTextSize(str(s), font, scale, thick)
    _text(img, s, (cx - tw // 2, baseline_y), scale, color, thick, font)


def _det_color(class_id: int):
    return DET_COLORS.get(class_id, CLR_OTHER)


def _path_color(acc: float):
    if acc < -5.0:
        return (93, 0, 255)
    if acc < -3.0:
        return (0, 102, 255)
    if acc < -1.0:
        return (0, 213, 255)
    return (174, 255, 0)


def _load_homography(path: Optional[str], key: str) -> Optional[np.ndarray]:
    if not path:
        return None
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"cannot open homography file: {path}")
    H = fs.getNode(key).mat()
    fs.release()
    if H is None or H.shape != (3, 3):
        raise RuntimeError(f"homography '{key}' missing/invalid in {path}")
    return H.astype(np.float64)


@dataclass(frozen=True)
class FrameGeometry:
    native_w: int
    native_h: int
    crop_top: int
    crop_h: int
    sx: float
    sy: float
    T_net_to_native: np.ndarray
    T_native_to_net: np.ndarray


def _compute_top_crop_2_1(height: int, width: int) -> int:
    """Exact positive-value equivalent of C++ std::lround(height - width/2)."""
    x = float(height) - float(width) / 2.0
    return max(0, int(math.floor(x + 0.5)))


def _frame_geometry(native_shape: tuple[int, ...]) -> FrameGeometry:
    h, w = native_shape[:2]
    crop_top = _compute_top_crop_2_1(h, w)
    crop_h = h - crop_top
    if crop_h <= 0:
        raise ValueError(f"invalid top crop for frame {w}x{h}: crop_top={crop_top}")
    sx = float(w) / float(NET_W)
    sy = float(crop_h) / float(NET_H)
    T = np.array(
        [[sx, 0.0, 0.0], [0.0, sy, float(crop_top)], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return FrameGeometry(
        native_w=w, native_h=h, crop_top=crop_top, crop_h=crop_h,
        sx=sx, sy=sy, T_net_to_native=T, T_native_to_net=np.linalg.inv(T),
    )


def _make_net_view(frame: np.ndarray, geom: FrameGeometry) -> np.ndarray:
    """Mirror ImagePreprocessor: top-crop to 2:1, then resize to 1024x512."""
    cropped = frame[geom.crop_top:, :]
    return cv2.resize(cropped, (NET_W, NET_H), interpolation=cv2.INTER_LINEAR)


def _net_point_to_display(
    u: float, v: float, geom: FrameGeometry, camera_space: str
) -> tuple[int, int]:
    if camera_space == "net":
        return int(round(u)), int(round(v))
    if camera_space != "native":
        raise ValueError("camera_space must be 'native' or 'net'")
    p = geom.T_net_to_native @ np.array([float(u), float(v), 1.0], dtype=np.float64)
    return int(round(p[0] / p[2])), int(round(p[1] / p[2]))


def _net_homography_px_to_world(
    provided_H: Optional[np.ndarray],
    homography_space: str,
    geom: FrameGeometry,
) -> tuple[np.ndarray, bool]:
    """Return H_resized: 1024x512 top-cropped inference pixels -> world metres.

    This mirrors InferencePipeline::set_H_resized():
      raw_px = T_net_to_native @ net_px
      world  = H_raw_to_world @ raw_px
      H_net_to_world = H_raw_to_world @ T_net_to_native
    """
    if provided_H is None:
        # Legacy fallback is only a warped-BEV calibration. It is useful for old
        # path rendering but is NOT a valid AutoSpeed resized-image calibration.
        return H_WARP_TO_WORLD.copy(), False

    H = provided_H.astype(np.float64)
    if homography_space == "net":
        return H, True
    if homography_space != "native":
        raise ValueError("homography_space must be 'net' or 'native'")
    return H @ geom.T_net_to_native, True


def _world_to_display_homography(
    H_net_to_world: np.ndarray, geom: FrameGeometry, camera_space: str
) -> np.ndarray:
    H_world_to_net = np.linalg.inv(H_net_to_world)
    if camera_space == "net":
        return H_world_to_net
    if camera_space == "native":
        return geom.T_net_to_native @ H_world_to_net
    raise ValueError("camera_space must be 'native' or 'net'")


def _project(H_world_to_px: np.ndarray, x: float, y: float) -> Optional[tuple[int, int]]:
    p = H_world_to_px @ np.array([x, y, 1.0], dtype=np.float64)
    if abs(float(p[2])) < 1e-9:
        return None
    return int(round(p[0] / p[2])), int(round(p[1] / p[2]))


def _steer_cmd(out: VisionPilotOutput) -> float:
    if len(out.plan.steering) > 1:
        return float(out.plan.steering[1])
    if out.plan.steering:
        return float(out.plan.steering[0])
    return 0.0


def _curvature_to_steer_deg(curv_1pm: float) -> float:
    return math.atan(curv_1pm * WHEELBASE_M) * STEER_RATIO * 180.0 / math.pi


# =============================================================================
# Production view: direct Python port of visualization.cpp
# =============================================================================


def _draw_top_vignette(img: np.ndarray) -> None:
    max_alpha = 0.60
    fade_h = img.shape[0] * 45 // 100
    if fade_h <= 0:
        return
    scales = 1.0 - max_alpha * (1.0 - np.arange(fade_h, dtype=np.float32) / float(fade_h))
    top = img[:fade_h].astype(np.float32)
    top *= scales[:, None, None]
    img[:fade_h] = np.clip(top, 0, 255).astype(np.uint8)


def _draw_path_corridor(img: np.ndarray, out: VisionPilotOutput, H_world_to_px: np.ndarray, enabled=True) -> None:
    lat = out.inference.lateral
    if not enabled or not lat.path_valid:
        return
    if lat.path_x_max_m <= lat.path_x_min_m + 1.0:
        return

    x_start = max(0.5, lat.path_x_min_m)
    x_end = lat.path_x_max_m
    lp, rp = [], []
    for x in np.arange(x_start, x_end + 0.01, 0.5):
        yc = lat.path_a * x * x + lat.path_b * x + lat.path_c
        l = _project(H_world_to_px, float(x), float(yc + 1.0))
        r = _project(H_world_to_px, float(x), float(yc - 1.0))
        if l is not None and r is not None:
            lp.append(l)
            rp.append(r)

    if len(lp) < 2:
        return
    overlay = img.copy()
    color = _path_color(out.plan.acceleration)
    H, W = img.shape[:2]
    for i in range(len(lp) - 1):
        quad = [lp[i], rp[i], rp[i + 1], lp[i + 1]]
        if not any(0 <= p[0] < W and 0 <= p[1] < H for p in quad):
            continue
        cv2.fillConvexPoly(overlay, np.asarray(quad, np.int32), color, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0.0, img)


def _draw_path_centerline(img: np.ndarray, out: VisionPilotOutput, H_world_to_px: np.ndarray) -> None:
    """Legacy --centerline-only behavior: fused polynomial without corridor fill."""
    lat = out.inference.lateral
    if not lat.path_valid or lat.path_x_max_m <= lat.path_x_min_m + 1.0:
        return
    pts: list[tuple[int, int]] = []
    for x in np.arange(max(0.5, lat.path_x_min_m), lat.path_x_max_m + 0.01, 0.5):
        y = lat.path_a * x * x + lat.path_b * x + lat.path_c
        p = _project(H_world_to_px, float(x), float(y))
        if p is not None and 0 <= p[0] < img.shape[1] and 0 <= p[1] < img.shape[0]:
            pts.append(p)
    if len(pts) >= 2:
        cv2.polylines(img, [np.asarray(pts, np.int32)], False, _path_color(out.plan.acceleration), 3, cv2.LINE_AA)


def _draw_cipo_boxes(img: np.ndarray, out: VisionPilotOutput, geom: FrameGeometry, camera_space: str, enabled=True) -> None:
    if not enabled:
        return
    detections = out.inference.auto_speed.detections
    if not detections:
        return

    overlay = img.copy()
    mapped = []
    for d in detections:
        c = _det_color(d.class_id)
        fill = tuple(_clip_u8(v * 0.35) for v in c)
        tl = _net_point_to_display(d.x1, d.y1, geom, camera_space)
        br = _net_point_to_display(d.x2, d.y2, geom, camera_space)
        mapped.append((d, tl, br, c))
        cv2.rectangle(overlay, tl, br, fill, -1)
    cv2.addWeighted(overlay, 0.32, img, 0.68, 0.0, img)

    for d, tl, br, c in mapped:
        cv2.rectangle(img, tl, br, c, 2, cv2.LINE_AA)

    cipo = out.inference.cipo
    if cipo.valid and cipo.distance_m > 0.0:
        l1 = [d for d in detections if d.class_id == 1]
        if l1:
            best = min(l1, key=lambda d: abs((d.x1 + d.x2) * 0.5 - NET_W * 0.5))
            lbl = f"{cipo.distance_m:.0f}m"
            tx, ty0 = _net_point_to_display(0.5 * (best.x1 + best.x2), best.y1, geom, camera_space)
            ty = ty0 - 6
            if ty > 12:
                (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(img, (tx - tw // 2 - 6, ty - th - 6), (tx + tw // 2 + 6, ty + 4), (20, 20, 20), -1)
                cv2.putText(img, lbl, (tx - tw // 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)


def _draw_speed(img: np.ndarray, speed_ms: float) -> None:
    num = f"{speed_ms * 2.23694:.0f}"
    unit = "mph"
    font = cv2.FONT_HERSHEY_DUPLEX
    (nw, nh), _ = cv2.getTextSize(num, font, 1.1, 2)
    (uw, _), _ = cv2.getTextSize(unit, font, 0.5, 1)
    num_tx = (img.shape[1] - nw) // 2
    unit_tx = (img.shape[1] - uw) // 2
    num_ty = 44
    unit_ty = num_ty + nh - 4
    overlay = img.copy()
    cv2.putText(overlay, num, (num_tx + 2, num_ty + 2), font, 1.1, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(overlay, num, (num_tx, num_ty), font, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(overlay, unit, (unit_tx + 2, unit_ty + 2), font, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(overlay, unit, (unit_tx, unit_ty), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 1.0, img, 0.5, 0.0, img)


def _draw_speed_limit(img: np.ndarray, speed_limit_ms: float) -> None:
    if speed_limit_ms <= 0.0:
        return
    mph = int(round(speed_limit_ms * 2.23694))
    box_x, box_y, box_w, box_h = 14, 14, 68, 76
    cv2.rectangle(img, (box_x, box_y), (box_x + box_w, box_y + box_h), (255, 255, 255), 3, cv2.LINE_AA)
    _text_centered(img, "MAX", box_x + box_w // 2, box_y + 20, 0.38, (255, 255, 255), 1)
    _text_centered(img, str(mph), box_x + box_w // 2, box_y + 58, 0.90, (255, 255, 255), 2)


def _paste_rgba(base: np.ndarray, icon: Optional[np.ndarray], cx: int, cy: int, px: int = 0) -> None:
    if icon is None or icon.size == 0:
        return
    draw = icon
    if px > 0 and (icon.shape[1] != px or icon.shape[0] != px):
        draw = cv2.resize(icon, (px, px), interpolation=cv2.INTER_AREA)
    x, y = cx - draw.shape[1] // 2, cy - draw.shape[0] // 2
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(base.shape[1], x + draw.shape[1]), min(base.shape[0], y + draw.shape[0])
    if x2 <= x1 or y2 <= y1:
        return
    roi = base[y1:y2, x1:x2]
    src = draw[y1 - y:y2 - y, x1 - x:x2 - x]
    if src.ndim == 3 and src.shape[2] == 4:
        alpha = src[:, :, 3:4].astype(np.float32) / 255.0
        rgb = src[:, :, :3].astype(np.float32)
        out = rgb * alpha + roi.astype(np.float32) * (1.0 - alpha)
        roi[:] = np.clip(out, 0, 255).astype(np.uint8)
    else:
        roi[:] = src[:, :, :3]


def _draw_alerts(img: np.ndarray, out: VisionPilotOutput, icons: dict[str, Optional[np.ndarray]]) -> None:
    warnings = set(int(w) for w in out.plan.warnings)
    W, H = img.shape[1], img.shape[0]
    sw = W * 14 // 100
    orange = (0, 120, 255)
    white = (255, 255, 255)
    alert_font = 0.50

    def side_alert(left: bool, icon, line1: str, line2: str):
        cx = sw // 2 if left else W - sw // 2
        x0 = 0 if left else W - sw
        _alpha_rect(img, (x0, 0), (x0 + sw, H), orange, 0.45)
        icon_px = min(52, sw - 20)
        icon_cy = H * 36 // 100
        _paste_rgba(img, icon, cx, icon_cy, icon_px)
        line1_y = icon_cy + icon_px // 2 + 22
        _text_centered(img, line1, cx, line1_y, alert_font, white, 1)
        _text_centered(img, line2, cx, line1_y + 20, alert_font, white, 1)

    def bottom_alert(icon, label: str, bg, bg_alpha: float, strip_pct: int):
        bh = H * strip_pct // 100
        y0 = H - bh
        _alpha_rect(img, (0, y0), (W, H), bg, bg_alpha)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, alert_font, 1)
        text_baseline = H - max(10, bh // 10)
        text_top = text_baseline - th
        icon_px = min(46, max(28, (text_top - y0 - 14) * 2))
        icon_cy = y0 + (text_top - y0) // 2
        _paste_rgba(img, icon, W // 2, icon_cy, icon_px)
        _text_centered(img, label, W // 2, text_baseline, alert_font, white, 1)

    if 3 in warnings:
        side_alert(True, icons.get("lld"), "Left Lane", "Departure")
    if 4 in warnings:
        side_alert(False, icons.get("rld"), "Right Lane", "Departure")
    if 2 in warnings:
        bottom_alert(icons.get("brake"), "Emergency Braking", (0, 0, 180), 0.50, 22)
    elif 1 in warnings:
        bottom_alert(icons.get("collision"), "Collision Alert", (0, 130, 230), 0.45, 18)


def _draw_ad_only_cipo(img: np.ndarray, out: VisionPilotOutput, H_world_to_px: np.ndarray) -> None:
    ad = out.inference.auto_drive
    cipo = out.inference.cipo
    if not (ad.valid and ad.flag_prob >= 0.40 and not cipo.cipo_raw_found):
        return
    dist = D_MAX * (1.0 - ad.dist_normalized)
    if dist <= 0.0 or dist >= D_MAX:
        return
    lat = out.inference.lateral
    yw = lat.path_a * dist * dist + lat.path_b * dist + lat.path_c if lat.path_valid else 0.0
    p = _project(H_world_to_px, dist, yw)
    if p is None or not (0 <= p[0] < img.shape[1] and 0 <= p[1] < img.shape[0]):
        return
    px, py = p
    aw, ah = 20, 25
    orange = (80, 180, 255)
    tri = np.asarray([(px, py - ah), (px - aw, py), (px + aw, py)], np.int32)
    cv2.fillConvexPoly(img, tri, orange, cv2.LINE_AA)
    cv2.polylines(img, [tri], True, orange, 1, cv2.LINE_AA)
    lbl = f"{dist:.0f}m"
    (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
    tx, ty = px - tw // 2, py - ah - 8
    if ty > th:
        cv2.rectangle(img, (tx - 4, ty - th - 2), (tx + tw + 4, ty + 4), (20, 20, 20), -1)
        cv2.putText(img, lbl, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.52, orange, 2, cv2.LINE_AA)


def draw_production_frame(
    frame_net: np.ndarray,
    out: VisionPilotOutput,
    speed_ms: float,
    speed_limit_ms: float,
    H_world_to_px: np.ndarray,
    geom: FrameGeometry,
    icons: dict[str, Optional[np.ndarray]],
    cfg: RenderConfig,
) -> np.ndarray:
    img = frame_net.copy()
    _draw_top_vignette(img)
    if cfg.show_fused_path:
        if cfg.path_corridor:
            _draw_path_corridor(img, out, H_world_to_px, True)
        else:
            _draw_path_centerline(img, out, H_world_to_px)
    _draw_cipo_boxes(img, out, geom, cfg.camera_space, cfg.show_detections)
    _draw_ad_only_cipo(img, out, H_world_to_px)
    _draw_alerts(img, out, icons)
    _draw_speed(img, speed_ms)
    _draw_speed_limit(img, speed_limit_ms)
    return img


# =============================================================================
# Engineering debug view: Python port of debug_draw.cpp
# =============================================================================


def _debug_draw_panel(img: np.ndarray, rect, title: str, accent) -> None:
    x, y, w, h = rect
    _alpha_rect(img, (x, y), (x + w, y + h), (0, 0, 0), 0.82)
    cv2.rectangle(img, (x, y), (x + w, y + h), accent, 1, cv2.LINE_AA)
    if title:
        cv2.putText(img, title, (x + 6, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.40, accent, 1, cv2.LINE_AA)


def _debug_draw_tag(img: np.ndarray, anchor, label: str, color) -> None:
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
    x, y = anchor
    x1, y1 = max(0, x), max(0, y - th - 6)
    x2, y2 = min(img.shape[1], x + tw + 10), min(img.shape[0], y + 4)
    _alpha_rect(img, (x1, y1), (x2, y2), (0, 0, 0), 0.88)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    cv2.putText(img, label, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)


def _debug_draw_detections(img: np.ndarray, out: VisionPilotOutput, geom: FrameGeometry, camera_space: str, enabled=True) -> None:
    if not enabled or not out.inference.auto_speed.valid:
        return
    for d in out.inference.auto_speed.detections:
        clr = _det_color(d.class_id)
        tl = _net_point_to_display(d.x1, d.y1, geom, camera_space)
        br = _net_point_to_display(d.x2, d.y2, geom, camera_space)
        cv2.rectangle(img, tl, br, clr, 2, cv2.LINE_AA)
        cv2.putText(img, f"L{d.class_id} {d.score*100:.0f}%", (tl[0] + 2, max(tl[1] - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, clr, 1, cv2.LINE_AA)


def _debug_draw_autosteer(img: np.ndarray, out: VisionPilotOutput, geom: FrameGeometry, camera_space: str, enabled=True) -> None:
    steer = out.inference.auto_steer
    if not enabled or not steer.valid:
        return
    n = min(len(steer.xp), len(steer.h_vector), 64)
    if n == 0:
        return
    ys_net = np.linspace(0.0, NET_H - 1.0, n)
    poly = []
    for i in range(n):
        if steer.h_vector[i] < 0.5:
            continue
        u_net = float(steer.xp[i]) * NET_W
        v_net = float(ys_net[i])
        u, v = _net_point_to_display(u_net, v_net, geom, camera_space)
        if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
            cv2.circle(img, (u, v), 3, CLR_GREEN, -1, cv2.LINE_AA)
            poly.append((u, v))
    if len(poly) >= 2:
        cv2.polylines(img, [np.asarray(poly, np.int32)], False, CLR_GREEN, 2, cv2.LINE_AA)
        tag = max(poly, key=lambda p: p[1])
        _debug_draw_tag(img, (tag[0] - 90, tag[1] - 8), "AutoSteer", CLR_GREEN)


def _debug_draw_fused_path(img: np.ndarray, out: VisionPilotOutput, H_world_to_px: np.ndarray, enabled=True) -> None:
    lat = out.inference.lateral
    if not enabled or not lat.path_valid:
        return
    x_start, x_end = max(0.0, lat.path_x_min_m), lat.path_x_max_m
    if x_end <= x_start + 0.5:
        return
    pts = []
    for x in np.arange(x_start, x_end + 0.01, 1.5):
        y = lat.path_a * x * x + lat.path_b * x + lat.path_c
        if abs(y) > 12.0:
            continue
        p = _project(H_world_to_px, float(x), float(y))
        if p is not None and 0 <= p[0] < img.shape[1] and 0 <= p[1] < img.shape[0]:
            pts.append(p)
    if len(pts) >= 2:
        cv2.polylines(img, [np.asarray(pts, np.int32)], False, CLR_FUSED, 2, cv2.LINE_AA)
        tag = max(pts, key=lambda p: p[1])
        _debug_draw_tag(img, (tag[0] + 8, tag[1] - 8), "Fused path", CLR_FUSED)


def _debug_draw_legend(img: np.ndarray, out: VisionPilotOutput) -> None:
    rect = (6, 26, 212, 100)
    _debug_draw_panel(img, rect, "OVERLAY KEY", (220, 220, 220))
    x0, y = rect[0] + 8, rect[1] + 28
    for clr, label in ((CLR_GREEN, "Green  AutoSteer waypoints"), (CLR_FUSED, "Yellow Fused path (RANSAC)")):
        cv2.rectangle(img, (x0, y - 9), (x0 + 14, y - 5), clr, -1, cv2.LINE_AA)
        cv2.putText(img, label, (x0 + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
        y += 16
    if out.inference.lateral.valid:
        st = _curvature_to_steer_deg(out.inference.lateral.curvature)
        cv2.putText(img, f"Fused steer {st:.1f} deg", (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (120, 240, 120), 1, cv2.LINE_AA)
        y += 16
    if out.inference.auto_drive.valid:
        curv = out.inference.auto_drive.curvature_raw * CURV_SCALE
        st = _curvature_to_steer_deg(curv)
        cv2.putText(img, f"AD steer {st:.1f} deg (white wheel)", (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (240, 240, 240), 1, cv2.LINE_AA)


def _debug_draw_bev(img: np.ndarray, out: VisionPilotOutput) -> None:
    lat = out.inference.lateral
    if not lat.path_valid:
        return
    rect = (img.shape[1] - 200 - 8, img.shape[0] - 104 - 148 - 6, 200, 148)
    _debug_draw_panel(img, rect, "FUSED PATH (top-down)", CLR_FUSED)
    x0, y0, pw, ph = rect
    x_end = lat.path_x_max_m if lat.path_x_max_m > lat.path_x_min_m + 1.0 else 0.0
    if x_end < 2.0:
        return
    px_per_m = 4.5
    lat_max = 7.0

    def w2p(x_fwd, y_lat):
        return int(round(x0 + pw / 2 - y_lat * px_per_m)), int(round(y0 + ph - 12 - x_fwd * px_per_m))

    poly = []
    for x in np.arange(0.0, x_end + 0.01, 1.0):
        y = lat.path_a * x * x + lat.path_b * x + lat.path_c
        if abs(y) > lat_max:
            continue
        p = w2p(float(x), float(y))
        if x0 + 2 <= p[0] < x0 + pw - 2 and y0 + 20 <= p[1] < y0 + ph - 2:
            poly.append(p)
    if len(poly) >= 2:
        cv2.polylines(img, [np.asarray(poly, np.int32)], False, CLR_FUSED, 2, cv2.LINE_AA)
    ego = w2p(0.0, 0.0)
    cv2.circle(img, ego, 4, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.putText(img, "ego", (ego[0] + 6, ego[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1, cv2.LINE_AA)


def _debug_draw_top_bar(img: np.ndarray, out: VisionPilotOutput, source_label: str) -> None:
    _alpha_rect(img, (0, 0), (img.shape[1], 22), (0, 0, 0), 0.65)
    inf = out.inference
    fps = 1000.0 / inf.wall_ms if inf.wall_ms > 0 else 0.0
    s = f"VisionPilot  #{inf.frame_id}  wall={inf.wall_ms:.1f} ms ({fps:.0f} fps)  pre={inf.pre_ms:.1f} ms  src={source_label}"
    cv2.putText(img, s, (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)


def _debug_draw_hud(img: np.ndarray, out: VisionPilotOutput) -> None:
    py, hud_h = img.shape[0] - 104, 104
    W = img.shape[1]
    _alpha_rect(img, (0, py), (W, py + hud_h), (0, 0, 0), 0.85)
    cv2.line(img, (0, py), (W, py), (70, 70, 70), 1, cv2.LINE_AA)
    col_w = W // 3
    cv2.line(img, (col_w, py + 4), (col_w, py + hud_h - 4), (70, 70, 70), 1, cv2.LINE_AA)
    cv2.line(img, (2 * col_w, py + 4), (2 * col_w, py + hud_h - 4), (70, 70, 70), 1, cv2.LINE_AA)
    c1x, c2x, c3x = 10, col_w + 10, 2 * col_w + 10
    line_h = 17

    def t(x, y, s, clr=(200, 200, 200), scale=0.38, thick=1):
        cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, clr, thick, cv2.LINE_AA)

    y = py + 16
    t(c1x, y, "NEURAL NET OUTPUTS", (220, 220, 220)); y += line_h
    ad = out.inference.auto_drive
    if ad.valid:
        dist = D_MAX * (1.0 - ad.dist_normalized)
        curv = ad.curvature_raw * CURV_SCALE
        flag = int(ad.flag_prob >= FLAG_THRESHOLD)
        t(c1x, y, f"AutoDrive  dist {dist:.1f} m"); y += line_h
        t(c1x, y, f"           curv {curv:.4f} 1/m"); y += line_h
        t(c1x, y, f"           CIPO flag {flag} p={ad.flag_prob:.2f}")
    else:
        t(c1x, y, "AutoDrive  (no output)")

    cipo = out.inference.cipo
    y = py + 16
    t(c2x, y, "FUSED LONGITUDINAL", (0, 220, 0)); y += line_h
    if cipo.valid:
        t(c2x, y, f"distance  {cipo.distance_m:.1f} m", (0, 220, 0), 0.42, 2); y += line_h
        t(c2x, y, f"velocity  {cipo.velocity_ms:.2f} m/s", (0, 220, 0), 0.42, 2); y += line_h
        t(c2x, y, f"uncert.   +/-{cipo.distance_stddev_m:.1f} m")
        if cipo.cut_in_detected:
            t(c2x, py + hud_h - 10, "CUT-IN", (0, 210, 255), 0.38, 2)
    else:
        t(c2x, y, "(waiting for tracker)")

    lat = out.inference.lateral
    y = py + 16
    t(c3x, y, "FUSED LATERAL", CLR_FUSED); y += line_h
    if lat.valid:
        t(c3x, y, f"CTE       {lat.cte_m:.2f} m  {lat.cte_rate_mps:.2f} m/s", CLR_FUSED, 0.42, 2); y += line_h
        t(c3x, y, f"yaw       {lat.yaw_rad:.3f} rad  {lat.yaw_rate_rps:.3f} rad/s", CLR_FUSED, 0.42, 2); y += line_h
        t(c3x, y, f"curvature {lat.curvature:.4f} 1/m", CLR_FUSED, 0.42, 2)
        if lat.path_valid:
            t(c3x, y + line_h, f"path fit  {lat.path_inliers} inliers / {lat.path_points} pts", (200, 200, 200), 0.36)
    else:
        t(c3x, y, "(no path / fusion)")


def _resolve_asset_dir(requested: Optional[str], candidates: list[Path]) -> Optional[Path]:
    if requested:
        p = Path(requested)
        if p.is_dir():
            return p
    for p in candidates:
        if p.is_dir():
            return p
    return None


def _load_rgba(path: Path, px: int) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    img = cv2.resize(img, (px, px), interpolation=cv2.INTER_AREA)
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def _rotate_rgba(src: Optional[np.ndarray], angle_deg: float) -> Optional[np.ndarray]:
    if src is None:
        return None
    center = (src.shape[1] / 2.0, src.shape[0] / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(src, M, (src.shape[1], src.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


def _debug_draw_wheels(img: np.ndarray, out: VisionPilotOutput, wheels: dict[str, Optional[np.ndarray]]) -> None:
    wheel_px = 92
    pad, y = 8, 28
    x_ad = img.shape[1] - pad - wheel_px
    x_fused = x_ad - pad - wheel_px
    lat = out.inference.lateral
    ad = out.inference.auto_drive
    if lat.valid and wheels.get("green") is not None:
        st = _curvature_to_steer_deg(lat.curvature)
        w = _rotate_rgba(wheels["green"], st)
        _paste_rgba(img, w, x_fused + wheel_px // 2, y + wheel_px // 2, wheel_px)
        cv2.putText(img, "fused", (x_fused + 18, y + wheel_px + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (120, 240, 120), 1, cv2.LINE_AA)
    if ad.valid and wheels.get("white") is not None:
        st = _curvature_to_steer_deg(ad.curvature_raw * CURV_SCALE)
        w = _rotate_rgba(wheels["white"], st)
        _paste_rgba(img, w, x_ad + wheel_px // 2, y + wheel_px // 2, wheel_px)
        cv2.putText(img, "AD", (x_ad + 34, y + wheel_px + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (240, 240, 240), 1, cv2.LINE_AA)


def draw_debug_frame(
    frame_net: np.ndarray,
    out: VisionPilotOutput,
    H_world_to_px: np.ndarray,
    geom: FrameGeometry,
    source_label: str,
    wheels: dict[str, Optional[np.ndarray]],
    cfg: RenderConfig,
) -> np.ndarray:
    img = frame_net.copy()
    _debug_draw_detections(img, out, geom, cfg.camera_space, cfg.show_detections)
    _debug_draw_autosteer(img, out, geom, cfg.camera_space, cfg.show_autosteer)
    _debug_draw_fused_path(img, out, H_world_to_px, cfg.show_fused_path)
    _debug_draw_legend(img, out)
    _debug_draw_bev(img, out)
    _debug_draw_wheels(img, out, wheels)
    _debug_draw_top_bar(img, out, source_label)
    _debug_draw_hud(img, out)
    return img


# =============================================================================
# Occupancy bridge: Python equivalent of occupancy_bridge.cpp
# =============================================================================


def make_occupancy_scene(out: VisionPilotOutput, H_px2world: Optional[np.ndarray]) -> OccupancyScene:
    s = OccupancyScene(H_px2world=None if H_px2world is None else H_px2world.astype(np.float32))
    lat = out.inference.lateral
    if lat.path_valid:
        s.path_a = lat.path_a
        s.path_b = lat.path_b
        s.path_c = lat.path_c
        s.path_valid = True
    if lat.valid:
        s.cte_m = lat.cte_m
        s.yaw_rad = lat.yaw_rad
    elif lat.path_valid:
        s.cte_m = lat.raw_cte_m
        s.yaw_rad = lat.raw_yaw_rad

    steer = out.inference.auto_steer
    if steer.valid and H_px2world is not None and steer.xp and steer.h_vector:
        n = min(64, len(steer.xp), len(steer.h_vector))
        src = []
        for i in range(n):
            if steer.h_vector[i] < 0.5:
                continue
            u = steer.xp[i] * NET_W
            v = i * ((NET_H - 1) / max(1, n - 1))
            src.append((u, v))
        if src:
            pts = np.asarray(src, dtype=np.float32).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, H_px2world.astype(np.float32)).reshape(-1, 2)
            for xw, yw in dst:
                if not np.isfinite(xw) or not np.isfinite(yw):
                    continue
                if xw < -1.0 or xw > 80.0:
                    continue
                s.lane_world.append((float(xw), float(yw)))

    for d in out.inference.auto_speed.detections:
        s.detections.append(SceneDetection(d.x1, d.y1, d.x2, d.y2, d.score, d.class_id))

    cipo = out.inference.cipo
    s.cipo_valid = cipo.valid
    s.cipo_distance_m = cipo.distance_m

    ad = out.inference.auto_drive
    if ad.valid and ad.flag_prob >= 0.40 and not cipo.cipo_raw_found:
        s.ad_cipo_only = True
        s.ad_distance_m = D_MAX * (1.0 - ad.dist_normalized)
    return s


# =============================================================================
# Right-side diagnostic panels
# =============================================================================


class _PanelWriter:
    def __init__(self, img: np.ndarray, line_h: int = 18):
        self.img = img
        self.y = 0
        self.line_h = line_h

    def title(self, title: str, subtitle: str) -> None:
        _text(self.img, title, (16, 25), 0.58, (240, 245, 250), 2)
        _text(self.img, subtitle, (16, 46), 0.35, CLR_MUTED, 1)
        self.y = 73

    def section(self, title: str, color=CLR_ACCENT) -> None:
        cv2.line(self.img, (12, self.y + 7), (self.img.shape[1] - 12, self.y + 7), CLR_LINE, 1)
        _text(self.img, title, (16, self.y), 0.43, color, 2)
        self.y += 25

    def kv(self, label: str, value: str, color=CLR_TEXT, bold=False, label_w=145) -> None:
        _text(self.img, label, (16, self.y), 0.36, CLR_MUTED, 1)
        _text(self.img, value, (label_w, self.y), 0.39, color, 2 if bold else 1)
        self.y += self.line_h

    def line(self, value: str, color=CLR_TEXT, scale=0.36, bold=False, indent=16) -> None:
        _text(self.img, value, (indent, self.y), scale, color, 2 if bold else 1)
        self.y += self.line_h


def _stats(vals: list[float]) -> tuple[float, float, float]:
    if not vals:
        return 0.0, 0.0, 0.0
    a = np.asarray(vals, dtype=np.float64)
    return float(a.min()), float(a.mean()), float(a.max())


def _build_panel_level1(height: int, width: int, out: VisionPilotOutput, speed_ms: float, metrics: HostMetrics, h_real: bool, cfg: RenderConfig) -> np.ndarray:
    p = np.full((height, width, 3), CLR_PANEL, dtype=np.uint8)
    w = _PanelWriter(p, 19)
    w.title("VISIONPILOT DEBUG L1", "runtime / fusion / planner")
    inf, cipo, lat = out.inference, out.inference.cipo, out.inference.lateral

    w.section("RUNTIME", CLR_ACCENT)
    w.kv("Frame / wire", f"#{inf.frame_id} / {out.wire_sequence}")
    w.kv("VisionPilot", f"{inf.visionpilot_ms:.2f} ms", CLR_ACCENT, True)
    w.kv("Wall / pre", f"{inf.wall_ms:.2f} / {inf.pre_ms:.2f} ms")
    w.kv("Wire RTT", f"{metrics.net_rtt_ms:.2f} ms  ema {metrics.rtt_ema_ms:.2f}")
    w.kv("Result rate", f"{metrics.result_fps:.1f} fps  ema {metrics.result_fps_ema:.1f}")
    w.kv("Pending", f"{metrics.pending_count}")
    w.kv("Geometry", f"{metrics.raw_w}x{metrics.raw_h} crop_top={metrics.crop_top} kept={metrics.crop_h}")

    w.section("FUSED LONGITUDINAL", (0, 220, 0))
    w.kv("Valid", str(bool(cipo.valid)), (0, 220, 0) if cipo.valid else CLR_MUTED, cipo.valid)
    w.kv("Distance", f"{cipo.distance_m:.2f} +/- {cipo.distance_stddev_m:.2f} m")
    w.kv("Velocity", f"{cipo.velocity_ms:+.3f} m/s")
    w.kv("Raw / cut-in", f"{int(cipo.cipo_raw_found)} / {int(cipo.cut_in_detected)}")

    w.section("FUSED LATERAL", CLR_FUSED)
    w.kv("Valid / path", f"{int(lat.valid)} / {int(lat.path_valid)}", CLR_FUSED if lat.valid else CLR_MUTED, lat.valid)
    w.kv("CTE", f"{lat.cte_m:+.3f} m   dot {lat.cte_rate_mps:+.3f}")
    w.kv("Yaw", f"{lat.yaw_rad:+.4f} rad  dot {lat.yaw_rate_rps:+.4f}")
    w.kv("Curvature", f"{lat.curvature:+.6f} 1/m")
    w.kv("Path fit", f"{lat.path_inliers}/{lat.path_points}  {lat.path_x_min_m:.1f}..{lat.path_x_max_m:.1f} m")

    w.section("PLANNER / HOST", (255, 200, 120))
    w.kv("Steering", f"{_steer_cmd(out):+.6f} rad", (255, 200, 120), True)
    w.kv("Acceleration", f"{out.plan.acceleration:+.4f} m/s2")
    w.kv("Ego speed", f"{speed_ms*3.6:.1f} km/h  |  {speed_ms*2.23694:.1f} mph")
    w.kv("Warnings", ",".join(map(str, out.plan.warnings)) or "none", (0, 120, 255) if out.plan.warnings else CLR_MUTED)
    w.kv("Homography", "provided" if h_real else "fallback", (0, 220, 0) if h_real else (0, 180, 255))
    w.kv("Camera view", f"{cfg.camera_view} / {cfg.camera_space}")
    return p


def _build_panel_level2(height: int, width: int, out: VisionPilotOutput, speed_ms: float, metrics: HostMetrics, h_real: bool, cfg: RenderConfig) -> np.ndarray:
    p = np.full((height, width, 3), CLR_PANEL, dtype=np.uint8)
    w = _PanelWriter(p, 16)
    w.title("VISIONPILOT DEBUG L2", "per-model outputs + fusion state")
    inf = out.inference
    ad, st, asp, cipo, lat = inf.auto_drive, inf.auto_steer, inf.auto_speed, inf.cipo, inf.lateral

    w.section("RUNTIME", CLR_ACCENT)
    w.kv("Frame", f"#{inf.frame_id}   pending={metrics.pending_count}", label_w=125)
    w.kv("VP / wall / pre", f"{inf.visionpilot_ms:.2f} / {inf.wall_ms:.2f} / {inf.pre_ms:.2f} ms", label_w=125)
    w.kv("RTT / jitter", f"{metrics.net_rtt_ms:.2f} / {metrics.rtt_jitter_ms:.2f} ms", label_w=125)
    w.kv("Wire", f"TX {metrics.tx_bytes/1024:.1f} KiB  RX {out.payload_bytes/1024:.1f} KiB", label_w=125)
    w.kv("Geometry", f"raw={metrics.raw_w}x{metrics.raw_h} top={metrics.crop_top} kept={metrics.crop_h}", label_w=125)

    w.section("AUTODRIVE", (235, 235, 235))
    w.kv("valid", str(ad.valid), label_w=125)
    w.kv("dist_normalized", f"{ad.dist_normalized:+.8f}", label_w=125)
    w.kv("curvature_raw", f"{ad.curvature_raw:+.8f}  -> {ad.curvature_raw*CURV_SCALE:+.6f} 1/m", label_w=125)
    w.kv("flag_prob", f"{ad.flag_prob:.8f}  flag={int(ad.flag_prob >= FLAG_THRESHOLD)}", label_w=125)

    w.section("AUTOSTEER", CLR_GREEN)
    xmin, xmean, xmax = _stats(st.xp)
    hmin, hmean, hmax = _stats(st.h_vector)
    active = sum(v >= 0.5 for v in st.h_vector)
    w.kv("valid / count", f"{int(st.valid)} / {len(st.xp)}", label_w=125)
    w.kv("xp min/mean/max", f"{xmin:+.4f} / {xmean:+.4f} / {xmax:+.4f}", label_w=125)
    w.kv("h min/mean/max", f"{hmin:+.4f} / {hmean:+.4f} / {hmax:+.4f}", label_w=125)
    w.kv("h >= 0.5", f"{active}/{len(st.h_vector)}", label_w=125)

    w.section("AUTOSPEED", (0, 210, 255))
    w.kv("valid / det", f"{int(asp.valid)} / {len(asp.detections)}", label_w=125)
    for i, d in enumerate(sorted(asp.detections, key=lambda x: x.score, reverse=True)[:5]):
        w.line(f"#{i:02d} L{d.class_id} p={d.score:.3f}  [{d.x1:.0f},{d.y1:.0f},{d.x2:.0f},{d.y2:.0f}]", _det_color(d.class_id), 0.31)

    w.section("FUSION", CLR_FUSED)
    w.line(f"CIPO v={int(cipo.valid)} d={cipo.distance_m:.2f}+/-{cipo.distance_stddev_m:.2f}m  vel={cipo.velocity_ms:+.2f}m/s", (0, 220, 0), 0.32)
    w.line(f"LAT  v={int(lat.valid)} path={int(lat.path_valid)} cte={lat.cte_m:+.3f} yaw={lat.yaw_rad:+.4f} k={lat.curvature:+.6f}", CLR_FUSED, 0.32)
    w.line(f"RAW  cte={lat.raw_cte_m:+.3f} yaw={lat.raw_yaw_rad:+.4f} pathK={lat.raw_path_curvature:+.6f} adK={lat.raw_ad_curvature:+.6f}", CLR_TEXT, 0.30)
    w.line(f"FIT  a={lat.path_a:+.6g} b={lat.path_b:+.6g} c={lat.path_c:+.6g}  {lat.path_inliers}/{lat.path_points}", CLR_TEXT, 0.30)

    w.section("PLAN", (255, 200, 120))
    w.line(f"steer={_steer_cmd(out):+.6f} rad   accel={out.plan.acceleration:+.4f} m/s2   ego={speed_ms:.2f} m/s", (255, 200, 120), 0.32)
    w.line(f"warnings={out.plan.warnings or []}   H={'provided' if h_real else 'fallback'}   view={cfg.camera_view}/{cfg.camera_space}", CLR_MUTED, 0.30)
    return p


def _build_panel_level3(height: int, width: int, out: VisionPilotOutput, speed_ms: float, metrics: HostMetrics, h_real: bool, cfg: RenderConfig) -> np.ndarray:
    p = np.full((height, width, 3), CLR_PANEL, dtype=np.uint8)
    inf = out.inference
    ad, st, asp, cipo, lat = inf.auto_drive, inf.auto_steer, inf.auto_speed, inf.cipo, inf.lateral

    _text(p, "VISIONPILOT DEBUG L3", (14, 23), 0.55, (240, 245, 250), 2)
    _text(p, "raw values actually transported V4M -> host", (14, 43), 0.33, CLR_MUTED, 1)
    _text(p, f"frame={inf.frame_id}  seq={out.wire_sequence}  VP={inf.visionpilot_ms:.2f}ms  RTT={metrics.net_rtt_ms:.2f}ms  pending={metrics.pending_count}", (14, 62), 0.31, CLR_ACCENT, 1)

    y = 86
    _text(p, "AUTODRIVE RAW", (14, y), 0.40, (235, 235, 235), 2); y += 18
    _text(p, f"valid={int(ad.valid)}  dist_norm={ad.dist_normalized:+.9f}  curvature_raw={ad.curvature_raw:+.9f}  flag_prob={ad.flag_prob:+.9f}", (14, y), 0.30, CLR_TEXT, 1); y += 23

    _text(p, "AUTOSTEER RAW   index: xp / h_vector", (14, y), 0.40, CLR_GREEN, 2); y += 18
    n = max(len(st.xp), len(st.h_vector))
    cols = 4
    rows = 16
    col_w = max(145, (width - 28) // cols)
    tiny = 0.275
    for r in range(rows):
        for c in range(cols):
            i = r + c * rows
            if i >= n:
                continue
            xp = st.xp[i] if i < len(st.xp) else float("nan")
            hv = st.h_vector[i] if i < len(st.h_vector) else float("nan")
            color = CLR_GREEN if np.isfinite(hv) and hv >= 0.5 else CLR_MUTED
            _text(p, f"{i:02d}: {xp:+.5f} / {hv:+.5f}", (14 + c * col_w, y + r * 15), tiny, color, 1)
    y += rows * 15 + 6

    _text(p, f"AUTOSPEED POST-NMS RAW   valid={int(asp.valid)}  detections={len(asp.detections)}", (14, y), 0.39, (0, 210, 255), 2); y += 17
    _text(p, "idx cls   score        x1       y1       x2       y2       w       h", (14, y), 0.28, CLR_MUTED, 1); y += 15
    max_det_rows = max(1, min(9, (height - y - 150) // 15))
    for i, d in enumerate(asp.detections[:max_det_rows]):
        _text(
            p,
            f"{i:02d}  {d.class_id:2d}  {d.score:0.6f}  {d.x1:7.2f}  {d.y1:7.2f}  {d.x2:7.2f}  {d.y2:7.2f}  {d.x2-d.x1:6.1f}  {d.y2-d.y1:6.1f}",
            (14, y), 0.27, _det_color(d.class_id), 1,
        )
        y += 15
    if len(asp.detections) > max_det_rows:
        _text(p, f"... {len(asp.detections)-max_det_rows} more detections", (14, y), 0.28, CLR_MUTED, 1); y += 15

    y += 4
    _text(p, "FUSION / PLAN FULL STATE", (14, y), 0.39, CLR_FUSED, 2); y += 17
    _text(p, f"CIPO valid={int(cipo.valid)} d={cipo.distance_m:+.5f} std={cipo.distance_stddev_m:+.5f} v={cipo.velocity_ms:+.5f} raw={int(cipo.cipo_raw_found)} raw_d={cipo.cipo_raw_dist_m:+.5f} cutin={int(cipo.cut_in_detected)}", (14, y), 0.27, (0, 220, 0), 1); y += 15
    _text(p, f"LAT valid={int(lat.valid)} cte={lat.cte_m:+.6f} cte_dot={lat.cte_rate_mps:+.6f} yaw={lat.yaw_rad:+.6f} yaw_dot={lat.yaw_rate_rps:+.6f}", (14, y), 0.27, CLR_FUSED, 1); y += 15
    _text(p, f"LAT std_cte={lat.cte_stddev_m:+.6f} std_yaw={lat.yaw_stddev_rad:+.6f} k={lat.curvature:+.7f} std_k={lat.curv_stddev:+.7f}", (14, y), 0.27, CLR_FUSED, 1); y += 15
    _text(p, f"PATH valid={int(lat.path_valid)} raw_cte={lat.raw_cte_m:+.6f} raw_yaw={lat.raw_yaw_rad:+.6f} raw_path_k={lat.raw_path_curvature:+.7f} raw_ad_k={lat.raw_ad_curvature:+.7f}", (14, y), 0.27, CLR_TEXT, 1); y += 15
    _text(p, f"FIT a={lat.path_a:+.8g} b={lat.path_b:+.8g} c={lat.path_c:+.8g} x=[{lat.path_x_min_m:.3f},{lat.path_x_max_m:.3f}] inliers={lat.path_inliers}/{lat.path_points}", (14, y), 0.27, CLR_TEXT, 1); y += 15
    _text(p, f"PLAN steer={out.plan.steering} accel={out.plan.acceleration:+.7f} warnings={out.plan.warnings} ego={speed_ms:+.4f}m/s", (14, y), 0.27, (255, 200, 120), 1); y += 15
    _text(p, f"WIRE tx={metrics.tx_bytes}B rx={out.payload_bytes}B  effTX={metrics.tx_mbps:.2f}Mb/s  H={'provided' if h_real else 'fallback'}  view={cfg.camera_view}/{cfg.camera_space}", (14, y), 0.27, CLR_MUTED, 1)
    return p


def build_diagnostic_panel(height: int, width: int, level: int, out: VisionPilotOutput, speed_ms: float, metrics: HostMetrics, h_real: bool, cfg: RenderConfig) -> np.ndarray:
    if level <= 1:
        return _build_panel_level1(height, width, out, speed_ms, metrics, h_real, cfg)
    if level == 2:
        return _build_panel_level2(height, width, out, speed_ms, metrics, h_real, cfg)
    return _build_panel_level3(height, width, out, speed_ms, metrics, h_real, cfg)


# =============================================================================
# High-level renderer / single-window dashboard
# =============================================================================


class VisionPilotRenderer:
    def __init__(self, cfg: RenderConfig):
        self.cfg = cfg
        self.cfg.debug_level = int(np.clip(self.cfg.debug_level, 0, 3))
        if self.cfg.camera_view not in ("production", "debug"):
            raise ValueError("camera_view must be 'production' or 'debug'")
        if self.cfg.camera_space not in ("native", "net"):
            raise ValueError("camera_space must be 'native' or 'net'")
        self.H_provided = _load_homography(cfg.homography_path, cfg.homography_key)
        self.occupancy = OccupancyRenderer()
        self._last_scale = 1.0
        self._last_occ_rect_base: Optional[tuple[int, int, int, int]] = None
        self._icons = self._load_icons()
        self._wheels = self._load_wheels()

    def _load_icons(self) -> dict[str, Optional[np.ndarray]]:
        cwd = Path.cwd()
        d = _resolve_asset_dir(
            self.cfg.icons_dir,
            [cwd / "../assets/icons", cwd / "assets/icons", cwd / "VisionPilot/assets/icons", Path("/usr/share/visionpilot/assets/icons")],
        )
        if d is None:
            return {"brake": None, "collision": None, "rld": None, "lld": None}
        rld = _load_rgba(d / "right_lane_departure.png", 80)
        lld = cv2.flip(rld, 1) if rld is not None else None
        return {
            "brake": _load_rgba(d / "brake.png", 80),
            "collision": _load_rgba(d / "collision.png", 80),
            "rld": rld,
            "lld": lld,
        }

    def _load_wheels(self) -> dict[str, Optional[np.ndarray]]:
        cwd = Path.cwd()
        d = _resolve_asset_dir(
            self.cfg.wheel_dir,
            [
                cwd / "../0.9/images",
                cwd / "../../development_releases/0.9/images",
                cwd / "VisionPilot/development_releases/0.9/images",
                cwd / "VisionPilot/production_release/images",
                Path("VisionPilot/development_releases/0.9/images"),
                Path("VisionPilot/production_release/images"),
            ],
        )
        if d is None:
            return {"white": None, "green": None}
        return {"white": _load_rgba(d / "wheel_white.png", 92), "green": _load_rgba(d / "wheel_green.png", 92)}

    def render(
        self,
        frame: np.ndarray,
        out: VisionPilotOutput,
        speed_ms: float,
        net_rtt_ms: float = 0.0,
        tx_mbps: float = 0.0,
        *,
        tx_bytes: int = 0,
        pending_count: int = 0,
        result_fps: float = 0.0,
        result_fps_ema: float = 0.0,
        rtt_ema_ms: float = 0.0,
        rtt_jitter_ms: float = 0.0,
    ) -> np.ndarray:
        if frame is None or frame.size == 0:
            raise ValueError("empty frame")

        geom = _frame_geometry(frame.shape)
        H_px2world, h_real = _net_homography_px_to_world(
            self.H_provided, self.cfg.homography_space, geom
        )
        H_world2display = _world_to_display_homography(H_px2world, geom, self.cfg.camera_space)

        # Two geometrically valid visualization spaces:
        #   native: original camera frame + inverse top-crop/resize mapping for model outputs
        #   net:    exactly the C++ resized input (top-crop -> resize 1024x512)
        camera_input = frame.copy() if self.cfg.camera_space == "native" else _make_net_view(frame, geom)

        if self.cfg.camera_view == "debug":
            camera = draw_debug_frame(
                camera_input, out, H_world2display, geom,
                self.cfg.source_label, self._wheels, self.cfg
            )
        else:
            camera = draw_production_frame(
                camera_input, out, speed_ms, self.cfg.speed_limit_ms,
                H_world2display, geom, self._icons, self.cfg
            )

        metrics = HostMetrics(
            net_rtt_ms=net_rtt_ms,
            tx_mbps=tx_mbps,
            tx_bytes=tx_bytes,
            pending_count=pending_count,
            result_fps=result_fps,
            result_fps_ema=result_fps_ema,
            rtt_ema_ms=rtt_ema_ms,
            rtt_jitter_ms=rtt_jitter_ms,
            raw_w=geom.native_w,
            raw_h=geom.native_h,
            crop_top=geom.crop_top,
            crop_h=geom.crop_h,
        )

        # Dashboard canonical height is the C++ occupancy height (700 px).
        dash_h = PANEL_H
        cam_w = int(round(camera.shape[1] * dash_h / camera.shape[0]))
        camera = cv2.resize(camera, (cam_w, dash_h), interpolation=cv2.INTER_LINEAR)
        pieces = [camera]
        x_cursor = cam_w
        self._last_occ_rect_base = None

        if self.cfg.show_occupancy:
            # AutoSpeed boxes can be projected into world only with the real
            # resized-image homography. The warped fallback is not valid here.
            scene = make_occupancy_scene(out, H_px2world if h_real else None)
            occ = self.occupancy.render(scene)
            pieces.append(occ)
            self._last_occ_rect_base = (x_cursor, 0, PANEL_W, PANEL_H)
            x_cursor += PANEL_W

        if self.cfg.debug_level > 0:
            side_w = max(self.cfg.side_panel_width, 680 if self.cfg.debug_level >= 3 else 430)
            panel = build_diagnostic_panel(dash_h, side_w, self.cfg.debug_level, out, speed_ms, metrics, h_real, self.cfg)
            pieces.append(panel)

        base = np.hstack(pieces)

        # Scale the assembled UI once; all internal geometry remains identical to
        # the 1024x512 / 560x700 reference renderers before this final display step.
        if self.cfg.display_width > 0 and base.shape[1] != self.cfg.display_width:
            scale = self.cfg.display_width / float(base.shape[1])
            out_h = max(1, int(round(base.shape[0] * scale)))
            display = cv2.resize(base, (self.cfg.display_width, out_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
            self._last_scale = scale
        else:
            display = base
            self._last_scale = 1.0
        return display

    def on_mouse(self, event: int, x: int, y: int, flags: int, _param=None) -> None:
        if self._last_occ_rect_base is None:
            return
        scale = self._last_scale if self._last_scale > 1e-9 else 1.0
        bx, by = x / scale, y / scale
        ox, oy, ow, oh = self._last_occ_rect_base
        if ox <= bx < ox + ow and oy <= by < oy + oh:
            self.occupancy.on_mouse(event, int(round(bx - ox)), int(round(by - oy)), flags)

    def on_key(self, key: int) -> bool:
        """Handle renderer hotkeys. Returns True when the key was consumed."""
        k = key & 0xFF
        if ord("0") <= k <= ord("3"):
            self.cfg.debug_level = k - ord("0")
            return True
        if k in (ord("v"), ord("V")):
            self.cfg.camera_view = "debug" if self.cfg.camera_view == "production" else "production"
            return True
        if k in (ord("o"), ord("O")):
            self.cfg.show_occupancy = not self.cfg.show_occupancy
            return True
        if k in (ord("r"), ord("R"), ord("+"), ord("="), ord("-"), ord("_")):
            self.occupancy.on_key(k)
            return True
        return False
