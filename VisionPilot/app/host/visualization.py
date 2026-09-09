from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from wire import VisionPilotOutput

NET_W = 1024
NET_H = 512
D_MAX = 150.0
CURV_SCALE = 0.21
FLAG_THRESHOLD = 0.65
WHEELBASE_M = 2.984
STEER_RATIO = 16.8

# Same warped-pixel -> world fallback matrix used by VisionPilot C++ rendering.
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
CLR_MUTED = (160, 160, 160)
CLR_PANEL = (15, 18, 22)
CLR_ACCENT = (180, 220, 255)


@dataclass
class RenderConfig:
    display_width: int = 1280
    side_panel_width: int = 360
    show_autosteer: bool = True
    show_fused_path: bool = True
    show_detections: bool = True
    path_corridor: bool = True
    homography_path: Optional[str] = None
    homography_key: str = "H"
    homography_space: str = "net"  # net or native


def _alpha_rect(img, p1, p2, color, alpha):
    overlay = img.copy()
    cv2.rectangle(overlay, p1, p2, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)


def _text(img, s, xy, scale=0.48, color=CLR_TEXT, thick=1):
    x, y = xy
    cv2.putText(img, s, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thick, cv2.LINE_AA)


def _fmt(v, digits=2):
    return f"{v:.{digits}f}"


def _det_color(class_id):
    return DET_COLORS.get(class_id, CLR_OTHER)


def _scale_point_from_net(x, y, W, H):
    return int(round(x * W / NET_W)), int(round(y * H / NET_H))


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


def _world_to_native_matrix(
    frame_shape,
    cfg: RenderConfig,
    provided_H: Optional[np.ndarray],
) -> np.ndarray:
    h, w = frame_shape[:2]

    if provided_H is None:
        # Fallback identical to the hard-coded C++ debug/production matrix, whose
        # inverse projects world coordinates into the 1024x512 VisionPilot plane.
        H_world_to_net = np.linalg.inv(H_WARP_TO_WORLD)
        S = np.array([[w / NET_W, 0, 0], [0, h / NET_H, 0], [0, 0, 1]], dtype=np.float64)
        return S @ H_world_to_net

    # Convention: supplied H maps pixels -> world, as in the C++ renderer's
    # H_resized. Invert it for world -> pixels.
    H_world_to_px = np.linalg.inv(provided_H)

    if cfg.homography_space == "net":
        S = np.array([[w / NET_W, 0, 0], [0, h / NET_H, 0], [0, 0, 1]], dtype=np.float64)
        H_world_to_px = S @ H_world_to_px
    elif cfg.homography_space != "native":
        raise ValueError("--homography-space must be net or native")

    return H_world_to_px


def _project(H, x, y):
    p = H @ np.array([x, y, 1.0], dtype=np.float64)
    if abs(p[2]) < 1e-9:
        return None
    return int(round(p[0] / p[2])), int(round(p[1] / p[2]))


def _path_color(acc):
    if acc < -5.0:
        return (93, 0, 255)
    if acc < -3.0:
        return (0, 102, 255)
    if acc < -1.0:
        return (0, 213, 255)
    return (174, 255, 0)


def draw_fused_path(frame, out: VisionPilotOutput, H_world_to_px, corridor=True):
    lat = out.inference.lateral
    if not lat.path_valid:
        return

    x0 = max(0.5, lat.path_x_min_m)
    x1 = lat.path_x_max_m
    if x1 <= x0 + 1.0:
        return

    xs = np.arange(x0, x1 + 0.01, 0.5)
    center = []
    left = []
    right = []

    for x in xs:
        y = lat.path_a * x * x + lat.path_b * x + lat.path_c
        pc = _project(H_world_to_px, x, y)
        pl = _project(H_world_to_px, x, y + 1.0)
        pr = _project(H_world_to_px, x, y - 1.0)
        if pc is not None:
            center.append(pc)
        if pl is not None and pr is not None:
            left.append(pl)
            right.append(pr)

    if corridor and len(left) >= 2 and len(right) >= 2:
        overlay = frame.copy()
        color = _path_color(out.plan.acceleration)
        for i in range(min(len(left), len(right)) - 1):
            q = np.asarray([left[i], right[i], right[i + 1], left[i + 1]], dtype=np.int32)
            cv2.fillConvexPoly(overlay, q, color, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

    visible = [(x, y) for x, y in center if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]]
    if len(visible) >= 2:
        cv2.polylines(frame, [np.asarray(visible, np.int32)], False, CLR_FUSED, 2, cv2.LINE_AA)


def draw_autosteer(frame, out: VisionPilotOutput):
    steer = out.inference.auto_steer
    if not steer.valid or not steer.xp or not steer.h_vector:
        return

    H, W = frame.shape[:2]
    n = min(len(steer.xp), len(steer.h_vector))
    ys_net = np.linspace(0, NET_H - 1, n)

    pts = []
    for xp, conf, yn in zip(steer.xp[:n], steer.h_vector[:n], ys_net):
        if conf < 0.5:
            continue
        xn = xp * NET_W
        x, y = _scale_point_from_net(xn, yn, W, H)
        if 0 <= x < W and 0 <= y < H:
            pts.append((x, y))
            cv2.circle(frame, (x, y), max(2, W // 500), CLR_GREEN, -1, cv2.LINE_AA)

    if len(pts) >= 2:
        cv2.polylines(frame, [np.asarray(pts, np.int32)], False, CLR_GREEN, 2, cv2.LINE_AA)


def draw_detections(frame, out: VisionPilotOutput):
    asp = out.inference.auto_speed
    if not asp.valid:
        return

    H, W = frame.shape[:2]
    overlay = frame.copy()

    scaled = []
    for d in asp.detections:
        x1, y1 = _scale_point_from_net(d.x1, d.y1, W, H)
        x2, y2 = _scale_point_from_net(d.x2, d.y2, W, H)
        color = _det_color(d.class_id)
        scaled.append((d, x1, y1, x2, y2, color))
        fill = tuple(int(c * 0.35) for c in color)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), fill, -1)

    cv2.addWeighted(overlay, 0.30, frame, 0.70, 0, frame)

    for d, x1, y1, x2, y2, color in scaled:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        _text(frame, f"L{d.class_id}  {d.score*100:.0f}%", (x1 + 3, max(16, y1 - 5)),
              0.42, color, 1)

    # Same idea as C++: put fused CIPO distance on the most centered class-1 box.
    cipo = out.inference.cipo
    l1 = [s for s in scaled if s[0].class_id == 1]
    if cipo.valid and cipo.distance_m > 0 and l1:
        best = min(l1, key=lambda s: abs(((s[1] + s[3]) * 0.5) - W * 0.5))
        _, x1, y1, x2, _, _ = best
        label = f"{cipo.distance_m:.0f} m"
        tx = (x1 + x2) // 2
        _alpha_rect(frame, (tx - 35, max(0, y1 - 31)), (tx + 35, max(1, y1 - 7)),
                    (10, 10, 10), 0.78)
        _text(frame, label, (tx - 25, max(16, y1 - 12)), 0.48, (255, 255, 255), 1)


def draw_top_bar(frame, out: VisionPilotOutput, speed_ms: float, net_rtt_ms: float, tx_mbps: float):
    inf = out.inference
    _alpha_rect(frame, (0, 0), (frame.shape[1], 33), (0, 0, 0), 0.72)
    fps = 1000.0 / inf.visionpilot_ms if inf.visionpilot_ms > 0 else 0.0
    s = (
        f"VISIONPILOT / R-CAR V4M    #{inf.frame_id}    "
        f"VP {inf.visionpilot_ms:.1f} ms ({fps:.1f} fps)    "
        f"pre {inf.pre_ms:.1f} ms    RTT {net_rtt_ms:.1f} ms    "
        f"TX {tx_mbps:.1f} Mb/s    ego {speed_ms*3.6:.1f} km/h"
    )
    _text(frame, s, (10, 22), 0.45, (225, 230, 235), 1)


def draw_warning_strip(frame, warnings):
    labels = {1: "FCW", 2: "AEB", 3: "LEFT LANE DEPARTURE", 4: "RIGHT LANE DEPARTURE"}
    active = [labels[w] for w in warnings if w in labels]
    if not active:
        return
    h, w = frame.shape[:2]
    color = (0, 0, 200) if 2 in warnings else (0, 120, 255)
    _alpha_rect(frame, (0, h - 42), (w, h), color, 0.48)
    msg = "   |   ".join(active)
    _text(frame, msg, (20, h - 14), 0.62, (255, 255, 255), 2)


def _panel_line(panel, label, value, y, color=CLR_TEXT, bold=False):
    _text(panel, label, (18, y), 0.42, CLR_MUTED, 1)
    _text(panel, value, (158, y), 0.46, color, 2 if bold else 1)


def _section(panel, title, y, color):
    cv2.line(panel, (14, y + 7), (panel.shape[1] - 14, y + 7), (55, 60, 68), 1)
    _text(panel, title, (18, y), 0.47, color, 2)
    return y + 27


def build_side_panel(height, width, out: VisionPilotOutput, speed_ms, net_rtt_ms, rx_kib):
    p = np.full((height, width, 3), CLR_PANEL, dtype=np.uint8)
    inf = out.inference
    ad = inf.auto_drive
    cipo = inf.cipo
    lat = inf.lateral

    _text(p, "VISIONPILOT TELEMETRY", (18, 27), 0.58, (240, 245, 250), 2)
    _text(p, "Renesas R-Car V4M / remote inference", (18, 48), 0.36, CLR_MUTED, 1)

    y = 78
    y = _section(p, "RUNTIME", y, CLR_ACCENT)
    _panel_line(p, "Frame", f"#{inf.frame_id}", y); y += 21
    _panel_line(p, "VisionPilot", f"{inf.visionpilot_ms:.2f} ms", y, CLR_ACCENT, True); y += 21
    _panel_line(p, "Wall / preprocess", f"{inf.wall_ms:.2f} / {inf.pre_ms:.2f} ms", y); y += 21
    _panel_line(p, "Wire RTT", f"{net_rtt_ms:.2f} ms", y); y += 21
    _panel_line(p, "Result payload", f"{rx_kib:.1f} KiB", y); y += 26

    y = _section(p, "AUTODRIVE", y, (235, 235, 235))
    if ad.valid:
        dist = D_MAX * (1.0 - ad.dist_normalized)
        _panel_line(p, "Distance", f"{dist:.1f} m", y); y += 21
        _panel_line(p, "Curvature raw", f"{ad.curvature_raw:.5f}", y); y += 21
        _panel_line(p, "CIPO probability", f"{ad.flag_prob:.3f}", y,
                    (0, 220, 255) if ad.flag_prob >= FLAG_THRESHOLD else CLR_TEXT, True); y += 26
    else:
        _panel_line(p, "State", "NO OUTPUT", y, (100, 100, 255), True); y += 26

    y = _section(p, "AUTOSTEER / AUTOSPEED", y, CLR_GREEN)
    steer = inf.auto_steer
    valid_pts = sum(1 for h in steer.h_vector if h >= 0.5) if steer.h_vector else 0
    _panel_line(p, "AutoSteer", f"{valid_pts}/64 points", y,
                CLR_GREEN if steer.valid else CLR_MUTED, steer.valid); y += 21
    _panel_line(p, "AutoSpeed", f"{len(inf.auto_speed.detections)} detections", y,
                (0, 210, 255) if inf.auto_speed.valid else CLR_MUTED, inf.auto_speed.valid); y += 26

    y = _section(p, "FUSED LONGITUDINAL", y, (0, 220, 0))
    if cipo.valid:
        _panel_line(p, "Distance", f"{cipo.distance_m:.1f} +/- {cipo.distance_stddev_m:.1f} m", y,
                    (0, 220, 0), True); y += 21
        _panel_line(p, "Velocity", f"{cipo.velocity_ms:+.2f} m/s", y); y += 21
        _panel_line(p, "Raw / cut-in", f"{int(cipo.cipo_raw_found)} / {int(cipo.cut_in_detected)}", y); y += 26
    else:
        _panel_line(p, "Tracker", "WAITING", y, CLR_MUTED); y += 26

    y = _section(p, "FUSED LATERAL", y, CLR_FUSED)
    if lat.valid:
        _panel_line(p, "CTE", f"{lat.cte_m:+.2f} m  ({lat.cte_rate_mps:+.2f} m/s)", y, CLR_FUSED, True); y += 21
        _panel_line(p, "Yaw", f"{lat.yaw_rad:+.3f} rad", y); y += 21
        _panel_line(p, "Curvature", f"{lat.curvature:+.5f} 1/m", y); y += 21
        _panel_line(p, "Path fit", f"{lat.path_inliers}/{lat.path_points}", y); y += 21
        _panel_line(p, "Extent", f"{lat.path_x_min_m:.1f} .. {lat.path_x_max_m:.1f} m", y); y += 26
    else:
        _panel_line(p, "Fusion", "NO PATH", y, CLR_MUTED); y += 26

    y = _section(p, "PLANNER", y, (255, 200, 120))
    steer_cmd = out.plan.steering[1] if len(out.plan.steering) > 1 else (
        out.plan.steering[0] if out.plan.steering else 0.0
    )
    _panel_line(p, "Acceleration", f"{out.plan.acceleration:+.3f} m/s2", y); y += 21
    _panel_line(p, "Steering", f"{steer_cmd:+.5f} rad", y, (255, 200, 120), True); y += 21
    _panel_line(p, "Ego speed", f"{speed_ms*3.6:.1f} km/h", y); y += 21

    warning_names = {1: "FCW", 2: "AEB", 3: "LLDW", 4: "RLDW"}
    warning_text = ", ".join(warning_names.get(w, str(w)) for w in out.plan.warnings) or "none"
    _panel_line(p, "Warnings", warning_text, y,
                (0, 120, 255) if out.plan.warnings else CLR_MUTED, bool(out.plan.warnings))

    return p


class VisionPilotRenderer:
    def __init__(self, cfg: RenderConfig):
        self.cfg = cfg
        self.H_px_to_world = _load_homography(cfg.homography_path, cfg.homography_key)

    def render(
        self,
        frame: np.ndarray,
        out: VisionPilotOutput,
        speed_ms: float,
        net_rtt_ms: float = 0.0,
        tx_mbps: float = 0.0,
    ) -> np.ndarray:
        img = frame.copy()
        H_world_to_px = _world_to_native_matrix(img.shape, self.cfg, self.H_px_to_world)

        # Scene overlay first.
        if self.cfg.show_fused_path:
            draw_fused_path(img, out, H_world_to_px, self.cfg.path_corridor)
        if self.cfg.show_autosteer:
            draw_autosteer(img, out)
        if self.cfg.show_detections:
            draw_detections(img, out)

        # HUD/chrome on top.
        draw_top_bar(img, out, speed_ms, net_rtt_ms, tx_mbps)
        draw_warning_strip(img, out.plan.warnings)

        panel_w = self.cfg.side_panel_width
        target_w = max(640, self.cfg.display_width - panel_w)
        scale = target_w / img.shape[1]
        target_h = int(round(img.shape[0] * scale))

        # Keep the camera image aspect ratio. The side panel exactly matches it.
        camera = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        panel = build_side_panel(
            target_h,
            panel_w,
            out,
            speed_ms,
            net_rtt_ms,
            out.payload_bytes / 1024.0,
        )
        return np.hstack([camera, panel])
