from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Optional

import cv2
import numpy as np


PANEL_W = 560
PANEL_H = 700
X_MAX_M = 150.0
Y_MAX_M = 12.0


@dataclass
class SceneDetection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int


@dataclass
class OccupancyScene:
    path_valid: bool = False
    path_a: float = 0.0
    path_b: float = 0.0
    path_c: float = 0.0

    cte_m: float = 0.0
    yaw_rad: float = 0.0

    # 1024x512 inference/display pixel -> world (x_forward, y_lateral), metres.
    H_px2world: Optional[np.ndarray] = None

    lane_world: list[tuple[float, float]] = field(default_factory=list)
    detections: list[SceneDetection] = field(default_factory=list)

    cipo_valid: bool = False
    cipo_distance_m: float = 0.0

    ad_cipo_only: bool = False
    ad_distance_m: float = 0.0


@dataclass
class OccCamera:
    # Same defaults as visualization/occupancy_view.cpp.
    yaw: float = 0.12
    pitch: float = 0.92
    dist: float = 52.0
    tx: float = 18.0
    ty: float = 0.0
    tz: float = 0.0
    mode: int = 0  # 0 idle, 1 orbit, 2 pan
    lx: int = 0
    ly: int = 0


@dataclass
class OccBasis:
    eye_x: float
    eye_y: float
    eye_z: float
    r: np.ndarray
    u: np.ndarray
    f: np.ndarray
    foc: float
    pw: int
    ph: int


@dataclass
class _Voxel:
    x: float
    y: float
    half_w: float
    half_l: float
    class_id: int
    is_truck: bool
    is_cipo: bool
    cam_depth: float


class OccupancyRenderer:
    """Python port of VisionPilot C++ heuristic occupancy_view.cpp.

    This is intentionally not a learned occupancy network. It projects the
    stock VisionPilot outputs into a 3-D-looking engineering scene, preserving
    the C++ camera, geometry, path ribbon, vehicle primitives and interactions.
    """

    def __init__(self) -> None:
        self.camera = OccCamera()
        self._cte_s = 0.0
        self._yaw_s = 0.0

    def reset_camera(self) -> None:
        self.camera = OccCamera()

    def on_key(self, key: int) -> None:
        k = key & 0xFF
        if k in (ord("r"), ord("R")):
            self.reset_camera()
        elif k in (ord("+"), ord("=")):
            self.camera.dist = float(np.clip(self.camera.dist * 0.92, 12.0, 160.0))
        elif k in (ord("-"), ord("_")):
            self.camera.dist = float(np.clip(self.camera.dist * 1.08, 12.0, 160.0))

    def on_mouse(self, event: int, x: int, y: int, flags: int) -> None:
        c = self.camera
        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.reset_camera()
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            c.mode = 1
            c.lx, c.ly = x, y
        elif event in (cv2.EVENT_RBUTTONDOWN, cv2.EVENT_MBUTTONDOWN):
            c.mode = 2
            c.lx, c.ly = x, y
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP, cv2.EVENT_MBUTTONUP):
            c.mode = 0
        elif event == cv2.EVENT_MOUSEMOVE and c.mode != 0:
            dx = float(x - c.lx)
            dy = float(y - c.ly)
            c.lx, c.ly = x, y
            if c.mode == 1:
                c.yaw += dx * 0.0075
                c.pitch = float(np.clip(c.pitch + dy * 0.0055, 0.18, 1.45))
            else:
                cy, sy = math.cos(c.yaw), math.sin(c.yaw)
                scale = c.dist * 0.0018
                c.tx += (-sy * dx + cy * dy) * scale
                c.ty += (cy * dx + sy * dy) * scale
                c.tx = float(np.clip(c.tx, -10.0, 90.0))
                c.ty = float(np.clip(c.ty, -25.0, 25.0))
        elif event in (cv2.EVENT_MOUSEWHEEL, cv2.EVENT_MOUSEHWHEEL):
            # OpenCV encodes the wheel delta in the high 16 bits of flags.
            # cv2.getMouseWheelDelta is not exposed in every Python build.
            raw = (int(flags) >> 16) & 0xFFFF
            delta = raw - 0x10000 if (raw & 0x8000) else raw
            factor = 0.90 if delta > 0 else 1.11
            c.dist = float(np.clip(c.dist * factor, 12.0, 160.0))

    def _make_basis(self, pw: int, ph: int) -> OccBasis:
        c = self.camera
        cp, sp = math.cos(c.pitch), math.sin(c.pitch)
        cy, sy = math.cos(c.yaw), math.sin(c.yaw)

        eye_x = c.tx - c.dist * cp * cy
        eye_y = c.ty - c.dist * cp * sy
        eye_z = c.tz + c.dist * sp

        f = np.array([c.tx - eye_x, c.ty - eye_y, c.tz - eye_z], dtype=np.float64)
        f /= np.linalg.norm(f) + 1e-6

        # C++: right = forward x world_up = (f_y, -f_x, 0).
        r = np.array([f[1], -f[0], 0.0], dtype=np.float64)
        r /= np.linalg.norm(r) + 1e-6
        u = np.cross(r, f)

        fov = math.radians(48.0)
        foc = 0.5 * float(ph) / math.tan(0.5 * fov)
        return OccBasis(eye_x, eye_y, eye_z, r, u, f, foc, pw, ph)

    @staticmethod
    def _project_xyz(b: OccBasis, x: float, y: float, z: float) -> tuple[bool, tuple[int, int], float]:
        d = np.array([x - b.eye_x, y - b.eye_y, z - b.eye_z], dtype=np.float64)
        cam_z = float(np.dot(d, b.f))
        if cam_z < 0.8:
            return False, (0, 0), cam_z
        cam_x = float(np.dot(d, b.r))
        cam_y = float(np.dot(d, b.u))
        px = int(round(0.5 * b.pw + b.foc * cam_x / cam_z))
        py = int(round(0.5 * b.ph - b.foc * cam_y / cam_z))
        return True, (px, py), cam_z

    @staticmethod
    def _scale_bgr(c: tuple[float, float, float], s: float) -> tuple[int, int, int]:
        return tuple(int(np.clip(v * s, 0.0, 255.0)) for v in c)

    @staticmethod
    def _detection_looks_like_truck(width_m: float, bbox_w: float, bbox_h: float) -> bool:
        if width_m >= 2.25:
            return True
        if width_m >= 1.95 and bbox_h > bbox_w * 0.85:
            return True
        if bbox_h > 140.0 and bbox_w > 90.0 and width_m >= 1.7:
            return True
        return False

    @staticmethod
    def _vehicle_paint(class_id: int, is_truck: bool) -> tuple[int, int, int]:
        del is_truck
        if class_id == 0:
            return (215, 212, 208)
        return (220, 200, 0)

    @staticmethod
    def _wall_shade_pts(a: tuple[int, int], b: tuple[int, int]) -> float:
        ex = float(b[0] - a[0])
        ey = float(b[1] - a[1])
        nx, ny = ey, -ex
        length = math.sqrt(nx * nx + ny * ny)
        if length < 1e-3:
            return 0.55
        nx /= length
        ny /= length
        lx, ly = 0.40, -0.85
        return float(np.clip(0.38 + 0.55 * (nx * lx + ny * ly), 0.22, 0.92))

    def _extrude_box_3d(
        self,
        layer: np.ndarray,
        basis: OccBasis,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        z0: float,
        z1: float,
        paint: tuple[int, int, int],
        fade: float,
        top_boost: float = 1.10,
    ) -> None:
        if z1 <= z0 + 0.05:
            return

        xs = [x0, x0, x1, x1]
        ys = [y1, y0, y0, y1]
        foot: list[tuple[int, int]] = []
        top: list[tuple[int, int]] = []
        fd: list[float] = []
        td: list[float] = []
        for i in range(4):
            ok0, p0, d0 = self._project_xyz(basis, xs[i], ys[i], z0)
            ok1, p1, d1 = self._project_xyz(basis, xs[i], ys[i], z1)
            if not ok0 or not ok1:
                return
            foot.append(p0)
            top.append(p1)
            fd.append(d0)
            td.append(d1)

        faces: list[tuple[float, int, int]] = []
        for i in range(4):
            j = (i + 1) % 4
            depth = 0.25 * (fd[i] + fd[j] + td[i] + td[j])
            faces.append((depth, 0, i))
        faces.append((0.25 * sum(td), 1, 0))
        faces.sort(key=lambda f: f[0], reverse=True)

        for _, kind, i in faces:
            if kind == 1:
                poly = np.asarray(top, dtype=np.int32)
                cv2.fillConvexPoly(layer, poly, self._scale_bgr(paint, top_boost * fade), cv2.LINE_AA)
                cv2.polylines(
                    layer,
                    [poly],
                    True,
                    self._scale_bgr((255, 255, 255), 0.35 * fade),
                    1,
                    cv2.LINE_AA,
                )
                cv2.line(
                    layer,
                    top[0],
                    top[1],
                    self._scale_bgr((255, 255, 255), 0.55 * fade),
                    1,
                    cv2.LINE_AA,
                )
            else:
                j = (i + 1) % 4
                sh = self._wall_shade_pts(foot[i], foot[j])
                wall = np.asarray([foot[i], foot[j], top[j], top[i]], dtype=np.int32)
                cv2.fillConvexPoly(layer, wall, self._scale_bgr(paint, sh * fade), cv2.LINE_AA)

    def _extrude_box_yaw(
        self,
        layer: np.ndarray,
        basis: OccBasis,
        x_fwd: float,
        y_lat: float,
        half_l: float,
        half_w: float,
        z0: float,
        z1: float,
        yaw: float,
        paint: tuple[int, int, int],
        fade: float,
        top_boost: float = 1.12,
    ) -> None:
        cy, sy = math.cos(yaw), math.sin(yaw)
        local = [(-half_l, -half_w), (+half_l, -half_w), (+half_l, +half_w), (-half_l, +half_w)]
        xyz = []
        for lx, ly in local:
            xyz.append((x_fwd + cy * lx - sy * ly, y_lat + sy * lx + cy * ly))

        foot: list[tuple[int, int]] = []
        top: list[tuple[int, int]] = []
        fd: list[float] = []
        for xw, yw in xyz:
            ok0, p0, d0 = self._project_xyz(basis, xw, yw, z0)
            ok1, p1, _ = self._project_xyz(basis, xw, yw, z1)
            if not ok0 or not ok1:
                return
            foot.append(p0)
            top.append(p1)
            fd.append(d0)

        roof = np.asarray(top, dtype=np.int32)
        cv2.fillConvexPoly(layer, roof, self._scale_bgr(paint, top_boost * fade), cv2.LINE_AA)

        faces = [(0.5 * (fd[i] + fd[(i + 1) % 4]), i) for i in range(4)]
        faces.sort(key=lambda f: f[0], reverse=True)
        for _, i in faces:
            j = (i + 1) % 4
            dx = float(foot[j][0] - foot[i][0])
            dy = float(foot[j][1] - foot[i][1])
            length = math.sqrt(dx * dx + dy * dy) + 1e-3
            nx = -dy / length
            sh = float(np.clip(0.55 + 0.45 * nx, 0.35, 1.0))
            wall = np.asarray([foot[i], foot[j], top[j], top[i]], dtype=np.int32)
            cv2.fillConvexPoly(layer, wall, self._scale_bgr(paint, sh * fade), cv2.LINE_AA)

    def _draw_extruded_vehicle(
        self,
        layer: np.ndarray,
        basis: OccBasis,
        x_fwd: float,
        y_lat: float,
        length_m: float,
        width_m: float,
        is_truck: bool,
        class_id: int,
        fade: float,
        is_cipo: bool = False,
        yaw: float = 0.0,
    ) -> None:
        paint = (40, 40, 220) if is_cipo else self._vehicle_paint(class_id, is_truck)
        half_l = 0.5 * length_m
        half_w = 0.5 * width_m

        ok, sc, _ = self._project_xyz(basis, x_fwd, y_lat, 0.0)
        if ok:
            d = max(0.4, fade)
            cv2.ellipse(
                layer,
                (sc[0] + 1, sc[1] + 2),
                (max(4, int(width_m * 5.0 * d)), max(2, int(length_m * 2.2 * d))),
                math.degrees(yaw),
                0,
                360,
                (14, 12, 10),
                -1,
                cv2.LINE_AA,
            )

        if abs(yaw) > 1e-3:
            self._extrude_box_yaw(
                layer,
                basis,
                x_fwd,
                y_lat,
                half_l,
                half_w,
                0.0,
                3.0 if is_truck else 1.45,
                yaw,
                paint,
                fade,
                1.05 if (is_truck or is_cipo) else 1.12,
            )
            return

        if not is_truck:
            self._extrude_box_3d(
                layer,
                basis,
                x_fwd - half_l,
                y_lat - half_w,
                x_fwd + half_l,
                y_lat + half_w,
                0.0,
                1.45,
                paint,
                fade,
                1.05 if is_cipo else 1.12,
            )
        else:
            split = x_fwd + half_l * 0.35
            self._extrude_box_3d(
                layer,
                basis,
                x_fwd - half_l,
                y_lat - half_w,
                split,
                y_lat + half_w,
                0.0,
                3.2,
                self._scale_bgr(paint, 1.0 if is_cipo else 0.90),
                fade,
                1.05,
            )
            self._extrude_box_3d(
                layer,
                basis,
                split - half_l * 0.02,
                y_lat - half_w * 0.88,
                x_fwd + half_l,
                y_lat + half_w * 0.88,
                0.0,
                2.5,
                self._scale_bgr(paint, 1.05),
                fade,
                1.12,
            )

    @staticmethod
    def _px_to_world(H: Optional[np.ndarray], u: float, v: float) -> Optional[tuple[float, float]]:
        if H is None or H.size == 0:
            return None
        src = np.asarray([[[u, v]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(src, H.astype(np.float32))[0, 0]
        xw, yw = float(dst[0]), float(dst[1])
        if not np.isfinite(xw) or not np.isfinite(yw):
            return None
        if xw < -2.0 or xw > X_MAX_M + 8.0:
            return None
        return xw, yw

    def render(self, scene: OccupancyScene) -> np.ndarray:
        pw, ph = PANEL_W, PANEL_H

        cte_tgt = float(np.clip(scene.cte_m, -4.5, 4.5))
        yaw_tgt = float(np.clip(scene.yaw_rad, -0.55, 0.55))
        smooth = 0.20
        self._cte_s += smooth * (cte_tgt - self._cte_s)
        self._yaw_s += smooth * (yaw_tgt - self._yaw_s)
        y_shift = self._cte_s
        ego_y = -self._cte_s
        ego_yaw = -self._yaw_s

        basis = self._make_basis(pw, ph)

        def depth_fade(x_fwd: float, y_lat: float) -> float:
            ok, _, d = self._project_xyz(basis, x_fwd, y_lat, 0.0)
            if not ok:
                d = 40.0
            rel = max(0.0, d - self.camera.dist * 0.75)
            return float(np.clip(1.0 - 0.006 * rel, 0.55, 1.0))

        sky_far = np.asarray((34, 27, 22), dtype=np.float32)
        ground_near = np.asarray((52, 43, 36), dtype=np.float32)
        grid_maj = (92, 83, 72)
        grid_min = (72, 63, 54)
        path_fill = (120, 190, 55)
        path_edge = (180, 240, 110)
        lane_color = (90, 88, 86)

        # Vectorized equivalent of the C++ per-pixel sky/ground gradient + vignette.
        ty = np.linspace(0.0, 1.0, ph, dtype=np.float32)[:, None, None]
        row_color = sky_far[None, None, :] + (ground_near - sky_far)[None, None, :] * ty
        nx = (np.arange(pw, dtype=np.float32) - pw * 0.5) / (pw * 0.5)
        edge = (1.0 - 0.18 * nx * nx)[None, :, None]
        panel = np.clip(row_color * edge, 0, 255).astype(np.uint8)

        ground = []
        for xw, yw in ((0.0, Y_MAX_M), (0.0, -Y_MAX_M), (X_MAX_M, -Y_MAX_M), (X_MAX_M, Y_MAX_M)):
            ok, p, _ = self._project_xyz(basis, xw, yw, 0.0)
            if ok:
                ground.append(p)
        if len(ground) == 4:
            cv2.fillConvexPoly(panel, np.asarray(ground, np.int32), (58, 48, 40), cv2.LINE_AA)

        for yw in np.arange(-Y_MAX_M, Y_MAX_M + 0.01, 1.0):
            major = int(round(abs(float(yw)))) % 2 == 0
            oka, a, _ = self._project_xyz(basis, 0.0, float(yw), 0.0)
            okb, b, _ = self._project_xyz(basis, X_MAX_M, float(yw), 0.0)
            if oka and okb:
                cv2.line(panel, a, b, grid_maj if major else grid_min, 1, cv2.LINE_AA)

        for xw in np.arange(0.0, X_MAX_M + 0.01, 5.0):
            fade = depth_fade(float(xw), 0.0)
            oka, a, _ = self._project_xyz(basis, float(xw), -Y_MAX_M, 0.0)
            okb, b, _ = self._project_xyz(basis, float(xw), Y_MAX_M, 0.0)
            if not (oka and okb):
                continue
            cv2.line(panel, a, b, self._scale_bgr(grid_maj, fade), 1, cv2.LINE_AA)
            if xw > 0.0 and int(xw) % 20 == 0:
                okp, p, _ = self._project_xyz(basis, float(xw), -Y_MAX_M + 0.6, 0.0)
                if okp:
                    cv2.putText(
                        panel,
                        f"{xw:.0f}",
                        (p[0] + 3, p[1] - 3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.33,
                        self._scale_bgr((140, 138, 134), fade),
                        1,
                        cv2.LINE_AA,
                    )

        for xw in (20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 150.0):
            fade = depth_fade(xw, 0.0) * 0.7
            oka, a, _ = self._project_xyz(basis, xw, -Y_MAX_M, 0.0)
            okb, b, _ = self._project_xyz(basis, xw, Y_MAX_M, 0.0)
            if oka and okb:
                cv2.line(panel, a, b, self._scale_bgr((82, 74, 64), fade), 1, cv2.LINE_AA)

        if scene.path_valid:
            lp, rp, mid = [], [], []
            for xw in np.arange(0.5, X_MAX_M + 0.01, 1.0):
                yc = scene.path_a * xw * xw + scene.path_b * xw + scene.path_c - y_shift
                okl, pl, _ = self._project_xyz(basis, float(xw), float(yc + 1.15), 0.02)
                okr, pr, _ = self._project_xyz(basis, float(xw), float(yc - 1.15), 0.02)
                okm, pm, _ = self._project_xyz(basis, float(xw), float(yc), 0.02)
                if okl and okr and okm:
                    lp.append(pl)
                    rp.append(pr)
                    mid.append(pm)
            if len(lp) >= 2:
                fill = panel.copy()
                for i in range(len(lp) - 1):
                    quad = np.asarray([lp[i], rp[i], rp[i + 1], lp[i + 1]], np.int32)
                    cv2.fillConvexPoly(fill, quad, path_fill, cv2.LINE_AA)
                cv2.addWeighted(fill, 0.22, panel, 0.78, 0.0, panel)
                cv2.polylines(panel, [np.asarray(lp, np.int32)], False, path_edge, 1, cv2.LINE_AA)
                cv2.polylines(panel, [np.asarray(rp, np.int32)], False, path_edge, 1, cv2.LINE_AA)
                if len(mid) >= 2:
                    cv2.polylines(panel, [np.asarray(mid, np.int32)], False, (210, 255, 160), 1, cv2.LINE_AA)

        if len(scene.lane_world) >= 2:
            lane_px = []
            for xw, yw in scene.lane_world:
                if xw < 0.0 or xw > X_MAX_M:
                    continue
                ok, p, _ = self._project_xyz(basis, xw, yw - y_shift, 0.03)
                if ok:
                    lane_px.append(p)
            if len(lane_px) >= 2:
                cv2.polylines(panel, [np.asarray(lane_px, np.int32)], False, lane_color, 2, cv2.LINE_AA)

        voxels: list[_Voxel] = []
        for d in scene.detections:
            bl = self._px_to_world(scene.H_px2world, d.x1, d.y2)
            br = self._px_to_world(scene.H_px2world, d.x2, d.y2)
            bc = self._px_to_world(scene.H_px2world, 0.5 * (d.x1 + d.x2), d.y2)
            if bl is None or br is None or bc is None:
                continue
            width_m = float(np.clip(abs(bl[1] - br[1]), 0.9, 3.2))
            bbox_w = max(1.0, d.x2 - d.x1)
            bbox_h = max(1.0, d.y2 - d.y1)
            truck = self._detection_looks_like_truck(width_m, bbox_w, bbox_h)
            xw, yw = bc[0], bc[1] - y_shift
            _, _, cam_depth = self._project_xyz(basis, xw, yw, 0.0)
            voxels.append(
                _Voxel(
                    x=xw,
                    y=yw,
                    half_l=5.0 if truck else 2.25,
                    half_w=1.20 if truck else 0.90,
                    class_id=d.class_id,
                    is_truck=truck,
                    is_cipo=False,
                    cam_depth=cam_depth,
                )
            )

        focus_x = (
            scene.cipo_distance_m
            if scene.cipo_valid and scene.cipo_distance_m > 0.5
            else (scene.ad_distance_m if scene.ad_cipo_only and scene.ad_distance_m > 0.5 else -1.0)
        )
        if focus_x > 0.0 and voxels:
            def path_y(xw: float) -> float:
                if not scene.path_valid:
                    return 0.0
                return scene.path_a * xw * xw + scene.path_b * xw + scene.path_c - y_shift

            best_i: Optional[int] = None
            best_score = 1e9
            for i, v in enumerate(voxels):
                dy = abs(v.y - path_y(v.x))
                if dy > 2.0:
                    continue
                dx = abs(v.x - focus_x)
                score = dx + (0.0 if v.class_id == 1 else 2.0)
                if score < best_score:
                    best_score, best_i = score, i
            if best_i is None:
                for i, v in enumerate(voxels):
                    dx = abs(v.x - focus_x)
                    if dx < best_score:
                        best_score, best_i = dx, i
            if best_i is not None and best_score < 25.0:
                voxels[best_i].is_cipo = True

        voxels.sort(key=lambda v: v.cam_depth, reverse=True)
        cipo_car = next((v for v in voxels if v.is_cipo), None)

        # Make projection failures explicit instead of silently showing an empty road.
        if scene.detections:
            if scene.H_px2world is None:
                cv2.putText(
                    panel,
                    "traffic projection disabled: missing camera H",
                    (12, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (80, 180, 255), 1, cv2.LINE_AA,
                )
            elif not voxels:
                cv2.putText(
                    panel,
                    f"traffic projection 0/{len(scene.detections)}: check H/crop",
                    (12, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (80, 180, 255), 1, cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    panel,
                    f"traffic {len(voxels)}/{len(scene.detections)} projected",
                    (12, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 190, 170), 1, cv2.LINE_AA,
                )

        for v in voxels:
            self._draw_extruded_vehicle(
                panel,
                basis,
                v.x,
                v.y,
                v.half_l * 2.0,
                v.half_w * 2.0,
                v.is_truck,
                v.class_id,
                depth_fade(v.x, v.y),
                v.is_cipo,
            )

        if cipo_car is not None:
            z_lbl = 3.4 if cipo_car.is_truck else 1.7
            ok, p, _ = self._project_xyz(basis, cipo_car.x, cipo_car.y, z_lbl)
            if ok:
                lbl = f"{cipo_car.x:.0f}m"
                (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
                origin = (p[0] - tw // 2, p[1] - 8)
                cv2.rectangle(
                    panel,
                    (origin[0] - 4, origin[1] - th - 2),
                    (origin[0] + tw + 4, origin[1] + 4),
                    (20, 20, 20),
                    -1,
                )
                cv2.putText(panel, lbl, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 80, 255), 1, cv2.LINE_AA)
        elif scene.ad_cipo_only and 0.5 < scene.ad_distance_m < X_MAX_M:
            yw = 0.0
            if scene.path_valid:
                xw = scene.ad_distance_m
                yw = scene.path_a * xw * xw + scene.path_b * xw + scene.path_c - y_shift
            ok, p, _ = self._project_xyz(basis, scene.ad_distance_m, yw, 0.4)
            if ok:
                cv2.arrowedLine(panel, (p[0], p[1] + 16), p, (40, 40, 220), 2, cv2.LINE_AA, 0, 0.35)

        self._draw_extruded_vehicle(panel, basis, 1.5, ego_y, 4.5, 1.8, False, 0, 1.0, False, ego_yaw)

        if abs(self._cte_s) > 0.35:
            cv2.putText(
                panel,
                f"lane {-self._cte_s:+0.1f}m",
                (12, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )

        cv2.rectangle(panel, (0, 0), (pw - 1, ph - 1), (70, 85, 50), 1, cv2.LINE_AA)
        self._fill_rect_alpha(panel, (0, 0, pw, 26), (10, 9, 8), 0.62)
        cv2.line(panel, (0, 26), (pw, 26), (90, 110, 55), 1, cv2.LINE_AA)
        cv2.putText(panel, "OCCUPANCY", (12, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 220, 180), 1, cv2.LINE_AA)
        range_lbl = f"0-{X_MAX_M:.0f}m"
        (tw, _), _ = cv2.getTextSize(range_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        cv2.putText(panel, range_lbl, (pw - tw - 12, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 160, 130), 1, cv2.LINE_AA)
        cv2.putText(
            panel,
            "L-drag orbit  R-drag pan  wheel zoom  R reset",
            (12, ph - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (140, 150, 120),
            1,
            cv2.LINE_AA,
        )
        return panel

    @staticmethod
    def _fill_rect_alpha(img: np.ndarray, rect: tuple[int, int, int, int], color, alpha: float) -> None:
        x, y, w, h = rect
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(img.shape[1], x + w), min(img.shape[0], y + h)
        if x2 <= x1 or y2 <= y1:
            return
        roi = img[y1:y2, x1:x2]
        block = np.full_like(roi, color)
        cv2.addWeighted(block, alpha, roi, 1.0 - alpha, 0.0, roi)
