#!/usr/bin/env python3
from __future__ import annotations

import argparse
import queue
import signal
import sys
import threading
import time
import traceback
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from wire import VisionPilotOutput, VisionPilotServer
from visualization import NET_H, NET_W, RenderConfig, VisionPilotRenderer


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class SpeedSource:
    def __init__(self, fixed_ms: float, path: str | None):
        self.fixed_ms = float(fixed_ms)
        self.values: list[float] | None = None
        if path:
            vals: list[float] = []
            for raw in Path(path).read_text().splitlines():
                raw = raw.strip()
                if not raw or raw.startswith("#"):
                    continue
                # Supports either "speed" or CSV where the final column is speed.
                vals.append(float(raw.split(",")[-1].strip()))
            if not vals:
                raise RuntimeError(f"no speed values found in {path}")
            self.values = vals

    def get(self, index: int) -> float:
        if self.values is None:
            return self.fixed_ms
        return self.values[min(index, len(self.values) - 1)]


class FrameSource:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.cap: cv2.VideoCapture | None = None
        self.files: list[Path] | None = None
        self.index = 0
        self.source_fps = float(args.fps)

        if args.video:
            self.cap = cv2.VideoCapture(args.video)
            if not self.cap.isOpened():
                raise RuntimeError(f"cannot open video: {args.video}")
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps and fps > 1e-3:
                self.source_fps = float(fps)
        elif args.camera is not None:
            self.cap = cv2.VideoCapture(args.camera)
            if not self.cap.isOpened():
                raise RuntimeError(f"cannot open camera: {args.camera}")
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps and fps > 1e-3:
                self.source_fps = float(fps)
        elif args.frames:
            folder = Path(args.frames)
            self.files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)
            if not self.files:
                raise RuntimeError(f"no images found in {folder}")
        else:
            raise RuntimeError("select one source: --video, --frames or --camera")

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.cap is not None:
            ok, frame = self.cap.read()
            if not ok:
                return False, None
            self.index += 1
            return True, frame

        assert self.files is not None
        if self.index >= len(self.files):
            return False, None
        frame = cv2.imread(str(self.files[self.index]), cv2.IMREAD_COLOR)
        self.index += 1
        return frame is not None, frame

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()


class ResultReceiver(threading.Thread):
    def __init__(self, server: VisionPilotServer, output_queue: queue.Queue, stop_event: threading.Event):
        super().__init__(name="v4m-result-rx", daemon=True)
        self.server = server
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.error: Exception | None = None

    def run(self) -> None:
        try:
            while not self.stop_event.is_set():
                result = self.server.recv_result()
                self.output_queue.put((time.monotonic_ns(), result))
        except Exception as exc:
            if not self.stop_event.is_set():
                self.error = exc
                print(
                    f"\n[ERROR] Result receiver crashed: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                traceback.print_exc()
                self.stop_event.set()


@dataclass
class RuntimeStats:
    last_rx_ns: int | None = None
    result_fps: float = 0.0
    result_fps_ema: float = 0.0
    rtt_ema_ms: float = 0.0
    rtt_jitter_ms: float = 0.0
    result_count: int = 0

    def update(self, rx_ns: int, rtt_ms: float) -> None:
        if self.last_rx_ns is not None:
            dt_s = (rx_ns - self.last_rx_ns) * 1e-9
            if dt_s > 1e-6:
                self.result_fps = 1.0 / dt_s
                self.result_fps_ema = (
                    self.result_fps
                    if self.result_count <= 1 or self.result_fps_ema <= 0.0
                    else 0.10 * self.result_fps + 0.90 * self.result_fps_ema
                )
        self.last_rx_ns = rx_ns

        if self.result_count == 0:
            self.rtt_ema_ms = rtt_ms
            self.rtt_jitter_ms = 0.0
        else:
            residual = abs(rtt_ms - self.rtt_ema_ms)
            self.rtt_ema_ms = 0.10 * rtt_ms + 0.90 * self.rtt_ema_ms
            self.rtt_jitter_ms = 0.10 * residual + 0.90 * self.rtt_jitter_ms
        self.result_count += 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "VisionPilot Linux host: stream camera frames to Renesas V4M and visualize "
            "remote inference with production/debug/occupancy views."
        ),
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="input video")
    src.add_argument("--frames", help="directory of image frames")
    src.add_argument("--camera", type=int, help="OpenCV/V4L2 camera index")

    p.add_argument("--speed", type=float, default=0.0, help="fixed ego speed [m/s]")
    p.add_argument("--speed-file", help="one speed [m/s] per line; CSV final column also accepted")
    p.add_argument("--speed-limit", type=float, default=0.0, help="speed limit [m/s] shown by production HUD")

    p.add_argument("--bind", default="0.0.0.0", help="host bind address")
    p.add_argument("--frame-port", type=int, default=8080)
    p.add_argument("--result-port", type=int, default=8081)

    p.add_argument("--fps", type=float, default=30.0, help="FPS for --frames or fallback video FPS")
    p.add_argument("--realtime", action="store_true", help="pace frames according to source FPS")
    p.add_argument("--loop", action="store_true")

    # Main requested interface. Level 0 intentionally means no diagnostic side panel.
    p.add_argument(
        "--debug-level",
        type=int,
        choices=(0, 1, 2, 3),
        default=0,
        help="0 production UI only; 1 runtime/fusion; 2 model detail; 3 full transmitted values",
    )
    p.add_argument(
        "--camera-view",
        choices=("production", "debug"),
        default="production",
        help="HUD style: stock VisionPilot production view or stock-style C++ debug view",
    )
    p.add_argument(
        "--camera-space",
        choices=("native", "net"),
        default="native",
        help=(
            "native overlays network outputs back onto the original camera frame; "
            "net shows the exact C++ top-cropped/resized 1024x512 inference view"
        ),
    )
    p.add_argument("--no-occupancy", action="store_true", help="hide the C++-style heuristic 3D occupancy panel")
    p.add_argument("--display-width", type=int, default=1920, help="final assembled window width; 0 keeps native dashboard size")
    p.add_argument("--side-panel-width", type=int, default=520, help="diagnostic panel width before final scaling")
    p.add_argument("--window", default="VisionPilot / Renesas V4M")
    p.add_argument("--source-label", default="remote/V4M", help="source label used by the debug camera view")
    p.add_argument("--icons-dir", help="VisionPilot production HUD icons directory")
    p.add_argument("--wheel-dir", help="VisionPilot debug steering-wheel assets directory")

    # Existing rendering switches retained.
    p.add_argument("--no-autosteer", action="store_true")
    p.add_argument("--no-fused-path", action="store_true")
    p.add_argument("--no-detections", action="store_true")
    p.add_argument("--centerline-only", action="store_true")

    p.add_argument("--homography", help="OpenCV YAML/XML matrix H (pixel -> world)")
    p.add_argument("--homography-key", default="H")
    p.add_argument(
        "--homography-space",
        choices=("net", "native"),
        default="native",
        help=(
            "coordinate space of H; VisionPilot H.yaml is normally native/raw and "
            "is converted with the same top-crop transform as C++ set_H_resized()"
        ),
    )

    p.add_argument(
        "--max-pending",
        type=int,
        default=12,
        help="number of frame images retained for result/frame synchronization",
    )
    p.add_argument(
        "--console-level",
        type=int,
        choices=(0, 1, 2, 3),
        default=0,
        help="terminal logging detail; independent from GUI debug level",
    )
    # Compatibility with the previous script: maps to console level 3.
    p.add_argument("--verbose-results", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


def _steer_cmd(out: VisionPilotOutput) -> float:
    if len(out.plan.steering) > 1:
        return float(out.plan.steering[1])
    if out.plan.steering:
        return float(out.plan.steering[0])
    return 0.0


def print_result(out: VisionPilotOutput, level: int, *, rtt_ms: float = 0.0, result_fps: float = 0.0) -> None:
    if level <= 0:
        return

    i = out.inference
    ad = i.auto_drive
    st = i.auto_steer
    asp = i.auto_speed
    c = i.cipo
    lat = i.lateral

    if level == 1:
        active = sum(v >= 0.5 for v in st.h_vector)
        print(
            f"[V4M #{i.frame_id:06d}] VP={i.visionpilot_ms:7.2f}ms "
            f"RTT={rtt_ms:7.2f}ms RX={result_fps:5.1f}fps | "
            f"AD p={ad.flag_prob:.3f} | ASteer={active:02d}/{len(st.h_vector):02d} "
            f"ASpeed={len(asp.detections):02d} | "
            f"CIPO={int(c.valid)} d={c.distance_m:6.1f}m v={c.velocity_ms:+6.2f}m/s | "
            f"LAT cte={lat.cte_m:+.2f} yaw={lat.yaw_rad:+.3f} k={lat.curvature:+.5f} | "
            f"PLAN steer={_steer_cmd(out):+.5f} acc={out.plan.acceleration:+.3f}",
            flush=True,
        )
        return

    print("\n" + "=" * 112)
    print(
        f" FRAME {i.frame_id} / wire seq {out.wire_sequence} | "
        f"total={i.total_ms:.3f} ms  pre={i.pre_ms:.3f} ms  "
        f"visionpilot={i.visionpilot_ms:.3f} ms  RTT={rtt_ms:.3f} ms"
    )
    print("=" * 112)
    print(
        f"[AUTODRIVE] valid={ad.valid}  dist_normalized={ad.dist_normalized:+.8f}  "
        f"curvature_raw={ad.curvature_raw:+.8f}  flag_prob={ad.flag_prob:+.8f}"
    )
    active = sum(v >= 0.5 for v in st.h_vector)
    if st.xp:
        print(
            f"[AUTOSTEER] valid={st.valid}  n={len(st.xp)}  active={active}/{len(st.h_vector)}  "
            f"xp[min/mean/max]={min(st.xp):+.6f}/{sum(st.xp)/len(st.xp):+.6f}/{max(st.xp):+.6f}"
        )
    else:
        print(f"[AUTOSTEER] valid={st.valid}  n=0")
    print(f"[AUTOSPEED] valid={asp.valid}  detections={len(asp.detections)}")
    for idx, d in enumerate(asp.detections):
        print(
            f"  det[{idx:02d}] cls={d.class_id:2d} score={d.score:.6f} "
            f"xyxy=({d.x1:.2f},{d.y1:.2f},{d.x2:.2f},{d.y2:.2f})"
        )
    print(
        f"[CIPO] valid={c.valid} d={c.distance_m:+.4f}m v={c.velocity_ms:+.4f}m/s "
        f"std={c.distance_stddev_m:.4f} raw={c.cipo_raw_found} raw_d={c.cipo_raw_dist_m:+.4f} cut_in={c.cut_in_detected}"
    )
    print(
        f"[LATERAL] valid={lat.valid} cte={lat.cte_m:+.5f} cte_dot={lat.cte_rate_mps:+.5f} "
        f"yaw={lat.yaw_rad:+.6f} yaw_dot={lat.yaw_rate_rps:+.6f} k={lat.curvature:+.7f} "
        f"path={lat.path_valid} fit={lat.path_inliers}/{lat.path_points}"
    )
    print(
        f"[PLAN] steer={out.plan.steering} acceleration={out.plan.acceleration:+.7f} "
        f"warnings={out.plan.warnings}"
    )

    if level >= 3:
        print("[AUTOSTEER xp]")
        print("  [" + ", ".join(f"{v:+.7f}" for v in st.xp) + "]")
        print("[AUTOSTEER h_vector]")
        print("  [" + ", ".join(f"{v:+.7f}" for v in st.h_vector) + "]")
        print(
            f"[LATERAL RAW] raw_cte={lat.raw_cte_m:+.7f} raw_yaw={lat.raw_yaw_rad:+.7f} "
            f"raw_path_k={lat.raw_path_curvature:+.8f} raw_ad_k={lat.raw_ad_curvature:+.8f} "
            f"std_cte={lat.cte_stddev_m:.7f} std_yaw={lat.yaw_stddev_rad:.7f} std_k={lat.curv_stddev:.8f}"
        )
        print(
            f"[PATH COEFF] a={lat.path_a:+.10g} b={lat.path_b:+.10g} c={lat.path_c:+.10g} "
            f"x=[{lat.path_x_min_m:.4f}, {lat.path_x_max_m:.4f}]"
        )
        print(f"[WIRE] result payload={out.payload_bytes} bytes")
    print("-" * 112, flush=True)


def _warmup_frame(frame: np.ndarray, display_width: int) -> np.ndarray:
    """Mirror the C++ no-inference behavior: display the plain 1024x512 resized frame."""
    img = cv2.resize(frame, (NET_W, NET_H), interpolation=cv2.INTER_LINEAR)
    if display_width > 0 and display_width != NET_W:
        h = max(1, int(round(NET_H * display_width / NET_W)))
        img = cv2.resize(img, (display_width, h), interpolation=cv2.INTER_LINEAR)
    return img


def main() -> int:
    args = parse_args()
    if args.verbose_results:
        args.console_level = 3

    stop_event = threading.Event()

    def stop(*_args) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    speed_source = SpeedSource(args.speed, args.speed_file)
    frame_source = FrameSource(args)

    renderer = VisionPilotRenderer(
        RenderConfig(
            display_width=args.display_width,
            side_panel_width=args.side_panel_width,
            show_autosteer=not args.no_autosteer,
            show_fused_path=not args.no_fused_path,
            show_detections=not args.no_detections,
            path_corridor=not args.centerline_only,
            homography_path=args.homography,
            homography_key=args.homography_key,
            homography_space=args.homography_space,
            debug_level=args.debug_level,
            camera_view=args.camera_view,
            camera_space=args.camera_space,
            show_occupancy=not args.no_occupancy,
            speed_limit_ms=args.speed_limit,
            icons_dir=args.icons_dir,
            wheel_dir=args.wheel_dir,
            source_label=args.source_label,
        )
    )

    result_queue: queue.Queue = queue.Queue()
    pending: OrderedDict[int, tuple[np.ndarray, float, int, int]] = OrderedDict()
    stats = RuntimeStats()

    server = VisionPilotServer(
        bind=args.bind,
        frame_port=args.frame_port,
        result_port=args.result_port,
    )

    print()
    print("======================================================================")
    print(" VisionPilot Remote Host")
    print(" Linux visualization  <->  Renesas R-Car V4M inference")
    print("======================================================================")
    print(f" Frame channel : {args.bind}:{args.frame_port}  HOST -> V4M")
    print(f" Result channel: {args.bind}:{args.result_port}  V4M -> HOST")
    print(" Protocol      : VPNT v2 / BGR8 raw / FlatBuffers VPO1")
    print(f" Camera view   : {renderer.cfg.camera_view} / {renderer.cfg.camera_space}")
    print(f" Debug level   : {renderer.cfg.debug_level}")
    print(f" Occupancy     : {'on' if renderer.cfg.show_occupancy else 'off'}")
    print(" Hotkeys       : 0..3 debug | V production/debug | O occupancy | R reset 3D | +/- zoom | q quit")
    print()

    try:
        server.open()
        print("Waiting for V4M TCP client on both channels ...")
        frame_peer, result_peer = server.accept()
        print(f"Frame socket connected : {frame_peer[0]}:{frame_peer[1]}")
        print(f"Result socket connected: {result_peer[0]}:{result_peer[1]}")
        print("Streaming started.")
        print()

        rx = ResultReceiver(server, result_queue, stop_event)
        rx.start()

        sequence = 0
        source_index = 0
        last_send_time: float | None = None
        last_display: np.ndarray | None = None

        cv2.namedWindow(args.window, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(
            args.window,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN,
        )
        cv2.setMouseCallback(args.window, renderer.on_mouse)

        while not stop_event.is_set():
            # Drain every result already available. TCP preserves wire order; frame_id
            # maps each result to the exact original host frame kept in `pending`.
            while True:
                try:
                    rx_time_ns, out = result_queue.get_nowait()
                except queue.Empty:
                    break

                item = pending.get(out.inference.frame_id)

                # A first two-frame AutoDrive warm-up may intentionally not yield a
                # result. Drop any stale host images once a newer result proves they
                # can no longer be matched.
                old_keys = [k for k in pending.keys() if k <= out.inference.frame_id]
                for k in old_keys:
                    if k != out.inference.frame_id:
                        pending.pop(k, None)

                if item is None:
                    continue

                frame_for_result, ego_speed, tx_time_ns, tx_bytes = item
                pending.pop(out.inference.frame_id, None)

                rtt_ms = (rx_time_ns - tx_time_ns) / 1e6
                tx_mbps = (tx_bytes * 8.0 / 1e6) / max(rtt_ms / 1000.0, 1e-6)
                stats.update(rx_time_ns, rtt_ms)

                last_display = renderer.render(
                    frame_for_result,
                    out,
                    speed_ms=ego_speed,
                    net_rtt_ms=rtt_ms,
                    tx_mbps=tx_mbps,
                    tx_bytes=tx_bytes,
                    pending_count=len(pending),
                    result_fps=stats.result_fps,
                    result_fps_ema=stats.result_fps_ema,
                    rtt_ema_ms=stats.rtt_ema_ms,
                    rtt_jitter_ms=stats.rtt_jitter_ms,
                )
                print_result(
                    out,
                    args.console_level,
                    rtt_ms=rtt_ms,
                    result_fps=stats.result_fps_ema,
                )

            if last_display is not None:
                cv2.imshow(args.window, last_display)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                if key != 255:
                    renderer.on_key(key)

            if rx.error is not None:
                raise RuntimeError(f"V4M result receiver failed: {rx.error}")

            # Bound host RAM and let the TCP stream provide natural back-pressure.
            if len(pending) >= args.max_pending:
                time.sleep(0.001)
                continue

            ok, frame = frame_source.read()
            if not ok:
                if args.loop:
                    frame_source.close()
                    frame_source = FrameSource(args)
                    source_index = 0
                    continue
                if pending:
                    time.sleep(0.002)
                    continue
                break

            assert frame is not None
            ego_speed = speed_source.get(source_index)
            source_index += 1
            sequence += 1

            if args.realtime and frame_source.source_fps > 0:
                period = 1.0 / frame_source.source_fps
                now = time.monotonic()
                if last_send_time is not None:
                    remain = period - (now - last_send_time)
                    if remain > 0:
                        time.sleep(remain)
                last_send_time = time.monotonic()

            timestamp_ns = time.time_ns()
            tx_start_ns = time.monotonic_ns()
            tx_bytes = server.send_frame(
                frame,
                sequence=sequence,
                timestamp_ns=timestamp_ns,
                vehicle_speed_ms=ego_speed,
            )
            pending[sequence] = (frame.copy(), ego_speed, tx_start_ns, tx_bytes)

            # Like the C++ application, show the plain resized image during the
            # initial inference warm-up rather than leaving the window empty.
            if last_display is None:
                warm = _warmup_frame(frame, args.display_width)
                cv2.imshow(args.window, warm)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                if key != 255:
                    renderer.on_key(key)

        stop_event.set()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt, exiting ...")
    finally:
        stop_event.set()
        server.close()
        frame_source.close()
        cv2.destroyAllWindows()
        print("VisionPilot host terminated.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
