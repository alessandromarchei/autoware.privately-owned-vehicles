#!/usr/bin/env python3
from __future__ import annotations

import argparse
import queue
import signal
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

from wire import VisionPilotOutput, VisionPilotServer
from visualization import RenderConfig, VisionPilotRenderer
import traceback

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class SpeedSource:
    def __init__(self, fixed_ms: float, path: str | None):
        self.fixed_ms = float(fixed_ms)
        self.values = None
        if path:
            vals = []
            for raw in Path(path).read_text().splitlines():
                raw = raw.strip()
                if not raw or raw.startswith("#"):
                    continue
                # Supports either "speed" or CSV where final column is speed.
                token = raw.split(",")[-1].strip()
                vals.append(float(token))
            if not vals:
                raise RuntimeError(f"no speed values found in {path}")
            self.values = vals

    def get(self, index: int) -> float:
        if self.values is None:
            return self.fixed_ms
        return self.values[min(index, len(self.values) - 1)]


class FrameSource:
    def __init__(self, args):
        self.args = args
        self.cap = None
        self.files = None
        self.index = 0
        self.source_fps = args.fps

        if args.video:
            self.cap = cv2.VideoCapture(args.video)
            if not self.cap.isOpened():
                raise RuntimeError(f"cannot open video: {args.video}")
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps and fps > 1e-3:
                self.source_fps = fps

        elif args.camera is not None:
            self.cap = cv2.VideoCapture(args.camera)
            if not self.cap.isOpened():
                raise RuntimeError(f"cannot open camera: {args.camera}")
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps and fps > 1e-3:
                self.source_fps = fps

        elif args.frames:
            folder = Path(args.frames)
            self.files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)
            if not self.files:
                raise RuntimeError(f"no images found in {folder}")

        else:
            raise RuntimeError("select one source: --video, --frames or --camera")

    def read(self):
        if self.cap is not None:
            ok, frame = self.cap.read()
            if not ok:
                return False, None
            self.index += 1
            return True, frame

        if self.index >= len(self.files):
            return False, None
        frame = cv2.imread(str(self.files[self.index]), cv2.IMREAD_COLOR)
        self.index += 1
        return frame is not None, frame

    def close(self):
        if self.cap is not None:
            self.cap.release()


class ResultReceiver(threading.Thread):
    def __init__(self, server, output_queue, stop_event):
        super().__init__(name="v4m-result-rx", daemon=True)
        self.server = server
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.error = None

    def run(self):
        try:
            while not self.stop_event.is_set():
                result = self.server.recv_result()
                self.output_queue.put(
                    (time.monotonic_ns(), result)
                )

        except Exception as exc:
            if not self.stop_event.is_set():
                self.error = exc

                print(
                    f"\n[ERROR] Result receiver crashed: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )

                traceback.print_exc()

                self.stop_event.set()

def parse_args():
    p = argparse.ArgumentParser(
        description="VisionPilot Linux host: stream camera frames to Renesas V4M and visualize remote inference."
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="input video")
    src.add_argument("--frames", help="directory of image frames")
    src.add_argument("--camera", type=int, help="OpenCV/V4L2 camera index")

    p.add_argument("--speed", type=float, default=0.0, help="fixed ego speed [m/s]")
    p.add_argument("--speed-file", help="one speed [m/s] per line; CSV final column also accepted")

    p.add_argument("--bind", default="0.0.0.0", help="host bind address")
    p.add_argument("--frame-port", type=int, default=8080)
    p.add_argument("--result-port", type=int, default=8081)

    p.add_argument("--fps", type=float, default=30.0, help="FPS for --frames or fallback video FPS")
    p.add_argument("--realtime", action="store_true", help="pace frames according to source FPS")
    p.add_argument("--loop", action="store_true")

    p.add_argument("--display-width", type=int, default=1420)
    p.add_argument("--side-panel-width", type=int, default=390)
    p.add_argument("--window", default="VisionPilot / Renesas V4M")
    p.add_argument("--no-autosteer", action="store_true")
    p.add_argument("--no-fused-path", action="store_true")
    p.add_argument("--no-detections", action="store_true")
    p.add_argument("--centerline-only", action="store_true")

    p.add_argument("--homography", help="OpenCV YAML/XML matrix H (pixel -> world)")
    p.add_argument("--homography-key", default="H")
    p.add_argument(
        "--homography-space",
        choices=("net", "native"),
        default="net",
        help="coordinate space of H: 1024x512 net image or native input image",
    )

    p.add_argument("--max-pending", type=int, default=12,
                   help="number of frame images retained for result/frame synchronization")
    p.add_argument("--verbose-results", action="store_true")
    return p.parse_args()


def print_result(out: VisionPilotOutput):
    i = out.inference
    lat = i.lateral
    c = i.cipo
    steer_cmd = out.plan.steering[1] if len(out.plan.steering) > 1 else (
        out.plan.steering[0] if out.plan.steering else 0.0
    )
    print(
        f"[V4M #{i.frame_id:06d}] "
        f"VP={i.visionpilot_ms:7.2f}ms wall={i.wall_ms:7.2f}ms pre={i.pre_ms:6.2f}ms | "
        f"AD(valid={int(i.auto_drive.valid)} p={i.auto_drive.flag_prob:.3f} "
        f"curvRaw={i.auto_drive.curvature_raw:+.5f}) | "
        f"ASteer(valid={int(i.auto_steer.valid)}) "
        f"ASpeed(det={len(i.auto_speed.detections)}) | "
        f"CIPO(valid={int(c.valid)} d={c.distance_m:6.1f}m v={c.velocity_ms:+6.2f}m/s) | "
        f"LAT(valid={int(lat.valid)} cte={lat.cte_m:+.2f}m yaw={lat.yaw_rad:+.3f} "
        f"k={lat.curvature:+.5f}) | "
        f"PLAN steer={steer_cmd:+.5f}rad acc={out.plan.acceleration:+.3f}"
    )


def main():
    args = parse_args()

    stop_event = threading.Event()

    def stop(*_):
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
        )
    )

    result_queue: queue.Queue = queue.Queue()
    pending: OrderedDict[int, tuple[np.ndarray, float, int, int]] = OrderedDict()

    server = VisionPilotServer(
        bind=args.bind,
        frame_port=args.frame_port,
        result_port=args.result_port,
    )

    print()
    print("==============================================================")
    print(" VisionPilot Remote Host")
    print(" Linux visualization  <->  Renesas R-Car V4M inference")
    print("==============================================================")
    print(f" Frame channel : {args.bind}:{args.frame_port}  HOST -> V4M")
    print(f" Result channel: {args.bind}:{args.result_port}  V4M -> HOST")
    print(" Protocol      : VPNT v2 / BGR8 raw / FlatBuffers VPO1")
    print()

    try:
        server.open()
        print("Waiting for V4M TCP client on both channels ...")
        frame_peer, result_peer = server.accept()
        print(f"Frame socket connected : {frame_peer[0]}:{frame_peer[1]}")
        print(f"Result socket connected: {result_peer[0]}:{result_peer[1]}")
        print("Streaming started. Press q or ESC to stop.")
        print()

        rx = ResultReceiver(server, result_queue, stop_event)
        rx.start()

        sequence = 0
        source_index = 0
        last_send_time = None
        last_display = None

        cv2.namedWindow(args.window, cv2.WINDOW_NORMAL)

        while not stop_event.is_set():
            # Drain all V4M outputs currently available.
            while True:
                try:
                    rx_time_ns, out = result_queue.get_nowait()
                except queue.Empty:
                    break

                item = pending.get(out.inference.frame_id)

                # Purge frames older than/equal to the returned result. This also
                # cleanly handles the first AutoDrive two-frame warm-up frame,
                # for which the V4M pipeline may intentionally send no result.
                old_keys = [k for k in pending.keys() if k <= out.inference.frame_id]
                for k in old_keys:
                    if k != out.inference.frame_id:
                        pending.pop(k, None)

                if item is not None:
                    frame_for_result, ego_speed, tx_time_ns, tx_bytes = item
                    pending.pop(out.inference.frame_id, None)

                    rtt_ms = (rx_time_ns - tx_time_ns) / 1e6
                    # Approximate frame-channel wire throughput for this transaction.
                    tx_mbps = (tx_bytes * 8.0 / 1e6) / max(rtt_ms / 1000.0, 1e-6)

                    last_display = renderer.render(
                        frame_for_result,
                        out,
                        speed_ms=ego_speed,
                        net_rtt_ms=rtt_ms,
                        tx_mbps=tx_mbps,
                    )
                    print_result(out)

            if last_display is not None:
                cv2.imshow(args.window, last_display)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            if rx.error is not None:
                raise RuntimeError(f"V4M result receiver failed: {rx.error}")

            # Keep only a small number of images awaiting matching result.
            # The TCP send itself provides back-pressure if V4M runs slower.
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

                # Input ended: continue displaying/draining already-sent results.
                if pending:
                    time.sleep(0.002)
                    continue
                break

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

            # Store the exact camera frame associated with sequence N.
            pending[sequence] = (frame.copy(), ego_speed, tx_start_ns, tx_bytes)

        stop_event.set()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt, exiting ...")
    finally:
        stop_event.set()
        if server is not None:
            server.close()
        frame_source.close()
        cv2.destroyAllWindows()
        print("VisionPilot host terminated.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
