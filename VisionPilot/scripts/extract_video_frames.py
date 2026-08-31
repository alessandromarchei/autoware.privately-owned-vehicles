#!/usr/bin/env python3
"""Recursively extract every frame from MP4 videos as lossless images."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

try:
    import cv2
except ImportError as exc:
    print(
        "OpenCV for Python is required. Install it in your host environment with:\n"
        "  python3 -m pip install opencv-python\n"
        "or use the OpenCV package already present in your Conda environment.",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find MP4 files recursively and extract their frames into a "
            "directory next to each video."
        )
    )
    parser.add_argument("input_dir", type=Path, help="Root directory containing MP4 files")
    parser.add_argument(
        "--frames-dir",
        default="frames",
        help="Output directory name created next to each video (default: frames)",
    )
    parser.add_argument(
        "--format",
        choices=("png", "bmp"),
        default="png",
        help="PNG is lossless/compressed; BMP is lossless/uncompressed (default: png)",
    )
    parser.add_argument(
        "--png-compression",
        type=int,
        choices=range(10),
        default=3,
        metavar="0..9",
        help="PNG compression level; affects size/speed, never image quality (default: 3)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing frames directory even if extraction is complete",
    )
    parser.add_argument(
        "--keep-partial",
        action="store_true",
        help="Keep a temporary partial directory after a decoding/write failure",
    )
    return parser.parse_args()


def finite_positive(value: float) -> float | None:
    return value if value > 0 and value < float("inf") else None


def extract_video(
    video_path: Path,
    frames_dir_name: str,
    image_format: str,
    png_compression: int,
    overwrite: bool,
    keep_partial: bool,
) -> bool:
    output_dir = video_path.parent / frames_dir_name
    partial_dir = video_path.parent / f".{frames_dir_name}.partial"
    metadata_path = output_dir / "metadata.json"

    if output_dir.exists() and metadata_path.is_file() and not overwrite:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("complete") is True:
                print(f"[SKIP] {video_path} -> already complete ({metadata.get('frame_count')} frames)")
                return True
        except (OSError, json.JSONDecodeError):
            pass

    if output_dir.exists() and not overwrite:
        print(
            f"[ERROR] Output exists but is not marked complete: {output_dir}\n"
            "        Use --overwrite to replace it.",
            file=sys.stderr,
        )
        return False

    if partial_dir.exists():
        shutil.rmtree(partial_dir)
    partial_dir.mkdir(parents=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}", file=sys.stderr)
        if not keep_partial:
            shutil.rmtree(partial_dir, ignore_errors=True)
        return False

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = finite_positive(float(capture.get(cv2.CAP_PROP_FPS)))
    reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    reported_frames = reported_frames if reported_frames > 0 else None
    suffix = f".{image_format}"
    write_params = (
        [cv2.IMWRITE_PNG_COMPRESSION, png_compression]
        if image_format == "png"
        else []
    )

    print(
        f"[START] {video_path}\n"
        f"        {width}x{height}, fps={fps or 'unknown'}, "
        f"reported_frames={reported_frames or 'unknown'}"
    )

    start_time = time.monotonic()
    frame_index = 0
    success = True

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame_path = partial_dir / f"frame_{frame_index:06d}{suffix}"
            if not cv2.imwrite(str(frame_path), frame, write_params):
                raise OSError(f"cv2.imwrite failed for {frame_path}")

            frame_index += 1
            if frame_index % 100 == 0:
                elapsed = time.monotonic() - start_time
                rate = frame_index / elapsed if elapsed > 0 else 0.0
                total = f"/{reported_frames}" if reported_frames is not None else ""
                print(f"        {frame_index}{total} frames, {rate:.1f} frame/s", end="\r", flush=True)
    except (OSError, cv2.error) as exc:
        success = False
        print(f"\n[ERROR] {video_path}: {exc}", file=sys.stderr)
    finally:
        capture.release()

    elapsed = time.monotonic() - start_time
    if frame_index == 0:
        success = False
        print(f"\n[ERROR] No frames decoded from: {video_path}", file=sys.stderr)

    if not success:
        if not keep_partial:
            shutil.rmtree(partial_dir, ignore_errors=True)
        return False

    metadata = {
        "complete": True,
        "source_video": video_path.name,
        "frame_count": frame_index,
        "width": width,
        "height": height,
        "fps": fps,
        "reported_frame_count": reported_frames,
        "pixel_order": "BGR when loaded with cv2.imread",
        "image_format": image_format,
        "filename_pattern": f"frame_%06d{suffix}",
        "first_frame_index": 0,
    }
    (partial_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    if output_dir.exists():
        shutil.rmtree(output_dir)
    partial_dir.rename(output_dir)

    print(
        f"\n[DONE]  {video_path} -> {output_dir} "
        f"({frame_index} frames in {elapsed:.1f} s)"
    )
    return True


def main() -> int:
    args = parse_args()
    root = args.input_dir.expanduser().resolve()
    if not root.is_dir():
        print(f"[ERROR] Not a directory: {root}", file=sys.stderr)
        return 2

    videos = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() == ".mp4"
    )
    if not videos:
        print(f"[ERROR] No MP4 files found under: {root}", file=sys.stderr)
        return 2

    print(f"Found {len(videos)} MP4 file(s) under {root}")
    failed: list[Path] = []
    for index, video_path in enumerate(videos, start=1):
        print(f"\n[{index}/{len(videos)}]")
        if not extract_video(
            video_path,
            args.frames_dir,
            args.format,
            args.png_compression,
            args.overwrite,
            args.keep_partial,
        ):
            failed.append(video_path)

    if failed:
        print(f"\nFailed videos ({len(failed)}):", file=sys.stderr)
        for path in failed:
            print(f"  {path}", file=sys.stderr)
        return 1

    print(f"\nAll {len(videos)} video(s) extracted successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
