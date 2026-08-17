#!/usr/bin/env python3

import argparse
import socket
import struct
import time
from pathlib import Path

import cv2

MAGIC = 0x56504E54  # VPNT
VERSION = 1
IMAGE = 1
RESULT = 2
BGR8 = 1
HEADER = struct.Struct("!IHHIQ")
IMAGE_METADATA = struct.Struct("!QIIIHHIf")
RESULT_PAYLOAD = struct.Struct("!Q8fI8x")


def recv_exact(sock: socket.socket, size: int) -> bytes:
    result = bytearray()
    while len(result) < size:
        chunk = sock.recv(size - len(result))
        if not chunk:
            raise ConnectionError("peer closed the connection")
        result.extend(chunk)
    return bytes(result)


def receive_result(sock: socket.socket) -> tuple[int, tuple]:
    magic, version, message_type, payload_size, sequence = HEADER.unpack(
        recv_exact(sock, HEADER.size)
    )
    if magic != MAGIC or version != VERSION or message_type != RESULT:
        raise RuntimeError("invalid result header")
    if payload_size != RESULT_PAYLOAD.size:
        raise RuntimeError(f"invalid result size: {payload_size}")
    values = RESULT_PAYLOAD.unpack(recv_exact(sock, payload_size))
    return sequence, values


def send_image(
    sock: socket.socket,
    image,
    frame_id: int,
    vehicle_speed_ms: float,
) -> None:
    if image is None or image.dtype.name != "uint8":
        raise ValueError("image must be uint8")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be BGR HxWx3")
    image = image.copy(order="C")
    height, width, channels = image.shape
    stride = width * channels
    data = image.tobytes()
    timestamp_ns = time.time_ns()
    metadata = IMAGE_METADATA.pack(
        timestamp_ns,
        width,
        height,
        stride,
        BGR8,
        0,
        len(data),
        vehicle_speed_ms,
    )
    header = HEADER.pack(
        MAGIC, VERSION, IMAGE, len(metadata) + len(data), frame_id
    )
    sock.sendall(header)
    sock.sendall(metadata)
    sock.sendall(data)


def image_paths(directory: Path):
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(p for p in directory.iterdir()
                  if p.is_file() and p.suffix.lower() in extensions)


def load_speeds(path: Path) -> list[float]:
    speeds = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        value = line.strip()
        if not value:
            continue
        try:
            speed = float(value)
        except ValueError as error:
            raise ValueError(
                f"invalid speed at {path}:{line_number}: {value!r}"
            ) from error
        if not 0.0 <= speed <= 150.0:
            raise ValueError(
                f"unreasonable speed at {path}:{line_number}: {speed} m/s"
            )
        speeds.append(speed)
    return speeds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--speed-file",
        type=Path,
        required=True,
        help="Text file containing one vehicle speed in m/s per image",
    )
    parser.add_argument("--bind", default="10.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=512)
    args = parser.parse_args()

    paths = image_paths(args.directory)
    if not paths:
        raise RuntimeError(f"no images found in {args.directory}")
    speeds = load_speeds(args.speed_file)
    if len(speeds) != len(paths):
        raise RuntimeError(
            f"image/speed count mismatch: {len(paths)} images, "
            f"{len(speeds)} speeds"
        )

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.bind, args.port))
        server.listen(1)
        print(f"Listening on {args.bind}:{args.port}")
        connection, peer = server.accept()
        with connection:

            connection.setsockopt(
                socket.IPPROTO_TCP,
                socket.TCP_NODELAY,
                1,
            )

            print(f"V4M connected from {peer}")

            for frame_id, (path, vehicle_speed_ms) in enumerate(
                zip(paths, speeds),
                start=1,
            ):
                image = cv2.imread(
                    str(path),
                    cv2.IMREAD_COLOR,
                )

                if image is None:
                    print(f"Skipping unreadable image: {path}")
                    continue

                image = cv2.resize(
                    image,
                    (args.width, args.height),
                    interpolation=cv2.INTER_LINEAR,
                )

                start = time.perf_counter()

                send_image(
                    connection,
                    image,
                    frame_id,
                    vehicle_speed_ms,
                )

                # Il primo frame inizializza solamente lo stato temporale
                # della pipeline. Non produce alcun risultato.
                if frame_id == 1:
                    print(
                        f"frame={frame_id} "
                        f"speed={vehicle_speed_ms:.4f}m/s "
                        f"sent as temporal warm-up"
                    )

                    continue

                sequence, result = receive_result(
                    connection
                )

                elapsed_ms = (time.perf_counter() - start) * 1000.0

                (
                    timestamp_ns,
                    steering,
                    acceleration,
                    cte,
                    yaw,
                    curvature,
                    cipo_distance,
                    cipo_velocity,
                    inference_ms,
                    flags,
                ) = result

                if sequence != frame_id:
                    raise RuntimeError(
                        f"result/frame mismatch: "
                        f"sent frame {frame_id}, "
                        f"received result {sequence}"
                    )

                print(
                    f"frame={sequence} "
                    f"speed={vehicle_speed_ms:.4f}m/s "
                    f"steering={steering:.3f} "
                    f"acceleration={acceleration:.3f} "
                    f"inference={inference_ms:.2f}ms "
                    f"rtt={elapsed_ms:.2f}ms"
                )


if __name__ == "__main__":
    main()
