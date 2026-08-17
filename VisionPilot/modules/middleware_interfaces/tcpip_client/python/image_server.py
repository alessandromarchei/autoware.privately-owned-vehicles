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
IMAGE_METADATA = struct.Struct("!QIIIHHII")
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


def send_image(sock: socket.socket, image, frame_id: int) -> None:
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
        timestamp_ns, width, height, stride, BGR8, 0, len(data), 0
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--bind", default="10.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=512)
    args = parser.parse_args()

    paths = image_paths(args.directory)
    if not paths:
        raise RuntimeError(f"no images found in {args.directory}")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.bind, args.port))
        server.listen(1)
        print(f"Listening on {args.bind}:{args.port}")
        connection, peer = server.accept()
        with connection:
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"V4M connected from {peer}")
            for frame_id, path in enumerate(paths, 1):
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is None:
                    print(f"Skipping {path}")
                    continue
                image = cv2.resize(image, (args.width, args.height))
                start = time.perf_counter()
                send_image(connection, image, frame_id)
                sequence, result = receive_result(connection)
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                timestamp_ns, steering, acceleration, *_ = result
                print(f"frame={sequence} steering={steering:.3f} "
                      f"acceleration={acceleration:.3f} rtt_ms={elapsed_ms:.2f}")


if __name__ == "__main__":
    main()
