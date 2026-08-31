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

# magic, version, type, payload_size, sequence
HEADER = struct.Struct("!IHHIQ")

# timestamp_ns, width, height, stride, encoding, reserved,
# data_size, vehicle_speed_ms
IMAGE_METADATA = struct.Struct("!QIIIHHIf")

# frame_id, timestamp_ns, 8 float values, cipo_valid, path_valid, reserved
RESULT_PAYLOAD = struct.Struct("!QQ8fBB2x")


def recv_exact(sock: socket.socket, size: int) -> bytes:
    data = bytearray(size)
    view = memoryview(data)
    received = 0

    while received < size:
        count = sock.recv_into(view[received:])
        if count == 0:
            raise ConnectionError("peer closed the connection")
        received += count

    return bytes(data)


def receive_result(sock: socket.socket) -> dict:
    header_bytes = recv_exact(sock, HEADER.size)
    magic, version, message_type, payload_size, sequence = HEADER.unpack(
        header_bytes
    )

    if magic != MAGIC:
        raise RuntimeError(f"invalid result magic: 0x{magic:08X}")
    if version != VERSION:
        raise RuntimeError(f"unsupported result version: {version}")
    if message_type != RESULT:
        raise RuntimeError(
            f"expected RESULT message ({RESULT}), got {message_type}"
        )
    if payload_size != RESULT_PAYLOAD.size:
        raise RuntimeError(
            f"invalid result payload size: got {payload_size}, "
            f"expected {RESULT_PAYLOAD.size}"
        )

    values = RESULT_PAYLOAD.unpack(recv_exact(sock, payload_size))

    (
        payload_frame_id,
        timestamp_ns,
        steering_rad,
        acceleration_ms2,
        cte_m,
        yaw_rad,
        curvature_1pm,
        cipo_distance_m,
        cipo_velocity_ms,
        inference_ms,
        cipo_valid,
        path_valid,
    ) = values

    if payload_frame_id != sequence:
        raise RuntimeError(
            "result header/payload mismatch: "
            f"header sequence={sequence}, payload frame_id={payload_frame_id}"
        )

    return {
        "frame_id": payload_frame_id,
        "timestamp_ns": timestamp_ns,
        "steering_rad": steering_rad,
        "acceleration_ms2": acceleration_ms2,
        "cte_m": cte_m,
        "yaw_rad": yaw_rad,
        "curvature_1pm": curvature_1pm,
        "cipo_distance_m": cipo_distance_m,
        "cipo_velocity_ms": cipo_velocity_ms,
        "inference_ms": inference_ms,
        "cipo_valid": bool(cipo_valid),
        "path_valid": bool(path_valid),
    }


def send_image(
    sock: socket.socket,
    image,
    frame_id: int,
    vehicle_speed_ms: float,
) -> int:
    if image is None or image.dtype.name != "uint8":
        raise ValueError("image must be uint8")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be BGR HxWx3")

    image = image.copy(order="C")
    height, width, channels = image.shape
    stride = width * channels
    image_data = image.tobytes()
    timestamp_ns = time.time_ns()

    metadata = IMAGE_METADATA.pack(
        timestamp_ns,
        width,
        height,
        stride,
        BGR8,
        0,
        len(image_data),
        vehicle_speed_ms,
    )

    header = HEADER.pack(
        MAGIC,
        VERSION,
        IMAGE,
        len(metadata) + len(image_data),
        frame_id,
    )

    sock.sendall(header)
    sock.sendall(metadata)
    sock.sendall(image_data)

    return timestamp_ns


def image_paths(directory: Path) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )


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


def create_server(bind_address: str, port: int) -> socket.socket:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((bind_address, port))
    server.listen(1)
    return server


def configure_connection(
    connection: socket.socket,
    timeout_seconds: float,
) -> None:
    connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    connection.settimeout(timeout_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Send BGR frames and vehicle speed to VisionPilot on V4M, "
            "then validate the returned VisionResult messages."
        )
    )
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--speed-file",
        type=Path,
        required=True,
        help="One vehicle speed in m/s per input image",
    )
    parser.add_argument(
        "--bind",
        default="0.0.0.0",
        help="Predator address on which both TCP servers listen",
    )
    parser.add_argument("--frame-port", type=int, default=8080)
    parser.add_argument("--result-port", type=int, default=8081)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument(
        "--result-timeout",
        type=float,
        default=60.0,
        help="Seconds to wait for each V4M result",
    )
    parser.add_argument(
        "--no-resize",
        action="store_true",
        help="Send images at their original resolution",
    )
    args = parser.parse_args()

    if not 1 <= args.frame_port <= 65535:
        raise ValueError(f"invalid frame port: {args.frame_port}")
    if not 1 <= args.result_port <= 65535:
        raise ValueError(f"invalid result port: {args.result_port}")
    if args.frame_port == args.result_port:
        raise ValueError("frame and result ports must be different")

    paths = image_paths(args.directory)
    if not paths:
        raise RuntimeError(f"no images found in {args.directory}")

    speeds = load_speeds(args.speed_file)
    if len(speeds) != len(paths):
        raise RuntimeError(
            f"image/speed count mismatch: {len(paths)} images, "
            f"{len(speeds)} speeds"
        )

    frame_server = create_server(args.bind, args.frame_port)
    result_server = create_server(args.bind, args.result_port)

    try:
        print(f"Frame server listening on {args.bind}:{args.frame_port}")
        print(f"Result server listening on {args.bind}:{args.result_port}")
        print("Start VisionPilot on the V4M now.")

        frame_connection, frame_peer = frame_server.accept()
        print(f"V4M frame connection from {frame_peer}")

        result_connection, result_peer = result_server.accept()
        print(f"V4M result connection from {result_peer}")

        with frame_connection, result_connection:
            configure_connection(frame_connection, args.result_timeout)
            configure_connection(result_connection, args.result_timeout)

            sent_frames = 0
            received_results = 0

            for path, vehicle_speed_ms in zip(paths, speeds):
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)

                if image is None:
                    print(f"Skipping unreadable image: {path}")
                    continue

                if not args.no_resize:
                    image = cv2.resize(
                        image,
                        (args.width, args.height),
                        interpolation=cv2.INTER_LINEAR,
                    )

                sent_frames += 1
                frame_id = sent_frames
                start = time.perf_counter()

                send_image(
                    frame_connection,
                    image,
                    frame_id,
                    vehicle_speed_ms,
                )

                # InferencePipeline uses two temporal frames. The first valid
                # frame initializes its previous-frame state and returns nullopt.
                if frame_id == 1:
                    print(
                        f"frame={frame_id} "
                        f"speed={vehicle_speed_ms:.4f}m/s "
                        "sent as temporal warm-up"
                    )
                    continue

                try:
                    result = receive_result(result_connection)
                except socket.timeout as error:
                    raise TimeoutError(
                        f"no VisionResult received for frame {frame_id} "
                        f"within {args.result_timeout:.1f}s"
                    ) from error

                elapsed_ms = (time.perf_counter() - start) * 1000.0

                if result["frame_id"] != frame_id:
                    raise RuntimeError(
                        "result/frame mismatch: "
                        f"sent frame {frame_id}, "
                        f"received frame {result['frame_id']}"
                    )

                received_results += 1

                print(
                    f"frame={result['frame_id']} "
                    f"speed={vehicle_speed_ms:.4f}m/s "
                    f"steering={result['steering_rad']:.4f}rad "
                    f"acceleration={result['acceleration_ms2']:.4f}m/s2 "
                    f"cte={result['cte_m']:.4f}m "
                    f"yaw={result['yaw_rad']:.4f}rad "
                    f"curvature={result['curvature_1pm']:.6f}1/m "
                    f"cipo_dist={result['cipo_distance_m']:.3f}m "
                    f"cipo_vel={result['cipo_velocity_ms']:.3f}m/s "
                    f"cipo_valid={result['cipo_valid']} "
                    f"path_valid={result['path_valid']} "
                    f"inference={result['inference_ms']:.2f}ms "
                    f"rtt={elapsed_ms:.2f}ms"
                )

            print(
                "Test completed successfully: "
                f"sent={sent_frames}, results={received_results}, "
                f"warmup={1 if sent_frames else 0}"
            )
    finally:
        frame_server.close()
        result_server.close()


if __name__ == "__main__":
    main()