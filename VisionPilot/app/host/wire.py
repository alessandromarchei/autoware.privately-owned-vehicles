from __future__ import annotations

import socket
import struct
import threading
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# VisionPilot wire protocol v2
# ---------------------------------------------------------------------------

PROTOCOL_MAGIC = 0x56504E54  # "VPNT"
PROTOCOL_VERSION = 2

WIRE_HEADER_SIZE = 20
WIRE_IMAGE_METADATA_SIZE = 32
MAX_IMAGE_BYTES = 64 * 1024 * 1024
MAX_RESULT_BYTES = 16 * 1024 * 1024

MSG_IMAGE = 1
MSG_RESULT = 2
MSG_PING = 3
MSG_PONG = 4

ENC_BGR8 = 1
ENC_RGB8 = 2
ENC_GRAY8 = 3

_HEADER = struct.Struct(">IHHIQ")
# timestamp_ns, width, height, stride, encoding, reserved, data_size, speed(float bits)
_IMAGE_META_PREFIX = struct.Struct(">QIIIHHI")
_U32 = struct.Struct(">I")


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int


@dataclass
class AutoDrive:
    dist_normalized: float = 0.0
    curvature_raw: float = 0.0
    flag_prob: float = 0.0
    valid: bool = False


@dataclass
class AutoSteer:
    xp: list[float] = field(default_factory=list)
    h_vector: list[float] = field(default_factory=list)
    valid: bool = False


@dataclass
class AutoSpeed:
    detections: list[Detection] = field(default_factory=list)
    valid: bool = False


@dataclass
class CIPO:
    valid: bool = False
    distance_m: float = 0.0
    velocity_ms: float = 0.0
    distance_stddev_m: float = 0.0
    cipo_raw_found: bool = False
    cipo_raw_dist_m: float = 0.0
    cut_in_detected: bool = False


@dataclass
class Lateral:
    valid: bool = False
    cte_m: float = 0.0
    cte_rate_mps: float = 0.0
    yaw_rad: float = 0.0
    yaw_rate_rps: float = 0.0
    cte_stddev_m: float = 0.0
    yaw_stddev_rad: float = 0.0
    curvature: float = 0.0
    curv_stddev: float = 0.0
    path_valid: bool = False
    raw_cte_m: float = 0.0
    raw_yaw_rad: float = 0.0
    raw_path_curvature: float = 0.0
    raw_ad_curvature: float = 0.0
    path_inliers: int = 0
    path_points: int = 0
    path_a: float = 0.0
    path_b: float = 0.0
    path_c: float = 0.0
    path_x_min_m: float = 0.0
    path_x_max_m: float = 0.0


@dataclass
class Inference:
    frame_id: int = 0
    total_ms: float = 0.0
    pre_ms: float = 0.0
    visionpilot_ms: float = 0.0
    auto_drive: AutoDrive = field(default_factory=AutoDrive)
    auto_steer: AutoSteer = field(default_factory=AutoSteer)
    auto_speed: AutoSpeed = field(default_factory=AutoSpeed)
    cipo: CIPO = field(default_factory=CIPO)
    lateral: Lateral = field(default_factory=Lateral)


@dataclass
class Plan:
    acceleration: float = 0.0
    steering: list[float] = field(default_factory=list)
    warnings: list[int] = field(default_factory=list)


@dataclass
class VisionPilotOutput:
    inference: Inference = field(default_factory=Inference)
    plan: Plan = field(default_factory=Plan)
    wire_sequence: int = 0
    payload_bytes: int = 0


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    out = bytearray(n)
    view = memoryview(out)
    pos = 0
    while pos < n:
        got = sock.recv_into(view[pos:], n - pos)
        if got == 0:
            raise ConnectionError("peer closed TCP connection")
        pos += got
    return bytes(out)


def encode_header(msg_type: int, payload_size: int, sequence: int) -> bytes:
    return _HEADER.pack(
        PROTOCOL_MAGIC,
        PROTOCOL_VERSION,
        msg_type,
        payload_size,
        sequence,
    )


def decode_header(raw: bytes) -> tuple[int, int, int]:
    if len(raw) != WIRE_HEADER_SIZE:
        raise ValueError(f"bad header size: {len(raw)}")
    magic, version, msg_type, payload_size, sequence = _HEADER.unpack(raw)
    if magic != PROTOCOL_MAGIC:
        raise ValueError(f"bad protocol magic 0x{magic:08x}")
    if version != PROTOCOL_VERSION:
        raise ValueError(f"unsupported protocol version {version}")
    return msg_type, payload_size, sequence


def encode_image_metadata(
    timestamp_ns: int,
    width: int,
    height: int,
    stride: int,
    encoding: int,
    data_size: int,
    vehicle_speed_ms: float,
) -> bytes:
    # C++ writes float bit-pattern with writeU32At(), i.e. network byte order.
    speed_u32 = struct.unpack(">I", struct.pack(">f", float(vehicle_speed_ms)))[0]
    return (
        _IMAGE_META_PREFIX.pack(
            int(timestamp_ns),
            int(width),
            int(height),
            int(stride),
            int(encoding),
            0,  # bytes 22-23 reserved
            int(data_size),
        )
        + _U32.pack(speed_u32)
    )


def encode_bgr_frame(
    frame: np.ndarray,
    sequence: int,
    timestamp_ns: int,
    vehicle_speed_ms: float,
) -> bytes:
    if frame is None or frame.size == 0:
        raise ValueError("empty frame")
    if frame.dtype != np.uint8:
        raise ValueError(f"expected uint8 frame, got {frame.dtype}")

    if frame.ndim == 2:
        encoding = ENC_GRAY8
        h, w = frame.shape
        channels = 1
    elif frame.ndim == 3 and frame.shape[2] == 3:
        encoding = ENC_BGR8
        h, w, channels = frame.shape
    else:
        raise ValueError(f"unsupported frame shape: {frame.shape}")

    # The V4M receiver supports arbitrary stride, but a packed contiguous payload
    # makes the host implementation deterministic and portable.
    packed = np.ascontiguousarray(frame)
    stride = w * channels
    data = packed.tobytes(order="C")

    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(f"image payload too large: {len(data)} bytes")

    metadata = encode_image_metadata(
        timestamp_ns=timestamp_ns,
        width=w,
        height=h,
        stride=stride,
        encoding=encoding,
        data_size=len(data),
        vehicle_speed_ms=vehicle_speed_ms,
    )

    payload_size = WIRE_IMAGE_METADATA_SIZE + len(data)
    return encode_header(MSG_IMAGE, payload_size, sequence) + metadata + data


def _import_generated():
    try:
        from visionpilot.wire.VisionPilotOutput import VisionPilotOutput as FBVisionPilotOutput
        return FBVisionPilotOutput
    except Exception as exc:
        raise RuntimeError(
            "FlatBuffers Python bindings are missing. Run:\n"
            "  python generate_bindings.py\n"
            "and install the runtime with:\n"
            "  python -m pip install flatbuffers"
        ) from exc


def decode_flatbuffer(payload: bytes, wire_sequence: int = 0) -> VisionPilotOutput:
    FBVisionPilotOutput = _import_generated()

    # Python FlatBuffers generated bindings verify the identifier by API only
    # in some generator versions. We also perform the cheap file-id check here.
    # Root table starts at byte 0; file identifier occupies bytes 4..7.
    if len(payload) < 8 or payload[4:8] != b"VPO1":
        raise ValueError("result payload is not a VPO1 FlatBuffer")

    root = FBVisionPilotOutput.GetRootAsVisionPilotOutput(payload, 0)

    out = VisionPilotOutput(
        wire_sequence=wire_sequence,
        payload_bytes=len(payload),
    )

    inf = root.Inference()
    if inf is None:
        raise ValueError("VisionPilotOutput.inference is missing")

    out.inference.frame_id = int(inf.FrameId())
    out.inference.total_ms = float(inf.TotalMs())
    out.inference.pre_ms = float(inf.PreMs())
    out.inference.visionpilot_ms = float(inf.VisionpilotMs())

    ad = inf.AutoDrive()
    if ad is not None:
        out.inference.auto_drive = AutoDrive(
            dist_normalized=float(ad.DistNormalized()),
            curvature_raw=float(ad.CurvatureRaw()),
            flag_prob=float(ad.FlagProb()),
            valid=bool(ad.Valid()),
        )

    ast = inf.AutoSteer()
    if ast is not None:
        xp = [float(ast.Xp(i)) for i in range(ast.XpLength())]
        hv = [float(ast.HVector(i)) for i in range(ast.HVectorLength())]
        out.inference.auto_steer = AutoSteer(
            xp=xp,
            h_vector=hv,
            valid=bool(ast.Valid()),
        )

    asp = inf.AutoSpeed()
    if asp is not None:
        detections: list[Detection] = []
        for i in range(asp.DetectionsLength()):
            d = asp.Detections(i)
            detections.append(
                Detection(
                    x1=float(d.X1()),
                    y1=float(d.Y1()),
                    x2=float(d.X2()),
                    y2=float(d.Y2()),
                    score=float(d.Score()),
                    class_id=int(d.ClassId()),
                )
            )
        out.inference.auto_speed = AutoSpeed(
            detections=detections,
            valid=bool(asp.Valid()),
        )

    c = inf.Cipo()
    if c is not None:
        out.inference.cipo = CIPO(
            valid=bool(c.Valid()),
            distance_m=float(c.DistanceM()),
            velocity_ms=float(c.VelocityMs()),
            distance_stddev_m=float(c.DistanceStddevM()),
            cipo_raw_found=bool(c.CipoRawFound()),
            cipo_raw_dist_m=float(c.CipoRawDistM()),
            cut_in_detected=bool(c.CutInDetected()),
        )

    lat = inf.Lateral()
    if lat is not None:
        out.inference.lateral = Lateral(
            valid=bool(lat.Valid()),
            cte_m=float(lat.CteM()),
            cte_rate_mps=float(lat.CteRateMps()),
            yaw_rad=float(lat.YawRad()),
            yaw_rate_rps=float(lat.YawRateRps()),
            cte_stddev_m=float(lat.CteStddevM()),
            yaw_stddev_rad=float(lat.YawStddevRad()),
            curvature=float(lat.Curvature()),
            curv_stddev=float(lat.CurvStddev()),
            path_valid=bool(lat.PathValid()),
            raw_cte_m=float(lat.RawCteM()),
            raw_yaw_rad=float(lat.RawYawRad()),
            raw_path_curvature=float(lat.RawPathCurvature()),
            raw_ad_curvature=float(lat.RawAdCurvature()),
            path_inliers=int(lat.PathInliers()),
            path_points=int(lat.PathPoints()),
            path_a=float(lat.PathA()),
            path_b=float(lat.PathB()),
            path_c=float(lat.PathC()),
            path_x_min_m=float(lat.PathXMinM()),
            path_x_max_m=float(lat.PathXMaxM()),
        )

    plan = root.Plan()
    if plan is None:
        raise ValueError("VisionPilotOutput.plan is missing")

    out.plan = Plan(
        acceleration=float(plan.Acceleration()),
        steering=[float(plan.Steering(i)) for i in range(plan.SteeringLength())],
        warnings=[int(plan.Warnings(i)) for i in range(plan.WarningsLength())],
    )

    if out.inference.frame_id != wire_sequence:
        raise ValueError(
            f"sequence mismatch: header={wire_sequence}, flatbuffer={out.inference.frame_id}"
        )

    return out


class VisionPilotServer:
    """Linux host side.

    V4M TCPClient connects to:
      frame_port  -> V4M receives Image messages
      result_port -> V4M sends Result messages
    """

    def __init__(
        self,
        bind: str = "0.0.0.0",
        frame_port: int = 8080,
        result_port: int = 8081,
    ):
        self.bind = bind
        self.frame_port = int(frame_port)
        self.result_port = int(result_port)
        self._frame_listener: Optional[socket.socket] = None
        self._result_listener: Optional[socket.socket] = None
        self.frame_sock: Optional[socket.socket] = None
        self.result_sock: Optional[socket.socket] = None

    @staticmethod
    def _listener(bind: str, port: int) -> socket.socket:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((bind, port))
        s.listen(1)
        return s

    @staticmethod
    def _tune(sock: socket.socket) -> None:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def open(self) -> None:
        # Both listeners must exist before accepting. The V4M client connects to
        # frame first, then result.
        self._frame_listener = self._listener(self.bind, self.frame_port)
        self._result_listener = self._listener(self.bind, self.result_port)

    def accept(self) -> tuple[tuple, tuple]:
        if self._frame_listener is None or self._result_listener is None:
            raise RuntimeError("call open() before accept()")

        self.frame_sock, frame_peer = self._frame_listener.accept()
        self._tune(self.frame_sock)

        self.result_sock, result_peer = self._result_listener.accept()
        self._tune(self.result_sock)

        return frame_peer, result_peer

    def send_frame(
        self,
        frame: np.ndarray,
        sequence: int,
        timestamp_ns: int,
        vehicle_speed_ms: float,
    ) -> int:
        if self.frame_sock is None:
            raise RuntimeError("frame socket is not connected")
        packet = encode_bgr_frame(
            frame=frame,
            sequence=sequence,
            timestamp_ns=timestamp_ns,
            vehicle_speed_ms=vehicle_speed_ms,
        )
        self.frame_sock.sendall(packet)
        return len(packet)

    def recv_result(self) -> VisionPilotOutput:
        if self.result_sock is None:
            raise RuntimeError("result socket is not connected")

        raw_header = _recv_exact(self.result_sock, WIRE_HEADER_SIZE)
        msg_type, payload_size, sequence = decode_header(raw_header)

        if msg_type != MSG_RESULT:
            raise ValueError(f"expected Result({MSG_RESULT}), got message type {msg_type}")
        if payload_size <= 0 or payload_size > MAX_RESULT_BYTES:
            raise ValueError(f"invalid result payload size: {payload_size}")

        payload = _recv_exact(self.result_sock, payload_size)
        return decode_flatbuffer(payload, sequence)

    def close(self):
        sockets = [
            self.frame_sock,
            self.result_sock,
            self._frame_listener,
            self._result_listener,
        ]

        for sock in sockets:
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

                try:
                    sock.close()
                except OSError:
                    pass

                    
    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()
