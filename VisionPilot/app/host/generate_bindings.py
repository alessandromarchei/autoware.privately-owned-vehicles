#!/usr/bin/env python3
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
FBS = "/home/sergey/DEV/AI/vision_pilot/VisionPilot/modules/middleware_interfaces/tcpip_client/schema/visionpilot.fbs"

flatc = shutil.which("flatc")
if flatc is None:
    print("ERROR: flatc not found.")
    print("Ubuntu/Debian: sudo apt install flatbuffers-compiler")
    sys.exit(1)

subprocess.run(
    [flatc, "--python", "-o", str(HERE), str(FBS)],
    check=True,
)

print(f"Generated Python FlatBuffers bindings under: {HERE / 'visionpilot' / 'wire'}")
