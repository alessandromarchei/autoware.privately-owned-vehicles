#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
BUILD_DIR="${PROJECT_ROOT}/build"


echo "==> Configuring build"
cmake -B build -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DRCAR_SOC=v4m \
    -DRCAR_TARGET_OS=LINUX \
    -DCMAKE_PREFIX_PATH=/opt/rcar-xos/v3.47.0/cmake \
    -DCMAKE_TOOLCHAIN_FILE=/opt/rcar-xos/v3.47.0/cmake/toolchain_poky_5_0_adas.cmake



echo "==> Building VisionPilot"
cmake --build "${BUILD_DIR}" --parallel 16