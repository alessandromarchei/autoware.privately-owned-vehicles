#!/usr/bin/env bash
set -Eeuo pipefail

VISION_PILOT_ROOT="/home/root/vision_pilot"
LOG_FILE="${VISION_PILOT_ROOT}/vision_pilot.log"

CLI_MODE=0
PASSTHROUGH_ARGS=()

# args parsing
for arg in "$@"; do
    if [[ "$arg" == "--cli" ]]; then
        CLI_MODE=1
    else
        PASSTHROUGH_ARGS+=("$arg")
    fi
done

# Redireziona l'output sul file log SOLO se non è attiva l'opzione --cli
if [[ "${CLI_MODE}" -eq 0 ]]; then
    exec >> "${LOG_FILE}" 2>&1
fi

echo
echo "============================================================"
echo "VisionPilot start requested at $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

OPENBLAS_LIBRARY="${VISION_PILOT_ROOT}/lib/libopenblas.so"

DEBUG_EXECUTABLE="${VISION_PILOT_ROOT}/bin/vision_pilot_v4m_d"
RELEASE_EXECUTABLE="${VISION_PILOT_ROOT}/bin/vision_pilot_v4m"

[[ -f "${OPENBLAS_LIBRARY}" ]] || {
    echo "ERROR: OpenBLAS not found: ${OPENBLAS_LIBRARY}" >&2
    exit 1
}

# Prefer Release when both executables are present.
if [[ -x "${RELEASE_EXECUTABLE}" ]]; then
    VISION_PILOT_EXECUTABLE="${RELEASE_EXECUTABLE}"
    BUILD_TYPE="Release"
elif [[ -x "${DEBUG_EXECUTABLE}" ]]; then
    VISION_PILOT_EXECUTABLE="${DEBUG_EXECUTABLE}"
    BUILD_TYPE="Debug"
else
    echo "ERROR: VisionPilot executable not found." >&2
    echo "Expected one of:" >&2
    echo "  ${DEBUG_EXECUTABLE}" >&2
    echo "  ${RELEASE_EXECUTABLE}" >&2
    exit 1
fi

export LBT_DEFAULT_LIBS="${OPENBLAS_LIBRARY}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export LD_LIBRARY_PATH="${VISION_PILOT_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

echo "Starting VisionPilot:"
echo "  build:      ${BUILD_TYPE}"
echo "  executable: ${VISION_PILOT_EXECUTABLE}"
echo "  BLAS:       ${LBT_DEFAULT_LIBS}"
echo "  arguments:  ${PASSTHROUGH_ARGS[*]:-none}"

cd "${VISION_PILOT_ROOT}"

exec "${VISION_PILOT_EXECUTABLE}" "${PASSTHROUGH_ARGS[@]+"${PASSTHROUGH_ARGS[@]}"}"
#exec gdb --args "${VISION_PILOT_EXECUTABLE}" "${PASSTHROUGH_ARGS[@]+"${PASSTHROUGH_ARGS[@]}"}"