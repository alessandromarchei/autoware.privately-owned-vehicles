#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
BUILD_DIR="${PROJECT_ROOT}/build"
TARGET_DIR="${BUILD_DIR}/target"
REMOTE="v4m"
REMOTE_DIR="/home/root/vision_pilot"

REACTION_MODEL_DIR="/opt/rcar-xos/v3.47.0/tools/hyco/reaction/work_dir"

# now send the model.msgpack files to the respective folders, creating dirs if needed
echo "==> Copying model.msgpack to V4M"
install_msgpack() {
    src="$1"
    dst="$2"
    dst_dir=$(dirname "${dst}")
    if [ ! -d "${dst_dir}" ]; then
        echo "    Creating directory ${dst_dir}"
        mkdir -p "${dst_dir}"
    fi
    cp "${src}" "${dst}"
}

install_msgpack "${REACTION_MODEL_DIR}/dummyautodrive/tvm-v4m/tvm_bundle/abundle.msgpack" "${PROJECT_ROOT}/modules/models/weights/autodrive_core0/autodrive.msgpack"
install_msgpack "${REACTION_MODEL_DIR}/dummyautospeed/tvm-v4m/tvm_bundle/abundle.msgpack" "${PROJECT_ROOT}/modules/models/weights/autospeed_core0/autospeed.msgpack"
install_msgpack "${REACTION_MODEL_DIR}/dummyautosteer/tvm-v4m/tvm_bundle/abundle.msgpack" "${PROJECT_ROOT}/modules/models/weights/autosteer_core0/autosteer.msgpack"
install_msgpack "${REACTION_MODEL_DIR}/dummyvisionpilotmerged/tvm-v4m/tvm_bundle/abundle.msgpack" "${PROJECT_ROOT}/modules/models/weights/visionpilot_core0/visionpilot.msgpack"




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

echo "==> Preparing deployment directory"
# rm -rf -- "${TARGET_DIR}"
cmake --install "${BUILD_DIR}" --prefix "${TARGET_DIR}"

echo "==> Synchronizing deployment to V4M"

rsync -az \
    --info=progress2 \
    --exclude='share/tests/*/frames/***' \
    "${TARGET_DIR}/" \
    "${REMOTE}:${REMOTE_DIR}/"

echo "==> Deployment completed successfully"
echo "    ${REMOTE}:${REMOTE_DIR}"

#make run_vision_pilot.sh executable on v4m (like chmod +x ${REMOTE_DIR}/run_vision_pilot.sh)
ssh "${REMOTE}" "chmod +x ${REMOTE_DIR}/run_vision_pilot.sh"


#now send the 3 model.msgpack to the respective folders on the v4m
# echo "==> Copying model.msgpack to V4M"
# scp "${REACTION_MODEL_DIR}/dummyautodrive/tvm-v4m/tvm_bundle/abundle.msgpack" "${REMOTE}:${REMOTE_DIR}/share/modules/models/weights/autodrive_core0/autodrive.msgpack"
# scp "${REACTION_MODEL_DIR}/dummyautospeed/tvm-v4m/tvm_bundle/abundle.msgpack" "${REMOTE}:${REMOTE_DIR}/share/modules/models/weights/autospeed_core0/autospeed.msgpack"
# scp "${REACTION_MODEL_DIR}/dummyautosteer/tvm-v4m/tvm_bundle/abundle.msgpack" "${REMOTE}:${REMOTE_DIR}/share/modules/models/weights/autosteer_core0/autosteer.msgpack"


# echo "==> Copying single merged visionpilot.msgpack to V4M"
# scp "${REACTION_MODEL_DIR}/dummyvisionpilotmerged/tvm-v4m/tvm_bundle/abundle.msgpack" "${REMOTE}:${REMOTE_DIR}/share/modules/models/weights/visionpilot_core0/visionpilot.msgpack"
