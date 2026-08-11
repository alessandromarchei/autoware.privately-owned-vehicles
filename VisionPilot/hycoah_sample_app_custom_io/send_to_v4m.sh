#!/bin/bash

set -e

echo "Building for V4M..."
#prepare build files 
cmake      -B build   -G "Unix Makefiles"   -DCMAKE_BUILD_TYPE=Debug   -DRCAR_SOC=v4m   -DRCAR_TARGET_OS=LINUX   -DCMAKE_PREFIX_PATH=/opt/rcar-xos/v3.47.0/cmake   -DCMAKE_TOOLCHAIN_FILE=/opt/rcar-xos/v3.47.0/cmake/toolchain_poky_5_0_adas.cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

#compile 
cmake --build build -j8 

#send data to target
echo "Sending files to V4M..."


TARGET="root@10.0.0.20"
REMOTE_DIR="/home/root/work/model_custom_io"

echo "Creating directories on V4M..."
ssh "$TARGET" "mkdir -p $REMOTE_DIR/data $REMOTE_DIR/"
echo -e

echo "Copying data..."
scp -r data "$TARGET:$REMOTE_DIR/"
echo -e 

echo "Copying binary..."
scp build/model_custom_io* "$TARGET:$REMOTE_DIR/"
echo -e

echo "Copying execution script..."
scp run_autosteer.sh "$TARGET:$REMOTE_DIR/"
echo -e

echo "Done."
