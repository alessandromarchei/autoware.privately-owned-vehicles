#!/bin/bash

set -e

TARGET="root@10.0.0.20"
REMOTE_DIR="/home/root/work/autosteer_hycoah"

echo "Creating directories on V4M..."
ssh "$TARGET" "mkdir -p $REMOTE_DIR/data $REMOTE_DIR/"
echo -e

echo "Copying data..."
scp -r data "$TARGET:$REMOTE_DIR/"
echo -e 

echo "Copying binary..."
scp build/autosteer_sample_app_v4m* "$TARGET:$REMOTE_DIR/"
echo -e

echo "Copying execution script..."
scp run_autosteer.sh "$TARGET:$REMOTE_DIR/"
echo -e

echo "Done."
