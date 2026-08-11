#!/bin/bash

binary="./autosteer_sample_app_v4m_d"
msgpack="data/autosteer.msgpack"
inputs="data/inputs.bin"
net_name="autosteer_hycoah"

echo "Running autosteer sample app..."
echo "Binary: $binary"
echo "Msgpack: $msgpack"
echo "Inputs: $inputs"
echo "Net name: $net_name"

echo -e "Command : $binary $msgpack $inputs $net_name"

./"$binary" "$msgpack" "$inputs" "$net_name"
