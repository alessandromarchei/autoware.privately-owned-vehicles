#!/bin/bash

binary="./model_custom_io_d"
msgpack="data/autosteer.msgpack"
inputs="data/inputs.bin"
net_name="autosteer_hycoah"

echo "Running autosteer custom io..."
echo "Binary: $binary"
echo "Msgpack: $msgpack"
echo "Inputs: $inputs"
echo "Net name: $net_name"

echo -e "Command : $binary $msgpack $inputs $net_name"

./"$binary" "$msgpack" "$inputs" "$net_name"
