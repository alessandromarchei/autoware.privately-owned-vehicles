# VisionPilot Remote Host

Linux host application for sending raw camera frames + ego speed to a Renesas
R-Car V4M target, receiving `VisionPilotOutput` over the result TCP channel, and
rendering the V4M inference/planning output locally.

## 1. Install

```bash
python -m pip install -r requirements.txt
sudo apt install flatbuffers-compiler
python generate_bindings.py
```

## 2. Network

The V4M executable is the TCP **client**. The Linux machine is the TCP **server**.

Default channels:

- `8080`: host -> V4M, raw BGR8 images
- `8081`: V4M -> host, `VisionPilotOutput` FlatBuffer

On the V4M executable use the Linux host IP, for example:

```bash
./visionpilot \
  --source-mode tcpip_frames \
  --tcp-server 10.0.0.1 \
  --tcp-frame-port 8080 \
  --tcp-result-port 8081
```

Start the Linux host first so both listeners already exist.

## 3. Video example

```bash
python host_app.py \
  --video /path/test.mp4 \
  --speed-file /path/vehicle_speed.csv \
  --bind 0.0.0.0 \
  --frame-port 8080 \
  --result-port 8081 \
  --realtime
```

## 4. Extracted frames

```bash
python host_app.py \
  --frames /path/frames \
  --speed 13.9 \
  --fps 30 \
  --realtime
```

## 5. Camera

```bash
python host_app.py \
  --camera 0 \
  --speed 0 \
  --realtime
```

## Homography

For the fused RANSAC path, pass the same pixel->world homography used for the
displayed camera plane if available:

```bash
python host_app.py \
  --video test.mp4 \
  --speed 12 \
  --homography H.yaml \
  --homography-key H \
  --homography-space net
```

`--homography-space net` means `H` operates on the 1024x512 VisionPilot image.
The renderer scales the projection back to the native camera frame.

If `--homography` is omitted, the renderer falls back to the hard-coded
VisionPilot matrix from the existing C++ visualization. If your production
pipeline's `H_resized` differs from this matrix, pass the actual matrix to get
pixel-identical fused-path projection.

## Display

The camera image contains:

- fused RANSAC path/corridor reconstructed from `path_a,b,c`
- AutoSteer 64-point path
- AutoSpeed bounding boxes
- CIPO distance
- warnings
- runtime/network top bar

The side telemetry panel shows:

- V4M wall/preprocess/VisionPilot latency
- AutoDrive raw output
- AutoSteer valid points
- AutoSpeed detection count
- fused CIPO distance/velocity/uncertainty/cut-in
- fused CTE/yaw/curvature and RANSAC inliers
- planner steering/acceleration/warnings

Press `q` or `Esc` to exit.
