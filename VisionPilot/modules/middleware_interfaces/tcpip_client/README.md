# TCP frame client for VisionPilot

This module implements the requested topology:

```text
Predator (10.0.0.1)             V4M (10.0.0.20)
Python image server  -------->  TCPClient
                     <--------  VisionResult
```

It uses the same version-1 `VPNT` wire format as the earlier
`tcp_middleware`, but reverses which host listens. The two C++ libraries should
not be linked into the same executable because they export the same public
protocol names and internal wire helper symbols.

## Build for V4M

Use the same Poky environment and CMake arguments as VisionPilot:

```bash
cmake -S . -B build-v4m \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=/opt/rcar-xos/v3.47.0/cmake/toolchain_poky_5_0_adas.cmake \
  -DCMAKE_PREFIX_PATH=/opt/rcar-xos/v3.47.0/cmake

cmake --build build-v4m --parallel 8
```

Copy `build-v4m/v4m_image_client` to the target.

## Run

On Predator:

```bash
python3 -m pip install opencv-python
python3 python/image_server.py /path/to/images --bind 10.0.0.1 --port 5000
```

On V4M:

```bash
./v4m_image_client 10.0.0.1 5000
```

The Python server sends one frame and waits for its corresponding result, so
the test cannot accumulate a frame backlog.

## Integrate into VisionPilot

Add the directory after OpenCV is available:

```cmake
add_subdirectory(tcp_frame_client)
target_link_libraries(vision_pilot_v4m PRIVATE visionpilot::tcp_frame_client)
```

For an embedded build without the example:

```bash
cmake ... -DTCP_FRAME_CLIENT_BUILD_EXAMPLE=OFF
```

Typical application flow:

```cpp
visionpilot::tcp::TCPClient source;
source.connect_to("10.0.0.1", 5000);

cv::Mat frame;
visionpilot::tcp::ReceivedFrame rx;
if (source.receive_frame(frame, rx, 5000)) {
    // pipeline.process(frame, ...)
    visionpilot::tcp::VisionResult result{};
    result.frame_id = rx.frame_id;
    result.timestamp_ns = rx.timestamp_ns;
    source.send_result(result);
}
```
