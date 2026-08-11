# VisionPilot V4M CMake Dependencies

This document summarizes the current CMake dependency graph and a safe progressive build order for R-Car V4M/Poky.

## Core rule

A name in `link_lib` must be either:

- a CMake target created with `add_subdirectory(...)`; or
- a library available in the toolchain sysroot/link paths.

For example, adding `config` to `link_lib` while `add_subdirectory(modules/config)` is disabled produces `-lconfig`, so the linker searches for `libconfig.so` or `libconfig.a` and fails.

## Target map

| Module | CMake target | Direct dependencies |
|---|---|---|
| Logging | `logging` | None |
| Common | `common` | OpenCV |
| Engine | `engine` | ONNX Runtime |
| Fusion | `fusion` | `models`, `logging`, OpenCV Core |
| Models | `models` | `common`, `engine`, `fusion`, `logging`, OpenCV Core/Imgproc |
| Config | `config` | `engine`, `models` |
| Image preprocessing | `image_preprocessing` | `common`, OpenCV |
| Vehicle interface | `vehicle_interface` | None declared |
| Camera interface | `camera_interface` | `config`, `logging`, OpenCV Core/VideoIO, Threads |
| Planning | `planning` | `common`, vendored Eigen, IPOPT |
| Debug | `vp_debug` | `models`, `fusion`, OpenCV |
| Visualization | `visualization` | `models`, `common`, OpenCV, Threads, optional WebRTC stack |
| ROS2 camera | `camera_subscriber` | `camera_interface`, ROS2, cv_bridge, OpenCV |
| ROS2 vehicle | `vehicle_ros2_interface` | `vehicle_interface`, ROS2 |
| Application | `VisionPilot` | Currently `config`, `logging`, `common` |
| V4M support | No target | Populates variables used by `rcar_configure_application()` |

## Critical issues

### `models` and `fusion` cycle

```text
models -> fusion -> models
```

Move shared DTOs such as `AutoDriveOutput`, `AutoSpeedOutput`, `Detection`, and fusion result types into an independent header-only target:

```cmake
add_library(visionpilot_types INTERFACE)
target_include_directories(visionpilot_types INTERFACE include)
```

Both `models` and `fusion` should depend on `visionpilot_types`, rather than on each other. If orchestration requires both, move it to a higher-level pipeline target.

### `config` is not lightweight

`config` currently pulls in:

```text
config -> engine -> ONNX Runtime
       -> models -> common, engine, fusion, logging, OpenCV
```

Therefore, it cannot be built immediately after `logging` and `common`. Move `EngineConfig` and model configuration types into a lightweight types/config API so `config` does not link full implementations.

### ONNX Runtime versus HYCO

`engine` requires:

```text
${ONNXRUNTIME_ROOT}/include
${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so
```

For V4M, that library must be AArch64/Poky-compatible. If inference uses HYCO/Artifact Helper, create a separate `engine_v4m` backend and avoid forcing ONNX Runtime into the V4M build.

### ROS2 camera cycle

With ROS2 enabled:

```text
camera_interface -> camera_subscriber -> camera_interface
```

The base camera interface should not link its ROS2 implementation. Keep the interface independent and let the final application select the implementation.

### V4M CMake scope and case

`modules/v4m/CMakeLists.txt` does not create a target. Its `source`, `header`, `include_dir`, and `link_lib` changes remain in the subdirectory scope unless explicitly propagated.

Use:

```bash
-DRCAR_TARGET_OS=linux
```

not `LINUX`, because `STREQUAL` is case-sensitive. Also initialize `source` before conditional `list(APPEND source ...)`; a later `set(source ...)` overwrites previous entries.

## Progressive build order

### 0. R-Car base

Build a dummy `main()` with only:

```cmake
osal
osal_wrapper
hwa_buffer_mngr
exfwk
atmlib
hycoah
```

This validates the cross-toolchain, sysroot, xOS, and HYCO linkage.

### 1. Independent modules

```cmake
add_subdirectory(modules/logging)
add_subdirectory(modules/sensing/vehicle_interface)
```

Link:

```cmake
logging
vehicle_interface
```

### 2. Common and OpenCV

```cmake
add_subdirectory(modules/common)
```

Link `common`. Prefer explicit OpenCV components instead of unrestricted `${OpenCV_LIBS}`.

### 3. Image preprocessing

```cmake
add_subdirectory(modules/sensing/image_preprocessing)
```

Link `image_preprocessing`; `common` and OpenCV propagate transitively.

### 4. Planning, optional

First verify that IPOPT is available for the Poky sysroot:

```bash
pkg-config --modversion ipopt
pkg-config --cflags --libs ipopt
```

Then enable:

```cmake
add_subdirectory(modules/safety_guardian/planning)
```

Prefer an imported pkg-config target:

```cmake
pkg_check_modules(IPOPT REQUIRED IMPORTED_TARGET ipopt)
target_link_libraries(planning PRIVATE PkgConfig::IPOPT)
```

The vendored Eigen tests, demos, and benchmarks are not application dependencies because Eigen is added with `EXCLUDE_FROM_ALL`.

### 5. Inference backend

Choose one backend:

- `engine` with an AArch64 ONNX Runtime build; or
- a dedicated `engine_v4m` using HYCO/Artifact Helper.

### 6. Models and fusion

Resolve the dependency cycle first. Recommended order after refactoring:

```text
visionpilot_types -> fusion -> models
```

### 7. Config

Enable `config` only after its dependencies exist, or after reducing it to lightweight type dependencies.

### 8. Camera

With ROS2 disabled:

```cmake
add_subdirectory(modules/sensing/camera_interface)
```

### 9. Optional final modules

Enable last:

```text
vp_debug -> visualization -> ROS2 adapters
```

## Minimal current test

For a dummy application that only logs:

```cmake
add_subdirectory(modules/logging)

set(link_lib
    osal
    osal_wrapper
    hwa_buffer_mngr
    exfwk
    atmlib
    hycoah
    logging
)
```

Use a minimal source:

```cpp
#include <logging/logger.hpp>

int main()
{
    VP_INFO("VisionPilot starting...");
    return 0;
}
```

Temporarily remove unused includes: commented implementation code does not disable active `#include` directives.

## Additional corrections

- Use `set(app_name "vision_pilot")`; `rcar_configure_application()` adds `_v4m` automatically.
- Do not create both the native `VisionPilot` executable and the executable generated by `rcar_configure_application()`.
- Fix the visualization definition:

```cmake
target_compile_definitions(visualization
    PUBLIC ENABLE_WEBRTC=1
    PRIVATE GST_USE_UNSTABLE_API=1
)
```

- Keep ROS2 and visualization disabled during the initial V4M port.

## Useful checks

```bash
# Available targets
cmake --build build --target help

# Verbose single-target build
cmake --build build --target <target> --verbose -j1

# Verify external library architecture
file /path/to/library.so

# Inspect dynamic dependencies without running the cross-built binary
aarch64-poky-linux-readelf -d build/vision_pilot_v4m_d
```

If the verbose link command contains `-lconfig` instead of a path to the generated config archive, the `config` CMake target was not created or recognized.

## Recommended immediate sequence

```text
R-Car base
-> logging
-> vehicle_interface
-> common + OpenCV
-> image_preprocessing
-> planning + IPOPT (optional)
-> V4M inference backend
-> refactored fusion/models
-> config
-> camera
-> debug/visualization/ROS2
```
