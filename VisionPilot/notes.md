## Modules & Dependencies

- [x] **Logging** (`logging`)
  - Direct dependencies: None

- [x] **Common** (`common`)
  - Direct dependencies: OpenCV

- [x] **Engine** (`engine`)
  - Direct dependencies: ONNX Runtime

- [x] **Fusion** (`fusion`)
  - Direct dependencies: `logging`, OpenCV Core

- [x] **Models** (`models`)
  - Direct dependencies: `common`, `engine`, `fusion`, `logging`, OpenCV Core/Imgproc

- [x] **Config** (`config`)
  - Direct dependencies: `engine`, `models`

- [x] **Image preprocessing** (`image_preprocessing`)
  - Direct dependencies: `common`, OpenCV

- [x] **Vehicle interface** (`vehicle_interface`)
  - Direct dependencies: None declared

- [x] **Camera interface** (`camera_interface`)
  - Direct dependencies: `config`, `logging`, OpenCV Core/VideoIO, Threads

- [ ] **Planning** (`planning`)
  - Direct dependencies: `common`, vendored Eigen, IPOPT

- [x] **Debug** (`vp_debug`)
  - Direct dependencies: `models`, `fusion`, OpenCV

- [not included ] **Visualization** (`visualization`)
  - Direct dependencies: `models`, `common`, OpenCV, Threads, optional WebRTC stack

- [not included ] **ROS2 camera** (`camera_subscriber`)
  - Direct dependencies: `camera_interface`, ROS2, cv_bridge, OpenCV

- [not included ] **ROS2 vehicle** (`vehicle_ros2_interface`)
  - Direct dependencies: `vehicle_interface`, ROS2

- [ ] **Application** (`VisionPilot`)
  - Direct dependencies: Currently `config`, `logging`, `common`

- [x] **V4M support** (No target)
  - Direct dependencies: Populates variables used by `rcar_configure_application()`



PLANNING Depends on 
-IPOPTS
-CPPAD

to avoid fetching them from the linux system, we can compile them from source and link them statically (with poky linux toolchain) to the planning module. 
this is done by adding thirdparty submodules for IPOPT and CPPAD, and adding them to the CMakeLists.txt of the planning module.