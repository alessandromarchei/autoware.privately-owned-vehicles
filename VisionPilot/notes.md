## Modules & Dependencies

- [x] **Logging** (`logging`)
  - Direct dependencies: None

- [x] **Common** (`common`)
  - Direct dependencies: OpenCV

- [ ] **Engine** (`engine`)
  - Direct dependencies: ONNX Runtime

- [ ] **Fusion** (`fusion`)
  - Direct dependencies: `models`, `logging`, OpenCV Core

- [ ] **Models** (`models`)
  - Direct dependencies: `common`, `engine`, `fusion`, `logging`, OpenCV Core/Imgproc

- [ ] **Config** (`config`)
  - Direct dependencies: `engine`, `models`

- [x] **Image preprocessing** (`image_preprocessing`)
  - Direct dependencies: `common`, OpenCV

- [x] **Vehicle interface** (`vehicle_interface`)
  - Direct dependencies: None declared

- [ ] **Camera interface** (`camera_interface`)
  - Direct dependencies: `config`, `logging`, OpenCV Core/VideoIO, Threads

- [ ] **Planning** (`planning`)
  - Direct dependencies: `common`, vendored Eigen, IPOPT

- [ ] **Debug** (`vp_debug`)
  - Direct dependencies: `models`, `fusion`, OpenCV

- [not included ] **Visualization** (`visualization`)
  - Direct dependencies: `models`, `common`, OpenCV, Threads, optional WebRTC stack

- [not included ] **ROS2 camera** (`camera_subscriber`)
  - Direct dependencies: `camera_interface`, ROS2, cv_bridge, OpenCV

- [not included ] **ROS2 vehicle** (`vehicle_ros2_interface`)
  - Direct dependencies: `vehicle_interface`, ROS2

- [ ] **Application** (`VisionPilot`)
  - Direct dependencies: Currently `config`, `logging`, `common`

- [ ] **V4M support** (No target)
  - Direct dependencies: Populates variables used by `rcar_configure_application()`