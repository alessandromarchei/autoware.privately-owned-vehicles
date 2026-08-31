cmake_minimum_required(VERSION 3.21.0)

find_package(OpenCV REQUIRED COMPONENTS core imgproc)

add_library(models
    # src/auto_drive.cpp
    # src/auto_steer.cpp
    # src/auto_speed.cpp
    src/inference.cpp
    src/visionpilot.cpp
)

target_include_directories(models
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include   # <models/...>
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/include   # <common/...>
        ${CMAKE_CURRENT_SOURCE_DIR}/../engine/include   # <engine/...>
)

# engine provides OnnxEngine + ORT headers/libs transitively
target_link_libraries(models
    PUBLIC
        common
        engine
        fusion
        logging
        opencv_core
        opencv_imgproc
)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/weights
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

target_compile_features(models PUBLIC cxx_std_17)