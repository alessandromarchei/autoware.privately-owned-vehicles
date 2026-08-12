cmake_minimum_required(VERSION 3.21.0)

find_package(OpenCV REQUIRED COMPONENTS core)

add_library(fusion
    src/longitudinal_fusion.cpp
    src/lateral_fusion.cpp
)

target_include_directories(fusion
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include   # <fusion/...>
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/include  # <common/...>
)

target_link_libraries(fusion
    PUBLIC
        common
        opencv_core     # cv::Mat in public header
    PRIVATE
        logging         # VP_INFO macros used in longitudinal_fusion.cpp
)

target_compile_features(fusion PUBLIC cxx_std_17)
