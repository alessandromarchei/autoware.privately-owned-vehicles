cmake_minimum_required(VERSION 3.21.0)

find_package(OpenCV REQUIRED)

add_library(image_preprocessing
        src/image_preprocessor.cpp
)

target_include_directories(image_preprocessing
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(image_preprocessing
        PUBLIC
        common
        ${OpenCV_LIBS}
)