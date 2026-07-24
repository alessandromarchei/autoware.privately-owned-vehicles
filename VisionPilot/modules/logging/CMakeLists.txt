cmake_minimum_required(VERSION 3.21.0)

add_library(logging
        src/logger.cpp
)

target_include_directories(logging
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)