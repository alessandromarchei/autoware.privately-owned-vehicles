cmake_minimum_required(VERSION 3.21)

add_library(engine
    src/v4m_engine.cpp
)


message(STATUS "Including directory: ${CMAKE_CURRENT_SOURCE_DIR}/include")

target_include_directories(engine
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(engine
    PUBLIC
        v4m_platform
)

target_compile_features(engine
    PUBLIC
        cxx_std_17
)