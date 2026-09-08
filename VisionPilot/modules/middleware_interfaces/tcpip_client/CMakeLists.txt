cmake_minimum_required(VERSION 3.21.0)

# -----------------------------------------------------------------------------
# FlatBuffers
# -----------------------------------------------------------------------------

if(NOT FLATBUFFERS_ROOT)
    message(FATAL_ERROR
        "Set -DFLATBUFFERS_ROOT=/absolute/path/to/FlatBuffers"
    )
endif()

if(NOT EXISTS "${FLATBUFFERS_ROOT}/include/flatbuffers/flatbuffers.h")
    message(FATAL_ERROR
        "FlatBuffers headers not found below ${FLATBUFFERS_ROOT}/include"
    )
endif()


find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)



add_library(tcpip_client
    src/tcpip_client.cpp
    src/flatbuffer_codec.cpp
)

target_include_directories(tcpip_client
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/schema/generated
        ${FLATBUFFERS_ROOT}/include
)

target_link_libraries(tcpip_client
        PUBLIC
        logging
        common
        ${OpenCV_LIBS}
        Threads::Threads
)