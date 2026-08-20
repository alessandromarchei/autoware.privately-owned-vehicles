cmake_minimum_required(VERSION 3.21.0)

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

add_library(tcpip_client
    src/tcpip_client.cpp
)

target_include_directories(tcpip_client
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(tcpip_client
        PUBLIC
        logging
        common
        ${OpenCV_LIBS}
        Threads::Threads
)