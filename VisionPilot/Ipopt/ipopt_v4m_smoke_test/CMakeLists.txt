cmake_minimum_required(VERSION 3.21)

project(ipopt_v4m_smoke_test VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

set(IPOPT_ROOT "" CACHE PATH "Root of the extracted AArch64 Ipopt JLL bundle")
if(NOT IPOPT_ROOT)
  message(FATAL_ERROR "Set -DIPOPT_ROOT=/absolute/path/to/ipopt")
endif()

if(NOT EXISTS "${IPOPT_ROOT}/include/coin-or/IpIpoptApplication.hpp")
  message(FATAL_ERROR "Ipopt headers not found below ${IPOPT_ROOT}")
endif()
if(NOT EXISTS "${IPOPT_ROOT}/lib/libipopt.so")
  message(FATAL_ERROR "libipopt.so not found below ${IPOPT_ROOT}")
endif()

add_library(Ipopt::Ipopt SHARED IMPORTED GLOBAL)
set_target_properties(Ipopt::Ipopt PROPERTIES
  IMPORTED_LOCATION "${IPOPT_ROOT}/lib/libipopt.so"
  INTERFACE_INCLUDE_DIRECTORIES "${IPOPT_ROOT}/include/coin-or"
)

add_executable(ipopt_v4m_smoke_test main.cpp)
target_link_libraries(ipopt_v4m_smoke_test PRIVATE Ipopt::Ipopt)
target_compile_options(ipopt_v4m_smoke_test PRIVATE -Wall -Wextra -Wpedantic -Werror)

# Let the cross-linker inspect Ipopt's dependent shared libraries. At runtime
# the board uses LD_LIBRARY_PATH, keeping the executable independent of a fixed
# installation directory.
target_link_options(ipopt_v4m_smoke_test PRIVATE
  "-Wl,-rpath-link,${IPOPT_ROOT}/lib"
)

message(STATUS "Target processor: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Ipopt bundle: ${IPOPT_ROOT}")
