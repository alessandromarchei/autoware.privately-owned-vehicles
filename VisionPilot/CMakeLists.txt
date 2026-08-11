cmake_minimum_required(VERSION 3.21.0)

project(VisionPilot VERSION 1.0 LANGUAGES C CXX)

set(app_name "vision_pilot")
set(app_version "1.0.0")


# Put all executables in build/
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# # add cmake prefix path for rcar-xos
# set(CMAKE_PREFIX_PATH "/opt/rcar-xos/v3.47.0/cmake" ${CMAKE_PREFIX_PATH})

# # add toolchain file for rcar-xos
# set(CMAKE_TOOLCHAIN_FILE "/opt/rcar-xos/v3.47.0/cmake/toolchain_poky_5_0_adas.cmake" ${CMAKE_TOOLCHAIN_FILE})

# add compile commands for IDEs
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


#----------------------------------------------------------------------------------------------------------------------
# R-Car xOS SoC/Board selection
#----------------------------------------------------------------------------------------------------------------------

#set V4M as default target SoC
if(NOT RCAR_SOC) # This guard allows to set RCAR_SOC externally, e.g. via CMake Commandline during configuration
  set(RCAR_SOC
    V4M
  )
endif()


#set linux as default target OS
if(NOT RCAR_TARGET_OS) # This guard allows to set RCAR_TARGET_OS externally, e.g. via CMake Commandline during configuration
  set(RCAR_TARGET_OS
    LINUX
  )
endif()

find_package(rcar-xos REQUIRED)

if(${rcar-xos_FOUND})
  message("Found xOS ${rcar-xos_VERSION}")
endif()

# set(CMAKE_CXX_STANDARD 14)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(RCAR_SOC STREQUAL "v4m")
    add_definitions(-DRCAR_V4M)
endif()

if(RGID_ENABLE)
    add_definitions(-DRGID_ENABLE)
endif()


message(STATUS "RCAR_TARGET_OS: ${RCAR_TARGET_OS}")
message(STATUS "RCAR_SOC: ${RCAR_SOC}")
message(STATUS "CMAKE_C_COMPILER: ${CMAKE_C_COMPILER}")
message(STATUS "CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
message(STATUS "CMAKE_C_COMPILER_ID: ${CMAKE_C_COMPILER_ID}")

#----------------------------------------------------------------------------------------------------------------------
# Add warning flags
#----------------------------------------------------------------------------------------------------------------------
if(RCAR_TARGET_OS STREQUAL "linux" AND CMAKE_C_COMPILER_ID STREQUAL "ARMClang")
    add_compile_options(-Wall -Wextra -pedantic -Werror
                        -Wno-sign-conversion)
                        # -Wno-unused-variable)
else()
    add_compile_options(-Wall -Wextra -pedantic -Werror)
endif()


set(Python3_EXECUTABLE /usr/bin/python3 CACHE FILEPATH "" FORCE)
find_package(Python3 REQUIRED)

add_custom_target(generate_config ALL
        COMMAND Python3::Interpreter
        ${CMAKE_SOURCE_DIR}/scripts/find_homography_C_matrix.py
        --output ${CMAKE_BINARY_DIR}/config/homography_C_matrix.yaml
)

# Modules
# add_subdirectory(modules/v4m)

add_subdirectory(modules/logging)

# add_subdirectory(modules/config)

add_subdirectory(modules/common)

add_subdirectory(modules/sensing/image_preprocessing)

# if(ENABLE_ROS2_INTERFACE)
#     add_subdirectory(modules/middleware_interfaces/ros2_interface/camera_subscriber)
#     add_subdirectory(modules/middleware_interfaces/ros2_interface/vehicle_ros2_interface)
# endif()

# add_subdirectory(modules/engine)
# add_subdirectory(modules/models)

# add_subdirectory(modules/safety_guardian/fusion)   # tracking is embedded inside fusion
# add_subdirectory(modules/safety_guardian/planning)

add_subdirectory(modules/sensing/vehicle_interface)
# add_subdirectory(modules/sensing/camera_interface)

# add_subdirectory(modules/visualization)   //not needed for now

# add_subdirectory(modules/debug)

# config
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/config
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# App
# add_subdirectory(app)     #TODO : reestablish previous app. for now a dummy app is created


# Architecture
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(ARCH_NAME "x86_64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(ARCH_NAME "arm64")
else()
    set(ARCH_NAME "${CMAKE_SYSTEM_PROCESSOR}")
endif()


set(vision_pilot_libraries
    # config           # VisionPilotConfig, load_vision_pilot_config
    logging          # VP_INFO / VP_WARN / VP_ERROR
    common           # Common module
    # engine           # OnnxEngine
    # models           # AutoDrive, AutoSteer, AutoSpeed
    # fusion           # LongitudinalFusion, LateralFusion
    # planning         # Planner
    vehicle_interface
    # camera_interface
    image_preprocessing
    # visualization
    # vp_debug         # Debug annotate_frame
    )

#----------------------------------------------------------------------------------------------------------------------
# Link libraries and Link directories used by the application
#----------------------------------------------------------------------------------------------------------------------
set(link_lib
    osal
    osal_wrapper
    hwa_buffer_mngr
    exfwk
    atmlib
    hycoah

    #vision pilot libraries
    ${vision_pilot_libraries}

)



list(APPEND source
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/v4m/src/target/linux/r_osal_configuration.c
)

list(APPEND header
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/v4m/src/file_common/inc/rcar-xos/osal_configuration/target/linux/r_osal_configuration.h
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/v4m/src/file_common/inc/rcar-xos/osal_configuration/target/linux/r_osal_cpu_cfg_info.h
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/v4m/src/file_common/inc/rcar-xos/osal_configuration/target/common/r_osal_mem_cfg_info.h
)

#print cmake source dir
message(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")

#----------------------------------------------------------------------------------------------------------------------
# Source files
#----------------------------------------------------------------------------------------------------------------------
set(source
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/v4m/src/file_common/src/common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/app/vision_pilot.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/modules/v4m/src/main.cpp
)

#----------------------------------------------------------------------------------------------------------------------
# Header files of the application
#----------------------------------------------------------------------------------------------------------------------
set(header
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/v4m/src/file_common/inc/common.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/v4m/src/vsfwk_common/inc/vsfwk_utils.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/v4m/src/vsfwk_common/inc/r_sample_init_config.hpp
)

#----------------------------------------------------------------------------------------------------------------------
# Include directories used by the application
#----------------------------------------------------------------------------------------------------------------------
set(include_dir
    ${rcar-xos_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/v4m/src/file_common/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/v4m/src/vsfwk_common/inc
    # add any additional 3rd party include directories your application requires
    # Note: Include directories from xOS Software components are automatically added when you list them in link_libs
)





#======================================================================================================================
# DO NOT CHANGE ANYTHING BELOW THIS LINE, except you know what to do ;)
#======================================================================================================================
rcar_configure_application()