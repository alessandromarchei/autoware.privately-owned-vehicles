#----------------------------------------------------------------------------------------------------------------------
# Set app name
#----------------------------------------------------------------------------------------------------------------------
set(app_name "autosteer_sample_app")
set(app_version "2.0.0")

cmake_minimum_required(VERSION 3.21.0)
project(app_name LANGUAGES C CXX)

#----------------------------------------------------------------------------------------------------------------------
# R-Car xOS SoC/Board selection
#----------------------------------------------------------------------------------------------------------------------

if(NOT RCAR_SOC) # This guard allows to set RCAR_SOC externally, e.g. via CMake Commandline during configuration
  set(RCAR_SOC
    V4H2 # Only V4H2 is available right now. Availability depends on the package you have.
  )
endif()

# Alternative: Selection via your board:
# if(NOT RCAR_BOARD) # This guard allows to set RCAR_BOARD externally, e.g. via CMake Commandline during configuration
#     set(RCAR_BOARD
#         v3hsk # eagle, v3msk, condor, v3hsk. Availability depends on the package you have.
#     )
# endif()

#----------------------------------------------------------------------------------------------------------------------
# Find R-Car xOS platform with your requested configuration
# - In case you need a specific version, usage: find_package(rcar-xos v0.3.0 EXACT REQUIRED)
# - In case you need at least version, usage: find_package(rcar-xos v0.3.0 REQUIRED)
#
# If the command fails to find the package:
#   Make sure rcar-xos config files can be found via standard variables set in your system:
#   PATH environment variable or rcar-xos_DIR (environment or CMake variable set via commandline)
#
# See R-Car xOS documentation for more help or https://cmake.org/cmake/help/latest/command/find_package.html
#----------------------------------------------------------------------------------------------------------------------
find_package(rcar-xos REQUIRED)

if(${rcar-xos_FOUND})
  message("Found xOS ${rcar-xos_VERSION}")
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(RCAR_SOC STREQUAL "v4m")
    add_definitions(-DRCAR_V4M)
endif()

if(RGID_ENABLE)
    add_definitions(-DRGID_ENABLE)
endif()

#----------------------------------------------------------------------------------------------------------------------
# Add warning flags
#----------------------------------------------------------------------------------------------------------------------
if(RCAR_TARGET_OS STREQUAL "linux" AND CMAKE_C_COMPILER_ID STREQUAL "ARMClang")
    add_compile_options(-Wall -Wextra -pedantic -Werror
                        -Wno-sign-conversion
                        -Wno-unused-variable)
else()
    add_compile_options(-Wall -Wextra -pedantic -Werror)
endif()

#----------------------------------------------------------------------------------------------------------------------
# Source files
#----------------------------------------------------------------------------------------------------------------------
set(source
    ${CMAKE_CURRENT_SOURCE_DIR}/application/src/file_common/src/common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/application/src/main.cpp
)

set(header
    ${CMAKE_CURRENT_SOURCE_DIR}/application/src/file_common/inc/common.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/application/src/vsfwk_common/inc/vsfwk_utils.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/application/src/vsfwk_common/inc/r_sample_init_config.hpp
)

#----------------------------------------------------------------------------------------------------------------------
# Include directories used by the application
#----------------------------------------------------------------------------------------------------------------------
set(include_dir
    ${rcar-xos_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/application/src/file_common/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/application/src/vsfwk_common/inc
    # add any additional 3rd party include directories your application requires
    # Note: Include directories from xOS Software components are automatically added when you list them in link_libs
)

#----------------------------------------------------------------------------------------------------------------------
# Include extra files in release package
#----------------------------------------------------------------------------------------------------------------------
set(sample_app_custom_file
    ${CMAKE_CURRENT_SOURCE_DIR}/Readme.md
)

#----------------------------------------------------------------------------------------------------------------------
# Link libraries and Link directories used by the application
#----------------------------------------------------------------------------------------------------------------------
set(link_lib
    # This sample does not need any other sw compoents
    # Examples could be: osal, osal_wrapper
    osal
    osal_wrapper
    hwa_buffer_mngr
    exfwk
    atmlib
    hycoah
)

if(RCAR_TARGET_OS STREQUAL "baremetal" )
    list(APPEND link_lib
        startup         # In baremetal environments the application requires at least startup code defined for xOS
    )
else() # Any other Target OS
    if(RCAR_XIL STREQUAL "hil")
        if(RCAR_TARGET_OS STREQUAL "qnx")
            list(APPEND link_lib
                socket
            )
            list(APPEND source
                ${CMAKE_CURRENT_SOURCE_DIR}/application/src/target/qnx/r_osal_resource_info.c
            )
        elseif(RCAR_TARGET_OS STREQUAL "linux")
            list(APPEND source
                ${CMAKE_CURRENT_SOURCE_DIR}/application/src/target/linux/r_osal_configuration.c
            )
        endif()
    endif()
endif()

# Copy header from osal_configuration module
if(EXISTS "${CMAKE_SOURCE_DIR}/renesas/os/osal_configuration/include")
    execute_process(COMMAND cmake -E copy_directory ${CMAKE_SOURCE_DIR}/renesas/os/osal_configuration/include/ ${CMAKE_CURRENT_SOURCE_DIR}/application/src/file_common/inc)
endif()
if(RCAR_TARGET_OS STREQUAL "qnx")
    list(APPEND header
        ${CMAKE_CURRENT_SOURCE_DIR}/application/src/file_common/inc/rcar-xos/osal_configuration/target/qnx/r_osal_configuration.h
        ${CMAKE_CURRENT_SOURCE_DIR}/application/src/file_common/inc/rcar-xos/osal_configuration/target/qnx/r_osal_cpu_cfg_info.h
        ${CMAKE_CURRENT_SOURCE_DIR}/application/src/file_common/inc/rcar-xos/osal_configuration/target/common/r_osal_mem_cfg_info.h
        ${CMAKE_CURRENT_SOURCE_DIR}/application/src/file_common/inc/rcar-xos/osal_configuration/target/common/r_osal_dev_cfg_info.h
        ${CMAKE_CURRENT_SOURCE_DIR}/application/src/file_common/inc/rcar-xos/osal_configuration/target/common/r_osal_res_cfg_info.h
    )
elseif(RCAR_TARGET_OS STREQUAL "linux")
    list(APPEND header
        ${CMAKE_CURRENT_SOURCE_DIR}/application/src/file_common/inc/rcar-xos/osal_configuration/target/linux/r_osal_configuration.h
        ${CMAKE_CURRENT_SOURCE_DIR}/application/src/file_common/inc/rcar-xos/osal_configuration/target/linux/r_osal_cpu_cfg_info.h
        ${CMAKE_CURRENT_SOURCE_DIR}/application/src/file_common/inc/rcar-xos/osal_configuration/target/common/r_osal_mem_cfg_info.h
    )
endif()

# [Optional] for 3rd Party libraries
set(link_dir
    # Additional Link directories for 3rd party libraries
)

#----------------------------------------------------------------------------------------------------------------------
# Test settings, e.g. which and how to run the tests for this application
#----------------------------------------------------------------------------------------------------------------------
set(test_args_config_file
    ${CMAKE_CURRENT_SOURCE_DIR}/test_args_${RCAR_SOC}.json
)
set(test_run_time_test_config_file
    ${CMAKE_CURRENT_SOURCE_DIR}/run_time_test_data_${RCAR_SOC}.json
)

#----------------------------------------------------------------------------------------------------------------------
# Set document file list
# If not set, it will search for document in ${CMAKE_CURRENT_SOURCE_DIR}/doc by default
#----------------------------------------------------------------------------------------------------------------------
set(document
    # List of document file to install to package
)
#----------------------------------------------------------------------------------------------------------------------
# [Optional] Any specifics that an application may require, like OS specifics, SoC specifics
#----------------------------------------------------------------------------------------------------------------------

# [Optional] Data directory at POST_BUILD


#======================================================================================================================
# DO NOT CHANGE ANYTHING BELOW THIS LINE, except you know what to do ;)
#======================================================================================================================
rcar_configure_application()
