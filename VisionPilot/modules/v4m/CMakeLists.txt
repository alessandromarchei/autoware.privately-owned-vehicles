cmake_minimum_required(VERSION 3.21)

# Usage requirements shared by every V4M target.
add_library(v4m_platform INTERFACE)

target_include_directories(v4m_platform
    INTERFACE
        ${rcar-xos_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/src/file_common/inc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/vsfwk_common/inc
)

target_compile_definitions(v4m_platform
    INTERFACE
        RCAR_V4M
)

if(RGID_ENABLE)
    target_compile_definitions(v4m_platform
        INTERFACE
            RGID_ENABLE
    )
endif()

target_compile_features(v4m_platform
    INTERFACE
        cxx_std_17
)

target_link_libraries(v4m_platform
    INTERFACE
        osal
        osal_wrapper
        hwa_buffer_mngr
        exfwk
        atmlib
        hycoah
)

# Actual reusable V4M implementation.
add_library(v4m_support STATIC
    src/file_common/src/common.cpp
    src/vsfwk_common/src/vsfwk_utils.cpp
)

target_link_libraries(v4m_support
    PUBLIC
        v4m_platform
)