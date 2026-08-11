# This file must be loaded with include(), not add_subdirectory().
# CMAKE_CURRENT_LIST_DIR always identifies modules/v4m here.

set(V4M_MODULE_DIR "${CMAKE_CURRENT_LIST_DIR}")

if(RCAR_TARGET_OS STREQUAL "linux")
    list(APPEND source
        ${V4M_MODULE_DIR}/src/target/linux/r_osal_configuration.c
    )

    list(APPEND header
        ${V4M_MODULE_DIR}/src/file_common/inc/rcar-xos/osal_configuration/target/linux/r_osal_configuration.h
        ${V4M_MODULE_DIR}/src/file_common/inc/rcar-xos/osal_configuration/target/linux/r_osal_cpu_cfg_info.h
        ${V4M_MODULE_DIR}/src/file_common/inc/rcar-xos/osal_configuration/target/common/r_osal_mem_cfg_info.h
    )

elseif(RCAR_TARGET_OS STREQUAL "qnx")
    list(APPEND source
        ${V4M_MODULE_DIR}/src/target/qnx/r_osal_resource_info.c
    )

    list(APPEND link_lib
        socket
    )

elseif(RCAR_TARGET_OS STREQUAL "baremetal")
    list(APPEND link_lib
        startup
    )
endif()

list(APPEND header
    ${V4M_MODULE_DIR}/src/file_common/inc/common.hpp
    ${V4M_MODULE_DIR}/src/vsfwk_common/inc/vsfwk_utils.hpp
    ${V4M_MODULE_DIR}/src/vsfwk_common/inc/r_sample_init_config.hpp
)

list(APPEND include_dir
    ${rcar-xos_INCLUDE_DIRS}
    ${V4M_MODULE_DIR}/src/file_common/inc
    ${V4M_MODULE_DIR}/src/vsfwk_common/inc
)

# Keep xOS components explicit because rcar_configure_application()
# may inspect them, not only pass them to the linker.
list(APPEND link_lib
    osal
    osal_wrapper
    hwa_buffer_mngr
    exfwk
    atmlib
    hycoah
    v4m_support
)