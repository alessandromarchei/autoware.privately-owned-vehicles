cmake_minimum_required(VERSION 3.21.0)

# -----------------------------------------------------------------------------
# CppAD
# -----------------------------------------------------------------------------

set(CPPAD_ROOT
    ""
    CACHE PATH
    "Root of the CppAD source or installation"
)

if(NOT CPPAD_ROOT)
    message(FATAL_ERROR
        "Set -DCPPAD_ROOT=/absolute/path/to/CppAD"
    )
endif()

if(NOT EXISTS "${CPPAD_ROOT}/include/cppad/cppad.hpp")
    message(FATAL_ERROR
        "CppAD headers not found below ${CPPAD_ROOT}/include"
    )
endif()

# -----------------------------------------------------------------------------
# Ipopt
# -----------------------------------------------------------------------------

if(CMAKE_CROSSCOMPILING)
    set(IPOPT_ROOT
        ""
        CACHE PATH
        "Root of the extracted AArch64 Ipopt bundle"
    )

    if(NOT IPOPT_ROOT)
        message(FATAL_ERROR
            "Cross-compilation requires "
            "-DIPOPT_ROOT=/absolute/path/to/ipopt"
        )
    endif()

    if(NOT EXISTS
       "${IPOPT_ROOT}/include/coin-or/IpIpoptApplication.hpp")
        message(FATAL_ERROR
            "Ipopt headers not found below ${IPOPT_ROOT}/include"
        )
    endif()

    if(NOT EXISTS "${IPOPT_ROOT}/lib/libipopt.so")
        message(FATAL_ERROR
            "libipopt.so not found below ${IPOPT_ROOT}/lib"
        )
    endif()

    add_library(Ipopt::Ipopt SHARED IMPORTED)

    set_target_properties(Ipopt::Ipopt PROPERTIES
        IMPORTED_LOCATION
            "${IPOPT_ROOT}/lib/libipopt.so"

        # CppAD uses #include <coin-or/IpIpoptApplication.hpp>
        INTERFACE_INCLUDE_DIRECTORIES
            "${IPOPT_ROOT}/include"
    )

else()
    find_package(PkgConfig REQUIRED)

    pkg_check_modules(
        IPOPT
        REQUIRED
        IMPORTED_TARGET
        ipopt
    )

    add_library(Ipopt::Ipopt ALIAS PkgConfig::IPOPT)
endif()

# -----------------------------------------------------------------------------
# Eigen
# -----------------------------------------------------------------------------

add_subdirectory(
    src/eigen-5.0.0
    EXCLUDE_FROM_ALL
)

# -----------------------------------------------------------------------------
# Planning
# -----------------------------------------------------------------------------

add_library(planning
    src/planning.cpp
    src/longitudinal_planning.cpp
    src/lateral_planning.cpp
)


target_include_directories(planning
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

#independently declare CppAD include directories as SYSTEM to avoid warnings from CppAD headers
target_include_directories(planning
    SYSTEM PRIVATE
        ${CPPAD_ROOT}/include
)

target_compile_definitions(planning
    PRIVATE
        HAVE_CSTDDEF
)

target_link_libraries(planning
    PUBLIC
        common
        Eigen3::Eigen

    PRIVATE
        Ipopt::Ipopt
)

if(CMAKE_CROSSCOMPILING)
    target_link_options(planning
        PRIVATE
            "-Wl,-rpath-link,${IPOPT_ROOT}/lib"
    )
endif()