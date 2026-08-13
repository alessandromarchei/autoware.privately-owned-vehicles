# app/CMakeLists.txt

function(configure_vision_pilot_app target_name)

    if(NOT TARGET ${target_name})
        message(FATAL_ERROR
            "Target '${target_name}' does not exist. "
            "Call configure_vision_pilot_app() after rcar_configure_application()."
        )
    endif()

    add_dependencies(${target_name}
        generate_config
    )

    # Keep the warning visible, but do not promote it to an error.
    target_compile_options(${target_name}
        PRIVATE
            -Wno-error=unused-function
    )

    # Ipopt runtime:
    # installed executable: <prefix>/bin/vision_pilot
    # installed libraries:  <prefix>/lib/*.so
    set_target_properties(${target_name} PROPERTIES
        INSTALL_RPATH "$ORIGIN/../lib"
    )

    # Suppress unresolved symbols from transitive shared libraries.
    # Keep this only if it is actually necessary for the V4M link.
    target_link_options(${target_name}
        PRIVATE
            -Wl,--allow-shlib-undefined
            "-Wl,-rpath-link,${IPOPT_ROOT}/lib"
    )

    install(
        TARGETS ${target_name}
        RUNTIME DESTINATION bin
    )

    set(VISIONPILOT_CONF_FILES
        "${CMAKE_SOURCE_DIR}/config/H.yaml"
        "${CMAKE_BINARY_DIR}/config/homography_C_matrix.yaml"
        "${CMAKE_SOURCE_DIR}/config/vision_pilot.conf"
        "${CMAKE_SOURCE_DIR}/config/vision_pilot_test.conf"
    )

    # Install the complete Ipopt runtime bundle:
    # Ipopt, MUMPS, SPRAL, METIS, OpenBLAS and Fortran runtime.
    install(
        DIRECTORY "${IPOPT_ROOT}/lib/"
        DESTINATION lib
    )
    
    install(
        FILES ${VISIONPILOT_CONF_FILES}
        DESTINATION share/visionpilot/config
    )

    install(
        DIRECTORY "${CMAKE_SOURCE_DIR}/modules/models/weights/"
        DESTINATION share/visionpilot/modules/models/weights
        #exlude .onnx files, since they are not used in the app
        PATTERN "*.onnx" EXCLUDE
    )

endfunction()