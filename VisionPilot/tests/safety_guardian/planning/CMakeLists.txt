find_package(OpenCV REQUIRED)
find_package(GTest REQUIRED)

add_executable(test_longitudinal_planning
    test_longitudinal_planning.cpp
)

target_link_libraries(test_longitudinal_planning
    PRIVATE
        planning
        GTest::gtest_main
)

include(GoogleTest)

gtest_discover_tests(test_longitudinal_planning)