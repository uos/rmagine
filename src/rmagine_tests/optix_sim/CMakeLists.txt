

add_executable(rmagine_tests_optix_sim Main.cpp)
target_link_libraries(rmagine_tests_optix_sim
    rmagine_optix
)

message(STATUS "ADD TEST!")
set(UNIT_TEST rmagine_tests_optix_sim)
add_test(NAME ${UNIT_TEST} COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${UNIT_TEST})
add_custom_command(
    TARGET ${UNIT_TEST}
    COMMENT "Run tests"
    POST_BUILD 
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMAND ${CMAKE_CTEST_COMMAND} -C $<CONFIGURATION> -R "^${UNIT_TEST}$" --output-on-failures
)