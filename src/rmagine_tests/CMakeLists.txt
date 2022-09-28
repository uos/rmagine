

add_subdirectory(optix_sim)

# add_executable(test1 tests/Test1.cpp)

# set(UNIT_TEST test1)
# add_test(NAME ${UNIT_TEST} COMMAND ${UNIT_TEST})
# add_custom_command(
#     TARGET ${UNIT_TEST}
#     COMMENT "Run tests"
#     POST_BUILD 
#     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
#     COMMAND ${CMAKE_CTEST_COMMAND} -C $<CONFIGURATION> -R "^${UNIT_TEST}$" --output-on-failures
# )