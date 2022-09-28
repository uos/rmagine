

# 1. MATH
add_executable(rmagine_tests_core_math math.cpp)
target_link_libraries(rmagine_tests_core_math
    rmagine
)

add_test(NAME core_math COMMAND rmagine_tests_core_math)


# 2. MEMORY
add_executable(rmagine_tests_core_memory memory.cpp)
target_link_libraries(rmagine_tests_core_memory
    rmagine
)

add_test(NAME core_memory COMMAND rmagine_tests_core_memory)


# 3. MEMORY SLICING
add_executable(rmagine_tests_core_memory_slicing memory_slicing.cpp)
target_link_libraries(rmagine_tests_core_memory_slicing
    rmagine
)

add_test(NAME core_memory_slicing COMMAND rmagine_tests_core_memory_slicing)

