

# 1. MATH
add_executable(rmagine_tests_core_math math.cpp)
target_link_libraries(rmagine_tests_core_math
    rmagine::core
)

add_test(NAME core_math COMMAND rmagine_tests_core_math)


# 2. MEMORY
add_executable(rmagine_tests_core_memory memory.cpp)
target_link_libraries(rmagine_tests_core_memory
    rmagine::core
)

add_test(NAME core_memory COMMAND rmagine_tests_core_memory)


# 3. MEMORY SLICING
add_executable(rmagine_tests_core_memory_slicing memory_slicing.cpp)
target_link_libraries(rmagine_tests_core_memory_slicing
    rmagine::core
)

add_test(NAME core_memory_slicing COMMAND rmagine_tests_core_memory_slicing)


# 3. MEMORY SLICING
add_executable(rmagine_tests_core_quaternion 
    quaternion.cpp)
target_link_libraries(rmagine_tests_core_quaternion
    rmagine::core
)

add_test(NAME core_quaternion COMMAND rmagine_tests_core_quaternion)

# 4. SVD
add_executable(rmagine_tests_core_math_svd math_svd.cpp)
target_link_libraries(rmagine_tests_core_math_svd
    rmagine::core
)

add_test(NAME core_math_svd COMMAND rmagine_tests_core_math_svd)
