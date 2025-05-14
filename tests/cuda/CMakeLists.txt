

# 1. MATH
add_executable(rmagine_tests_cuda_math math.cpp)
target_link_libraries(rmagine_tests_cuda_math
    rmagine::cuda
)

add_test(NAME cuda_math COMMAND rmagine_tests_cuda_math)


# 2. MEMORY
add_executable(rmagine_tests_cuda_memory memory.cpp)
target_link_libraries(rmagine_tests_cuda_memory
    rmagine::cuda
)

add_test(NAME cuda_memory COMMAND rmagine_tests_cuda_memory)


# 3. MEMORY SLICING
add_executable(rmagine_tests_cuda_memory_slicing memory_slicing.cpp)
target_link_libraries(rmagine_tests_cuda_memory_slicing
    rmagine::cuda
)

add_test(NAME cuda_memory_slicing COMMAND rmagine_tests_cuda_memory_slicing)


# 4. SVD
add_executable(rmagine_tests_cuda_math_svd math_svd.cpp)
target_link_libraries(rmagine_tests_cuda_math_svd
    rmagine::cuda
)

add_test(NAME cuda_math_svd COMMAND rmagine_tests_cuda_math_svd)

# 5. Statistics
add_executable(rmagine_tests_cuda_math_statistics math_statistics.cpp)
target_link_libraries(rmagine_tests_cuda_math_statistics
    rmagine::cuda
)

add_test(NAME cuda_math_statistics COMMAND rmagine_tests_cuda_math_statistics)

# 6. Reduction
add_executable(rmagine_tests_cuda_math_reduction math_reduction.cpp)
target_link_libraries(rmagine_tests_cuda_math_reduction
    rmagine::cuda
)

add_test(NAME cuda_math_reduction COMMAND rmagine_tests_cuda_math_reduction)

# 7. Shared Function
add_executable(rmagine_tests_cuda_shared_functions shared_functions.cu)
target_link_libraries(rmagine_tests_cuda_shared_functions
    # rmagine::core
    rmagine::cuda
)

add_test(NAME cuda_shared_functions COMMAND rmagine_tests_cuda_shared_functions)