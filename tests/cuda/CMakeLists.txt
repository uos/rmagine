

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

# 7. Reduction correctness (asserting: rm::sum/mean/cov vs CPU ref + determinism)
add_executable(rmagine_tests_cuda_math_reduction_correctness math_reduction_correctness.cpp)
target_link_libraries(rmagine_tests_cuda_math_reduction_correctness
    rmagine::cuda
)

add_test(NAME cuda_math_reduction_correctness COMMAND rmagine_tests_cuda_math_reduction_correctness)

# 8. Public header hygiene (plain C++ TU, headers first, no GPU compiler)
add_executable(rmagine_tests_cuda_public_headers public_headers.cpp)
target_link_libraries(rmagine_tests_cuda_public_headers
    rmagine::cuda
)

add_test(NAME cuda_public_headers COMMAND rmagine_tests_cuda_public_headers)