

# 1. MATH
add_executable(rmagine_tests_cuda_math math.cpp)
target_link_libraries(rmagine_tests_cuda_math
    rmagine_cuda
)

add_test(NAME cuda_math COMMAND rmagine_tests_cuda_math)


# 2. MEMORY
add_executable(rmagine_tests_cuda_memory memory.cpp)
target_link_libraries(rmagine_tests_cuda_memory
    rmagine_cuda
)

add_test(NAME cuda_memory COMMAND rmagine_tests_cuda_memory)


# 3. MEMORY SLICING
add_executable(rmagine_tests_cuda_memory_slicing memory_slicing.cpp)
target_link_libraries(rmagine_tests_cuda_memory_slicing
    rmagine_cuda
)

add_test(NAME cuda_memory_slicing COMMAND rmagine_tests_cuda_memory_slicing)

