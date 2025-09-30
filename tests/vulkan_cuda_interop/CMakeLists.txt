
# 1. INTEROP
add_executable(rmagine_tests_vulkan_cuda_interop_interop interop.cpp)
target_link_libraries(rmagine_tests_vulkan_cuda_interop_interop
    rmagine::vulkan
    rmagine::cuda
    rmagine::vulkanCudaInterop
)

add_test(NAME vulkan_cuda_interop_tests COMMAND rmagine_tests_vulkan_cuda_interop_interop)


# 2. VULKAN_AS_CUDA
add_executable(rmagine_tests_vulkan_cuda_interop_vulkan_as_cuda vulkan_as_cuda.cpp)
target_link_libraries(rmagine_tests_vulkan_cuda_interop_vulkan_as_cuda
    rmagine::vulkan
    rmagine::cuda
    rmagine::vulkanCudaInterop
)

add_test(NAME vulkan_as_cuda_tests COMMAND rmagine_tests_vulkan_cuda_interop_vulkan_as_cuda)
