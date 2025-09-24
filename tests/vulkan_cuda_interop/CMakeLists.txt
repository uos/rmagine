
# 1. INTEROP
add_executable(rmagine_tests_vulkan_cuda_interop_interop interop.cpp)
target_link_libraries(rmagine_tests_vulkan_cuda_interop_interop
    rmagine::vulkan
    rmagine::cuda
    rmagine::vulkanCudaInterop
)

add_test(NAME vulkan_cuda_interop_tests COMMAND rmagine_tests_vulkan_cuda_interop_interop)
