
message(STATUS "Building tests for Rmagine core")

# ADD TESTS FOR CORE HERE
add_subdirectory(core)



if(embree_FOUND)
message(STATUS "Building tests for Embree backend")

# ADD TESTS FOR EMBREE HERE
add_subdirectory(embree)

endif(embree_FOUND)



if(CUDA_FOUND)
message(STATUS "Building tests for Cuda backend")

# ADD TESTS FOR CUDA HERE
add_subdirectory(cuda)

endif(CUDA_FOUND)



if(OptiX_FOUND)
message(STATUS "Building tests for OptiX backend")

# ADD TESTS FOR OPTIX HERE
add_subdirectory(optix)

endif(OptiX_FOUND)



if(Vulkan_FOUND)
message(STATUS "Building tests for Vulkan backend")

# ADD TESTS FOR VULKAN HERE
add_subdirectory(vulkan)

endif(Vulkan_FOUND)



if(Vulkan_FOUND AND CUDA_FOUND)
message(STATUS "Building tests for Vulkan Cuda interop")

# ADD TESTS FOR VULKAN HERE
add_subdirectory(vulkan_cuda_interop)

endif(Vulkan_FOUND AND CUDA_FOUND)
