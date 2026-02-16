
message(STATUS "Building tests for rmagine::core")
add_subdirectory(core)

if(TARGET rmagine-embree)
message(STATUS "Building tests for rmagine::embree")
add_subdirectory(embree)
endif(TARGET rmagine-embree)

if(TARGET rmagine-cuda)
message(STATUS "Building tests for rmagine::cuda")
add_subdirectory(cuda)
endif(TARGET rmagine-cuda)

if(TARGET rmagine-optix)
message(STATUS "Building tests for rmagine::optix")
add_subdirectory(optix)
endif(TARGET rmagine-optix)

# if(TARGET rmagine-vulkan)
# message(STATUS "Building tests for rmagine::vulkan")
# add_subdirectory(vulkan)
# endif(TARGET rmagine-vulkan)

if(TARGET rmagine-vulkan-cuda-interop)
message(STATUS "Building tests for rmagine::vulkan-cuda-interop")
add_subdirectory(vulkan_cuda_interop)
endif(TARGET rmagine-vulkan-cuda-interop)
