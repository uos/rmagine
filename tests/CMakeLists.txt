


# ADD TESTS FOR CORE HERE
message(STATUS "Building tests for Rmagine core")


add_subdirectory(core)

if(embree_FOUND)
message(STATUS "Building tests for Embree backend")

# ADD TESTS FOR EMBREE HERE
add_subdirectory(embree_sim)

endif(embree_FOUND)

if(CUDA_FOUND)
add_subdirectory(cuda)
endif(CUDA_FOUND)



if(OptiX_FOUND)
message(STATUS "Building tests for OptiX backend")

# ADD TESTS FOR OPTIX HERE

add_subdirectory(optix_sim)

endif(OptiX_FOUND)

