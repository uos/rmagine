
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
