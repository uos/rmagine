
add_executable(rmagine_benchmark Main.cpp)

target_link_libraries(rmagine_benchmark
    rmagine
)

if(CUDA_FOUND AND OptiX_FOUND)
    target_link_libraries(rmagine_benchmark
        rmagine_optix
    )
    target_compile_definitions(rmagine_benchmark PUBLIC WITH_OPTIX)
endif(CUDA_FOUND AND OptiX_FOUND)