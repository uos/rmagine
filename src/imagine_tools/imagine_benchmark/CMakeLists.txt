
add_executable(imagine_benchmark Main.cpp)

target_link_libraries(imagine_benchmark
    imagine
)

if(CUDA_FOUND AND OptiX_FOUND)
    target_link_libraries(imagine_benchmark
        imagine_optix
    )
    target_compile_definitions(imagine_benchmark PUBLIC WITH_OPTIX)
endif(CUDA_FOUND AND OptiX_FOUND)