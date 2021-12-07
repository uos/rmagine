
add_executable(imagine_evaluation Main.cpp)

target_link_libraries(imagine_evaluation
    imagine
)

if(CUDA_FOUND AND OptiX_FOUND)
    target_link_libraries(imagine_evaluation
        imagine_optix
    )
    target_compile_definitions(imagine_evaluation PUBLIC WITH_OPTIX)
endif(CUDA_FOUND AND OptiX_FOUND)