

if(embree_FOUND OR OptiX_FOUND)

add_executable(rmagine_benchmark 
    Main.cpp
)

target_link_libraries(rmagine_benchmark
    rmagine_static
)

if(embree_FOUND)
    target_link_libraries(rmagine_benchmark
        rmagine_embree_static
    )
    target_compile_definitions(rmagine_benchmark PUBLIC WITH_EMBREE)
endif(embree_FOUND)

if(CUDA_FOUND AND OptiX_FOUND)
    target_link_libraries(rmagine_benchmark
        rmagine_optix_static
    )
    target_compile_definitions(rmagine_benchmark PUBLIC WITH_OPTIX)
endif(CUDA_FOUND AND OptiX_FOUND)


##### INSTALL

install(TARGETS rmagine_benchmark
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

endif(embree_FOUND OR OptiX_FOUND)

