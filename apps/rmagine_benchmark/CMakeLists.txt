


if(embree_FOUND)

add_executable(rmagine_benchmark_cpu
    Main.cpp
)

target_link_libraries(rmagine_benchmark_cpu
    rmagine::core
    rmagine::embree
)

target_compile_definitions(rmagine_benchmark_cpu PUBLIC WITH_EMBREE)

##### INSTALL
install(TARGETS rmagine_benchmark_cpu
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT embree
)

endif(embree_FOUND)

if(OptiX_FOUND)

add_executable(rmagine_benchmark_gpu 
    Main.cpp
)

target_link_libraries(rmagine_benchmark_gpu
    rmagine::core
    rmagine::optix
)

target_compile_definitions(rmagine_benchmark_gpu PUBLIC WITH_OPTIX)

##### INSTALL
install(TARGETS rmagine_benchmark_gpu
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT optix
)

endif(OptiX_FOUND)
