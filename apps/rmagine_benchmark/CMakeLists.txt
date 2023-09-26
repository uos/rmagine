


if(embree_FOUND)

add_executable(rmagine_benchmark_cpu
    Main.cpp
)

target_link_libraries(rmagine_benchmark_cpu
    rmagine::core-static
    rmagine::embree-static
)

target_compile_definitions(rmagine_benchmark_cpu PUBLIC WITH_EMBREE)

##### INSTALL
install(TARGETS rmagine_benchmark_cpu
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT embree
)

install(TARGETS rmagine_benchmark_cpu
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT embree-static
)

endif(embree_FOUND)

if(OptiX_FOUND)

add_executable(rmagine_benchmark_gpu 
    Main.cpp
)

target_link_libraries(rmagine_benchmark_gpu
    rmagine::core-static
    rmagine::optix-static
)

target_compile_definitions(rmagine_benchmark_gpu PUBLIC WITH_OPTIX)

##### INSTALL
install(TARGETS rmagine_benchmark_gpu
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT optix
)

install(TARGETS rmagine_benchmark_gpu
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT optix-static
)

endif(OptiX_FOUND)
