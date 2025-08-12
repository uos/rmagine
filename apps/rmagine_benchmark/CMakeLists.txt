


if(TARGET rmagine::embree)

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

endif(TARGET rmagine::embree)

if(TARGET rmagine::optix)

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

endif(TARGET rmagine::optix)
