


if(TARGET rmagine::embree)

add_executable(rmagine_benchmark_embree
    benchmark_embree.cpp
)

target_link_libraries(rmagine_benchmark_embree
    rmagine::core
    rmagine::embree
)

##### INSTALL
install(TARGETS rmagine_benchmark_embree
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT embree
)

endif(TARGET rmagine::embree)

if(TARGET rmagine::optix)

add_executable(rmagine_benchmark_optix
    benchmark_optix.cpp
)

target_link_libraries(rmagine_benchmark_optix
    rmagine::core
    rmagine::optix
)

##### INSTALL
install(TARGETS rmagine_benchmark_optix
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT optix
)

endif(TARGET rmagine::optix)
