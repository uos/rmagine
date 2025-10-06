if(TARGET rmagine::vulkan AND TARGET rmagine::cuda AND TARGET rmagine::optix AND TARGET rmagine::embree)

add_executable(rmagine_simulation_diff_calc
    Main.cpp
)

target_link_libraries(rmagine_simulation_diff_calc
    rmagine::core
    rmagine::vulkan
    rmagine::cuda
    rmagine::optix
    rmagine::embree
)

# target_compile_definitions(rmagine_simulation_diff_calc PUBLIC VDEBUG)

##### INSTALL
install(TARGETS rmagine_simulation_diff_calc
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT vulkan
)

endif(TARGET rmagine::vulkan AND TARGET rmagine::cuda AND TARGET rmagine::optix AND TARGET rmagine::embree)