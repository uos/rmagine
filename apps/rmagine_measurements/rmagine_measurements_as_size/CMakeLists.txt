if(TARGET rmagine::vulkan AND TARGET rmagine::cuda AND TARGET rmagine::optix)

add_executable(rmagine_measurements_as_size
    Main.cpp
)

target_link_libraries(rmagine_measurements_as_size
    rmagine::core
    rmagine::cuda
    rmagine::optix
    rmagine::vulkan
)

##### INSTALL
install(TARGETS rmagine_measurements_as_size
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT vulkan
)

endif(TARGET rmagine::vulkan AND TARGET rmagine::cuda AND TARGET rmagine::optix)
