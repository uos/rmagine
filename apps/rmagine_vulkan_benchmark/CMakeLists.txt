if(Vulkan_FOUND)

add_executable(rmagine_vulkan_benchmark
    Main.cpp
)

target_link_libraries(rmagine_vulkan_benchmark
    rmagine::core
    rmagine::vulkan
)

# target_compile_definitions(rmagine_vulkan_benchmark PUBLIC VDEBUG)

##### INSTALL
install(TARGETS rmagine_vulkan_benchmark
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT vulkan
)

endif(Vulkan_FOUND)