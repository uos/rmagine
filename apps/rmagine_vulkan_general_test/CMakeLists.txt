if(Vulkan_FOUND)

add_executable(rmagine_vulkan_general_test
    Main.cpp
)

target_link_libraries(rmagine_vulkan_general_test
    rmagine::core
    rmagine::vulkan
)

# target_compile_definitions(rmagine_vulkan_general_test PUBLIC VDEBUG)

##### INSTALL
install(TARGETS rmagine_vulkan_general_test
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT vulkan
)

endif(Vulkan_FOUND)