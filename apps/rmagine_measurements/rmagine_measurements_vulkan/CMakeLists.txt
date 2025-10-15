if(TARGET rmagine::vulkan)

add_executable(rmagine_measurements_vulkan
    Main.cpp
)

target_link_libraries(rmagine_measurements_vulkan
    rmagine::core
    rmagine::vulkan
)

##### INSTALL
install(TARGETS rmagine_measurements_vulkan
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT vulkan
)

endif(TARGET rmagine::vulkan)