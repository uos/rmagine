if(TARGET rmagine::optix)

add_executable(rmagine_measurements_optix
    Main.cpp
)

target_link_libraries(rmagine_measurements_optix
    rmagine::core
    rmagine::cuda
    rmagine::optix
)

##### INSTALL
install(TARGETS rmagine_measurements_optix
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT optix
)

endif(TARGET rmagine::optix)
