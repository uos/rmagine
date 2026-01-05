if(TARGET rmagine::embree)

add_executable(rmagine_measurements_embree
    Main.cpp
)

target_link_libraries(rmagine_measurements_embree
    rmagine::core
    rmagine::embree
)

##### INSTALL
install(TARGETS rmagine_measurements_embree
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT embree
)

endif(TARGET rmagine::embree)
