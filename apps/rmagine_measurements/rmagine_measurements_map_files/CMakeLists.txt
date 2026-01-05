if(TARGET rmagine::core)

add_executable(rmagine_measurements_map_files
    Main.cpp
)

target_link_libraries(rmagine_measurements_map_files
    rmagine::core
)

##### INSTALL
install(TARGETS rmagine_measurements_map_files
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT embree
)

endif(TARGET rmagine::core)
