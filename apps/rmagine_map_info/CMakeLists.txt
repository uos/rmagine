
add_executable(rmagine_map_info Main.cpp)

target_link_libraries(rmagine_map_info
    rmagine::core
)

install(TARGETS rmagine_map_info
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})