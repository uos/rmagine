
add_executable(rmagine_info Main.cpp)

target_link_libraries(rmagine_info
    rmagine
)

install(TARGETS rmagine_info
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT core
)