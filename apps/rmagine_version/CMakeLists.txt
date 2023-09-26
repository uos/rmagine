
add_executable(rmagine_version Main.cpp)

target_link_libraries(rmagine_version
    rmagine
)

install(TARGETS rmagine_version
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT core
)