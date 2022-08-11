
add_executable(rmagine_synthetic Main.cpp)

target_link_libraries(rmagine_synthetic
    rmagine
)

install(TARGETS rmagine_synthetic
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})