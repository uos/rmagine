add_executable(rmagine_tests_memory Main.cpp)

target_link_libraries(rmagine_tests_memory
    rmagine
    rmagine_cuda
)