add_executable(rmagine_tests_memory_slicing_cuda Main.cpp)

target_link_libraries(rmagine_tests_memory_slicing_cuda
    rmagine
    rmagine_cuda
)