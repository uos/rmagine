
# Build CUDA Library
include_directories(include)
add_library(my_lib 
    src/my_lib.cu)

target_link_libraries(my_lib
    ${CUDA_LIBRARIES}
)


# Use own cuda library
add_executable(rmagine_tests_memory_cuda Main.cpp)

target_link_libraries(rmagine_tests_memory_cuda
    rmagine
    rmagine_cuda
    my_lib
)
