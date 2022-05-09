

include_directories(include)

# Build normal library
add_library(my_lib
    src/my_lib.cpp)
    
target_link_libraries(my_lib
    rmagine
)

# Build CUDA Library
add_library(my_lib_cuda 
    src/my_lib.cu)

target_link_libraries(my_lib_cuda
    rmagine_cuda
)

# Use own cuda library
add_executable(rmagine_tests_memory_cuda Main.cpp)

target_link_libraries(rmagine_tests_memory_cuda
    my_lib_cuda
    my_lib
)
