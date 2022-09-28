
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

add_executable(rmagine_tests_svd Main.cpp)
target_link_libraries(rmagine_tests_svd 
    rmagine_cuda 
    Eigen3::Eigen)