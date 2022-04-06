
find_package(Eigen3 REQUIRED)

if(Eigen3_FOUND)
include_directories(${EIGEN3_INCLUDE_DIR})
endif(Eigen3_FOUND)


add_executable(rmagine_tests_svd Main.cpp)
target_link_libraries(rmagine_tests_svd 
    rmagine_cuda 
    Eigen3::Eigen)