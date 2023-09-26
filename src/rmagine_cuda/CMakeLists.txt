message(STATUS "Building Cuda dependend code. Library: rmagine-cuda")

set(RMAGINE_CUDA_SRCS
    src/types/MemoryCuda.cu
    # math
    src/math/math.cu
    src/math/math_batched.cu
    src/math/SVDCuda.cpp
    # util
    src/util/cuda/CudaContext.cpp
    src/util/cuda/CudaDebug.cpp
    src/util/cuda/CudaStream.cpp
    src/util/cuda/random.cu
    # noise
    src/noise/NoiseCuda.cu
    src/noise/GaussianNoiseCuda.cu
    src/noise/RelGaussianNoiseCuda.cu
    src/noise/UniformDustNoiseCuda.cu
    # map
    src/map/mesh_preprocessing.cu
)

add_library(rmagine-cuda SHARED
    ${RMAGINE_CUDA_SRCS}
)

target_include_directories(rmagine-cuda
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# TODO: use CUDA language instead of find_package
target_link_libraries(rmagine-cuda
    rmagine-core
    ${CUDA_LIBRARIES}
    ${CUDA_cusolver_LIBRARY}
    ${CUDA_cublas_LIBRARY}
    cuda
)

add_dependencies(rmagine-cuda
    rmagine-core
)

set_target_properties(rmagine-cuda
    PROPERTIES
    EXPORT_NAME cuda
    SOVERSION ${rmagine_VERSION_MAJOR}
    VERSION ${rmagine_VERSION}
)

add_library(rmagine::cuda ALIAS rmagine-cuda)

list(APPEND RMAGINE_LIBRARIES rmagine-cuda)
set(RMAGINE_LIBRARIES ${RMAGINE_LIBRARIES} PARENT_SCOPE)



add_library(rmagine-cuda-static STATIC
    ${RMAGINE_CUDA_SRCS}
)

target_include_directories(rmagine-cuda-static
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)


# TODO: use CUDA language instead of find_package
target_link_libraries(rmagine-cuda-static
    rmagine-core-static
    ${CUDA_LIBRARIES}
    ${CUDA_cusolver_LIBRARY}
    ${CUDA_cublas_LIBRARY}
    cuda
)


add_dependencies(rmagine-cuda-static
    rmagine-core-static
)

set_target_properties(rmagine-cuda-static
    PROPERTIES
    EXPORT_NAME cuda-static
    SOVERSION ${rmagine_VERSION_MAJOR}
    VERSION ${rmagine_VERSION}
)

add_library(rmagine::cuda-static ALIAS rmagine-cuda-static)

list(APPEND RMAGINE_STATIC_LIBRARIES rmagine-cuda-static)
set(RMAGINE_STATIC_LIBRARIES ${RMAGINE_STATIC_LIBRARIES} PARENT_SCOPE)

###########
## CUDA
############
install(TARGETS rmagine-cuda
    EXPORT rmagine-cuda-targets
    COMPONENT cuda
)

install(EXPORT rmagine-cuda-targets
    FILE rmagine-cuda-targets.cmake
    COMPONENT cuda
    NAMESPACE rmagine::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rmagine
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/rmagine-cuda-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(cmake/rmagine-cuda-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/rmagine-cuda-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rmagine
)

install(
    FILES
    ${CMAKE_CURRENT_BINARY_DIR}/rmagine-cuda-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/rmagine-cuda-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rmagine
    COMPONENT cuda
)

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/rmagine
  COMPONENT cuda
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(TARGETS rmagine-cuda-static 
    EXPORT rmagine-cuda-static-targets
    COMPONENT cuda-static
)

install(EXPORT rmagine-cuda-static-targets
    FILE rmagine-cuda-static-targets.cmake
    COMPONENT cuda-static
    NAMESPACE rmagine::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rmagine
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/rmagine-cuda-static-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(cmake/rmagine-cuda-static-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/rmagine-cuda-static-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rmagine
)

install(
    FILES
    ${CMAKE_CURRENT_BINARY_DIR}/rmagine-cuda-static-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/rmagine-cuda-static-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rmagine
    COMPONENT cuda-static
)

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/rmagine
  COMPONENT cuda-static
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

set(CPACK_DEBIAN_CUDA_PACKAGE_DEPENDS "rmagine-core" PARENT_SCOPE)
set(CPACK_DEBIAN_CUDA-STATIC_PACKAGE_DEPENDS "rmagine-core-static" PARENT_SCOPE)

list(APPEND CPACK_COMPONENTS_ALL cuda)
list(APPEND CPACK_COMPONENTS_ALL cuda-static)
set(CPACK_COMPONENTS_ALL ${CPACK_COMPONENTS_ALL} PARENT_SCOPE)