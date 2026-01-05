####################################
## Package: rmagine               ##
## Component: vulkan_cuda_interop ##
####################################

##################################
## Required external libraries: ##
## - Vulkan                     ##
## - CUDA                       ##
##################################

message(STATUS "Checking dependencies of rmagine::vulkan-cuda-interop ...")

find_package(Vulkan QUIET)

find_program(GLSLANG_BIN glslang)
if(GLSLANG_BIN)
  find_package(glslang QUIET)
else()
  message(WARNING "glslang binary missing; not importing glslang CMake package. Have you installed glslang-tools?")
endif()

# default flags are not set when including cuda
if (NOT EXISTS ${CMAKE_BINARY_DIR}/CMakeCache.txt)

  if(NOT CUDA_NVCC_FLAGS_DEBUG)
    set(CUDA_NVCC_FLAGS_DEBUG "-g" CACHE STRING "" FORCE)
  endif()

  if(NOT CUDA_NVCC_FLAGS_MINSIZEREL)
    set(CUDA_NVCC_FLAGS_MINSIZEREL "-Os -DNDEBUG" CACHE STRING "" FORCE)
  endif()

  if(NOT CUDA_NVCC_FLAGS_RELEASE)
    set(CUDA_NVCC_FLAGS_RELEASE "-O3 -DNDEBUG" CACHE STRING "" FORCE)
  endif()

  if(NOT CUDA_NVCC_FLAGS_RELWITHDEBINFO)
    set(CUDA_NVCC_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG" CACHE STRING "" FORCE)
  endif()

endif()

# set(CUDA_STANDARD 14)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)

    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        enable_language(CUDA)
        set(CUDA_FOUND True)
        set(CUDA_LIBRARIES CUDA::cudart)
        set(CUDA_cusolver_LIBRARY CUDA::cusolver)
        set(CUDA_cublas_LIBRARY CUDA::cublas)
        set(CUDA_DRIVER_LIBRARY CUDA::cuda_driver)
        set(CUDA_INCLUDE_DIRS "") # is in target instead
        set(CUDA_VERSION ${CUDAToolkit_VERSION})
    else()
        find_package(CUDA)
        if(CUDA_FOUND)
            enable_language(CUDA)
            set(CUDA_DRIVER_LIBRARY cuda)
        else()
            message(STATUS "Neither CudaToolkit nor CUDA found!")
        endif(CUDA_FOUND)
    endif(CUDAToolkit_FOUND)
endif(CMAKE_CUDA_COMPILER)


if(Vulkan_FOUND AND CUDA_FOUND AND TARGET rmagine::vulkan AND TARGET rmagine::cuda)

message(STATUS "${Green}rmagine::vulkan-cuda-interop${ColourReset}: Building")

set(RMAGINE_VULKAN_CUDA_INTEROP_SRCS
    #Types
    src/types/VulkanCudaInterop.cpp
    src/types/MemoryVulkanCudaInterop.cpp
)

## SHARED ##
add_library(rmagine-vulkan-cuda-interop SHARED
    ${RMAGINE_VULKAN_CUDA_INTEROP_SRCS}
)

target_include_directories(rmagine-vulkan-cuda-interop
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/rmagine-${rmagine_VERSION}>
        # ${Vulkan_INCLUDE_DIRS}
)

target_link_libraries(rmagine-vulkan-cuda-interop
    rmagine-core
    rmagine-vulkan
    rmagine-cuda
)

add_dependencies(rmagine-vulkan-cuda-interop
    rmagine-core
    rmagine-vulkan
    rmagine-cuda
)

# target_compile_definitions(rmagine-vulkan-cuda-interop PUBLIC VDEBUG)

# target_compile_features(rmagine-vulkan-cuda-interop PUBLIC cxx_std_17)

set_target_properties(rmagine-vulkan-cuda-interop
    PROPERTIES
        EXPORT_NAME vulkan-cuda-interop
        SOVERSION ${rmagine_VERSION_MAJOR}
        VERSION ${rmagine_VERSION}
        # CXX_STANDARD 17
)

# set CUDA_ARCHITECTURES with CMake >= 3.23 when CMAKE_* is not set
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.23" AND NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set_target_properties(rmagine-optix PROPERTIES CUDA_ARCHITECTURES all)
endif()

add_library(rmagine::vulkan-cuda-interop ALIAS rmagine-vulkan-cuda-interop)

list(APPEND RMAGINE_LIBRARIES rmagine-vulkan-cuda-interop)
set(RMAGINE_LIBRARIES ${RMAGINE_LIBRARIES} PARENT_SCOPE)

###########
## INSTALL
############

include(CMakePackageConfigHelpers)

install(TARGETS rmagine-vulkan-cuda-interop EXPORT rmagine-vulkan-cuda-interop-targets
    COMPONENT vulkan-cuda-interop
)

install(EXPORT rmagine-vulkan-cuda-interop-targets
    FILE rmagine-vulkan-cuda-interop-targets.cmake
    COMPONENT vulkan-cuda-interop
    NAMESPACE rmagine::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rmagine-${rmagine_VERSION}
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/rmagine-vulkan-cuda-interop-config-version.cmake
    VERSION ${rmagine_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(cmake/rmagine-vulkan-cuda-interop-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/rmagine-vulkan-cuda-interop-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rmagine-${rmagine_VERSION}
)

# CMAKE FIND PACKAGE FILES
install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/rmagine-vulkan-cuda-interop-config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/rmagine-vulkan-cuda-interop-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rmagine-${rmagine_VERSION}
    COMPONENT vulkan-cuda-interop
)

# HEADERS
install(
    DIRECTORY include/rmagine
    COMPONENT vulkan-cuda-interop
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rmagine-${rmagine_VERSION}
)

set(CPACK_DEBIAN_VULKAN_CUDA_INTEROP_PACKAGE_DEPENDS "rmagine-core" PARENT_SCOPE)
# set(CPACK_DEBIAN_VULKAN_CUDA_INTEROP_PACKAGE_DEPENDS "rmagine-vulkan" PARENT_SCOPE)
# set(CPACK_DEBIAN_VULKAN_CUDA_INTEROP_PACKAGE_DEPENDS "rmagine-cuda" PARENT_SCOPE)
# TODO: was davon wäre richtig?

list(APPEND CPACK_COMPONENTS_ALL vulkan-cuda-interop)
set(CPACK_COMPONENTS_ALL ${CPACK_COMPONENTS_ALL} PARENT_SCOPE)

else(Vulkan_FOUND AND CUDA_FOUND AND TARGET rmagine::vulkan AND TARGET rmagine::cuda)

message(STATUS "${Yellow}rmagine::vulkan-cuda-interop${ColourReset}: Skipping. Missing Dependencies.")

endif(Vulkan_FOUND AND CUDA_FOUND AND TARGET rmagine::vulkan AND TARGET rmagine::cuda)
