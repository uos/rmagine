# rmagine_hiprt -- HIPRT ray-tracing backend (AMD ROCm)
# Implements Pinhole, Spherical, O1Dn, and OnDn sensor simulators

cmake_minimum_required(VERSION 3.21)

# Guard: only build on HIP
if(NOT USE_HIP)
    message(STATUS "rmagine_hiprt: USE_HIP not set, skipping")
    return()
endif()

# Find HIP runtime (the target should already be available from rmagine_cuda)
find_package(hip QUIET)

# HIPRT SDK path (vendored or external)
if(NOT DEFINED HIPRT_PATH)
    # Try environment variable first
    if(DEFINED ENV{HIPRT_PATH})
        set(HIPRT_PATH "$ENV{HIPRT_PATH}" CACHE PATH "HIPRT SDK root")
    else()
        set(HIPRT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/HIPRT" CACHE PATH "HIPRT SDK root")
    endif()
endif()

if(NOT EXISTS "${HIPRT_PATH}/hiprt/hiprt.h")
    message(WARNING "rmagine_hiprt: HIPRT SDK not found at ${HIPRT_PATH}, skipping. "
                    "Set HIPRT_PATH to the HIPRT SDK root (containing hiprt/hiprt.h).")
    return()
endif()

# Find the HIPRT library
set(HIPRT_LIB_DIR "${HIPRT_PATH}/dist/bin/Release")
find_library(HIPRT_LIBRARY
    NAMES hiprt0300164 hiprt03001
    PATHS ${HIPRT_LIB_DIR}
    NO_DEFAULT_PATH
)

if(NOT HIPRT_LIBRARY)
    message(WARNING "rmagine_hiprt: HIPRT library not found in ${HIPRT_LIB_DIR}. "
                    "Build HIPRT first: cmake --build <hiprt>/build --target hiprt03001")
    return()
endif()

message(STATUS "rmagine_hiprt: Building HIPRT backend (Pinhole, Spherical, O1Dn, OnDn sensors)")
message(STATUS "  HIPRT_PATH: ${HIPRT_PATH}")
message(STATUS "  HIPRT_LIBRARY: ${HIPRT_LIBRARY}")

set(RMAGINE_HIPRT_SRCS
    src/util/HiprtContext.cpp
    src/map/HiprtGeometry.cpp
    src/map/HiprtMesh.cpp
    src/map/HiprtScene.cpp
    src/simulation/SimulatorHiprt.cpp
    src/simulation/PinholeSimulatorHiprt.cpp
    src/simulation/SphericalSimulatorHiprt.cpp
    src/simulation/O1DnSimulatorHiprt.cpp
    src/simulation/OnDnSimulatorHiprt.cpp
)

add_library(rmagine_hiprt SHARED ${RMAGINE_HIPRT_SRCS})

target_include_directories(rmagine_hiprt
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${HIPRT_PATH}
        ${HIPRT_PATH}/hiprt
        ${HIPRT_PATH}/contrib/Orochi
)

target_link_libraries(rmagine_hiprt
    PUBLIC
        rmagine::core
        rmagine::cuda  # reuse rmagine_cuda types/memory
        hip::host
    PRIVATE
        ${HIPRT_LIBRARY}
)

target_compile_definitions(rmagine_hiprt PRIVATE USE_HIP=1)

# RPATH so the runtime can find libhiprt
set_target_properties(rmagine_hiprt PROPERTIES
    INSTALL_RPATH "${HIPRT_LIB_DIR}"
    BUILD_RPATH "${HIPRT_LIB_DIR}"
)

# The trace kernels are JIT-compiled at runtime via hiprtc
# No .cu sources needed here -- the kernel source lives in include/rmagine/simulation/hiprt/
