### FindOptiX
# Variables Set:
# Pure Optix:
# - OptiX_INCLUDE_DIR
# - OptiX_LIBRARY
# - OptiX_FOUND
# Additional:
# - OptiX_INCLUDE_DIRS
# - Optix_LIBRARIES
find_package(PkgConfig)
pkg_check_modules(OptiX QUIET optix)


# INCLUDE


find_path(OptiX_INCLUDE_DIR
  NAMES optix.h
  PATHS "${OptiX_INSTALL_DIR}/include"
  NO_DEFAULT_PATH
)

if(NOT OptiX_INCLUDE_DIR)
find_path(OptiX_INCLUDE_DIR
  NAMES optix.h
)
endif()

if(NOT OptiX_INCLUDE_DIR)
find_path(OptiX_INCLUDE_DIR
  NAMES optix.h
  PATHS "/opt/optix/include"
  NO_DEFAULT_PATH
)
endif()

if(NOT OptiX_INCLUDE_DIR)
find_path(OptiX_INCLUDE_DIR
  NAMES optix.h
  PATHS "~/optix/include"
  NO_DEFAULT_PATH
)
endif()

# LIBRARY

# 1. general search
find_library(OptiX_LIBRARY
  NAMES optix.1 optix
  )

# 2. more special search
if(NOT OptiX_LIBRARY)
find_library(OptiX_LIBRARY
  NAMES nvoptix.1 nvoptix libnvoptix.so.1 libnvoptix.so
  )
endif()
# 3. more special search
if(NOT OptiX_LIBRARY)
find_library(OptiX_LIBRARY
  NAMES libnvoptix.so.1 libnvoptix.so
  )
endif()

if(NOT OptiX_LIBRARY)
  message(WARNING "optix library not found.  Please locate before proceeding." TRUE)
endif()
if(NOT OptiX_INCLUDE_DIR)
  message(WARNING "OptiX headers (optix.h and friends) not found.  Please locate before proceeding." TRUE)
endif()

if(OptiX_LIBRARY AND OptiX_INCLUDE_DIR)
  set(OptiX_FOUND TRUE)
else()
  message(STATUS "Could not found OptiX")
endif()

message(STATUS "Include: ${OptiX_INCLUDE_DIR}")
set(OptiX_VERSION ${optix_VERSION})

if(OptiX_FOUND)
  set(OptiX_LIBRARIES ${OptiX_LIBRARY})
  set(OptiX_INCLUDE_DIRS ${OptiX_INCLUDE_DIR})
  # set(OptiX_DEFINITIONS ${PC_Foo_CFLAGS_OTHER})
endif()