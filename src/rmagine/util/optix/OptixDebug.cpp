#include "rmagine/util/optix/OptixDebug.hpp"

#include <stdexcept>
#include <sstream>
#include <iostream>

#include <optix.h>
#include <cuda_runtime.h>
#include <optix_stubs.h>

namespace rmagine
{

OptixException::OptixException( const char* msg )
: std::runtime_error( msg )
{

}

} // namespace rmagine