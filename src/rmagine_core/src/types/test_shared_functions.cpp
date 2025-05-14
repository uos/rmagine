#include "rmagine/types/test_shared_functions.h"

namespace rmagine
{

RMAGINE_HOST_FUNCTION
size_t test_function()
{
  return 10;
}

RMAGINE_HOST_FUNCTION
size_t test_host_function()
{
  return 10;
}

} // namespace rmagine