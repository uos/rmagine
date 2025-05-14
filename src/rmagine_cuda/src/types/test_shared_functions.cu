#include "rmagine/types/test_shared_functions.h"

namespace rmagine
{

RMAGINE_DEVICE_FUNCTION
size_t test_function()
{
  return 20;
}

RMAGINE_DEVICE_FUNCTION
size_t test_device_function()
{
  return 20;
}

} // namespace rmagine