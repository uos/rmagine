#ifndef RMAGINE_TYPES_TEST_SHARED_FUNCTIONS_H
#define RMAGINE_TYPES_TEST_SHARED_FUNCTIONS_H

#include <rmagine/types/shared_functions.h>

namespace rmagine
{

RMAGINE_FUNCTION
size_t test_function();

RMAGINE_INLINE_FUNCTION
size_t test_inline_function()
{
  return 10;
}

RMAGINE_HOST_FUNCTION
size_t test_host_function();


RMAGINE_INLINE_HOST_FUNCTION
size_t test_inline_host_function()
{
  return 10;
}

RMAGINE_DEVICE_FUNCTION
size_t test_device_function();

RMAGINE_INLINE_DEVICE_FUNCTION
size_t test_inline_device_function()
{
  return 10;
}

} // namespace rmagine

#endif // RMAGINE_TYPES_TEST_SHARED_FUNCTIONS_H