#include "rmagine/types/mesh_types.h"

namespace rmagine
{

} // namespace rmagine


unsigned int adaptorF_custom_accessVector3Value(
  const rmagine::Face& f, 
  unsigned int ind) 
{
  return f[ind];
}

size_t adaptorF_size(const rmagine::Face& f)
{
  return 3;
}