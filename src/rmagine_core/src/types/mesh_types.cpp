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