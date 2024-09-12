#include "mesh_types.h"

namespace rmagine
{

RMAGINE_INLINE_FUNCTION
const unsigned int& Face::operator[](const size_t& idx) const
{
    return *((&v0)+idx);
}

RMAGINE_INLINE_FUNCTION
unsigned int& Face::operator[](const size_t& idx)
{
    return *((&v0)+idx);
}

} // namespace rmagine
