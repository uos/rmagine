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

// namespace std
// {

// template <std::size_t I>
// unsigned int& get(rmagine::Face& f) {
//     if constexpr (I == 0) return (f.v0);
//     else if constexpr (I == 1) return (f.v1);
//     else if constexpr (I == 2) return (f.v2);
//     else static_assert(I < 3, "Index out of bounds");
// }

// template <std::size_t I>
// const unsigned int& get(const rmagine::Face& f) {
//     if constexpr (I == 0) return (f.v0);
//     else if constexpr (I == 1) return (f.v1);
//     else if constexpr (I == 2) return (f.v2);
//     else static_assert(I < 3, "Index out of bounds");
// }


// } // namespace std