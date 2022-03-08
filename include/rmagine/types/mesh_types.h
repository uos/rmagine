#ifndef RMAGINE_TYPES_MESH_TYPES_H
#define RMAGINE_TYPES_MESH_TYPES_H

#include <rmagine/math/types.h>

namespace rmagine
{

using Vertex = Point;

struct Face {
    unsigned int v0;
    unsigned int v1;
    unsigned int v2;
};

} // namespace rmagine

#endif // RMAGINE_TYPES_MESH_TYPES_H