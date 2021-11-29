#ifndef IMAGINE_TYPES_MESH_TYPES_H
#define IMAGINE_TYPES_MESH_TYPES_H

#include <imagine/math/types.h>

namespace imagine
{

using Vertex = Point;

struct Face {
    unsigned int v0;
    unsigned int v1;
    unsigned int v2;
};

} // namespace imagine

#endif // IMAGINE_TYPES_MESH_TYPES_H