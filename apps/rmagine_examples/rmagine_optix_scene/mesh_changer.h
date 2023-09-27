#ifndef RMAGINE_EXAMPLES_MESH_CHANGER_H
#define RMAGINE_EXAMPLES_MESH_CHANGER_H

#include <rmagine/math/types.h>
#include <rmagine/types/MemoryCuda.hpp>

namespace rmagine
{

void moveVertices(MemoryView<Vector, VRAM_CUDA>& vertices, const Vector3 vec);

} // namespace rmagine

#endif // RMAGINE_EXAMPLES_MESH_CHANGER_H