#ifndef RMAGINE_MAP_MESH_PREPROCESSING_CUH
#define RMAGINE_MAP_MESH_PREPROCESSING_CUH

#include <rmagine/types/MemoryCuda.hpp>
#include <rmagine/types/mesh_types.h>
#include <rmagine/math/types.h>

namespace rmagine
{

void computeFaceNormals(
    const MemoryView<Vector3, VRAM_CUDA>& vertices,
    const MemoryView<Face, VRAM_CUDA>& faces,
    MemoryView<Vector3, VRAM_CUDA>& face_normals);

Memory<Vector3, VRAM_CUDA> computeFaceNormals(
    const MemoryView<Vector3, VRAM_CUDA>& vertices,
    const MemoryView<Face, VRAM_CUDA>& faces);

} // namespace rmagine

#endif // RMAGINE_MAP_MESH_PREPROCESSING_CUH