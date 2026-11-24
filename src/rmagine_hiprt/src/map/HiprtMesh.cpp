/**
 * @file HiprtMesh.cpp
 * @brief HIPRT mesh implementation
 */

#include "rmagine/map/hiprt/HiprtMesh.hpp"

#include <hip/hip_runtime.h>

namespace rmagine
{

HiprtMesh::HiprtMesh()
{
    // Initialize pre_transform to identity
    pre_transform_h[0] = 1.0f;  pre_transform_h[1] = 0.0f;  pre_transform_h[2] = 0.0f;  pre_transform_h[3] = 0.0f;
    pre_transform_h[4] = 0.0f;  pre_transform_h[5] = 1.0f;  pre_transform_h[6] = 0.0f;  pre_transform_h[7] = 0.0f;
    pre_transform_h[8] = 0.0f;  pre_transform_h[9] = 0.0f;  pre_transform_h[10] = 1.0f; pre_transform_h[11] = 0.0f;
}

HiprtMesh::~HiprtMesh()
{
    if (pre_transform) {
        hipFree(pre_transform);
        pre_transform = nullptr;
    }
}

void HiprtMesh::computeFaceNormals()
{
    // This should call rmagine's existing computeFaceNormals from rmagine_cuda
    // For now, allocate the buffer -- actual computation is done by rmagine_cuda
    if (face_normals.size() != faces.size()) {
        face_normals.resize(faces.size());
    }
    // TODO: call rmagine::computeFaceNormals(vertices, faces, face_normals);
}

void HiprtMesh::commit()
{
    mesh_data.vertex_normals = vertex_normals.raw();
    mesh_data.face_normals = face_normals.raw();
    mesh_data.id = id;
}

void HiprtMesh::apply(const Matrix4x4& M)
{
    // Convert 4x4 matrix to 3x4 row-major
    pre_transform_h[0] = M(0, 0);  pre_transform_h[1] = M(0, 1);  pre_transform_h[2] = M(0, 2);  pre_transform_h[3] = M(0, 3);
    pre_transform_h[4] = M(1, 0);  pre_transform_h[5] = M(1, 1);  pre_transform_h[6] = M(1, 2);  pre_transform_h[7] = M(1, 3);
    pre_transform_h[8] = M(2, 0);  pre_transform_h[9] = M(2, 1);  pre_transform_h[10] = M(2, 2); pre_transform_h[11] = M(2, 3);

    if (!pre_transform) {
        hipMalloc(&pre_transform, sizeof(float) * 12);
    }

    hipMemcpy(pre_transform, pre_transform_h, sizeof(float) * 12, hipMemcpyHostToDevice);
}

} // namespace rmagine
