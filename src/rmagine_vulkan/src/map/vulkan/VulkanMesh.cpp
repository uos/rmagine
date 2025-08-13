#include "rmagine/map/vulkan/VulkanMesh.hpp"

namespace rmagine
{

VulkanMesh::VulkanMesh(/* args */)
{

}

VulkanMesh::~VulkanMesh()
{

}

void VulkanMesh::apply()
{
    transformMatrix = {.matrix = {{1.0, 0.0, 0.0, 0.0},
                                  {0.0, 1.0, 0.0, 0.0},
                                  {0.0, 0.0, 1.0, 0.0}}};
}

void VulkanMesh::commit(){}

unsigned int VulkanMesh::depth() const
{
    return 0;
}

void VulkanMesh::computeFaceNormals()//TODO
{
    if(face_normals.size() != faces.size())
    {
        face_normals.resize(faces.size());
    }
    // rmagine::computeFaceNormals(vertices, faces, face_normals);
}



VulkanMeshPtr make_vulkan_mesh(Memory<float, RAM>& vertexMem_ram, Memory<uint32_t, RAM>& indexMem_ram)
{

}

// VulkanMeshPtr make_vulkan_mesh(const aiMesh* amesh)
// {
//     // (parts) taken from OptixMap.hpp

//     VulkanMeshPtr ret = std::make_shared<VulkanMesh>();

//     const aiVector3D* ai_vertices = amesh->mVertices;
//     unsigned int num_vertices = amesh->mNumVertices;
//     const aiFace* ai_faces = amesh->mFaces;
//     unsigned int num_faces = amesh->mNumFaces;

//     Memory<Point, RAM> vertices_cpu(num_vertices);
//     Memory<Face, RAM> faces_cpu(num_faces);
//     Memory<Vector, RAM> face_normals_cpu(num_faces);

//     // convert
//     for(size_t i=0; i<num_vertices; i++)
//     {
//         vertices_cpu[i] = {
//                 ai_vertices[i].x,
//                 ai_vertices[i].y,
//                 ai_vertices[i].z};
//     }
//     ret->vertices = vertices_cpu;

//     for(size_t i=0; i<num_faces; i++)
//     {
//         faces_cpu[i].v0 = ai_faces[i].mIndices[0];
//         faces_cpu[i].v1 = ai_faces[i].mIndices[1];
//         faces_cpu[i].v2 = ai_faces[i].mIndices[2];
//     }
//     ret->faces = faces_cpu;

//     // ret->computeFaceNormals();
//     for(size_t i=0; i<num_faces; i++)//TODO
//     {
//         face_normals_cpu[i] = Vector3{0,0,0};
//     }
//     ret->face_normals = face_normals_cpu;

//     if(amesh->HasNormals())
//     {
//         // has vertex normals
//         Memory<Vector, RAM> vertex_normals_cpu(num_faces);
//         vertex_normals_cpu.resize(num_vertices);
//         for(size_t i=0; i<num_vertices; i++)
//         {
//             vertex_normals_cpu[i] = convert(amesh->mNormals[i]);
//         }
//         // upload
//         ret->vertex_normals = vertex_normals_cpu;
//     }

//     ret->apply();

//     return ret;
// }

} // namespace rmagine