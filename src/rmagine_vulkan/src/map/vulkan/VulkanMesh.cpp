#include "rmagine/map/vulkan/VulkanMesh.hpp"

namespace rmagine
{

VulkanMesh::VulkanMesh() : Base(),
    transformMatrix_ram(1),
    transformMatrix(1, VulkanMemoryUsage::Usage_AccelerationStructureInstanceData),
    vertices(0, VulkanMemoryUsage::Usage_AccelerationStructureMeshData),
    faces(0, VulkanMemoryUsage::Usage_AccelerationStructureMeshData), 
    face_normals(0),
    vertex_normals(0)
{
    transformMatrix_ram[0] = {{{1.0, 0.0, 0.0, 0.0},
                               {0.0, 1.0, 0.0, 0.0},
                               {0.0, 0.0, 1.0, 0.0}}};
}

VulkanMesh::~VulkanMesh()
{

}

void VulkanMesh::apply()
{
    Matrix4x4 M = matrix();
    transformMatrix_ram[0] = {{{M(0,0), M(0,1), M(0,2), M(0,3)},
                               {M(1,0), M(1,1), M(1,2), M(1,3)},
                               {M(2,0), M(2,1), M(2,2), M(2,3)}}};
    m_changed = true;
}

void VulkanMesh::commit()
{
    transformMatrix = transformMatrix_ram;
}

unsigned int VulkanMesh::depth() const
{
    return 0;
}

void VulkanMesh::computeFaceNormals()
{
    throw std::runtime_error("[VulkanMesh::computeFaceNormals()] ERROR - currently does not work, as data is already on the gpu and there is currently no function rmagine::computeFaceNormals() for DEVICE_LOCAL_VULKAN memory. please fill face_normals manually.");
    // if(face_normals.size() != faces.size())
    // {
    //     face_normals.resize(faces.size());
    // }
    // rmagine::computeFaceNormals(vertices, faces, face_normals);
}



VulkanMeshPtr make_vulkan_mesh(Memory<Point, RAM>& vertices_ram, Memory<Face, RAM>& faces_ram)
{
    VulkanMeshPtr ret = std::make_shared<VulkanMesh>();

    unsigned int num_vertices = vertices_ram.size();
    unsigned int num_faces = faces_ram.size();

    ret->vertices.resize(vertices_ram.size());
    ret->vertices = vertices_ram;

    ret->faces.resize(faces_ram.size());
    ret->faces = faces_ram;

    // ret->computeFaceNormals();
    Memory<Vector, RAM> face_normals_ram(num_faces);
    for(size_t i=0; i<num_faces; i++)
    {
        const Vector v0 = vertices_ram[faces_ram[i].v0];
        const Vector v1 = vertices_ram[faces_ram[i].v1];
        const Vector v2 = vertices_ram[faces_ram[i].v2];
        face_normals_ram[i] = (v1 - v0).normalize().cross((v2 - v0).normalize() ).normalize();
    }
    ret->face_normals.resize(num_faces);
    ret->face_normals = face_normals_ram;

    ret->apply();

    return ret;
}

VulkanMeshPtr make_vulkan_mesh(const aiMesh* amesh)
{
    VulkanMeshPtr ret = std::make_shared<VulkanMesh>();

    const aiVector3D* ai_vertices = amesh->mVertices;
    unsigned int num_vertices = amesh->mNumVertices;
    const aiFace* ai_faces = amesh->mFaces;
    unsigned int num_faces = amesh->mNumFaces;

    Memory<Point, RAM> vertices_cpu(num_vertices);
    Memory<Face, RAM> faces_cpu(num_faces);
    Memory<Vector, RAM> face_normals_cpu(num_faces);

    // convert
    for(size_t i=0; i<num_vertices; i++)
    {
        vertices_cpu[i] = {
            ai_vertices[i].x,
            ai_vertices[i].y,
            ai_vertices[i].z};
    }
    ret->vertices.resize(vertices_cpu.size());
    ret->vertices = vertices_cpu;

    for(size_t i=0; i<num_faces; i++)
    {
        faces_cpu[i].v0 = ai_faces[i].mIndices[0];
        faces_cpu[i].v1 = ai_faces[i].mIndices[1];
        faces_cpu[i].v2 = ai_faces[i].mIndices[2];
    }
    ret->faces.resize(faces_cpu.size());
    ret->faces = faces_cpu;

    // ret->computeFaceNormals();
    for(size_t i=0; i<num_faces; i++)
    {
        const Vector v0 = vertices_cpu[faces_cpu[i].v0];
        const Vector v1 = vertices_cpu[faces_cpu[i].v1];
        const Vector v2 = vertices_cpu[faces_cpu[i].v2];
        face_normals_cpu[i] = (v1 - v0).normalize().cross((v2 - v0).normalize() ).normalize();
    }
    ret->face_normals.resize(num_faces);
    ret->face_normals = face_normals_cpu;

    if(amesh->HasNormals())
    {
        // has vertex normals
        Memory<Vector, RAM> vertex_normals_cpu(num_faces);
        vertex_normals_cpu.resize(num_vertices);
        for(size_t i=0; i<num_vertices; i++)
        {
            vertex_normals_cpu[i] = convert(amesh->mNormals[i]);
        }
        // upload
        ret->vertex_normals.resize(num_vertices);
        ret->vertex_normals = vertex_normals_cpu;
    }

    ret->apply();

    return ret;
}

} // namespace rmagine