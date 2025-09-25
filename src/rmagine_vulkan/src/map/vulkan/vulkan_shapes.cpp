#include "rmagine/map/vulkan/vulkan_shapes.hpp"



namespace rmagine
{

VulkanSphere::VulkanSphere(unsigned int num_long, unsigned int num_lat) : Base()
{
    std::vector<Vector3> _vertices;
    std::vector<Face> _faces;

    genSphere(_vertices, _faces, num_long, num_lat);

    Memory<Vector3, RAM> vertex_mem(_vertices.size());
    Memory<Face, RAM> face_mem(_faces.size());

    std::copy(_vertices.begin(), _vertices.end(), vertex_mem.raw());
    std::copy(_faces.begin(), _faces.end(), face_mem.raw());

    // upload to GPU
    vertices = vertex_mem;
    faces = face_mem;

    computeFaceNormals();
    apply();
}

VulkanSphere::~VulkanSphere()
{
    
}



VulkanCube::VulkanCube(unsigned int side_triangles_exp) : Base()
{
    std::vector<Vector3> _vertices;
    std::vector<Face> _faces;

    genCube(_vertices, _faces, side_triangles_exp);

    Memory<Vector3, RAM> vertex_mem(_vertices.size());
    Memory<Face, RAM> face_mem(_faces.size());

    std::copy(_vertices.begin(), _vertices.end(), vertex_mem.raw());
    std::copy(_faces.begin(), _faces.end(), face_mem.raw());

    // upload to GPU
    vertices = vertex_mem;
    faces = face_mem;

    computeFaceNormals();

    apply();
}

VulkanCube::~VulkanCube()
{
    
}



VulkanPlane::VulkanPlane(unsigned int side_triangles_exp) : Base()
{
    std::vector<Vector3> _vertices;
    std::vector<Face> _faces;

    genPlane(_vertices, _faces, side_triangles_exp);

    Memory<Vector3, RAM> vertex_mem(_vertices.size());
    Memory<Face, RAM> face_mem(_faces.size());

    std::copy(_vertices.begin(), _vertices.end(), vertex_mem.raw());
    std::copy(_faces.begin(), _faces.end(), face_mem.raw());

    // upload to GPU
    vertices = vertex_mem;
    faces = face_mem;

    computeFaceNormals();

    apply();
}

VulkanPlane::~VulkanPlane()
{
    
}



VulkanCylinder::VulkanCylinder(unsigned int side_faces) : Base()
{
    std::vector<Vector3> _vertices;
    std::vector<Face> _faces;

    genCylinder(_vertices, _faces, side_faces);

    Memory<Vector3, RAM> vertex_mem(_vertices.size());
    Memory<Face, RAM> face_mem(_faces.size());

    std::copy(_vertices.begin(), _vertices.end(), vertex_mem.raw());
    std::copy(_faces.begin(), _faces.end(), face_mem.raw());

    // upload to GPU
    vertices = vertex_mem;
    faces = face_mem;

    computeFaceNormals();

    apply();
}

VulkanCylinder::~VulkanCylinder()
{
    
}

} // namespace rmagine
