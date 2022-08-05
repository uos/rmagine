#include "rmagine/map/optix/optix_shapes.h"

#include <rmagine/util/synthetic.h>

namespace rmagine
{

OptixSphere::OptixSphere(
    unsigned int num_long,
    unsigned int num_lat,
    OptixContextPtr context)
:Base(context)
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

OptixSphere::~OptixSphere()
{
    
}

} // namespace rmagine