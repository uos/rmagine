#include "rmagine/map/embree/embree_shapes.h"
#include <rmagine/util/synthetic.h>

namespace rmagine
{

EmbreeSphere::EmbreeSphere(
    float radius, 
    unsigned int num_long, 
    unsigned int num_lat,
    EmbreeDevicePtr device)
:Base(device)
{
    std::vector<Vector3> _vertices;
    std::vector<Face> _faces;

    genSphere(_vertices, _faces, num_long, num_lat);
    init(_vertices.size(), _faces.size());

    std::copy(_vertices.begin(), _vertices.end(), vertices.raw());
    std::copy(_faces.begin(), _faces.end(), faces);

    float diameter = radius * 2.0;
    setScale({diameter, diameter, diameter});

    computeFaceNormals();

    apply();
}

EmbreeCube::EmbreeCube(
    unsigned int side_triangles_exp,
    EmbreeDevicePtr device)
:Base(device)
{
    std::vector<Vector3> _vertices;
    std::vector<Face> _faces;

    genCube(_vertices, _faces, side_triangles_exp);
    init(_vertices.size(), _faces.size());

    std::copy(_vertices.begin(), _vertices.end(), vertices.raw());
    std::copy(_faces.begin(), _faces.end(), faces);

    computeFaceNormals();

    apply();
}

EmbreePlane::EmbreePlane(
    unsigned int side_triangles_exp,
    EmbreeDevicePtr device)
:Base(device)
{
    std::vector<Vector3> _vertices;
    std::vector<Face> _faces;

    genPlane(_vertices, _faces, side_triangles_exp);
    init(_vertices.size(), _faces.size());

    std::copy(_vertices.begin(), _vertices.end(), vertices.raw());
    std::copy(_faces.begin(), _faces.end(), faces);

    computeFaceNormals();

    apply();
}

EmbreeCylinder::EmbreeCylinder(
    unsigned int side_faces,
    EmbreeDevicePtr device)
:Base(device)
{
    std::vector<Vector3> _vertices;
    std::vector<Face> _faces;

    genCylinder(_vertices, _faces, side_faces);
    init(_vertices.size(), _faces.size());

    std::copy(_vertices.begin(), _vertices.end(), vertices.raw());
    std::copy(_faces.begin(), _faces.end(), faces);

    computeFaceNormals();

    apply();
}



} // namespace rmagine