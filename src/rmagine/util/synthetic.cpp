#include "rmagine/util/synthetic.h"
#include <iostream>

namespace rmagine {

aiScene createAiScene(
    const std::vector<Vector3>& vertices,
    const std::vector<Face>& faces)
{
    // construct aiScene
    aiScene scene;

    scene.mRootNode = new aiNode();

    scene.mMaterials = new aiMaterial*[1];
    scene.mMaterials[0] = nullptr;
    scene.mNumMaterials = 1;

    scene.mMaterials[0] = new aiMaterial();

    scene.mMeshes = new aiMesh*[1];
    scene.mMeshes[0] = nullptr;
    scene.mNumMeshes = 1;
    scene.mMeshes[0] = new aiMesh();
    scene.mMeshes[0]->mMaterialIndex = 0;

    scene.mRootNode->mMeshes = new unsigned int[1];
    scene.mRootNode->mMeshes[0] = 0;
    scene.mRootNode->mNumMeshes = 1;

    auto pMesh = scene.mMeshes[0];

    pMesh->mVertices = new aiVector3D[vertices.size()];
    pMesh->mNumVertices = vertices.size();

    pMesh->mNumUVComponents[ 0 ] = 0;

    for(unsigned int i=0; i<vertices.size(); i++)
    {
        auto v = vertices[i];
        pMesh->mVertices[i].x = v.x;
        pMesh->mVertices[i].y = v.y;
        pMesh->mVertices[i].z = v.z; 
    }

    pMesh->mFaces = new aiFace[faces.size()];
    pMesh->mNumFaces = faces.size();

    for(unsigned int i=0; i<faces.size(); i++)
    {
        auto f = faces[i];
        aiFace& face = pMesh->mFaces[i];

        face.mIndices = new unsigned int[3];
        face.mNumIndices = 3;

        face.mIndices[0] = f.v0;
        face.mIndices[1] = f.v1;
        face.mIndices[2] = f.v2;
    }

    return scene;
}

void genSphere(
    std::vector<Vector3>& vertices,
    std::vector<Face>& faces,
    unsigned int num_long,
    unsigned int num_lat)
{
    float radius = 0.5;

    vertices.resize(0);
    faces.resize(0);

    float long_inc = M_PI / static_cast<float>(num_long + 1);
    float lat_inc = (2 * M_PI) / static_cast<float>(num_lat);

    // add first and last point

    // add first vertex manually
    vertices.push_back({
        radius * sinf(0.0f) * cosf(0.0f),
        radius * sinf(0.0f) * sinf(0.0f),
        radius * cosf(0.0f)
        });

    // add first faces manually
    for(unsigned int i=0; i<num_lat; i++)
    {
        unsigned int id_bl = 0;
        unsigned int id_tl = i;
        unsigned int id_tr = i + 1;

        if(i == num_lat - 1)
        {
            id_tr -= num_lat;
        }

        faces.push_back({id_bl, id_tl + 1, id_tr + 1});
    }

    for(unsigned int i=0; i<num_long; i++)
    {
        float alpha = long_inc * (i+1);

        for(unsigned int j=0; j<num_lat; j++)
        {
            float beta = lat_inc * (j+1);

            vertices.push_back({
                radius * sinf(alpha) * cosf(beta),
                radius * sinf(alpha) * sinf(beta),
                radius * cosf(alpha)
            });

            if(i > 0)
            {

                unsigned int id_bl = num_lat * (i-1) + j;
                unsigned int id_br = num_lat * (i-1) + j + 1;
                unsigned int id_tl = num_lat * (i)   + j;
                unsigned int id_tr = num_lat * (i)   + j + 1;

                if(j == num_lat - 1)
                {
                    id_br -= num_lat;
                    id_tr -= num_lat;
                }

                // clockwise
                
                // first face
                faces.push_back({id_br + 1,id_bl + 1,id_tl + 1});

                // second face
                faces.push_back({id_tl + 1,id_tr + 1,id_br + 1});
            }
        }
    }

    // add last vertex
    vertices.push_back({
        radius * sinf(M_PI) * cosf(2.0f * M_PI),
        radius * sinf(M_PI) * sinf(2.0f * M_PI),
        radius * cosf(M_PI)
        });

    unsigned int num_vertices = vertices.size();
    for(unsigned int i=num_vertices-1-num_lat; i<num_vertices-1; i++)
    {
        unsigned int id_bl = i;
        unsigned int id_br = i+1;
        unsigned int id_tl = num_vertices-1;

        if(id_br == id_tl)
        {
            id_br -= num_lat;
        }

        faces.push_back({id_br,id_bl,id_tl});
    }
}

aiScene genSphere(unsigned int num_long, unsigned int num_lat)
{
    // x = cx + r * sin(alpha) * cos(beta)
    // y = cy + r * sin(alpha) * sin(beta)
    // z = cz + r * cos(alpha)

    // alpha [-pi, pi)
    // beta [-pi, pi)

    std::vector<Vector3> vertices;
    std::vector<Face> faces;

    genSphere(vertices, faces, num_long, num_lat);
    return createAiScene(vertices, faces);
}


void genCube(std::vector<Vector3>& vertices,
    std::vector<Face>& faces,
    unsigned int side_triangles_exp)
{

    if(side_triangles_exp > 1)
    {
        std::cout << "WARNING: side_triangles_exp other than default is not yet supported. Setting to default = 1 instead." << std::endl;
    }

    vertices.resize(0);
    faces.resize(0);

    vertices.push_back({-0.5, -0.5, -0.5});
    vertices.push_back({-0.5, -0.5,  0.5});
    vertices.push_back({-0.5,  0.5,  0.5});
    vertices.push_back({-0.5,  0.5, -0.5});

    faces.push_back({0, 1, 2});
    faces.push_back({2, 3, 0});

    vertices.push_back({0.5,  0.5, -0.5});
    vertices.push_back({0.5,  0.5,  0.5});
    vertices.push_back({0.5, -0.5,  0.5});
    vertices.push_back({0.5, -0.5, -0.5});

    faces.push_back({0+4, 1+4, 2+4});
    faces.push_back({2+4, 3+4, 0+4});

    faces.push_back({0, 7, 1});
    faces.push_back({1, 7, 6});
    faces.push_back({1, 6, 5});
    faces.push_back({5, 2, 1});
    faces.push_back({2, 5, 4});
    faces.push_back({3, 2, 4});
    faces.push_back({3, 4, 7});
    faces.push_back({3, 7, 0});

}

aiScene genCube(unsigned int side_triangles_exp)
{
    std::vector<Vector3> vertices;
    std::vector<Face> faces;
    genCube(vertices, faces, side_triangles_exp);
    return createAiScene(vertices, faces);
}


void genPlane(
    std::vector<Vector3>& vertices,
    std::vector<Face>& faces,
    unsigned int side_triangles_exp)
{
    if(side_triangles_exp > 1)
    {
        std::cout << "WARNING: side_triangles_exp other than default is not yet supported. Setting to default = 1 instead." << std::endl;
    }

    vertices.resize(4);
    vertices[0] = {-0.5, 0.5, 0.0};
    vertices[1] = { 0.5, 0.5, 0.0};
    vertices[2] = { 0.5, -0.5, 0.0};
    vertices[3] = {-0.5, -0.5, 0.0};

    faces.resize(2);
    faces[0] = {1, 0, 3};
    faces[1] = {3, 2, 1};
}

aiScene genPlane(unsigned int side_triangles_exp)
{
    std::vector<Vector3> vertices;
    std::vector<Face> faces;
    genPlane(vertices, faces, side_triangles_exp);
    return createAiScene(vertices, faces);
}


void genCylinder(
    std::vector<Vector3>& vertices,
    std::vector<Face>& faces,
    unsigned int side_faces)
{

    // parameters
    float diameter = 1.0;
    float height = 1.0;



    float increment = 2.f * M_PI / static_cast<float>(side_faces);
    float radius = diameter / 2.f;

    // make vertices
    vertices.reserve(side_faces * 2 + 2);
    for(size_t i=0; i<side_faces; i++)
    {
        float angle = static_cast<float>(i) * increment;
        Vertex v_up = {cos(angle) * radius, sin(angle) * radius, height / 2.f};
        Vertex v_bottom = {cos(angle) * radius, sin(angle) * radius, - height / 2.f};
        vertices.push_back(v_up);
        vertices.push_back(v_bottom);
    }

    // add top and bottom vertices
    vertices.push_back({0.0, 0.0, height / 2.f});
    vertices.push_back({0.0, 0.0, - height / 2.f});

    unsigned int vid_top = vertices.size() - 2;
    unsigned int vid_bottom = vertices.size() - 1; 

    // connect vertices
    faces.resize(0);

    // side
    unsigned int n_side_triangles = side_faces * 2;
    for(size_t i=0; i<side_faces; i++)
    {
        unsigned int vid_lt = i * 2;
        unsigned int vid_lb = i * 2 + 1;
        unsigned int vid_rt = i * 2 + 2;
        unsigned int vid_rb = i * 2 + 3;

        if(vid_rt >= n_side_triangles)
        {
            vid_rt %= n_side_triangles;
            vid_rb %= n_side_triangles;
        }

        faces.push_back({vid_rt, vid_lt, vid_lb});
        faces.push_back({vid_lb, vid_rb, vid_rt});
    }

    // top
    for(size_t i=0; i<side_faces; i++)
    {
        unsigned int vid_lt = i * 2;
        unsigned int vid_rt = i * 2 + 2;
        if(vid_rt >= n_side_triangles)
        {
            vid_rt %= n_side_triangles;
        }

        faces.push_back({vid_rt, vid_top, vid_lt});
    }

    // bottom
    for(size_t i=0; i<side_faces; i++)
    {
        unsigned int vid_lb = i * 2 + 1;
        unsigned int vid_rb = i * 2 + 3;
        if(vid_rb >= n_side_triangles)
        {
            vid_rb %= n_side_triangles;
        }

        faces.push_back({vid_lb, vid_bottom, vid_rb});
    }
}

aiScene genCylinder(unsigned int side_faces)
{
    std::vector<Vector3> vertices;
    std::vector<Face> faces;
    genCylinder(vertices, faces, side_faces);
    return createAiScene(vertices, faces);
}


} // namespace rmagine