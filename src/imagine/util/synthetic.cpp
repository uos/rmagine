#include "imagine/util/synthetic.h"

namespace imagine {

void genSphere(
    std::vector<float>& vertices, 
    std::vector<unsigned int>& faces,
    unsigned int num_long,
    unsigned int num_lat)
{
    vertices.resize(0);
    faces.resize(0);



    float long_inc = M_PI / static_cast<float>(num_long + 1);
    float lat_inc = (2 * M_PI) / static_cast<float>(num_lat);

    // add first and last point


    // add first vertex manually
    vertices.push_back(sin(0.0) * cos(0.0));
    vertices.push_back(sin(0.0) * sin(0.0));
    vertices.push_back(cos(0.0));

    // add first faces manually
    for(int i=0; i<num_lat; i++)
    {
        int id_bl = 0;
        int id_tl = i;
        int id_tr = i + 1;

        if(i == num_lat - 1)
        {
            id_tr -= num_lat;
        }

        faces.push_back(id_bl);
        faces.push_back(id_tl + 1);
        faces.push_back(id_tr + 1);
    }

    for(int i=0; i<num_long; i++)
    {
        float alpha = long_inc * (i+1);

        for(int j=0; j<num_lat; j++)
        {
            float beta = lat_inc * (j+1);

            vertices.push_back(sin(alpha) * cos(beta));
            vertices.push_back(sin(alpha) * sin(beta));
            vertices.push_back(cos(alpha));

            if(i > 0)
            {

                int id_bl = num_lat * (i-1) + j;
                int id_br = num_lat * (i-1) + j + 1;
                int id_tl = num_lat * (i)   + j;
                int id_tr = num_lat * (i)   + j + 1;

                if(j == num_lat - 1)
                {
                    id_br -= num_lat;
                    id_tr -= num_lat;
                }

                // clockwise
                
                // first face
                faces.push_back(id_br + 1);
                faces.push_back(id_bl + 1);
                faces.push_back(id_tl + 1);

                // second face
                faces.push_back(id_tl + 1);
                faces.push_back(id_tr + 1);
                faces.push_back(id_br + 1);
            }
        }
    }

    // add last vertex
    vertices.push_back(sin(M_PI) * cos(2.0 * M_PI));
    vertices.push_back(sin(M_PI) * sin(2.0 * M_PI));
    vertices.push_back(cos(M_PI));

    int num_vertices = vertices.size() / 3;
    for(int i=num_vertices-1-num_lat; i<num_vertices-1; i++)
    {
        int id_bl = i;
        int id_br = i+1;
        int id_tl = num_vertices-1;

        if(id_br == id_tl)
        {
            id_br -= num_lat;
        }

        faces.push_back(id_br);
        faces.push_back(id_bl);
        faces.push_back(id_tl);
    }
}

void genSphere(
    std::vector<Vector3>& vertices,
    std::vector<Face>& faces,
    unsigned int num_long,
    unsigned int num_lat)
{
    vertices.resize(0);
    faces.resize(0);

    float long_inc = M_PI / static_cast<float>(num_long + 1);
    float lat_inc = (2 * M_PI) / static_cast<float>(num_lat);

    // add first and last point

    // add first vertex manually
    vertices.push_back({
        sin(0.0) * cos(0.0),
        sin(0.0) * sin(0.0),
        cos(0.0)
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
                sin(alpha) * cos(beta),
                sin(alpha) * sin(beta),
                cos(alpha)
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
        sin(M_PI) * cos(2.0 * M_PI),
        sin(M_PI) * sin(2.0 * M_PI),
        cos(M_PI)
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

} // namespace imagine