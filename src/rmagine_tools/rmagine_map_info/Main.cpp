#include <iostream>
#include <sstream>

// General rmagine includes
#include <rmagine/types/sensor_models.h>
#include <rmagine/util/StopWatch.hpp>
#include <rmagine/util/synthetic.h>

#include <assimp/Importer.hpp>

#include <unordered_map>
#include <functional>

#include <boost/algorithm/string/join.hpp>

// #include <rmagine/map/EmbreeMap.hpp>
#include <rmagine/util/prints.h>
#include <rmagine/math/assimp_conversions.h>




using namespace rmagine;

struct Indent
{
    int indent;
    Indent(int indent) : indent(indent) {}

    inline Indent operator+(int oindent) const
    {
        return Indent(indent + oindent);
    }
};

inline std::ostream& operator<<(std::ostream& os, const Indent& ind)
{
    for(int i=0; i<ind.indent; i++)
    {
        os << " ";
    }
    return os;
}

std::vector<std::string> getPrimTypes(unsigned int primType )
{
    std::vector<std::string> ret;

    if(primType & aiPrimitiveType_POINT)
    {
        ret.push_back("POINT");
    }
    if(primType & aiPrimitiveType_LINE)
    {
        ret.push_back("LINE");
    }
    if(primType & aiPrimitiveType_TRIANGLE)
    {
        ret.push_back("TRIANGLE");
    }
    if(primType & aiPrimitiveType_POLYGON)
    {
        ret.push_back("POLYGON");
    }

    return ret;
}

void print(const aiMesh* amesh, int indent = 0)
{
    auto ind = Indent(indent);
    std::cout << ind << "- name: " << amesh->mName.C_Str() << std::endl;
    std::cout << ind << "- vertices, faces: " << amesh->mNumVertices << ", " << amesh->mNumFaces << std::endl;
    std::vector<std::string> prim_types = getPrimTypes(amesh->mPrimitiveTypes);
    std::cout << ind << "- primitives: " << boost::algorithm::join(prim_types, ", ") << std::endl;
    std::cout << ind << "- normals: " << ((amesh->HasNormals())? "yes" : "no") << std::endl;
    std::cout << ind << "- vertex color channels: " << amesh->GetNumColorChannels() << std::endl; 
    std::cout << ind << "- uv channels: " << amesh->GetNumUVChannels() << std::endl;
    std::cout << ind << "- bones: " << amesh->mNumBones << std::endl;
    std::cout << ind << "- material index: " << amesh->mMaterialIndex << std::endl;
    std::cout << ind << "- tangents and bitangents: " << ((amesh->HasTangentsAndBitangents())? "yes" : "no") << std::endl;
    AABB box = convert(amesh->mAABB);
    std::cout << ind << "- aabb: " << box << std::endl;
}

void print(const Matrix4x4& M, int indent = 0)
{
    auto ind = Indent(indent);
    std::cout << ind << "M4x4[\n";
    std::cout << ind + 2 << M(0, 0) << " " << M(0, 1) << " " << M(0, 2) << " " << M(0, 3) << "\n";
    std::cout << ind + 2 << M(1, 0) << " " << M(1, 1) << " " << M(1, 2) << " " << M(1, 3) << "\n";
    std::cout << ind + 2 << M(2, 0) << " " << M(2, 1) << " " << M(2, 2) << " " << M(2, 3) << "\n";
    std::cout << ind + 2 << M(3, 0) << " " << M(3, 1) << " " << M(3, 2) << " " << M(3, 3) << "\n";
    std::cout << ind << "]\n";
}

void print(const aiNode* node, int indent = 0)
{
    auto ind = Indent(indent);

    Matrix4x4 T;
    convert(node->mTransformation, T);

    std::cout << ind << "- name: " << node->mName.C_Str() << std::endl;
    std::cout << ind << "- transform: " << std::endl;
    print(T, indent + 2);
    // std::cout << T << std::endl;

    std::cout << ind << "- meshes: " << node->mNumMeshes << std::endl;
    for(int i=0; i<node->mNumMeshes; i++)
    {
        std::cout << ind + 2 << "- mesh ref " << i << " -> " << node->mMeshes[i] << std::endl; 
    }

    std::cout << ind << "- children: " << node->mNumChildren << std::endl;
    for(int i=0; i<node->mNumChildren; i++)
    {
        std::cout << ind + 2 << "Node " << i << std::endl;
        print(node->mChildren[i], indent + 4);
    }
}

void print(const aiScene* ascene)
{
    std::cout << "Meshes: " << ascene->mNumMeshes << std::endl;

    for(size_t i=0; i<ascene->mNumMeshes; i++)
    {
        std::cout << Indent(2) << "Mesh " << i << std::endl;
        print(ascene->mMeshes[i], 4);
    }
    
    std::cout << "Textures: " << ascene->mNumTextures << std::endl;

    std::cout << "Scene Graph: " << std::endl;
    const aiNode* root_node = ascene->mRootNode;
    
    print(root_node);
}

int main(int argc, char** argv)
{
    std::cout << "Rmagine Map Info" << std::endl;


    // minimum 2 arguments
    if(argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " mesh_file" << std::endl;
        return 0;
    }

    std::string filename = argv[1];

    std::cout << "Inputs: " << std::endl;
    std::cout << "- filename: " << filename << std::endl;

    Assimp::Importer importer;
    const aiScene* ascene = importer.ReadFile(filename, 0);

    print(ascene);

    return 0;
}
