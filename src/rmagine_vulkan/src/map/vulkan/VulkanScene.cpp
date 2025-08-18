#include "rmagine/map/vulkan/VulkanScene.hpp"



namespace rmagine
{

void VulkanScene::createScene(Memory<float, RAM>& vertexMem_ram, Memory<uint32_t, RAM>& indexMem_ram)
{
    vertexMem = vertexMem_ram;
    indexMem = indexMem_ram;

    numVerticies = vertexMem_ram.size()/3;
    numTriangles = indexMem_ram.size()/3;
    
    //create acceleration structure
    bottomLevelAccelerationStructure->createAccelerationStructure(numVerticies, vertexMem, numTriangles, indexMem);

    VkTransformMatrixKHR transformMatrix = {{{1.0, 0.0, 0.0, 0.0},
                                             {0.0, 1.0, 0.0, 0.0},
                                             {0.0, 0.0, 1.0, 0.0}}};
    bottomLevelAccelerationStructureInstance->createBottomLevelAccelerationStructureInstance(transformMatrix, 0xFF, bottomLevelAccelerationStructure);

    topLevelAccelerationStructure->createAccelerationStructure(bottomLevelAccelerationStructureInstance);
}


void VulkanScene::commit() {}


void VulkanScene::cleanup()
{
    std::cout << "cleaning up..." << std::endl;

    bottomLevelAccelerationStructure->cleanup();
    topLevelAccelerationStructure->cleanup();
    bottomLevelAccelerationStructureInstance->cleanup();
    std::cout << "cleaned up acceleration structure." << std::endl;

    std::cout << "done." << std::endl;
}



Memory<float, VULKAN_DEVICE_LOCAL>& VulkanScene::getVertexMem()
{
    return vertexMem;
}

Memory<uint32_t, VULKAN_DEVICE_LOCAL>& VulkanScene::getIndexMem()
{
    return indexMem;
}

TopLevelAccelerationStructurePtr VulkanScene::getTopLevelAccelerationStructure()
{
    return topLevelAccelerationStructure;
}



VulkanScenePtr make_vulkan_scene(Memory<Point, RAM>& vertices_ram, Memory<Face, RAM>& faces_ram)
{
    VulkanScenePtr scene = std::make_shared<VulkanScene>();

    VulkanMeshPtr mesh = make_vulkan_mesh(vertices_ram, faces_ram);
    mesh->commit();

    //TODO: create one instance of the mesh

    return scene;
}

VulkanScenePtr make_vulkan_scene(const aiScene* ascene)
{
    VulkanScenePtr scene = std::make_shared<VulkanScene>();

    //NOTE: meshes entsprechen den bottom level acceleration structures
    // 1. meshes
    std::map<unsigned int, VulkanMeshPtr> meshes;
    std::cout << "[make_vulkan_scene()] Loading Meshes..." << std::endl;

    for(size_t i=0; i<ascene->mNumMeshes; i++)
    {
        std::cout << "Make Mesh " << i+1 << "/" << ascene->mNumMeshes << std::endl;
        const aiMesh* amesh = ascene->mMeshes[i];

        if(amesh->mPrimitiveTypes & aiPrimitiveType_TRIANGLE)
        {
            // triangle mesh
            VulkanMeshPtr mesh = make_vulkan_mesh(amesh);
            mesh->commit();
            meshes[i] = mesh;
            std::cout << "Mesh " << i << "(" << mesh->name << ") added." << std::endl;
        }
        else
        {
            std::cout << "[ make_vulkan_scene(aiScene) ] WARNING: Could not construct geometry " << i << " prim type " << amesh->mPrimitiveTypes << " not supported yet. Skipping." << std::endl;
        }
    }

    // //NOTE: nodes entsprechen den bottom level geometry instances (und sind somit die instanzen bottom level acceleration structures)
    // // 2. instances (if available)
    // std::unordered_set<VulkanGeometryPtr> instanciated_meshes;
    // const aiNode* root_node = ascene->mRootNode;
    // std::vector<const aiNode*> mesh_nodes = get_nodes_with_meshes(root_node);
    // for(size_t i=0; i<mesh_nodes.size(); i++)
    // {
    //     const aiNode* node = mesh_nodes[i];
        
    //     Matrix4x4 M = global_transform(node);
    //     Transform T;
    //     Vector3 scale;
    //     decompose(M, T, scale);

    //     VulkanScenePtr mesh_scene = std::make_shared<VulkanScene>();

    //     for(unsigned int i = 0; i<node->mNumMeshes; i++)
    //     {
    //         unsigned int mesh_id = node->mMeshes[i];
    //         auto mesh_it = meshes.find(mesh_id);
    //         if(mesh_it != meshes.end())
    //         {
    //             // mesh found
    //             VulkanMeshPtr mesh = mesh_it->second;
    //             instanciated_meshes.insert(mesh);
    //             mesh_scene->add(mesh);
    //             mesh_scene->commit();
    //         }
    //         else
    //         {
    //             std::cout << "[make_vulkan_scene()] WARNING: could not find mesh_id " 
    //                 << mesh_id << " in meshes during instantiation" << std::endl;
    //         }
    //     }

    //     mesh_scene->commit();

    //     // std::cout << "--- mesh added to mesh_scene" << std::endl;
    //     VulkanInstPtr mesh_instance = std::make_shared<VulkanInst>();
    //     mesh_instance->set(mesh_scene);
    //     mesh_instance->name = node->mName.C_Str();
    //     mesh_instance->setTransform(T);
    //     mesh_instance->setScale(scale);
    //     mesh_instance->apply();
    //     mesh_instance->commit();
    //     // std::cout << "--- mesh_instance created" << std::endl;
    //     unsigned int inst_id = scene->add(mesh_instance);
    //     // std::cout << "Instance " << inst_id << " (" << mesh_instance->name << ") added" << std::endl;
    // }

    // // ADD MESHES THAT ARE NOT INSTANCIATED
    // for(auto elem : meshes)
    // {
    //     auto mesh = elem.second;
    //     if(instanciated_meshes.find(mesh) == instanciated_meshes.end())
    //     {
    //         // mesh was never instanciated. add to scene
    //         if(scene->type() != VulkanSceneType::INSTANCES)
    //         {
    //             scene->add(mesh);
    //         }
    //         else
    //         {
    //             // mesh->instantiate();
    //             VulkanInstPtr geom_inst = mesh->instantiate();
    //             geom_inst->apply();
    //             geom_inst->commit();
    //             scene->add(geom_inst);
    //         }
    //     }
    // }

    return scene;
}

} // namespace rmagine
