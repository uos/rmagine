#include "rmagine/map/vulkan/VulkanScene.hpp"
#include "VulkanScene.hpp"



namespace rmagine
{
VulkanScene::VulkanScene() : VulkanEntity()
{

}

VulkanScene::~VulkanScene()
{
    std::cout << "destroying VulkanScene" << std::endl;
    cleanup();
}


unsigned int VulkanScene::add(VulkanGeometryPtr geom)
{
    auto it = m_ids.find(geom);
    if(it == m_ids.end())
    {
        if(m_geometries.empty())
        {
            // first element
            m_geom_type = geom->type();
            if(m_geom_type == VulkanGeometryType::INSTANCE)
            {
                m_type = VulkanSceneType::INSTANCES;
            }
            else
            {
                m_type = VulkanSceneType::GEOMETRIES;
            }
        }

        if(geom->type() != m_geom_type)
        {
            std::cout << "[VulkanScene::add()] WARNING - used mixed types of geometries. NOT supported" << std::endl;
            throw std::runtime_error("[VulkanScene::add()] WARNING - used mixed types of geometries. NOT supported");//just in case
        }

        // add geom to self
        unsigned int id = gen.get();
        m_geometries[id] = geom;
        m_ids[geom] = id;

        // add self to geom
        geom->addParent(this_shared<VulkanScene>());

        m_geom_added = true;

        return id;
    }
    else
    {
        return it->second;
    }
}

unsigned int VulkanScene::get(VulkanGeometryPtr geom) const
{
    return m_ids.at(geom);
}

std::optional<unsigned int> VulkanScene::getOpt(VulkanGeometryPtr geom) const
{
    auto it = m_ids.find(geom);
    if(it != m_ids.end())
    {
        return it->second;
    }

    return {};
}

bool VulkanScene::has(VulkanGeometryPtr geom) const
{
    return (m_ids.find(geom) != m_ids.end());
}

bool VulkanScene::has(unsigned int geom_id) const
{
    return (m_geometries.find(geom_id) != m_geometries.end());
}

bool VulkanScene::remove(VulkanGeometryPtr geom)
{
    bool ret = false;

    auto it = m_ids.find(geom);
    if(it != m_ids.end())
    {
        unsigned int geom_id = it->second;

        m_ids.erase(it);
        m_geometries.erase(geom_id);
        geom->removeParent(this_shared<VulkanScene>());
        gen.give_back(geom_id);

        m_geom_removed = true;
        ret = true;
    }

    return ret;
}

VulkanGeometryPtr VulkanScene::remove(unsigned int geom_id)
{
    VulkanGeometryPtr ret;

    auto it = m_geometries.find(geom_id);
    if(it != m_geometries.end())
    {
        VulkanGeometryPtr geom = it->second;

        m_geometries.erase(it);
        m_ids.erase(geom);
        geom->removeParent(this_shared<VulkanScene>());
        gen.give_back(geom_id);

        m_geom_removed = true;

        ret = geom;
    }

    return ret;
}

std::map<unsigned int, VulkanGeometryPtr> VulkanScene::geometries() const
{
    return m_geometries;
}

std::unordered_map<VulkanGeometryPtr, unsigned int> VulkanScene::ids() const
{
    return m_ids;
}

void VulkanScene::commit()
{
    std::vector<VkAccelerationStructureGeometryKHR> accelerationStructureGeometrys;
    std::vector<VkAccelerationStructureBuildRangeInfoKHR> accelerationStructureBuildRangeInfos;

    for (auto const& geometry : m_geometries)
    {
        accelerationStructureGeometrys.push_back(geometry->getASGeometry());
        accelerationStructureBuildRangeInfos.push_back(geometry->getASBuildRangeInfo());
    }
    
    if(m_type == VulkanSceneType::INSTANCES)
    {
        // create top level AS
        // get all the instances and add the to the top level AS
        m_as = std::make_shared<TopLevelAccelerationStructure>();
        m_as->createAccelerationStructure(accelerationStructureGeometrys, accelerationStructureBuildRangeInfos);
    }
    else if(m_type == VulkanSceneType::GEOMETRIES)
    {
        // create bottom level AS
        // get all the meshes and add the to the bottom level AS
        m_as = std::make_shared<BottomLevelAccelerationStructure>();
        m_as->createAccelerationStructure(accelerationStructureGeometrys, accelerationStructureBuildRangeInfos);
    }
}

VulkanInstPtr VulkanScene::instantiate()
{
    VulkanInstPtr ret = std::make_shared<VulkanInst>();
    ret->set(this_shared<VulkanScene>());
    return ret;
}

void VulkanScene::cleanupParents()
{
    for(auto it = m_parents.cbegin(); it != m_parents.cend();)
    {
        if (it->expired())
        {
            m_parents.erase(it++);    // or "it = m.erase(it)" since C++11
        } else {
            ++it;
        }
    }
}

std::unordered_set<VulkanInstPtr> VulkanScene::parents() const
{
    std::unordered_set<VulkanInstPtr> ret;

    for(VulkanInstWPtr elem : m_parents)
    {
        if(auto tmp = elem.lock())
        {
            ret.insert(tmp);
        }
    }
    
    return ret;
}

void VulkanScene::addParent(VulkanInstPtr parent)
{
    m_parents.insert(parent);
}


void VulkanScene::cleanup()
{
    std::cout << "cleaning up..." << std::endl;

    m_as->cleanup();
    std::cout << "cleaned up acceleration structure." << std::endl;

    std::cout << "done." << std::endl;
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
