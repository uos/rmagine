#include "rmagine/map/VulkanMap.hpp"



namespace rmagine
{

VulkanMap::VulkanMap() : VulkanEntity()
{

}

VulkanMap::VulkanMap(VulkanScenePtr scene) : VulkanEntity(), m_scene(scene)
{

}

VulkanMap::~VulkanMap()
{
    
}


void VulkanMap::setScene(VulkanScenePtr scene)
{
    if(scene->type() == VulkanSceneType::INSTANCES)
    {
        m_scene = scene;
    }
    else if(scene->type() == VulkanSceneType::GEOMETRIES)
    {
        throw std::runtime_error("[VulkanMap::setScene()] ERROR - Map needs a scene containing instances not one containing meshes.");
        // TODO: maybe create topLevelScene from bottomLevelScene
        // m_scene = std::make_shared<VulkanScene>();
        // VulkanInstPtr inst = scene->instantiate();
        // m_scene->add(inst);
    }
    else
    {
        throw std::runtime_error("[VulkanMap::setScene()] ERROR - This should never happen.");
    }
}


VulkanScenePtr VulkanMap::scene() const
{
    return m_scene;
}



VulkanMapPtr import_vulkan_map(Memory<Point, RAM>& vertices_ram, Memory<Face, RAM>& faces_ram)
{
    VulkanScenePtr scene = make_vulkan_scene(vertices_ram, faces_ram);
    scene->commit();
    return std::make_shared<VulkanMap>(scene);
}

VulkanMapPtr import_vulkan_map(const std::string& meshfile)
{
    AssimpIO io;
    // aiProcess_GenNormals does not work!
    const aiScene* ascene = io.ReadFile(meshfile, 0);

    if(!ascene)
    {
        std::cerr << io.Importer::GetErrorString() << std::endl;
    }

    if(!ascene->HasMeshes())
    {
        std::cerr << "ERROR: file '" << meshfile << "' contains no meshes" << std::endl;
    }

    VulkanScenePtr scene = make_vulkan_scene(ascene);
    scene->commit();
    return std::make_shared<VulkanMap>(scene);
}

} // namespace rmagine
