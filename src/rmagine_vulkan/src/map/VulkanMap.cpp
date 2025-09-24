#include "rmagine/map/VulkanMap.hpp"



namespace rmagine
{

VulkanMap::VulkanMap() : VulkanEntity()
{

}

VulkanMap::VulkanMap(VulkanScenePtr scene) : VulkanEntity()
{
    setScene(scene);
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
        // create an instance-scene from the mesh-scene
        // by instanitating the mesh-scene and adding it to a newly created instance-scene

        VulkanInstPtr inst = scene->instantiate();
        inst->apply();
        inst->commit();

        m_scene = std::make_shared<VulkanScene>();
        m_scene->add(inst);
        m_scene->commit();
    }
    else
    {
        throw std::invalid_argument("[VulkanMap::setScene()] ERROR - invalid scene type, this should never happen.");
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
