#include "rmagine/map/VulkanMap.hpp"
#include "VulkanMap.hpp"



namespace rmagine
{

VulkanMap::VulkanMap() : VulkanEntity()
{}

VulkanMap::VulkanMap(VulkanScenePtr scene) : VulkanEntity(), m_scene(scene)
{}

VulkanMap::~VulkanMap()
{
    std::cout << "destroying VulkanMap" << std::endl;
    cleanup();
}


void VulkanMap::setScene(VulkanScenePtr scene)
{
    m_scene = scene;
}


VulkanScenePtr VulkanMap::scene() const
{
    return m_scene;
}


void VulkanMap::cleanup()
{
    std::cout << "cleaning up..." << std::endl;

    m_scene->cleanup();
    std::cout << "cleaned up scene." << std::endl;

    std::cout << "done." << std::endl;
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
