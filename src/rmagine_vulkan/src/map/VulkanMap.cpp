#include "VulkanMap.hpp"
#include "../util/VulkanContext.hpp"



namespace rmagine
{

void VulkanMap::cleanup()
{
    std::cout << "cleaning up..." << std::endl;

    m_scene->cleanup();
    std::cout << "cleaned up scene." << std::endl;

    std::cout << "done." << std::endl;
}


VulkanScenePtr VulkanMap::scene() const
{
    return m_scene;
}



VulkanMapPtr import_vulkan_map(Memory<float, RAM>& vertexMem_ram, Memory<uint32_t, RAM>& indexMem_ram)
{
    VulkanScenePtr scene = std::make_shared<VulkanScene>(vertexMem_ram, indexMem_ram);
    VulkanMapPtr map = std::make_shared<VulkanMap>(scene);
    return map;
}

// VulkanMapPtr import_vulkan_map(const std::string& meshfile)
// {
//     // (parts) taken from OptixMap.hpp

//     AssimpIO io;
//     // aiProcess_GenNormals does not work!
//     const aiScene* ascene = io.ReadFile(meshfile, 0);

//     if(!ascene)
//     {
//         std::cerr << io.Importer::GetErrorString() << std::endl;
//     }

//     if(!ascene->HasMeshes())
//     {
//         std::cerr << "ERROR: file '" << meshfile << "' contains no meshes" << std::endl;
//     }

//     VulkanScenePtr scene = make_vulkan_scene(ascene);
//     scene->commit();
//     return std::make_shared<VulkanMap>(scene);
// }

} // namespace rmagine
