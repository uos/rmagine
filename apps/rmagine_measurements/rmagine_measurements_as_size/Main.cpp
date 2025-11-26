#include <filesystem>

// Core rmagine includes
#include <rmagine/types/Memory.hpp>
#include <rmagine/map/OptixMap.hpp>
#include <rmagine/types/MemoryCuda.hpp>
#include <rmagine/map/optix/optix_shapes.h>
#include <rmagine/map/VulkanMap.hpp>
#include <rmagine/types/MemoryVulkan.hpp>
#include <rmagine/map/vulkan/vulkan_shapes.hpp>




using namespace rmagine;


VulkanMapPtr make_sphere_map_vulkan(unsigned int num_long, unsigned int num_lat)
{
    VulkanScenePtr scene = std::make_shared<VulkanScene>();

    VulkanMeshPtr mesh = std::make_shared<VulkanSphere>(num_long, num_lat);
    mesh->commit();
    scene->add(mesh);
    scene->commit();

    return std::make_shared<VulkanMap>(scene);
}

OptixMapPtr make_sphere_map_optix(unsigned int num_long, unsigned int num_lat)
{
    OptixScenePtr scene = std::make_shared<OptixScene>();

    OptixMeshPtr mesh = std::make_shared<OptixSphere>(num_long, num_lat);
    mesh->commit();
    scene->add(mesh);
    scene->commit();

    return std::make_shared<OptixMap>(scene);
}


size_t num_maps = 20;
size_t map_param = 500;

int main(int argc, char** argv)
{
    //measure different num of faces
    std::vector<std::string> results_vulkan;
    std::vector<std::string> results_optix;
    std::vector<std::string> mesh_sizes_vert;
    std::vector<std::string> mesh_sizes_ind;
    std::vector<std::string> mesh_sizes_total;
    for(size_t i = 0; i <= num_maps; i++)
    {
        unsigned int num_lon_and_lat = static_cast<unsigned int>(static_cast<double>(map_param)*sqrt(static_cast<double>(i)));
        if(i == 0)
            num_lon_and_lat = 10;
        
        {
            VulkanMapPtr map_vulkan = make_sphere_map_vulkan(num_lon_and_lat, num_lon_and_lat);
            std::cout << "Num Faces Vulkan: " << map_vulkan->scene()->geometries().begin()->second->this_shared<VulkanInst>()->scene()->geometries().begin()->second->this_shared<VulkanMesh>()->faces.size() << std::endl;
            size_t map_size_bytes_vulkan = map_vulkan->scene()->getAsSize();

            std::stringstream point_vulkan;
            point_vulkan << "(" << static_cast<double>(map_vulkan->scene()->geometries().begin()->second->this_shared<VulkanInst>()->scene()->geometries().begin()->second->this_shared<VulkanMesh>()->faces.size())/1000000.0 << ", " << map_size_bytes_vulkan << ")";
            results_vulkan.push_back(point_vulkan.str());
        }

        {
            OptixMapPtr map_optix = make_sphere_map_optix(num_lon_and_lat, num_lon_and_lat);
            std::cout << "Num Faces Optix:  " << map_optix->scene()->geometries().begin()->second->this_shared<OptixMesh>()->faces.size() << std::endl;
            size_t map_size_bytes_optix = map_optix->scene()->getAsSize();

            std::stringstream point_optix;
            point_optix << "(" << static_cast<double>(map_optix->scene()->geometries().begin()->second->this_shared<OptixMesh>()->faces.size())/1000000.0 << ", " << map_size_bytes_optix << ")";
            results_optix.push_back(point_optix.str());

            //these two are the same for vulkan and optix:
            uint64_t vert_size = map_optix->scene()->geometries().begin()->second->this_shared<OptixMesh>()->vertices.size()*sizeof(rmagine::Point);
            uint64_t ind_size = map_optix->scene()->geometries().begin()->second->this_shared<OptixMesh>()->faces.size()*sizeof(rmagine::Face);
            uint64_t total_size = vert_size + ind_size;

            std::stringstream point_vert;
            point_vert << "(" << static_cast<double>(map_optix->scene()->geometries().begin()->second->this_shared<OptixMesh>()->faces.size())/1000000.0 << ", " << vert_size << ")";
            mesh_sizes_vert.push_back(point_vert.str());

            std::stringstream point_ind;
            point_ind << "(" << static_cast<double>(map_optix->scene()->geometries().begin()->second->this_shared<OptixMesh>()->faces.size())/1000000.0 << ", " << ind_size << ")";
            mesh_sizes_ind.push_back(point_ind.str());

            std::stringstream point_total;
            point_total << "(" << static_cast<double>(map_optix->scene()->geometries().begin()->second->this_shared<OptixMesh>()->faces.size())/1000000.0 << ", " << total_size << ")";
            mesh_sizes_total.push_back(point_total.str());
        }
    }
    std::cout << std::endl;
    std::cout << "Vulkan:" << std::endl;
    for(size_t i = 0; i < results_vulkan.size(); i++)
    {
        std::cout << results_vulkan[i];
    }
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "Optix:" << std::endl;
    for(size_t i = 0; i < results_optix.size(); i++)
    {
        std::cout << results_optix[i];
    }
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "Vertex Buffer Sizes:" << std::endl;
    for(size_t i = 0; i < mesh_sizes_vert.size(); i++)
    {
        std::cout << mesh_sizes_vert[i];
    }
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "Face Buffer Sizes:" << std::endl;
    for(size_t i = 0; i < mesh_sizes_ind.size(); i++)
    {
        std::cout << mesh_sizes_ind[i];
    }
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "Total Mesh Sizes:" << std::endl;
    for(size_t i = 0; i < mesh_sizes_total.size(); i++)
    {
        std::cout << mesh_sizes_total[i];
    }
    std::cout << std::endl;

    std::cout << "\nFinished." << std::endl;

    return EXIT_SUCCESS;
}
