#include <iostream>

#include <rmagine/simulation/SphereSimulatorVulkan.hpp>
#include <rmagine/map/vulkan/vulkan_shapes.hpp>
#include <rmagine/map/VulkanMap.hpp>
#include <rmagine/types/sensors.h>
#include <rmagine/util/exceptions.h>


#include <stdexcept>
#include <cassert>

using namespace rmagine;

VulkanMapPtr make_map()
{
    // A raw mesh added directly to the top-level scene (no instance
    // wrapping) is not a valid top-level acceleration structure for
    // simulate() to trace against -- always wrap meshes in an instance,
    // same as make_vulkan_scene(const aiScene*) does.
    VulkanScenePtr mesh_scene = std::make_shared<VulkanScene>();
    auto cube = std::make_shared<VulkanCube>();
    VulkanGeometryPtr mesh = cube;
    mesh->commit();

    // DIAGNOSTIC: dump the raw vertex/face buffers to confirm the cube
    // geometry actually contains sane data.
    Memory<Point, RAM> verts_ram(cube->vertices.size());
    verts_ram = cube->vertices;
    Memory<Face, RAM> faces_ram(cube->faces.size());
    faces_ram = cube->faces;
    std::cout << "[DIAG] cube vertices: " << verts_ram.size()
              << ", faces: " << faces_ram.size() << std::endl;
    for(size_t i = 0; i < std::min<size_t>(8, verts_ram.size()); i++)
    {
      std::cout << "[DIAG] v" << i << " = (" << verts_ram[i].x << ", "
                << verts_ram[i].y << ", " << verts_ram[i].z << ")" << std::endl;
    }
    for(size_t i = 0; i < std::min<size_t>(4, faces_ram.size()); i++)
    {
      std::cout << "[DIAG] f" << i << " = (" << faces_ram[i].v0 << ", "
                << faces_ram[i].v1 << ", " << faces_ram[i].v2 << ")" << std::endl;
    }

    mesh_scene->add(mesh);
    mesh_scene->commit();

    VulkanScenePtr scene = std::make_shared<VulkanScene>();
    VulkanInstPtr inst = mesh_scene->instantiate();
    inst->apply();
    inst->commit();
    scene->add(inst);
    scene->commit();

    return std::make_shared<VulkanMap>(scene);
}

int main(int argc, char** argv)
{
    SphereSimulatorVulkan sim;

    // make synthetic map
    VulkanMapPtr map = make_map();
    sim.setMap(map);
    
    auto model = example_spherical();
    sim.setModel(model);
    sim.setTsb(Transform::Identity()); // DIAG: test whether Tsb defaults to zero instead of identity

    IntAttrAll<DEVICE_LOCAL_VULKAN> result;
    resize_memory_bundle<DEVICE_LOCAL_VULKAN>(result, model.getWidth(), model.getHeight(), 100);

    Memory<Transform, RAM> T(100);
    for(size_t i=0; i<T.size(); i++)
    {
      T[i] = Transform::Identity();
    }

    Memory<Transform, DEVICE_LOCAL_VULKAN> T_ = T;

    std::cout << "Simulate!" << std::endl;

    for(size_t i=0; i<100; i++)
    {
      sim.simulate(T_, result);

      if(i == 0)
      {
        Memory<uint8_t, RAM> hits_ram(model.size());
        hits_ram = result.hits(0, model.size());
        size_t n_hits = 0;
        for(size_t j = 0; j < hits_ram.size(); j++)
        {
          if(hits_ram[j] != 0) { n_hits++; }
        }
        std::cout << "[DIAG] hits: " << n_hits << " / " << hits_ram.size() << std::endl;
      }

      Memory<float, RAM> last_scan = result.ranges(
        model.size() * 99,
        model.size() * 100
      );

      float range = last_scan[model.getBufferId((model.phi.size) / 2, 0)];
      std::cout << range << std::endl;
      float error = std::fabs(range - 0.500076);
                                                      
      if(error > 0.0001)                                              
      {                             
        std::stringstream ss;
        ss << "Simulated scan error is too high: " << error;
        RM_THROW(VulkanException, ss.str());
      }
    }

    std::cout << "Done simulating." << std::endl;

    return 0;
}