#include <iostream>
#include <sstream>
#include <cmath>

#include <rmagine/simulation/SphereSimulatorVulkan.hpp>
#include <rmagine/map/vulkan/vulkan_shapes.hpp>
#include <rmagine/map/VulkanMap.hpp>
#include <rmagine/types/sensors.h>
#include <rmagine/util/exceptions.h>

using namespace rmagine;

namespace
{

void check(const std::string& label, float got, float expected, float tol = 0.0005f)
{
    float error = std::fabs(got - expected);
    std::cout << label << ": got=" << got << ", expected=" << expected << std::endl;
    if(error > tol)
    {
        std::stringstream ss;
        ss << "[scene_update] " << label << " - error too high: " << error
           << " (got=" << got << ", expected=" << expected << ")";
        RM_THROW(VulkanException, ss.str());
    }
}

} // namespace

/**
 * @brief regression test for VulkanScene::commit()'s incremental (refit) update path
 *
 * Adding/removing geometry to/from a VulkanScene must trigger a full acceleration
 * structure rebuild, but moving existing instances/meshes around (no add/remove)
 * must instead refit the existing TLAS/BLAS in place - this is what makes rmagine's
 * Vulkan backend usable for scenes with moving objects (e.g. a gazebo plugin
 * updating object poses every tick) instead of paying for a full rebuild on every
 * single change, mirroring the existing Embree/OptiX backends.
 *
 * The cube is placed offset along the ray's travel direction (rather than at the
 * origin) so the sensor origin stays outside the cube for the whole test - if it
 * were inside, moving the cube across the origin would swap which face (entry vs.
 * exit) is hit first, which looks like a broken update but is actually just a
 * different, but equally correct, ray/geometry intersection.
 */
int main(int argc, char** argv)
{
    // mesh scene (BLAS) - stays untouched for the whole test: only the outer
    // scene's instance is moved around, so BLAS builds/updates are not exercised
    // here directly (they share the exact same build-vs-update logic as the TLAS).
    VulkanScenePtr mesh_scene = std::make_shared<VulkanScene>();
    auto cube = std::make_shared<VulkanCube>();
    VulkanGeometryPtr mesh = cube;
    mesh->commit();
    mesh_scene->add(mesh);
    mesh_scene->commit();

    // top level scene (TLAS) with a single instance we will move around.
    VulkanScenePtr scene = std::make_shared<VulkanScene>();
    VulkanInstPtr inst = mesh_scene->instantiate();
    Transform T_inst = Transform::Identity();
    T_inst.t.x = -2.0;
    inst->setTransform(T_inst);
    inst->apply();
    inst->commit();
    scene->add(inst);
    scene->commit(); // first commit -> full BUILD (no acceleration structure exists yet)

    VulkanMapPtr map = std::make_shared<VulkanMap>(scene);

    SphereSimulatorVulkan sim;
    sim.setMap(map);

    auto model = example_spherical();
    sim.setModel(model);
    sim.setTsb(Transform::Identity());

    IntAttrAll<DEVICE_LOCAL_VULKAN> result;
    resize_memory_bundle<DEVICE_LOCAL_VULKAN>(result, model.getWidth(), model.getHeight(), 1);

    Memory<Transform, RAM> T(1);
    T[0] = Transform::Identity();
    Memory<Transform, DEVICE_LOCAL_VULKAN> T_ = T;

    // ray at (phi.size/2, theta=0) of this spherical model travels along -x from the
    // sensor origin - moving the cube along -x therefore shifts the measured range 1:1.
    const size_t buffer_id = model.getBufferId(model.phi.size / 2, 0);

    auto center_range = [&]() -> float
    {
        sim.simulate(T_, result);
        Memory<float, RAM> scan = result.ranges(0, model.size());
        return scan[buffer_id];
    };

    check("initial BUILD (x=-2)", center_range(), 1.500076f);

    // single move -> UPDATE (refit) path: instance count is unchanged since the last commit
    T_inst.t.x -= 1.0; // -3
    inst->setTransform(T_inst);
    inst->apply();
    inst->commit();
    scene->commit();
    check("after single move -1 (x=-3), UPDATE", center_range(), 2.500076f);

    // BATCHED moves: apply()+commit() the instance several times, calling
    // scene->commit() only ONCE at the end - the classic "move many objects, commit
    // once" pattern used with Embree/OptiX. Only the state at commit time may matter;
    // intermediate per-instance apply()/commit() calls must not touch the AS at all.
    T_inst.t.x -= 1.0; // -4 (not committed to the scene yet)
    inst->setTransform(T_inst);
    inst->apply();
    inst->commit();

    T_inst.t.x -= 1.0; // -5
    inst->setTransform(T_inst);
    inst->apply();
    inst->commit();

    T_inst.t.x += 3.0; // back to -2, still not committed to the scene
    inst->setTransform(T_inst);
    inst->apply();
    inst->commit();

    scene->commit(); // single UPDATE reflecting only the final batched state (x=-2)
    check("after batched moves settling back to x=-2, single UPDATE", center_range(), 1.500076f);

    // forced full BUILD path once more (add+remove a throwaway instance) - must still
    // produce the correct result after a run of UPDATEs.
    VulkanInstPtr throwaway = mesh_scene->instantiate();
    throwaway->apply();
    throwaway->commit();
    scene->add(throwaway);
    scene->remove(throwaway->this_shared<VulkanGeometry>());
    scene->commit(); // added+removed since last commit -> full BUILD path
    check("after forced full rebuild (x=-2)", center_range(), 1.500076f);

    std::cout << "Done." << std::endl;
    return 0;
}
