/**
 * Regression test: different sensor types simulating on the SAME scene
 * with the SAME result bundle must not share a cached pipeline.
 *
 * Each sensor type is first simulated on its own fresh scene (reference).
 * Then all sensor types are simulated one after another on one shared scene.
 * The results must be identical to the references.
 */

#include <iostream>
#include <sstream>
#include <string>
#include <cmath>

#include <rmagine/simulation/SphereSimulatorOptix.hpp>
#include <rmagine/simulation/PinholeSimulatorOptix.hpp>
#include <rmagine/simulation/O1DnSimulatorOptix.hpp>
#include <rmagine/simulation/OnDnSimulatorOptix.hpp>
#include <rmagine/map/optix/optix_shapes.h>
#include <rmagine/map/OptixMap.hpp>
#include <rmagine/types/sensors.h>
#include <rmagine/util/exceptions.h>

using namespace rmagine;

static constexpr size_t N_POSES = 10;

OptixMapPtr make_map()
{
  OptixScenePtr scene = std::make_shared<OptixScene>();

  OptixGeometryPtr mesh = std::make_shared<OptixCube>();
  mesh->commit();
  scene->add(mesh);
  scene->commit();

  return std::make_shared<OptixMap>(scene);
}

Memory<Transform, VRAM_CUDA> make_poses()
{
  Memory<Transform, RAM> T(N_POSES);
  for(size_t i=0; i<T.size(); i++)
  {
    T[i] = Transform::Identity();
    // small offsets so that the poses differ but stay inside the cube
    const float s = static_cast<float>(i) / static_cast<float>(N_POSES);
    T[i].t = {0.2f * s - 0.1f, 0.1f * s - 0.05f, 0.05f * s};
  }
  return T;
}

template<typename SimT, typename ModelT>
Memory<float, RAM> simulate_ranges(
  const std::string& name,
  OptixMapPtr map,
  const ModelT& model,
  const Memory<Transform, VRAM_CUDA>& Tbm)
{
  std::cout << "- " << name << std::endl;
  SimT sim;
  sim.setMap(map);
  sim.setModel(model);

  using ResultT = Bundle<Ranges<VRAM_CUDA> >;
  ResultT res;
  resize_memory_bundle<VRAM_CUDA>(res, model.getWidth(), model.getHeight(), Tbm.size());

  sim.simulate(Tbm, res);
  cudaDeviceSynchronize();

  Memory<float, RAM> ranges = res.ranges;
  return ranges;
}

void compare(
  const std::string& name,
  const Memory<float, RAM>& reference,
  const Memory<float, RAM>& shared)
{
  if(reference.size() != shared.size())
  {
    std::stringstream ss;
    ss << name << ": size mismatch " << reference.size() << " vs " << shared.size();
    RM_THROW(OptixException, ss.str());
  }

  size_t n_mismatch = 0;
  size_t first_mismatch = 0;
  for(size_t i=0; i<reference.size(); i++)
  {
    const float a = reference[i];
    const float b = shared[i];
    const bool same = (std::isnan(a) && std::isnan(b)) || (a == b);
    if(!same)
    {
      if(n_mismatch == 0)
      {
        first_mismatch = i;
      }
      n_mismatch++;
    }
  }

  if(n_mismatch > 0)
  {
    std::stringstream ss;
    ss << name << ": " << n_mismatch << "/" << reference.size()
       << " ranges differ when simulating on a scene shared with other sensor types."
       << " First mismatch at " << first_mismatch << ": "
       << reference[first_mismatch] << " (reference) vs " << shared[first_mismatch] << " (shared)";
    RM_THROW(OptixException, ss.str());
  }

  std::cout << name << ": OK" << std::endl;
}

int main(int argc, char** argv)
{
  const auto sphere_model  = example_spherical();
  const auto pinhole_model = example_pinhole();
  const auto o1dn_model    = example_o1dn();
  const auto ondn_model    = example_ondn();

  const Memory<Transform, VRAM_CUDA> Tbm = make_poses();

  // 1. references: every sensor type on its own fresh scene
  std::cout << "Reference simulations (one scene per sensor type)" << std::endl;
  const auto sphere_ref  = simulate_ranges<SphereSimulatorOptix>("Sphere", make_map(), sphere_model, Tbm);
  const auto pinhole_ref = simulate_ranges<PinholeSimulatorOptix>("Pinhole", make_map(), pinhole_model, Tbm);
  const auto o1dn_ref    = simulate_ranges<O1DnSimulatorOptix>("O1Dn", make_map(), o1dn_model, Tbm);
  const auto ondn_ref    = simulate_ranges<OnDnSimulatorOptix>("OnDn", make_map(), ondn_model, Tbm);

  // 2. all sensor types one after another on one shared scene
  std::cout << "Shared-scene simulations (all sensor types on one scene)" << std::endl;
  OptixMapPtr shared_map = make_map();
  const auto sphere_shared  = simulate_ranges<SphereSimulatorOptix>("Sphere", shared_map, sphere_model, Tbm);
  const auto pinhole_shared = simulate_ranges<PinholeSimulatorOptix>("Pinhole", shared_map, pinhole_model, Tbm);
  const auto o1dn_shared    = simulate_ranges<O1DnSimulatorOptix>("O1Dn", shared_map, o1dn_model, Tbm);
  const auto ondn_shared    = simulate_ranges<OnDnSimulatorOptix>("OnDn", shared_map, ondn_model, Tbm);

  // 3. compare
  compare("Sphere",  sphere_ref,  sphere_shared);
  compare("Pinhole", pinhole_ref, pinhole_shared);
  compare("O1Dn",    o1dn_ref,    o1dn_shared);
  compare("OnDn",    ondn_ref,    ondn_shared);

  return 0;
}
