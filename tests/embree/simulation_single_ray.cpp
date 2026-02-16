#include <iostream>

#include <rmagine/simulation/OnDnSimulatorEmbree.hpp>
#include <rmagine/map/embree/embree_shapes.h>
#include <rmagine/map/EmbreeMap.hpp>
#include <rmagine/types/sensors.h>
#include <rmagine/util/exceptions.h>
#include <rmagine/util/prints.h>

using namespace rmagine;

EmbreeMapPtr make_map()
{
  EmbreeScenePtr scene = std::make_shared<EmbreeScene>();

  EmbreeGeometryPtr mesh = std::make_shared<EmbreeCube>();
  mesh->commit();
  scene->add(mesh);
  scene->commit();

  return std::make_shared<EmbreeMap>(scene);
}

int main(int argc, char** argv)
{
  OnDnSimulatorEmbree sim;

  // make synthetic map
  EmbreeMapPtr map = make_map();
  sim.setMap(map);
  
  OnDnModel model;
  model.dirs.resize(1);
  model.dirs[0] = {1.0, 0.0, 0.0};
  model.origs.resize(1);
  model.origs[0] = {0.0, 0.0, 0.0};
  model.width = 1;
  model.height = 1;
  model.range.min = 0.0;
  model.range.max = 100.0;
  sim.setModel(model);

  IntAttrAll<RAM> result;
  resize_memory_bundle<RAM>(result, model.getWidth(), model.getHeight(), 100);

  Memory<Transform, RAM> T(100);
  for(size_t i=0; i<T.size(); i++)
  {
    T[i] = Transform::Identity();
  }

  std::cout << "Simulate!" << std::endl;
  
  auto res = sim.simulate<IntAttrAll<RAM> >(T);

  float range = res.ranges[0];
  std::cout << "Range: " << res.ranges[0] << std::endl;

  float error = std::fabs(range - 0.5);
  if(error > 0.0001)                                              
  {
    std::stringstream ss;
    ss << "Simulated scan error is too high. " << range << " (simulated) vs. " << 0.5 << " (expected) (error: " << error << ")";
    RM_THROW(EmbreeException, ss.str());
  }

  std::cout << "Done simulating." << std::endl;

  return 0;
}