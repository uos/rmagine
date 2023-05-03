[[Code](https://github.com/uos/rmagine)] [[Wiki](https://github.com/uos/rmagine/wiki)]

# Rmagine - 3D Range Sensor Simulation in Polygonal Maps via Raytracing for Embedded Hardware on Mobile Robots

Library for fast sensor data simulation in large 3D environments.

## Overview

Mainly designed for robotic applications:

- Perform multiple sensor simulations simultaneously and in realtime
- Perform computations at specific computing devices (CPU, GPU, RTX..)
    - Hold data at device of computation
    - Minimal graphical overhead (offscreen-rendering)
- Runtime critical operations


| Spherical, Pinhole or fully customizable models. | Query several attributes at intersection. |
|:----:|:----:|
|  ![rmagine_models_3d](dat/doc/sensor_models_3d.png) |   ![rmagine_attributes](dat/doc/simulation_attributes.png)   |

## Installation / Usage

See [Wiki](https://github.com/uos/rmagine/wiki) page for further instructions.

## Code
Is available on Github: [rmagine](https://github.com/uos/rmagine)

## Rmagine-accelerated Applications
- [rmagine_gazebo_plugins](https://github.com/uos/rmagine_gazebo_plugins)
- [RMCL](https://github.com/uos/rmcl)

## Publications

Please reference the following papers when using the Rmagine library in your scientific work.

**Rmagine: 3D Range Sensor Simulation in Polygonal Maps via Raytracing for Embedded Hardware on Mobile Robots**
- State: Accepted at ICRA 2023 - Citation is updated soon
- Preprint: https://arxiv.org/abs/2209.13397


## Example

This examples shows how to simulate ranges of 100 Velodyne VLP-16 sensor using Embree backbone. First, the following headers need to be included:


```c++
// Map
#include <rmagine/map/EmbreeMap.hpp>
// Sensor Models
#include <rmagine/types/sensor_models.h>
// Simulators
#include <rmagine/simulation/SphereSimulatorEmbree.hpp>

namespace rm = rmagine;
```

The following code loads a map "my_mesh.ply" and simulates 100 Velodyne VLP-16 scans from certain predefined poses. Hits and Ranges are chosen as return attributes.

```c++
// loading a map from disk
std::string path_to_mesh = "my_mesh.ply";
rm::EmbreeMapPtr map = rm::load_embree_map(path_to_mesh);

// defining a model
rm::SphericalModel velo_model = rm::vlp16_900();

// construct a simulator
rm::SphereSimulatorEmbree sim;
sim.setMap(map);
sim.setModel(velo_model);

// 100 Transformations between base and map. e.g. poses of the robot
rm::Memory<rm::Transform, rm::RAM> Tbm(100);

for(size_t i=0; i < Tbm.size(); i++)
{
    rm::Transform T = rm::Transform::Identity();
    T.t = {2.0, 0.0, 0.0}; // position (2,0,0)
    rm::EulerAngles e = {0.0, 0.0, 1.0}; // orientation (0,0,1) radian - as euler angles
    T.R.set(e); // euler internally converted to quaternion
    Tbm[i] = T; // Write Transform/Pose to Memory
}

// add your desired attributes at intersection here
// - optimizes the code at compile time
// Possible Attributes (rmagine/simulation/SimulationResults.hpp):
// - Hits
// - Ranges
// - Points
// - Normals
// - FaceIds
// - GeomIds
// - ObjectIds
using ResultT = rm::Bundle<
    rm::Hits<rm::RAM>, 
    rm::Ranges<rm::RAM>
>;
// querying every attribute with 'rm::IntAttrAny' instead of 'ResultT'

ResultT result = sim.simulate<ResultT>(poses);
// result.hits, result.ranges contain the resulting attribute buffers
std::cout << "printing the first ray's range: " << result.ranges[0] << std::endl;

// or slice the results for the scan of pose 5
auto ranges5 = result.ranges(5 * model.size(), 6 * model.size());
std::cout << "printing the first ray's range of the fifth scan: " << ranges5[0] << std::endl;
```

More detailed examples explaining each step and how to customize it to your needs are explained in the [Wiki](https://github.com/uos/rmagine/wiki).
