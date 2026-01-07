<div align="center">
<h1>
Rmagine
</h1>
<h4 align="center">Robots want to simulate too</h4>
</div>

<div align="center">
  <a href="https://github.com/uos/rmagine">Code</a>
  <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
  <a href="https://uos.github.io/rmagine_docs">Documentation</a>
  <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
  <a href="https://youtube.com/playlist?list=PL9wBuzh6ev07faQ13tXH9mhL5Wk6r34JM">Videos</a>
  <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
  <a href="https://github.com/uos/rmagine/issues">Issues</a>
  <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
  <a href="https://github.com/amock/rmagine_examples">Examples</a>
  <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
  <a href="https://github.com/amock/rmagine_viewer">Viewer</a>
  <br />
</div>

## Description

Rmagine is a library for fast and accurate simulation of range sensors in large 3D environments using ray tracing.
These simulations can also run on embedded devices mounted on a robot.
Rmagine has been specifically developed to:

- perform multiple sensor simulations simultaneously and in real time
- distribute computations across devices (CPU, GPU, RTX, …)
- keep data local to the device performing the computation
- minimize graphical overhead (off-screen rendering)
- support runtime-critical tasks


| Spherical, pinhole, or fully customizable models | Query multiple attributes at intersections |
|:----:|:----:|
|  ![rmagine_models_3d](dat/doc/sensor_models_3d.png) |   ![rmagine_attributes](dat/doc/simulation_attributes.png)   |

## Installation and Usage

Minimal instructions for installing the Rmagine library on your system:

### Dependencies

Rmagine depends on **TBB, Boost, Eigen, Assimp, and CMake**. Install them as follows:

<details>
<summary>Ubuntu</summary>

```bash
sudo apt install -y libtbb-dev libboost-dev libeigen3-dev libassimp-dev cmake
```

</details>

<details>
<summary>macOS</summary>

```bash
brew install tbb boost eigen assimp cmake
```

</details>

### Build and Install

Rmagine can be built either with a standard CMake workflow **or** by placing it directly into your ROS workspace. 

<details>
<summary>Standard CMake Build</summary>

```bash
mkdir -p rmagine/build
cd rmagine/build
cmake ..
make
```

```bash
make install
```
</details>

<details>
<summary>ROS Workspace</summary>

Download this library and place it into the `src` folder of your ROS workspace.

```bash
colcon build
```

</details>

For more advanced options and detailed instructions, visit the [Wiki](https://uos.github.io/rmagine_docs/installation/).


## Backends

Rmagine provides multiple computation backends that enable seamless switching between different hardware devices available on a robot. This makes it possible to either offload ray casting workloads to dedicated devices and free resources for other computations, or to utilize all available compute power for large-scale ray casting.

The supported backends differ in terms of device compatibility, acceleration capabilities, and platform availability.


| Backend    | Supported Devices                    | Acceleration         | Notes                                                                                                   |
| ---------- | ------------------------------------ | -------------------- | ------------------------------------------------------------------------------------------------------- |
| **Embree** | CPU                                  | Software (CPU)       | High-performance CPU-based ray tracing                                                                  |
| **OptiX**  | NVIDIA GPUs with OptiX support       | Hardware or software | Uses hardware ray tracing when available, otherwise software emulation; not available on Jetson devices |
| **Vulkan** | GPUs with Vulkan ray tracing support | Hardware ray tracing | Cross-vendor solution; supported on desktop GPUs and embedded platforms such as Jetson                  |


Each backend is compiled as an optional CMake component and is only enabled if all required dependencies are available on the system. Detailed instructions and further information on building the optional backends can be found in the [Wiki](https://uos.github.io/rmagine_docs).


## Example

This example demonstrates how to simulate ranges for 100 Velodyne VLP-16 sensors using the Embree backend.  
First, include the following headers:

```c++
// Map handling
#include <rmagine/map/EmbreeMap.hpp>
// Sensor models
#include <rmagine/types/sensor_models.h>
// Predefined sensor models (e.g. VLP-16)
#include <rmagine/types/sensors.h>
// Simulators
#include <rmagine/simulation/SphereSimulatorEmbree.hpp>

namespace rm = rmagine;
```

The following code loads the map `"my_mesh.ply"` and simulates 100 Velodyne VLP-16 scans from predefined poses. Hits and ranges are selected as intersection attributes:

```c++
// Load a map from disk
std::string path_to_mesh = "my_mesh.ply";
rm::EmbreeMapPtr map = rm::import_embree_map(path_to_mesh);

// Define a sensor model
// Here we use the predefined VLP-16 sensor model
rm::SphericalModel sensor_model = rm::vlp16_900();

// Construct a simulator, set sensor model and map
rm::SphereSimulatorEmbree sim;
sim.setModel(sensor_model);
sim.setMap(map);

// Create 100 transformations between base and map (robot poses)
rm::Memory<rm::Transform, rm::RAM> Tbm(100);
for(size_t i = 0; i < Tbm.size(); i++)
{
    rm::Transform T = rm::Transform::Identity();
    T.t = {2.0, 0.0, 0.0}; // Position (2, 0, 0)
    rm::EulerAngles e = {0.0, 0.0, 1.0}; // Orientation (0, 0, 1) radians
    T.R.set(e); // Euler angles internally converted to quaternion
    Tbm[i] = T;
}

// Add desired attributes at intersections
// Optimized at compile time
// Possible attributes (rmagine/simulation/SimulationResults.hpp):
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
// To query all attributes at once, use 'rm::IntAttrAll' instead of 'ResultT'

ResultT result = sim.simulate<ResultT>(Tbm);
// result.hits, result.ranges contain the resulting attribute buffers
std::cout << "First ray’s range: " << result.ranges[0] << std::endl;

// Slice the results for the scan at pose 5
auto ranges5 = result.ranges(5 * sensor_model.size(), 6 * sensor_model.size());
std::cout << "First ray’s range of the fifth scan: " << ranges5[0] << std::endl;
```

More detailed examples explaining each step and how to customize them are available in the [Wiki](https://uos.github.io/rmagine_docs).  
Additional examples can be found here: https://github.com/amock/rmagine_examples.

## Citation

Please reference the following paper when using the Rmagine library in your scientific work:

```bib
@inproceedings{mock2023rmagine,
  title     = {Rmagine: 3D Range Sensor Simulation in Polygonal Maps via Ray Tracing for Embedded Hardware on Mobile Robots}, 
  author    = {Mock, Alexander and Wiemann, Thomas and Hertzberg, Joachim},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)}, 
  year      = {2023},
  doi       = {10.1109/ICRA48891.2023.10161388}
}
```

The paper is available on [IEEE Xplore](https://ieeexplore.ieee.org/document/10161388) and as a preprint on [arXiv](https://arxiv.org/abs/2209.13397). For an overview of how to integrate this library into robotics applications we recommend the following work:

```bib
@phdthesis{amock2025inverse,
  title  = {Inverse Sensor Modeling for 6D Mobile Robot Localization in Scene Graphs via Hardware-Accelerated Ray Tracing},
  author = {Mock, Alexander},
  school = {Universität Osnabrück},
  year   = {2025},
  doi    = {10.48693/802}
}
```

which is open-access available to read [**here**](https://osnadocs.ub.uni-osnabrueck.de/handle/ds-2025112613801).

## Rmagine-Accelerated Applications
- [rmagine_viewer](https://github.com/amock/rmagine_viewer)
- [rmagine_gazebo_plugins](https://github.com/uos/rmagine_gazebo_plugins)
- [RMCL](https://github.com/uos/rmcl)
- [radarays_ros](https://github.com/uos/radarays_ros)

## Contributions

We welcome contributions of all kinds: issues, pull requests, and feedback. Feel free to help us improve this open-source project.  
If you would like to enhance the [documentation](https://uos.github.io/rmagine_docs/), whether by fixing typos or adding examples, please submit issues or pull requests at https://github.com/uos/rmagine_docs.

## Maintainment

Maintainers:
- [Alexander Mock](https://github.com/amock) (Nature Robots)

Contact the people listed above if you want and feel capable to help maintaining this piece of open-source software.
