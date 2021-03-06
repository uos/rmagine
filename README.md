[[Code](https://github.com/uos/rmagine)] [[Wiki](https://github.com/uos/rmagine/wiki)]

# rmagine - Fast Depth-Sensor simulation in 3D environments

Library for fast sensor data simulation in large 3D environments.

## Design Goals

Mainly designed for robotic applications:

- Perform multiple sensor simulations simultaneously and in realtime
- Perform computations at specific computing devices (CPU, GPU, RTX..)
    - Hold data at device of computation
    - Minimal graphical overhead (offscreen-rendering)
- Runtime critical operations

## Sensor Models

Spherical, Pinhole or fully customizable models.

![rmagine_models_3d](dat/doc/sensor_models_3d.png)

![rmagine_models_ortho](dat/doc/sensor_models_ortho.png)

## Intersection Attributes

Query several attributes at intersection.

![rmagine_attributes](dat/doc/simulation_attributes.png)

## Usage

See [Wiki](https://github.com/uos/rmagine/wiki) page for further instructions.

## Code
Is available on Github: [rmagine](https://github.com/uos/rmagine)
