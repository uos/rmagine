# Rmagine - Vulkan

Using Vulkan ray tracing shader as additional backend for Rmagine. This makes Rmagine compatible with any Vulkan-capable GPU and we can for the first time use the hardware acceleration on Jetson devices through the embedded RTX cores!

It introduces:

- Vulkan buffer as rmagine memory object `Memory<DEVICE_LOCAL_VULKAN>`
    - copyable to `Memory<CPU>`
    - copyable to `Memory<VRAM_CUDA>` via [rmagine_vulkan_cuda_interop](/src/rmagine_vulkan_cuda_interop).
- Vulkan Scene description (currently only static scenes are possible, TODO for the future)
- Vulkan Simulators (Spherical, Pinhole, O1Dn, OnDn)

## Requirements

- GLSLANG, Vulkan

```bash
sudo apt install glslang-dev glslang-tools libvulkan-dev
```

### Acknowledgements

This work was initially developed during the Bachelor's thesis of Fabian Geers at Osnabrück University under supervision of Thomas Wiemann and Alexander Mock.

