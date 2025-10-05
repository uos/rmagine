#pragma once

#define _USE_MATH_DEFINES // needed on some devices for some reason
#include <math.h>
#include <memory>

#include <vulkan/vulkan.h>



namespace rmagine
{

//forward declarations

class VulkanContext;
using VulkanContextPtr = std::shared_ptr<VulkanContext>;
using VulkanContextWPtr = std::weak_ptr<VulkanContext>;

class Device;
using DevicePtr = std::shared_ptr<Device>;
using DeviceWPtr = std::weak_ptr<Device>;

class DescriptorSetLayout;
using DescriptorSetLayoutPtr = std::shared_ptr<DescriptorSetLayout>;
using DescriptorSetLayoutWPtr = std::weak_ptr<DescriptorSetLayout>;

class RayTracingPipelineLayout;
using RayTracingPipelineLayoutPtr = std::shared_ptr<RayTracingPipelineLayout>;
using RayTracingPipelineLayoutWPtr = std::weak_ptr<RayTracingPipelineLayout>;

} // namespace rmagine
