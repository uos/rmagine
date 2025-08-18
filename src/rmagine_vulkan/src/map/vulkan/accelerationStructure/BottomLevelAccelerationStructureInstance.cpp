#include "rmagine/map/vulkan/accelerationStructure/BottomLevelAccelerationStructureInstance.hpp"
#include "rmagine/util/VulkanContext.hpp"



namespace rmagine
{

void BottomLevelAccelerationStructureInstance::createBottomLevelAccelerationStructureInstance(BottomLevelAccelerationStructurePtr bottomLevelAccelerationStructure)
{
    VkTransformMatrixKHR transformMatrix = {{{1.0, 0.0, 0.0, 0.0},
                                             {0.0, 1.0, 0.0, 0.0},
                                             {0.0, 0.0, 1.0, 0.0}}};
    createBottomLevelAccelerationStructureInstance(transformMatrix, 0xFF, bottomLevelAccelerationStructure);
}

void BottomLevelAccelerationStructureInstance::createBottomLevelAccelerationStructureInstance(VkTransformMatrixKHR transformMatrix, uint32_t mask, BottomLevelAccelerationStructurePtr bottomLevelAccelerationStructure)
{
    VkAccelerationStructureInstanceKHR bottomLevelAccelerationStructureInstance{};
    bottomLevelAccelerationStructureInstance.transform = transformMatrix;
    bottomLevelAccelerationStructureInstance.mask = mask;
    bottomLevelAccelerationStructureInstance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    bottomLevelAccelerationStructureInstance.accelerationStructureReference = bottomLevelAccelerationStructure->getDeviceAddress();

    instanceMemory_ram[0] = bottomLevelAccelerationStructureInstance;
    instanceMemory = instanceMemory_ram;
}


void BottomLevelAccelerationStructureInstance::updateTransformMatrix(VkTransformMatrixKHR transformMatrix)
{
    instanceMemory_ram[0].transform = transformMatrix;
    instanceMemory = instanceMemory_ram;
}


void BottomLevelAccelerationStructureInstance::updateMask(uint32_t mask)
{
    instanceMemory_ram[0].mask = mask;
    instanceMemory = instanceMemory_ram;
}


void BottomLevelAccelerationStructureInstance::updateTransformMatrixAndMask(VkTransformMatrixKHR transformMatrix, uint32_t mask)
{
    instanceMemory_ram[0].transform = transformMatrix;
    instanceMemory_ram[0].mask = mask;
    instanceMemory = instanceMemory_ram;
}


Memory<VkAccelerationStructureInstanceKHR, VULKAN_DEVICE_LOCAL> &BottomLevelAccelerationStructureInstance::getInstanceMemory()
{
    return instanceMemory;
}


void BottomLevelAccelerationStructureInstance::cleanup()
{
    // ...
}

} // namespace rmagine
