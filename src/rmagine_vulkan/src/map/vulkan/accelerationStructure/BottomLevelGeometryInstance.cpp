#include "BottomLevelGeometryInstance.hpp"
#include "../../../util/VulkanContext.hpp"



namespace rmagine
{

void BottomLevelGeometryInstance::createBottomLevelAccelerationStructureInstance(VkTransformMatrixKHR transformMatrix, BottomLevelAccelerationStructurePtr bottomLevelAccelerationStructure)
{
    VkAccelerationStructureInstanceKHR bottomLevelAccelerationStructureInstance{};
    bottomLevelAccelerationStructureInstance.transform = transformMatrix;
    // bottomLevelAccelerationStructureInstance.transform = {.matrix = {{1.0, 0.0, 0.0, 0.0},
    //                                                                  {0.0, 1.0, 0.0, 0.0},
    //                                                                  {0.0, 0.0, 1.0, 0.0}}};
    bottomLevelAccelerationStructureInstance.mask = 0xFF;
    bottomLevelAccelerationStructureInstance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    bottomLevelAccelerationStructureInstance.accelerationStructureReference = bottomLevelAccelerationStructure->getDeviceAddress();

    Memory<VkAccelerationStructureInstanceKHR, RAM> geometryInstanceMemory_ram(1);
    geometryInstanceMemory_ram[0] = bottomLevelAccelerationStructureInstance;
    geometryInstanceMemory = geometryInstanceMemory_ram;
}

Memory<VkAccelerationStructureInstanceKHR, VULKAN_DEVICE_LOCAL>& BottomLevelGeometryInstance::getGeometryInstanceMemory()
{
    return geometryInstanceMemory;
}

void BottomLevelGeometryInstance::cleanup()
{
    // ...
}

} // namespace rmagine
