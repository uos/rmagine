#pragma once

#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>

#include <vulkan/vulkan.h>

#include "AccelerationStructure.hpp"
#include "BottomLevelGeometryInstance.hpp"
#include <rmagine/util/VulkanUtil.hpp>



namespace rmagine
{

class TopLevelAccelerationStructure final : public AccelerationStructure
{
private:
    size_t tlasID = 0;

public:
    TopLevelAccelerationStructure() : AccelerationStructure()
    {
        tlasID = getNewTlasID();
        std::cout << "New top level acceleration structure with tlasID: " << tlasID << std::endl;
    }

    TopLevelAccelerationStructure(DevicePtr device, ExtensionFunctionsPtr extensionFunctionsPtr) :
        AccelerationStructure(device, extensionFunctionsPtr)
    {
        tlasID = getNewTlasID();
        std::cout << "New top level acceleration structure with tlasID: " << tlasID << std::endl;
    }
    
    ~TopLevelAccelerationStructure(){}

    TopLevelAccelerationStructure(const TopLevelAccelerationStructure&) = delete;

    void createAccelerationStructure(BottomLevelGeometryInstancePtr bottomLevelGeometryInstance);

    size_t getID();

private:
    static size_t tlasIDcounter;

    static size_t getNewTlasID();
};

using TopLevelAccelerationStructurePtr = std::shared_ptr<TopLevelAccelerationStructure>;

} // namespace rmagine
