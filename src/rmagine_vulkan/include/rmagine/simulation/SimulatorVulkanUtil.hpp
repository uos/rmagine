#include <rmagine/math/types.h>
#include <rmagine/types/Bundle.hpp>
#include <rmagine/simulation/SimulationResults.hpp>
#include <rmagine/types/MemoryVulkan.hpp>
#include <rmagine/util/vulkan/ShaderUtil.hpp>



namespace rmagine
{

template<typename MemT>
using PrimitiveIds = FaceIds<MemT>;

template<typename MemT>
using GeometryIds = GeomIds<MemT>;

template<typename MemT>
using InstanceIds = ObjectIds<MemT>;



struct VulkanResultsAddresses
{
    //TODO: maybe add mapData address here as well

    VkDeviceAddress hitsAddress = 0;
    VkDeviceAddress rangesAddress = 0;
    VkDeviceAddress pointsAddress = 0;
    VkDeviceAddress normalsAddress = 0;
    VkDeviceAddress primitiveIdAddress = 0;
    VkDeviceAddress instanceIdAddress = 0;
    VkDeviceAddress geometryIdAddress = 0;
};


struct VulkanTbmAndSensorSpecificAddresses
{
    VkDeviceAddress tbmAddress = 0;
    
    VkDeviceAddress origsAddress = 0;
    VkDeviceAddress dirsAddress = 0;
};


struct VulkanDimensions
{
    uint64_t width = 0;
    uint64_t height = 0;
    uint64_t depth = 0;
}



template<typename BundleT>
static void set_vulkan_results_data(BundleT& res, VulkanResultsAddresses& mem)
{
    if constexpr(BundleT::template has<Hits<VULKAN_DEVICE_LOCAL> >())
    {
        mem.hitsAddress = res.Hits<VULKAN_DEVICE_LOCAL>::hits.getBuffer()->getBufferDeviceAddress();
    }

    if constexpr(BundleT::template has<Ranges<VULKAN_DEVICE_LOCAL> >())
    {
        mem.rangesAddress = res.Ranges<VULKAN_DEVICE_LOCAL>::ranges.getBuffer()->getBufferDeviceAddress();
    }

    if constexpr(BundleT::template has<Points<VULKAN_DEVICE_LOCAL> >())
    {
        mem.pointsAddress = res.Points<VULKAN_DEVICE_LOCAL>::points.getBuffer()->getBufferDeviceAddress();
    }

    if constexpr(BundleT::template has<Normals<VULKAN_DEVICE_LOCAL> >())
    {
        mem.normalsAddress = res.Normals<VULKAN_DEVICE_LOCAL>::normals.getBuffer()->getBufferDeviceAddress();
    }

    if constexpr(BundleT::template has<FaceIds<VULKAN_DEVICE_LOCAL> >())
    {
        mem.primitiveIdAddress = res.FaceIds<VULKAN_DEVICE_LOCAL>::face_ids.getBuffer()->getBufferDeviceAddress();
    }

    if constexpr(BundleT::template has<GeomIds<VULKAN_DEVICE_LOCAL> >())
    {
        mem.geometryIdAddress = res.GeomIds<VULKAN_DEVICE_LOCAL>::geom_ids.getBuffer()->getBufferDeviceAddress();
    }

    if constexpr(BundleT::template has<ObjectIds<VULKAN_DEVICE_LOCAL> >())
    {
        mem.instanceIdAddress = res.ObjectIds<VULKAN_DEVICE_LOCAL>::object_ids.getBuffer()->getBufferDeviceAddress();
    }
}


template<typename BundleT>
static bool check_vulkan_bundle_sizes(BundleT& res, size_t size)
{
    if constexpr(BundleT::template has<Hits<VULKAN_DEVICE_LOCAL> >())
    {
        if(res.hits.size() != 0 && res.hits.size() < size)
        {
            return false;
        }
    }

    if constexpr(BundleT::template has<Ranges<VULKAN_DEVICE_LOCAL> >())
    {
        if(res.ranges.size() != 0 && res.ranges.size() < size)
        {
            return false;
        }
    }

    if constexpr(BundleT::template has<Points<VULKAN_DEVICE_LOCAL> >())
    {
        if(res.points.size() != 0 && res.points.size() < size)
        {
            return false;
        }
    }

    if constexpr(BundleT::template has<Normals<VULKAN_DEVICE_LOCAL> >())
    {
        if(res.normals.size() != 0 && res.normals.size() < size)
        {
            return false;
        }
    }

    if constexpr(BundleT::template has<FaceIds<VULKAN_DEVICE_LOCAL> >())
    {
        //primitiveID
        if(res.face_ids.size() != 0 && res.face_ids.size() < size)
        {
            return false;
        }
    }

    if constexpr(BundleT::template has<GeomIds<VULKAN_DEVICE_LOCAL> >())
    {
        //geometryID
        if(res.geom_ids.size() != 0 && res.geom_ids.size() < size)
        {
            return false;
        }
    }

    if constexpr(BundleT::template has<ObjectIds<VULKAN_DEVICE_LOCAL> >())
    {
        //instanceID
        if(res.object_ids.size() != 0 && res.object_ids.size() < size)
        {
            return false;
        }
    }

    return true;
}


// template<typename BundleT>
// static ShaderDefineFlags get_result_flags(BundleT& res)
// {
//     ShaderDefineFlags resultFlags = 0;

//     if constexpr(BundleT::template has<Hits<VULKAN_DEVICE_LOCAL> >())
//     {
//         if(res.hits.size() != 0)
//         {
//             resultFlags = resultFlags | ShaderDefines::Def_Hits;
//         }
//     }

//     if constexpr(BundleT::template has<Ranges<VULKAN_DEVICE_LOCAL> >())
//     {
//         if(res.ranges.size() != 0)
//         {
//             resultFlags = resultFlags | ShaderDefines::Def_Ranges;
//         }
//     }

//     if constexpr(BundleT::template has<Points<VULKAN_DEVICE_LOCAL> >())
//     {
//         if(res.points.size() != 0)
//         {
//             resultFlags = resultFlags | ShaderDefines::Def_Points;
//         }
//     }

//     if constexpr(BundleT::template has<Normals<VULKAN_DEVICE_LOCAL> >())
//     {
//         if(res.normals.size() != 0)
//         {
//             resultFlags = resultFlags | ShaderDefines::Def_Normals;
//         }
//     }

//     if constexpr(BundleT::template has<FaceIds<VULKAN_DEVICE_LOCAL> >())
//     {
//         //primitiveID
//         if(res.face_ids.size() != 0)
//         {
//             resultFlags = resultFlags | ShaderDefines::Def_PrimitiveID;
//         }
//     }

//     if constexpr(BundleT::template has<GeomIds<VULKAN_DEVICE_LOCAL> >())
//     {
//         //geometryID
//         if(res.geom_ids.size() != 0)
//         {
//             resultFlags = resultFlags | ShaderDefines::Def_GeometryID;
//         }
//     }

//     if constexpr(BundleT::template has<ObjectIds<VULKAN_DEVICE_LOCAL> >())
//     {
//         //instanceID
//         if(res.object_ids.size() != 0)
//         {
//             resultFlags = resultFlags | ShaderDefines::Def_InstanceID;
//         }
//     }

//     return resultFlags;
// }


static ShaderDefineFlags get_result_flags(const VulkanResultsAddresses& res)
{
    ShaderDefineFlags resultFlags = 0;

    if(res.hitsAddress != 0)
    {
        resultFlags = resultFlags | ShaderDefines::Def_Hits;
    }

    if(res.rangesAddress != 0)
    {
        resultFlags = resultFlags | ShaderDefines::Def_Ranges;
    }

    if(res.pointsAddress != 0)
    {
        resultFlags = resultFlags | ShaderDefines::Def_Points;
    }

    if(res.normalsAddress != 0)
    {
        resultFlags = resultFlags | ShaderDefines::Def_Normals;
    }

    if(res.primitiveIdAddress != 0)
    {
        resultFlags = resultFlags | ShaderDefines::Def_PrimitiveID;
    }

    if(res.geometryIdAddress != 0)
    {
        resultFlags = resultFlags | ShaderDefines::Def_GeometryID;
    }

    if(res.instanceIdAddress != 0)
    {
        resultFlags = resultFlags | ShaderDefines::Def_InstanceID;
    }

    return resultFlags;
}


static inline ShaderDefineFlags get_result_flags(const Memory<VulkanResultsAddresses, RAM>& resultsMem_ram)
{
    VulkanResultsAddresses res = resultsMem_ram[0];
    return get_result_flags(res);
}

} // namespace rmagine