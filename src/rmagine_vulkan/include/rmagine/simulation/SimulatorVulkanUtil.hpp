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
};



template<typename BundleT>
static void set_vulkan_results_data(BundleT& res, VulkanResultsAddresses& mem)
{
    if constexpr(BundleT::template has<Hits<DEVICE_LOCAL_VULKAN> >())
    {
        mem.hitsAddress = res.Hits<DEVICE_LOCAL_VULKAN>::hits.getBuffer()->getBufferDeviceAddress();
    }

    if constexpr(BundleT::template has<Ranges<DEVICE_LOCAL_VULKAN> >())
    {
        mem.rangesAddress = res.Ranges<DEVICE_LOCAL_VULKAN>::ranges.getBuffer()->getBufferDeviceAddress();
    }

    if constexpr(BundleT::template has<Points<DEVICE_LOCAL_VULKAN> >())
    {
        mem.pointsAddress = res.Points<DEVICE_LOCAL_VULKAN>::points.getBuffer()->getBufferDeviceAddress();
    }

    if constexpr(BundleT::template has<Normals<DEVICE_LOCAL_VULKAN> >())
    {
        mem.normalsAddress = res.Normals<DEVICE_LOCAL_VULKAN>::normals.getBuffer()->getBufferDeviceAddress();
    }

    if constexpr(BundleT::template has<FaceIds<DEVICE_LOCAL_VULKAN> >())
    {
        mem.primitiveIdAddress = res.FaceIds<DEVICE_LOCAL_VULKAN>::face_ids.getBuffer()->getBufferDeviceAddress();
    }

    if constexpr(BundleT::template has<GeomIds<DEVICE_LOCAL_VULKAN> >())
    {
        mem.geometryIdAddress = res.GeomIds<DEVICE_LOCAL_VULKAN>::geom_ids.getBuffer()->getBufferDeviceAddress();
    }

    if constexpr(BundleT::template has<ObjectIds<DEVICE_LOCAL_VULKAN> >())
    {
        mem.instanceIdAddress = res.ObjectIds<DEVICE_LOCAL_VULKAN>::object_ids.getBuffer()->getBufferDeviceAddress();
    }
}


template<typename BundleT>
static bool check_vulkan_bundle_sizes(BundleT& res, size_t size)
{
    if constexpr(BundleT::template has<Hits<DEVICE_LOCAL_VULKAN> >())
    {
        if(res.hits.size() != 0 && res.hits.size() < size)
        {
            return false;
        }
    }

    if constexpr(BundleT::template has<Ranges<DEVICE_LOCAL_VULKAN> >())
    {
        if(res.ranges.size() != 0 && res.ranges.size() < size)
        {
            return false;
        }
    }

    if constexpr(BundleT::template has<Points<DEVICE_LOCAL_VULKAN> >())
    {
        if(res.points.size() != 0 && res.points.size() < size)
        {
            return false;
        }
    }

    if constexpr(BundleT::template has<Normals<DEVICE_LOCAL_VULKAN> >())
    {
        if(res.normals.size() != 0 && res.normals.size() < size)
        {
            return false;
        }
    }

    if constexpr(BundleT::template has<FaceIds<DEVICE_LOCAL_VULKAN> >())
    {
        //primitiveID
        if(res.face_ids.size() != 0 && res.face_ids.size() < size)
        {
            return false;
        }
    }

    if constexpr(BundleT::template has<GeomIds<DEVICE_LOCAL_VULKAN> >())
    {
        //geometryID
        if(res.geom_ids.size() != 0 && res.geom_ids.size() < size)
        {
            return false;
        }
    }

    if constexpr(BundleT::template has<ObjectIds<DEVICE_LOCAL_VULKAN> >())
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

//     if constexpr(BundleT::template has<Hits<DEVICE_LOCAL_VULKAN> >())
//     {
//         if(res.hits.size() != 0)
//         {
//             resultFlags = resultFlags | ShaderDefines::Def_Hits;
//         }
//     }

//     if constexpr(BundleT::template has<Ranges<DEVICE_LOCAL_VULKAN> >())
//     {
//         if(res.ranges.size() != 0)
//         {
//             resultFlags = resultFlags | ShaderDefines::Def_Ranges;
//         }
//     }

//     if constexpr(BundleT::template has<Points<DEVICE_LOCAL_VULKAN> >())
//     {
//         if(res.points.size() != 0)
//         {
//             resultFlags = resultFlags | ShaderDefines::Def_Points;
//         }
//     }

//     if constexpr(BundleT::template has<Normals<DEVICE_LOCAL_VULKAN> >())
//     {
//         if(res.normals.size() != 0)
//         {
//             resultFlags = resultFlags | ShaderDefines::Def_Normals;
//         }
//     }

//     if constexpr(BundleT::template has<FaceIds<DEVICE_LOCAL_VULKAN> >())
//     {
//         //primitiveID
//         if(res.face_ids.size() != 0)
//         {
//             resultFlags = resultFlags | ShaderDefines::Def_PrimitiveID;
//         }
//     }

//     if constexpr(BundleT::template has<GeomIds<DEVICE_LOCAL_VULKAN> >())
//     {
//         //geometryID
//         if(res.geom_ids.size() != 0)
//         {
//             resultFlags = resultFlags | ShaderDefines::Def_GeometryID;
//         }
//     }

//     if constexpr(BundleT::template has<ObjectIds<DEVICE_LOCAL_VULKAN> >())
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