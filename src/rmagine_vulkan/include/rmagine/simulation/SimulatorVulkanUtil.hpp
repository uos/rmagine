#include "../../rmagine_core/types/Bundle.hpp"
#include "../../rmagine_core/simulation/SimulationResults.hpp"

#include "../util/MemoryVulkan.hpp"
#include "../../rmagine_core/math/Types.hpp"
#include "../util/contextComponents/ShaderUtil.hpp"



namespace rmagine
{

template<typename MemT>
using PrimitiveIds = FaceIds<MemT>;

template<typename MemT>
using GeometryIds = GeomIds<MemT>;

template<typename MemT>
using InstanceIds = ObjectIds<MemT>;



struct VulkanResultsData
{
    Memory<uint8_t, VULKAN_DEVICE_LOCAL> hits;
    Memory<float, VULKAN_DEVICE_LOCAL> ranges;
    Memory<Vec3, VULKAN_DEVICE_LOCAL> points;
    Memory<Vec3, VULKAN_DEVICE_LOCAL> normals;
    Memory<unsigned int, VULKAN_DEVICE_LOCAL> primitiveID;
    Memory<unsigned int, VULKAN_DEVICE_LOCAL> instanceID;
    Memory<unsigned int, VULKAN_DEVICE_LOCAL> geometryID;
};

/**
 * TODO: maybe put these buffers here, instead of using the descriptor set for them...
 */
struct VulkanSensorData
{
    // Memory<Transform, VULKAN_DEVICE_LOCAL> tsb;
    // Memory<Transform, VULKAN_DEVICE_LOCAL> tbm;

    // Memory<SensorModelUnion, VULKAN_DEVICE_LOCAL> sensor;
};



template<typename BundleT>
static void set_vulkan_results_data(BundleT& res, VulkanResultsData& mem)
{
    if constexpr(BundleT::template has<Hits<VULKAN_DEVICE_LOCAL> >())
    {
        mem.hits = res.Hits<VULKAN_DEVICE_LOCAL>::hits;
    }

    if constexpr(BundleT::template has<Ranges<VULKAN_DEVICE_LOCAL> >())
    {
        mem.ranges = res.Ranges<VULKAN_DEVICE_LOCAL>::ranges;
    }

    if constexpr(BundleT::template has<Points<VULKAN_DEVICE_LOCAL> >())
    {
        mem.points = res.Points<VULKAN_DEVICE_LOCAL>::points;
    }

    if constexpr(BundleT::template has<Normals<VULKAN_DEVICE_LOCAL> >())
    {
        mem.normals = res.Normals<VULKAN_DEVICE_LOCAL>::normals;
    }

    if constexpr(BundleT::template has<FaceIds<VULKAN_DEVICE_LOCAL> >())
    {
        mem.primitiveID = res.FaceIds<VULKAN_DEVICE_LOCAL>::face_ids;
    }

    if constexpr(BundleT::template has<GeomIds<VULKAN_DEVICE_LOCAL> >())
    {
        mem.geometryID = res.GeomIds<VULKAN_DEVICE_LOCAL>::geom_ids;
    }

    if constexpr(BundleT::template has<ObjectIds<VULKAN_DEVICE_LOCAL> >())
    {
        mem.instanceID = res.ObjectIds<VULKAN_DEVICE_LOCAL>::object_ids;
    }
}


static ShaderDefineFlags get_result_flags(const VulkanResultsData& res)
{
    ShaderDefineFlags resultFlags = 0;

    if(res.hits.size() > 0)
    {
        resultFlags = resultFlags | ShaderDefines::Def_Hits;
    }

    if(res.ranges.size() > 0)
    {
        resultFlags = resultFlags | ShaderDefines::Def_Ranges;
    }

    if(res.points.size() > 0)
    {
        resultFlags = resultFlags | ShaderDefines::Def_Points;
    }

    if(res.normals.size() > 0)
    {
        resultFlags = resultFlags | ShaderDefines::Def_Normals;
    }

    if(res.primitiveID.size() > 0)
    {
        resultFlags = resultFlags | ShaderDefines::Def_PrimitiveID;
    }

    if(res.geometryID.size() > 0)
    {
        resultFlags = resultFlags | ShaderDefines::Def_GeometryID;
    }

    if(res.instanceID.size() > 0)
    {
        resultFlags = resultFlags | ShaderDefines::Def_InstanceID;
    }

    return resultFlags;
}


static inline ShaderDefineFlags get_result_flags(const Memory<VulkanResultsData, RAM>& resultsMem_ram)
{
    VulkanResultsData res = resultsMem_ram[0];
    return get_result_flags(res);
}


static bool check_results_data_size(const VulkanResultsData& res, size_t size)
{
    if(res.hits.size() != 0 && res.hits.size() < size)
    {
        return false;
    }

    if(res.ranges.size() != 0 && res.ranges.size() < size)
    {
        return false;
    }

    if(res.points.size() != 0 && res.points.size() < size)
    {
        return false;
    }

    if(res.normals.size() != 0 && res.normals.size() < size)
    {
        return false;
    }

    if(res.primitiveID.size() != 0 && res.primitiveID.size() < size)
    {
        return false;
    }

    if(res.geometryID.size() != 0 && res.geometryID.size() < size)
    {
        return false;
    }

    if(res.instanceID.size() != 0 && res.instanceID.size() < size)
    {
        return false;
    }

    return true;
}


static inline bool check_results_data_size(const Memory<VulkanResultsData, RAM>& resultsMem_ram, size_t size)
{
    VulkanResultsData res = resultsMem_ram[0];
    return check_results_data_size(res, size);
}

} // namespace rmagine