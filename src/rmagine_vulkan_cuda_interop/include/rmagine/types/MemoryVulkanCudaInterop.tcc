#include "MemoryVulkanCudaInterop.hpp"



namespace rmagine
{

template<typename DataT>
template<typename MemT2>
MemoryView<DataT, VULKAN_AS_CUDA>::MemoryView(const MemoryView<DataT, MemT2>& vulkanMemView) : m_mem(nullptr), m_size(vulkanMemView.size()), m_externalMemory(new CudaExternalMemory)
{
    static_assert(std::is_same<MemT2, HOST_VISIBLE_VULKAN>::value ||
                  std::is_same<MemT2, DEVICE_LOCAL_VULKAN>::value, 
                  "ERROR - constructed invalid MemoryView<DataT, VULKAN_AS_CUDA>");

    m_mem = reinterpret_cast<DataT*>(vulkanCudaInterop::importVulkanMemToCuda(vulkanMemView.getDeviceMemory(), m_externalMemory, vulkanMemView.size() * sizeof(DataT), vulkanMemView.offset() * sizeof(DataT)));
}

template<typename DataT>
MemoryView<DataT, VULKAN_AS_CUDA>::MemoryView(DataT* mem, size_t N, CudaExternalMemoryPtr externalMemory) : m_mem(mem), m_size(N)
{
    
}

template<typename DataT>
MemoryView<DataT, VULKAN_AS_CUDA>::~MemoryView()
{
    // Do nothing on destruction
    // The external memory destroys itself when there is no view left that needs it
}

template<typename DataT>
MemoryView<DataT, VULKAN_AS_CUDA>& MemoryView<DataT, VULKAN_AS_CUDA>::operator=(const MemoryView<DataT, VULKAN_AS_CUDA>& o)
{
    #if !defined(NDEBUG)
    if(o.size() != this->size())
    {
        throw std::runtime_error("MemoryView VULKAN_AS_CUDA assignment of different sizes");
    }
    #endif // NDEBUG
    copy(o, *this);
    return *this;
}

template<typename DataT>
template<typename MemT2>
MemoryView<DataT, VULKAN_AS_CUDA>& MemoryView<DataT, VULKAN_AS_CUDA>::operator=(const MemoryView<DataT, MemT2>& o)
{
    #if !defined(NDEBUG)
    if(o.size() != this->size())
    {
        throw std::runtime_error("MemoryView MemT2 assignment of different sizes");
    }
    #endif // NDEBUG
    copy(o, *this);
    return *this;
}

template<typename DataT>
DataT* MemoryView<DataT, VULKAN_AS_CUDA>::raw()
{
    return m_mem;
}

template<typename DataT>
const DataT* MemoryView<DataT, VULKAN_AS_CUDA>::raw() const {
    return m_mem;
}






//// VULKAN_AS_CUDA

template<typename DataT>
DataT* VULKAN_AS_CUDA::alloc(size_t N)
{
    (void) N;
    throw std::runtime_error("VULKAN_AS_CUDA (alloc): The program should never call this function and always use the template specialization instead!");
    return nullptr;
}

template<typename DataT>
DataT* VULKAN_AS_CUDA::realloc(DataT* mem, size_t Nold, size_t Nnew)
{
    (void) mem;
    (void) Nold;
    (void) Nnew;
    throw std::runtime_error("VULKAN_AS_CUDA (realloc): The program should never call this function and always use the template specialization instead!");
    return nullptr;
}

template<typename DataT>
void VULKAN_AS_CUDA::free(DataT* mem, size_t N)
{
    (void) mem;
    (void) N;
    throw std::runtime_error("VULKAN_AS_CUDA (free): The program should never call this function and always use the template specialization instead!");
}

} // namespace rmagine