#include "rmagine/util/cuda/CudaContext.hpp"
#include <iostream>

#include "rmagine/util/cuda/CudaStream.hpp"

namespace rmagine {

bool cuda_initialized_ = false;

bool cuda_initialized()
{
    return cuda_initialized_;
}

void cuda_initialize()
{
    cuInit(0);
    cuda_initialized_ = true;
}

CudaContext::CudaContext(int device_id)
{
    if(!cuda_initialized())
    {
        std::cout << "[CudaContext] Init Cuda" << std::endl;
        cuda_initialize();
    }

    cudaDeviceProp info = cuda::getDeviceInfo(device_id);
    std::cout << "[CudaContext] Construct context on device " << device_id << " " << info.name << " " << info.luid << std::endl;

    cuCtxCreate(&m_context, 0, device_id);
}

CudaContext::CudaContext(CUcontext ctx)
:m_context(ctx)
{
    
}

CudaContext::~CudaContext()
{
    cuCtxDestroy(m_context);
    std::cout << "[CudaContext::~CudaContext()] destroyed." << std::endl;
}

int CudaContext::getDeviceId() const
{
    CUcontext old;
    cuCtxGetCurrent(&old);
    cuCtxSetCurrent(m_context);

    int device_id = -1;
    cuCtxGetDevice(&device_id);

    // restore old context
    cuCtxSetCurrent(old);
    return device_id;
}

cudaDeviceProp CudaContext::getDeviceInfo() const
{
    return cuda::getDeviceInfo(getDeviceId());
}

void CudaContext::use()
{
    cuCtxSetCurrent(m_context);
}

void CudaContext::enqueue()
{
    cuCtxPushCurrent(m_context);
}

bool CudaContext::isActive() const
{
    CUcontext old;
    cuCtxGetCurrent(&old);
    return (old == m_context);
}

CudaStreamPtr CudaContext::createStream(unsigned int flags) const
{
    CUcontext old;
    cuCtxGetCurrent(&old);
    cuCtxSetCurrent(m_context);

    CudaStreamPtr ret = std::make_shared<CudaStream>(flags);

    // restore old
    cuCtxSetCurrent(old);
    return ret;
}

void CudaContext::setSharedMemBankSize(unsigned int bytes)
{
    CUcontext old;
    cuCtxGetCurrent(&old);
    cuCtxSetCurrent(m_context);

    CUresult status;
    
    if(bytes == 4)
    {
        status = cuCtxSetSharedMemConfig(CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE);
    } 
    else if(bytes == 8)
    {
        status = cuCtxSetSharedMemConfig(CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE);
    }

    if(status != CUDA_SUCCESS)
    {
        std::cout << "WARNING: Could not set SMEM Size to " << bytes << std::endl; 
    } 

    // restore old
    cuCtxSetCurrent(old);
}

unsigned int CudaContext::getSharedMemBankSize() const
{
    CUcontext old;
    cuCtxGetCurrent(&old);
    cuCtxSetCurrent(m_context);
    CUsharedconfig config;
    cuCtxGetSharedMemConfig(&config);

    unsigned int bytes = 4;

    if(config == CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE)
    {
        bytes = 4;
    }
    else if(config == CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE)
    {
        bytes = 8;
    }

    cuCtxSetCurrent(old);

    return bytes;
}

void CudaContext::synchronize()
{
    CUcontext old;
    cuCtxGetCurrent(&old);
    cuCtxSetCurrent(m_context);

    cuCtxSynchronize();

    cuCtxSetCurrent(old);
}

CUcontext CudaContext::ref()
{
    return m_context;
}

std::ostream& operator<<(std::ostream& os, const CudaContext& ctx)
{
    
    cudaDeviceProp info = ctx.getDeviceInfo();
    int device = ctx.getDeviceId();

    os << "[CudaContext]\n";
    os << "- Device: " << info.name << "\n";
    os << "- SMemSize: " << ctx.getSharedMemBankSize() << "B\n";
    os << "- Active: " << (ctx.isActive()? "true" : "false") << "\n";

    return os;
}

CudaContextPtr cuda_def_ctx(new CudaContext(0));

CudaContextPtr cuda_current_context()
{
    // if(!cuda_def_ctx->defaultStream())
    // {
    //     std::cout << "initDefaultStream" << std::endl;
    //     cuda_def_ctx->initDefaultStream();
    //     std::cout << "initDefaultStream done." << std::endl;
    // }
    
    return cuda_def_ctx;
}

} // namespace rmagine