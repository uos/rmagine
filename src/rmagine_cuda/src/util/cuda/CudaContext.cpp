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

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

static void ensure_cuda_init(int device_id)
{
    if(!cuda_initialized())
    {
        // std::cout << "[Rmagine - CudaContext] Init Cuda" << std::endl;
        printCudaInfo();
        cuda_initialize();
    }

    cudaDeviceProp info = cuda::getDeviceInfo(device_id);
    std::cout << "[Rmagine - CudaContext] Construct context on device " << device_id << " - " << info.name << " " << info.luid << std::endl;
}

CudaContext::CudaContext(CUcontext ctx)
:m_context(ctx)
{
}

// Destructor is intentionally a no-op. Cleanup is handled by
// the custom deleters in the static factory methods.
CudaContext::~CudaContext()
{
}

CudaContextPtr CudaContext::createPrimary(int device_id)
{
    ensure_cuda_init(device_id);

    CUcontext ctx;
    cuDevicePrimaryCtxRetain(&ctx, device_id);
    cuCtxSetCurrent(ctx);

    return CudaContextPtr(new CudaContext(ctx), [device_id](CudaContext* p) {
        cuDevicePrimaryCtxRelease(device_id);
        delete p;
    });
}

CudaContextPtr CudaContext::createStandalone(int device_id)
{
    ensure_cuda_init(device_id);

    CUcontext ctx;
    #if CUDA_VERSION >= 13000
    cuCtxCreate(&ctx, nullptr, 0, device_id);
    #else
    cuCtxCreate(&ctx, 0, device_id);
    #endif

    return CudaContextPtr(new CudaContext(ctx), [](CudaContext* p) {
        cuCtxDestroy(p->ref());
        delete p;
    });
}

CudaContextPtr CudaContext::fromExternal(CUcontext ctx)
{
    // No-op deleter: caller owns the context
    return CudaContextPtr(new CudaContext(ctx), [](CudaContext* p) {
        delete p;
    });
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

// Default global context: uses primary context for runtime API interop.
CudaContextPtr cuda_def_ctx = CudaContext::createPrimary(0);

CudaContextPtr cuda_current_context()
{
    return cuda_def_ctx;
}

} // namespace rmagine
