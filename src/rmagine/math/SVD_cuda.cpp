#include "rmagine/math/SVD_cuda.hpp"
#include "rmagine/types/MemoryCuda.hpp"
#include <assert.h>

namespace rmagine {


SVD_cuda::SVD_cuda()
{
    cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cuda_status = cudaSuccess;

    cuda_status = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    assert(cudaSuccess == cuda_status);

    /* step 1: create cusolver handle, bind a stream  */
    status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    status = cusolverDnSetStream(cusolverH, stream);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    /* step 2: configuration of gesvdj */
    status = cusolverDnCreateGesvdjInfo(&gesvdj_params);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    const double tol = 1.e-6;
    const int max_sweeps = 15;
    const int sort_svd  = 0;   /* don't sort singular values */

    /* default value of tolerance is machine zero */
    status = cusolverDnXgesvdjSetTolerance(
        gesvdj_params,
        tol);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    /* default value of max. sweeps is 100 */
    status = cusolverDnXgesvdjSetMaxSweeps(
        gesvdj_params,
        max_sweeps);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    /* disable sorting */
    status = cusolverDnXgesvdjSetSortEig(
        gesvdj_params,
        sort_svd);
    assert(CUSOLVER_STATUS_SUCCESS == status);
}

SVD_cuda::SVD_cuda(cudaStream_t stream)
{
    this->stream = stream;

    cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cuda_status = cudaSuccess;

    /* step 1: create cusolver handle, bind a stream  */
    status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    status = cusolverDnSetStream(cusolverH, stream);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    /* step 2: configuration of gesvdj */
    status = cusolverDnCreateGesvdjInfo(&gesvdj_params);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    const double tol = 1.e-6;
    const int max_sweeps = 15;
    const int sort_svd  = 0;   /* don't sort singular values */

    /* default value of tolerance is machine zero */
    status = cusolverDnXgesvdjSetTolerance(
        gesvdj_params,
        tol);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    /* default value of max. sweeps is 100 */
    status = cusolverDnXgesvdjSetMaxSweeps(
        gesvdj_params,
        max_sweeps);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    /* disable sorting */
    status = cusolverDnXgesvdjSetSortEig(
        gesvdj_params,
        sort_svd);
    assert(CUSOLVER_STATUS_SUCCESS == status);
}

SVD_cuda::~SVD_cuda()
{
    if (cusolverH) cusolverDnDestroy(cusolverH);
    if (stream      ) cudaStreamDestroy(stream);
    if (gesvdj_params) cusolverDnDestroyGesvdjInfo(gesvdj_params);
}

void SVD_cuda::calcUV(
    const Memory<Matrix3x3, VRAM_CUDA>& As,
    Memory<Matrix3x3, VRAM_CUDA>& Us,
    Memory<Matrix3x3, VRAM_CUDA>& Vs) const
{
    Memory<Vector, VRAM_CUDA> Ss(Us.size());
    calcUSV(As, Us, Ss, Vs);
}

void SVD_cuda::calcUSV(const Memory<Matrix3x3, VRAM_CUDA>& As,
        Memory<Matrix3x3, VRAM_CUDA>& Us,
        Memory<Vector, VRAM_CUDA>& Ss,
        Memory<Matrix3x3, VRAM_CUDA>& Vs) const
{
    cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cuda_status = cudaSuccess;

    if(Vs.size() != As.size())
    {
        Vs.resize(As.size());
    }

    if(Us.size() != As.size())
    {
        Us.resize(As.size());
    }

    // height or rows
    const int m = 3; /* 1 <= m <= 32 */
    // width or cols
    const int n = 3; /* 1 <= n <= 32 */
    const int lda = m; /* lda >= m */
    const int ldu = m; /* ldu >= m */
    const int ldv = n; /* ldv >= n */
    const int batchSize = As.size();
    const int minmn = (m < n)? m : n; /* min(m,n) */

    // Create Buffer
    int* d_info  = NULL; /* batchSize */
    int lwork = 0;       /* size of workspace */
    float *d_work = NULL; /* device workspace for gesvdjBatched */
    
    const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; /* compute singular vectors */

    float residual = 0;
    int executed_sweeps = 0;

    const float* d_A = reinterpret_cast<const float*>(As.raw());
    
    // reinterpret data
    float* d_U = reinterpret_cast<float*>(Us.raw());
    float* d_S = reinterpret_cast<float*>(Ss.raw());
    float* d_V = reinterpret_cast<float*>(Vs.raw());
    
    // cuda_status = cudaMalloc ((void**)&d_S   , sizeof(float)*minmn*batchSize);
    // assert(cudaSuccess == cuda_status);
    cuda_status = cudaMalloc ((void**)&d_info, sizeof(int   )*batchSize);
    assert(cudaSuccess == cuda_status);

    /////////////////////
    // SOLVING
    ////

    /* step 4: query working space of gesvdjBatched */
    status = cusolverDnSgesvdjBatched_bufferSize(
        cusolverH,
        jobz,
        m,
        n,
        d_A,
        lda,
        d_S,
        d_U,
        ldu,
        d_V,
        ldv,
        &lwork,
        gesvdj_params,
        batchSize
    );
    assert(CUSOLVER_STATUS_SUCCESS == status);

    cuda_status = cudaMalloc((void**)&d_work, sizeof(float)*lwork);
    assert(cudaSuccess == cuda_status);

    // cusolverDnDgesvdjBatched wants A to be none const? 
    // until they change(/fix?) it we need const_cast  

    /* step 5: compute singular values of A0 and A1 */
    status = cusolverDnSgesvdjBatched(
        cusolverH,
        jobz,
        m,
        n,
        const_cast<float*>(d_A),
        lda,
        d_S,
        d_U,
        ldu,
        d_V,
        ldv,
        d_work,
        lwork,
        d_info,
        gesvdj_params,
        batchSize
    );

    // cuda_status = cudaDeviceSynchronize();

    // std::cout << "batchSize: " << batchSize << std::endl;
    // float *h_S = (float*)malloc(batchSize * minmn * sizeof(float));
    // cudaMemcpy(h_S, d_S, batchSize * minmn * sizeof(float), cudaMemcpyDeviceToHost);
    // cuda_status = cudaDeviceSynchronize();

    // std::cout << "Singular Values: " << std::endl;
    // std::cout << h_S[0] << ", " << h_S[1] << ", " << h_S[2] << std::endl;

    // free(h_S);

    assert(CUSOLVER_STATUS_SUCCESS == status);
    cudaFree(d_work);
    cudaFree(d_info);
}

} // namespace rmagine