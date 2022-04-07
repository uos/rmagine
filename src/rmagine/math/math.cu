#include "rmagine/math/math.cuh"
#include "rmagine/math/math.h"
#include "rmagine/math/types.h"

namespace rmagine 
{

////////
// Generic Kernel
///

template<typename In1T, typename In2T, typename ResT>
__global__ void multNxN_kernel(
    const In1T* A,
    const In2T* B,
    ResT* C,
    unsigned int N
)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N)
    {
        C[id] = A[id] * B[id];
    }
}

template<typename In1T, typename In2T, typename ResT>
__global__ void multNx1_kernel(
    const In1T* A,
    const In2T* b,
    ResT* C,
    unsigned int N)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N)
    {
        C[id] = A[id] * b[0];
    }
}

template<typename In1T, typename In2T, typename ResT>
__global__ void mult1xN_kernel(
    const In1T* a,
    const In2T* B,
    ResT* C,
    unsigned int N)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N)
    {
        C[id] = a[0] * B[id];
    }
}


template<typename In1T, typename In2T, typename ResT>
__global__ void addNxN_kernel(
    const In1T* A,
    const In2T* B,
    ResT* C,
    unsigned int N
)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N)
    {
        C[id] = A[id] + B[id];
    }
}

template<typename In1T, typename In2T, typename ResT>
__global__ void subNxN_kernel(
    const In1T* A,
    const In2T* B,
    ResT* C,
    unsigned int N
)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N)
    {
        C[id] = A[id] - B[id];
    }
}

template<typename In1T, typename In2T, typename ResT>
__global__ void subNx1_kernel(
    const In1T* A,
    const In2T* b,
    ResT* C,
    unsigned int N
)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N)
    {
        C[id] = A[id] - b[0];
    }
}


template<typename T>
__global__ void transpose_kernel(
    const T* A,
    T* B,
    unsigned int N
)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N)
    {
        B[id] = A[id].transpose();
    }
}

template<typename T>
__global__ void transposeInplace_kernel(
    T* A,
    unsigned int N
)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N)
    {
        A[id].transposeInplace();
    }
}


template<typename T>
__global__ void invert_kernel(
    const T* A,
    T* B,
    unsigned int N
)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N)
    {
        B[id] = A[id].inv();
    }
}

template<typename In1T, typename In2T, typename ResT>
__global__ void divNxN_kernel(
    const In1T* A, 
    const In2T* B,
    ResT* C,
    unsigned int N)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N)
    {
        C[id] = A[id] / B[id];
    }
}

template<typename ConvT, typename In1T, typename In2T, typename ResT>
__global__ void divNxN_conv_kernel(
    const In1T* A, 
    const In2T* B,
    ResT* C,
    unsigned int N)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N)
    {
        C[id] = A[id] / static_cast<ConvT>(B[id]);
    }
}

template<typename ConvT>
__global__ void divNxNIgnoreZeros_conv_kernel(
    const Vector* A, 
    const unsigned int* B,
    Vector* C,
    unsigned int N)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N)
    {
        if(B[id] > 0)
        {
            C[id] = A[id] / static_cast<ConvT>(B[id]);
        } else {
            C[id].setZeros();
        }
    }
}

template<typename ConvT>
__global__ void divNxNIgnoreZeros_conv_kernel(
    const Matrix3x3* A,
    const unsigned int* B,
    Matrix3x3* C,
    unsigned int N)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N)
    {
        if(B[id] > 0)
        {
            C[id] = A[id] / static_cast<ConvT>(B[id]);
        } else {
            C[id].setZeros();
        }
    }
}

template<typename ConvT>
__global__ void divNxNIgnoreZeros_conv_kernel(
    const float* A, 
    const unsigned int* B,
    float* C,
    unsigned int N)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N)
    {
        if(B[id] > 0)
        {
            C[id] = A[id] / static_cast<ConvT>(B[id]);
        } else {
            C[id] = 0.0;
        }
    }
}

__global__ void divNxNInplace_kernel(
    Vector* A, 
    const float* B,
    unsigned int N)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N)
    {
        A[id] /= B[id];
    }
}

__global__ void divNxNInplace_kernel(
    Matrix3x3* A, 
    const unsigned int* B,
    unsigned int N)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N)
    {
        A[id] /= static_cast<float>(B[id]);
    }
}

template<typename T>
__global__ void divNx1Inplace_kernel(
    T* A,
    unsigned int b,
    unsigned int N)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N)
    {
        A[id] /= static_cast<float>(b);
    }
}

__global__ void convert_kernel(const uint8_t* from, float* to, unsigned int N)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N)
    {
        to[id] = static_cast<float>(from[id]);
    }
}

__global__
void convert_kernel(const bool* from, unsigned int* to, unsigned int N)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N)
    {
        to[id] = static_cast<unsigned int>(from[id]);
    }
}

__global__
void convert_kernel(const unsigned int* from, bool* to, unsigned int N)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N)
    {
        to[id] = (from[id] > 0);
    }
}

__global__ void pack_kernel(
    const Matrix3x3* R, 
    const Vector* t, // Vector3d / Vector3f
    Transform* T,
    unsigned int N)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N)
    {
        T[id].R.set(R[id]);
        T[id].t = t[id];
    }
}

__global__ void pack_kernel(
    const Quaternion* R, 
    const Vector* t, // Vector3d / Vector3f
    Transform* T,
    unsigned int N)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N)
    {
        T[id].R = R[id];
        T[id].t = t[id];
    }
}

__global__ void covParts_kernel(
    const Vector* a, 
    const Vector* b,
    Matrix3x3* C,
    unsigned int N)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N)
    {
        C[id](0,0) = a[id].x * b[id].x;
        C[id](1,0) = a[id].x * b[id].y;
        C[id](2,0) = a[id].x * b[id].z;
        C[id](0,1) = a[id].y * b[id].x;
        C[id](1,1) = a[id].y * b[id].y;
        C[id](2,1) = a[id].y * b[id].z;
        C[id](0,2) = a[id].z * b[id].x;
        C[id](1,2) = a[id].z * b[id].y;
        C[id](2,2) = a[id].z * b[id].z;
    }
}

__global__ void covParts_kernel(
    const Vector* a, 
    const Vector* b,
    const bool* corr,
    Matrix3x3* C,
    unsigned int N)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N)
    {
        if(corr[id])
        {
            C[id](0,0) = a[id].x * b[id].x;
            C[id](1,0) = a[id].x * b[id].y;
            C[id](2,0) = a[id].x * b[id].z;
            C[id](0,1) = a[id].y * b[id].x;
            C[id](1,1) = a[id].y * b[id].y;
            C[id](2,1) = a[id].y * b[id].z;
            C[id](0,2) = a[id].z * b[id].x;
            C[id](1,2) = a[id].z * b[id].y;
            C[id](2,2) = a[id].z * b[id].z;
        } else {
            C[id].setZeros();
        }
    }
}

template<unsigned int blockSize, typename T>
__device__ void warpReduce(volatile T* sdata, unsigned int tid)
{
    if(blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if(blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if(blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if(blockSize >=  8) sdata[tid] += sdata[tid + 4];
    if(blockSize >=  4) sdata[tid] += sdata[tid + 2];
    if(blockSize >=  2) sdata[tid] += sdata[tid + 1];
}


template<unsigned int blockSize, typename T>
__global__ void sum_kernel(
    const T* data,
    T* res,
    unsigned int N)
{
    __shared__ T sdata[blockSize];
    
    const unsigned int tid = threadIdx.x;
    const unsigned int globId = N * blockIdx.x + threadIdx.x;
    const unsigned int rows = (N + blockSize - 1) / blockSize;

    sdata[tid] *= 0.0;
    for(unsigned int i=0; i<rows; i++)
    {
        if(tid + blockSize * i < N)
        {
            sdata[threadIdx.x] += data[globId + blockSize * i];
        }
    }
    __syncthreads();

    for(unsigned int s = blockSize / 2; s > 32; s >>= 1)
    {
        if(tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if(tid < blockSize / 2 && tid < 32)
    {
        warpReduce<blockSize>(sdata, tid);
    }

    if(tid == 0)
    {
        res[blockIdx.x] = sdata[0];
    }
}

template<unsigned int blockSize>
__global__ void cov_kernel(
    const Vector* v1,
    const Vector* v2,
    Matrix3x3* res,
    unsigned int N)
{
    __shared__ Matrix3x3 sdata[blockSize];
    
    const unsigned int tid = threadIdx.x;
    const unsigned int globId = N * blockIdx.x + threadIdx.x;
    const unsigned int rows = (N + blockSize - 1) / blockSize;

    sdata[tid].setZeros();
    for(unsigned int i=0; i<rows; i++)
    {
        if(tid + blockSize * i < N)
        {
            const Vector& a = v1[globId + blockSize * i];
            const Vector& b = v2[globId + blockSize * i];
            sdata[tid](0,0) += a.x * b.x;
            sdata[tid](1,0) += a.x * b.y;
            sdata[tid](2,0) += a.x * b.z;
            sdata[tid](0,1) += a.y * b.x;
            sdata[tid](1,1) += a.y * b.y;
            sdata[tid](2,1) += a.y * b.z;
            sdata[tid](0,2) += a.z * b.x;
            sdata[tid](1,2) += a.z * b.y;
            sdata[tid](2,2) += a.z * b.z;
        }
    }
    __syncthreads();

    for(unsigned int s = blockSize / 2; s > 32; s >>= 1)
    {
        if(tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if(tid < blockSize / 2 && tid < 32)
    {
        warpReduce<blockSize>(sdata, tid);
    }

    if(tid == 0)
    {
        res[blockIdx.x] = sdata[0] / static_cast<float>(N);
    }
}

////////////
// #multNxN
void multNxN(
    const Memory<Quaternion, VRAM_CUDA>& A,
    const Memory<Quaternion, VRAM_CUDA>& B,
    Memory<Quaternion, VRAM_CUDA>& C)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (A.size() + blockSize - 1) / blockSize;
    multNxN_kernel<<<gridSize, blockSize>>>(A.raw(), B.raw(), C.raw(), A.size());
}

Memory<Quaternion, VRAM_CUDA> multNxN(
    const Memory<Quaternion, VRAM_CUDA>& A, 
    const Memory<Quaternion, VRAM_CUDA>& B)
{
    Memory<Quaternion, VRAM_CUDA> C(A.size());
    // mult
    multNxN(A, B, C);
    return C;
}

void multNxN(
    const Memory<Quaternion, VRAM_CUDA>& A,
    const Memory<Vector, VRAM_CUDA>& b, 
    Memory<Vector, VRAM_CUDA>& c)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (A.size() + blockSize - 1) / blockSize;
    multNxN_kernel<<<gridSize, blockSize>>>(A.raw(), b.raw(), c.raw(), A.size());
}

Memory<Vector, VRAM_CUDA> multNxN(
    const Memory<Quaternion, VRAM_CUDA>& A,
    const Memory<Vector, VRAM_CUDA>& b)
{
    Memory<Vector, VRAM_CUDA> c(A.size());
    multNxN(A, b, c);
    return c;
}

/// TRANSFORM
void multNxN(
    const Memory<Transform, VRAM_CUDA>& T1,
    const Memory<Transform, VRAM_CUDA>& T2,
    Memory<Transform, VRAM_CUDA>& Tr)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (T1.size() + blockSize - 1) / blockSize;
    multNxN_kernel<<<gridSize, blockSize>>>(T1.raw(), T2.raw(), Tr.raw(), T1.size());
}

Memory<Transform, VRAM_CUDA> multNxN(
    const Memory<Transform, VRAM_CUDA>& T1,
    const Memory<Transform, VRAM_CUDA>& T2)
{
    Memory<Transform, VRAM_CUDA> Tr(T1.size());
    multNxN(T1,T2,Tr);
    return Tr;
}

void multNxN(
    const Memory<Transform, VRAM_CUDA>& T,
    const Memory<Vector, VRAM_CUDA>& x,
    Memory<Vector, VRAM_CUDA>& c)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (T.size() + blockSize - 1) / blockSize;
    multNxN_kernel<<<gridSize, blockSize>>>(T.raw(), x.raw(), c.raw(), T.size());
}

Memory<Vector, VRAM_CUDA> multNxN(
    const Memory<Transform, VRAM_CUDA>& T,
    const Memory<Vector, VRAM_CUDA>& x)
{
    Memory<Vector, VRAM_CUDA> c(T.size());
    multNxN(T,x,c);
    return c;
}

void multNxN(
    const Memory<Matrix3x3, VRAM_CUDA>& M1,
    const Memory<Matrix3x3, VRAM_CUDA>& M2,
    Memory<Matrix3x3, VRAM_CUDA>& Mr)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (M1.size() + blockSize - 1) / blockSize;
    multNxN_kernel<<<gridSize, blockSize>>>(M1.raw(), M2.raw(), Mr.raw(), M1.size());
}

Memory<Matrix3x3, VRAM_CUDA> multNxN(
    const Memory<Matrix3x3, VRAM_CUDA>& M1,
    const Memory<Matrix3x3, VRAM_CUDA>& M2)
{
    Memory<Matrix3x3, VRAM_CUDA> Mr(M1.size());
    multNxN(M1,M2,Mr);
    return Mr;
}

void multNxN(
    const Memory<Matrix3x3, VRAM_CUDA>& M,
    const Memory<Vector, VRAM_CUDA>& x,
    Memory<Vector, VRAM_CUDA>& c)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (M.size() + blockSize - 1) / blockSize;
    multNxN_kernel<<<gridSize, blockSize>>>(M.raw(), x.raw(), c.raw(), M.size());
}

Memory<Vector, VRAM_CUDA> multNxN(
    const Memory<Matrix3x3, VRAM_CUDA>& M,
    const Memory<Vector, VRAM_CUDA>& x)
{
    Memory<Vector, VRAM_CUDA> c(M.size());
    multNxN(M, x, c);
    return c;
}

////////
// #multNx1
///
void multNx1(
    const Memory<Quaternion, VRAM_CUDA>& A,
    const Memory<Quaternion, VRAM_CUDA>& b,
    Memory<Quaternion, VRAM_CUDA>& C)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (A.size() + blockSize - 1) / blockSize;
    multNxN_kernel<<<gridSize, blockSize>>>(A.raw(), b.raw(), C.raw(), A.size());
}

Memory<Quaternion, VRAM_CUDA> multNx1(
    const Memory<Quaternion, VRAM_CUDA>& A, 
    const Memory<Quaternion, VRAM_CUDA>& b)
{
    Memory<Quaternion, VRAM_CUDA> C(A.size());
    // mult
    multNx1(A, b, C);
    return C;
}

void multNx1(
    const Memory<Quaternion, VRAM_CUDA>& A,
    const Memory<Vector, VRAM_CUDA>& b, 
    Memory<Vector, VRAM_CUDA>& C)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (A.size() + blockSize - 1) / blockSize;
    multNx1_kernel<<<gridSize, blockSize>>>(A.raw(), b.raw(), C.raw(), A.size());
}

Memory<Vector, VRAM_CUDA> multNx1(
    const Memory<Quaternion, VRAM_CUDA>& A,
    const Memory<Vector, VRAM_CUDA>& b)
{
    Memory<Vector, VRAM_CUDA> C(A.size());
    multNx1(A, b, C);
    return C;
}

void multNx1(
    const Memory<Transform, VRAM_CUDA>& T1,
    const Memory<Transform, VRAM_CUDA>& t2,
    Memory<Transform, VRAM_CUDA>& Tr)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (T1.size() + blockSize - 1) / blockSize;
    multNx1_kernel<<<gridSize, blockSize>>>(T1.raw(), t2.raw(), Tr.raw(), T1.size());
}

Memory<Transform, VRAM_CUDA> multNx1(
    const Memory<Transform, VRAM_CUDA>& T1,
    const Memory<Transform, VRAM_CUDA>& t2)
{
    Memory<Transform, VRAM_CUDA> Tr(T1.size());
    multNx1(T1,t2,Tr);
    return Tr;
}

void multNx1(
    const Memory<Transform, VRAM_CUDA>& T,
    const Memory<Vector, VRAM_CUDA>& x,
    Memory<Vector, VRAM_CUDA>& c)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (T.size() + blockSize - 1) / blockSize;
    multNx1_kernel<<<gridSize, blockSize>>>(T.raw(), x.raw(), c.raw(), T.size());
}

Memory<Vector, VRAM_CUDA> multNx1(
    const Memory<Transform, VRAM_CUDA>& T,
    const Memory<Vector, VRAM_CUDA>& x)
{
    Memory<Vector, VRAM_CUDA> C(T.size());
    multNx1(T,x,C);
    return C;
}

void multNx1(
    const Memory<Matrix3x3, VRAM_CUDA>& M1,
    const Memory<Matrix3x3, VRAM_CUDA>& m2,
    Memory<Matrix3x3, VRAM_CUDA>& Mr)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (M1.size() + blockSize - 1) / blockSize;
    multNx1_kernel<<<gridSize, blockSize>>>(M1.raw(), m2.raw(), Mr.raw(), M1.size());
}

Memory<Matrix3x3, VRAM_CUDA> multNx1(
    const Memory<Matrix3x3, VRAM_CUDA>& M1,
    const Memory<Matrix3x3, VRAM_CUDA>& m2)
{
    Memory<Matrix3x3, VRAM_CUDA> Mr(M1.size());
    multNx1(M1,m2,Mr);
    return Mr;
}

void multNx1(
    const Memory<Matrix3x3, VRAM_CUDA>& M,
    const Memory<Vector, VRAM_CUDA>& x,
    Memory<Vector, VRAM_CUDA>& C)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (M.size() + blockSize - 1) / blockSize;
    multNx1_kernel<<<gridSize, blockSize>>>(M.raw(), x.raw(), C.raw(), M.size());
}

Memory<Vector, VRAM_CUDA> multNx1(
    const Memory<Matrix3x3, VRAM_CUDA>& M,
    const Memory<Vector, VRAM_CUDA>& x)
{
    Memory<Vector, VRAM_CUDA> c(M.size());
    multNx1(M, x, c);
    return c;
}

/////////////
// #mult1xN
////////
void mult1xN(
    const Memory<Quaternion, VRAM_CUDA>& a,
    const Memory<Quaternion, VRAM_CUDA>& B,
    Memory<Quaternion, VRAM_CUDA>& C)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (B.size() + blockSize - 1) / blockSize;
    mult1xN_kernel<<<gridSize, blockSize>>>(a.raw(), B.raw(), C.raw(), B.size());
}

Memory<Quaternion, VRAM_CUDA> mult1xN(
    const Memory<Quaternion, VRAM_CUDA>& a, 
    const Memory<Quaternion, VRAM_CUDA>& B)
{
    Memory<Quaternion, VRAM_CUDA> C(B.size());
    mult1xN(a, B, C);
    return C;
}

void mult1xN(
    const Memory<Quaternion, VRAM_CUDA>& a,
    const Memory<Vector, VRAM_CUDA>& B, 
    Memory<Vector, VRAM_CUDA>& C)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (B.size() + blockSize - 1) / blockSize;
    mult1xN_kernel<<<gridSize, blockSize>>>(a.raw(), B.raw(), C.raw(), B.size());
}

Memory<Vector, VRAM_CUDA> mult1xN(
    const Memory<Quaternion, VRAM_CUDA>& a,
    const Memory<Vector, VRAM_CUDA>& B)
{
    Memory<Vector, VRAM_CUDA> C(B.size());
    mult1xN(a, B, C);
    return C;
}

void mult1xN(
    const Memory<Transform, VRAM_CUDA>& t1,
    const Memory<Transform, VRAM_CUDA>& T2,
    Memory<Transform, VRAM_CUDA>& Tr)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (T2.size() + blockSize - 1) / blockSize;
    mult1xN_kernel<<<gridSize, blockSize>>>(t1.raw(), T2.raw(), Tr.raw(), T2.size());
}

Memory<Transform, VRAM_CUDA> mult1xN(
    const Memory<Transform, VRAM_CUDA>& t1,
    const Memory<Transform, VRAM_CUDA>& T2)
{
    Memory<Transform, VRAM_CUDA> Tr(T2.size());
    mult1xN(t1, T2, Tr);
    return Tr;
}

void mult1xN(
    const Memory<Transform, VRAM_CUDA>& t,
    const Memory<Vector, VRAM_CUDA>& X,
    Memory<Vector, VRAM_CUDA>& C)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (X.size() + blockSize - 1) / blockSize;
    mult1xN_kernel<<<gridSize, blockSize>>>(t.raw(), X.raw(), C.raw(), X.size());
}

Memory<Vector, VRAM_CUDA> mult1xN(
    const Memory<Transform, VRAM_CUDA>& t,
    const Memory<Vector, VRAM_CUDA>& X)
{
    Memory<Vector, VRAM_CUDA> C(X.size());
    mult1xN(t, X, C);
    return C;
}

void mult1xN(
    const Memory<Matrix3x3, VRAM_CUDA>& m1,
    const Memory<Matrix3x3, VRAM_CUDA>& M2,
    Memory<Matrix3x3, VRAM_CUDA>& Mr)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (M2.size() + blockSize - 1) / blockSize;
    mult1xN_kernel<<<gridSize, blockSize>>>(m1.raw(), M2.raw(), Mr.raw(), M2.size());
}

Memory<Matrix3x3, VRAM_CUDA> mult1xN(
    const Memory<Matrix3x3, VRAM_CUDA>& m1,
    const Memory<Matrix3x3, VRAM_CUDA>& M2)
{
    Memory<Matrix3x3, VRAM_CUDA> Mr(M2.size());
    mult1xN(m1, M2, Mr);
    return Mr;
}

void mult1xN(
    const Memory<Matrix3x3, VRAM_CUDA>& m,
    const Memory<Vector, VRAM_CUDA>& X,
    Memory<Vector, VRAM_CUDA>& C)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (X.size() + blockSize - 1) / blockSize;
    mult1xN_kernel<<<gridSize, blockSize>>>(m.raw(), X.raw(), C.raw(), X.size());
}

Memory<Vector, VRAM_CUDA> mult1xN(
    const Memory<Matrix3x3, VRAM_CUDA>& m,
    const Memory<Vector, VRAM_CUDA>& X)
{
    Memory<Vector, VRAM_CUDA> C(X.size());
    mult1xN(m, X, C);
    return C;
}

///////
// #add
void addNxN(
    const Memory<Vector, VRAM_CUDA>& A,
    const Memory<Vector, VRAM_CUDA>& B,
    Memory<Vector, VRAM_CUDA>& C)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (A.size() + blockSize - 1) / blockSize;
    addNxN_kernel<<<gridSize, blockSize>>>(A.raw(), B.raw(), C.raw(), A.size());
}

Memory<Vector, VRAM_CUDA> addNxN(
    const Memory<Vector, VRAM_CUDA>& A,
    const Memory<Vector, VRAM_CUDA>& B)
{
    Memory<Vector, VRAM_CUDA> C(A.size());
    addNxN(A, B, C);
    return C;
}

////////
// #sub
void subNxN(
    const Memory<Vector, VRAM_CUDA>& A,
    const Memory<Vector, VRAM_CUDA>& B,
    Memory<Vector, VRAM_CUDA>& C)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (A.size() + blockSize - 1) / blockSize;
    subNxN_kernel<<<gridSize, blockSize>>>(A.raw(), B.raw(), C.raw(), A.size());
}

Memory<Vector, VRAM_CUDA> subNxN(
    const Memory<Vector, VRAM_CUDA>& A,
    const Memory<Vector, VRAM_CUDA>& B)
{
    Memory<Vector, VRAM_CUDA> C(A.size());
    subNxN(A, B, C);
    return C;
}

void subNx1(
    const Memory<Vector, VRAM_CUDA>& A,
    const Memory<Vector, VRAM_CUDA>& b,
    Memory<Vector, VRAM_CUDA>& C)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (A.size() + blockSize - 1) / blockSize;
    subNx1_kernel<<<gridSize, blockSize>>>(A.raw(), b.raw(), C.raw(), A.size());
}

Memory<Vector, VRAM_CUDA> subNx1(
    const Memory<Vector, VRAM_CUDA>& A,
    const Memory<Vector, VRAM_CUDA>& b)
{
    Memory<Vector, VRAM_CUDA> C(A.size());
    subNx1(A, b, C);
    return C;
}

/////
// #transpose
void transpose(
    const Memory<Matrix3x3, VRAM_CUDA>& A, 
    Memory<Matrix3x3, VRAM_CUDA>& B)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (A.size() + blockSize - 1) / blockSize;
    transpose_kernel<<<gridSize, blockSize>>>(A.raw(), B.raw(), A.size());
}

Memory<Matrix3x3, VRAM_CUDA> transpose(
    const Memory<Matrix3x3, VRAM_CUDA>& A)
{
    Memory<Matrix3x3, VRAM_CUDA> B(A.size());
    transpose(A, B);
    return B;
}

void transpose(
    const Memory<Matrix4x4, VRAM_CUDA>& A,
    Memory<Matrix4x4, VRAM_CUDA>& B)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (A.size() + blockSize - 1) / blockSize;
    transpose_kernel<<<gridSize, blockSize>>>(A.raw(), B.raw(), A.size());
}

Memory<Matrix4x4, VRAM_CUDA> transpose(
    const Memory<Matrix4x4, VRAM_CUDA>& A)
{
    Memory<Matrix4x4, VRAM_CUDA> B(A.size());
    transpose(A, B);
    return B;
}

///////
// #transposeInplace
void transposeInplace(
    Memory<Matrix3x3, VRAM_CUDA>& A)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (A.size() + blockSize - 1) / blockSize;
    transposeInplace_kernel<<<gridSize, blockSize>>>(A.raw(), A.size());
}

//////
// #invert
void invert(
    const Memory<Matrix3x3, VRAM_CUDA>& A, 
    Memory<Matrix3x3, VRAM_CUDA>& B)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (A.size() + blockSize - 1) / blockSize;
    invert_kernel<<<gridSize, blockSize>>>(A.raw(), B.raw(), A.size());
}

Memory<Matrix3x3, VRAM_CUDA> invert(
    const Memory<Matrix3x3, VRAM_CUDA>& A)
{
    Memory<Matrix3x3, VRAM_CUDA> B(A.size());
    invert(A, B);
    return B;
}

void invert(
    const Memory<Matrix4x4, VRAM_CUDA>& A,
    Memory<Matrix4x4, VRAM_CUDA>& B)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (A.size() + blockSize - 1) / blockSize;
    invert_kernel<<<gridSize, blockSize>>>(A.raw(), B.raw(), A.size());
}

Memory<Matrix4x4, VRAM_CUDA> invert(
    const Memory<Matrix4x4, VRAM_CUDA>& A)
{
    Memory<Matrix4x4, VRAM_CUDA> B(A.size());
    invert(A, B);
    return B;
}

void invert(
    const Memory<Transform, VRAM_CUDA>& A,
    Memory<Transform, VRAM_CUDA>& B)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (A.size() + blockSize - 1) / blockSize;
    invert_kernel<<<gridSize, blockSize>>>(A.raw(), B.raw(), A.size());
}

Memory<Transform, VRAM_CUDA> invert(
    const Memory<Transform, VRAM_CUDA>& A)
{
    Memory<Transform, VRAM_CUDA> B(A.size());
    invert(A, B);
    return B;
}

//////
// #divNxN
void divNxN(
    const Memory<Vector, VRAM_CUDA>& A, 
    const Memory<unsigned int, VRAM_CUDA>& B,
    Memory<Vector, VRAM_CUDA>& C)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (A.size() + blockSize - 1) / blockSize;
    divNxN_conv_kernel<float><<<gridSize, blockSize>>>(A.raw(), B.raw(), C.raw(), A.size());
}

Memory<Vector, VRAM_CUDA> divNxN(
    const Memory<Vector, VRAM_CUDA>& A, 
    const Memory<unsigned int, VRAM_CUDA>& B)
{
    Memory<Vector, VRAM_CUDA> C(A.size());
    divNxN(A, B, C);
    return C;
}

void divNxN(
    const Memory<Matrix3x3, VRAM_CUDA>& A, 
    const Memory<unsigned int, VRAM_CUDA>& B, 
    Memory<Matrix3x3, VRAM_CUDA>& C)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (A.size() + blockSize - 1) / blockSize;
    divNxN_conv_kernel<float><<<gridSize, blockSize>>>(A.raw(), B.raw(), C.raw(), A.size());
}

Memory<Matrix3x3, VRAM_CUDA> divNxN(
    const Memory<Matrix3x3, VRAM_CUDA>& A,
    const Memory<unsigned int, VRAM_CUDA>& B)
{
    Memory<Matrix3x3, VRAM_CUDA> C(A.size());
    divNxN(A, B, C);
    return C;
}

///////
// #divNxNIgnoreZeros
void divNxNIgnoreZeros(
    const Memory<Vector, VRAM_CUDA>& A, 
    const Memory<unsigned int, VRAM_CUDA>& B,
    Memory<Vector, VRAM_CUDA>& C)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (A.size() + blockSize - 1) / blockSize;
    divNxNIgnoreZeros_conv_kernel<float><<<gridSize, blockSize>>>(A.raw(), B.raw(), C.raw(), A.size());
}

Memory<Vector, VRAM_CUDA> divNxNIgnoreZeros(
    const Memory<Vector, VRAM_CUDA>& A, 
    const Memory<unsigned int, VRAM_CUDA>& B)
{
    Memory<Vector, VRAM_CUDA> C(A.size());
    divNxNIgnoreZeros(A, B, C);
    return C;
}

void divNxNIgnoreZeros(
    const Memory<Matrix3x3, VRAM_CUDA>& A, 
    const Memory<unsigned int, VRAM_CUDA>& B,
    Memory<Matrix3x3, VRAM_CUDA>& C)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (A.size() + blockSize - 1) / blockSize;
    divNxNIgnoreZeros_conv_kernel<float><<<gridSize, blockSize>>>(A.raw(), B.raw(), C.raw(), A.size());
}

Memory<Matrix3x3, VRAM_CUDA> divNxNIgnoreZeros(
    const Memory<Matrix3x3, VRAM_CUDA>& A, 
    const Memory<unsigned int, VRAM_CUDA>& B)
{
    Memory<Matrix3x3, VRAM_CUDA> C(A.size());
    divNxNIgnoreZeros(A, B, C);
    return C;
}

void divNxNIgnoreZeros(
    const Memory<float, VRAM_CUDA>& A, 
    const Memory<unsigned int, VRAM_CUDA>& B,
    Memory<float, VRAM_CUDA>& C)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (A.size() + blockSize - 1) / blockSize;
    divNxNIgnoreZeros_conv_kernel<float><<<gridSize, blockSize>>>(A.raw(), B.raw(), C.raw(), A.size());
}

Memory<float, VRAM_CUDA> divNxNIgnoreZeros(
    const Memory<float, VRAM_CUDA>& A, 
    const Memory<unsigned int, VRAM_CUDA>& B)
{
    Memory<float, VRAM_CUDA> C(A.size());
    divNxNIgnoreZeros(A, B, C);
    return C;
}

////////
// #divNxNInplace
void divNxNInplace(
    Memory<Vector, VRAM_CUDA>& A, 
    const Memory<float, VRAM_CUDA>& B)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (A.size() + blockSize - 1) / blockSize;
    divNxNInplace_kernel<<<gridSize, blockSize>>>(A.raw(), B.raw(), A.size());
}

void divNxNInplace(
    Memory<Matrix3x3, VRAM_CUDA>& A, 
    const Memory<unsigned int, VRAM_CUDA>& B)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (A.size() + blockSize - 1) / blockSize;
    divNxNInplace_kernel<<<gridSize, blockSize>>>(A.raw(), B.raw(), A.size());
}

////////
// #divNx1Inplace
void divNx1Inplace(
    Memory<Matrix3x3, VRAM_CUDA>& A, 
    const unsigned int& B)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (A.size() + blockSize - 1) / blockSize;
    divNx1Inplace_kernel<<<gridSize, blockSize>>>(A.raw(), B, A.size());
}

void divNx1Inplace(
    Memory<Vector, VRAM_CUDA>& A, 
    const unsigned int& B)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (A.size() + blockSize - 1) / blockSize;
    divNx1Inplace_kernel<<<gridSize, blockSize>>>(A.raw(), B, A.size());
}


////////
// #convert
void convert(
    const Memory<uint8_t, VRAM_CUDA>& from, 
    Memory<float, VRAM_CUDA>& to)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (from.size() + blockSize - 1) / blockSize;
    convert_kernel<<<gridSize, blockSize>>>(from.raw(), to.raw(), from.size());
}

void convert(
    const Memory<bool, VRAM_CUDA>& from, 
    Memory<unsigned int, VRAM_CUDA>& to)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (from.size() + blockSize - 1) / blockSize;
    convert_kernel<<<gridSize, blockSize>>>(from.raw(), to.raw(), from.size());
}

void copy(const Memory<unsigned int, VRAM_CUDA>& from, 
    Memory<bool, VRAM_CUDA>& to)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (from.size() + blockSize - 1) / blockSize;
    convert_kernel<<<gridSize, blockSize>>>(from.raw(), to.raw(), from.size());
}

////////
// #pack
void pack(
    const Memory<Matrix3x3, VRAM_CUDA>& R,
    const Memory<Vector, VRAM_CUDA>& t,
    Memory<Transform, VRAM_CUDA>& T)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (R.size() + blockSize - 1) / blockSize;
    pack_kernel<<<gridSize, blockSize>>>(R.raw(), t.raw(), T.raw(), R.size());
}

void pack(
    const Memory<Quaternion, VRAM_CUDA>& R,
    const Memory<Vector, VRAM_CUDA>& t,
    Memory<Transform, VRAM_CUDA>& T)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (R.size() + blockSize - 1) / blockSize;
    pack_kernel<<<gridSize, blockSize>>>(R.raw(), t.raw(), T.raw(), R.size());
}

////////
// #multNxNTransposed
void multNxNTransposed(
    const Memory<Vector, VRAM_CUDA>& m1,
    const Memory<Vector, VRAM_CUDA>& m2,
    Memory<Matrix3x3, VRAM_CUDA>& Cs)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (m1.size() + blockSize - 1) / blockSize;
    covParts_kernel<<<gridSize, blockSize>>>(m1.raw(), m2.raw(), Cs.raw(), m1.size());
}

Memory<Matrix3x3, VRAM_CUDA> multNxNTransposed(
    const Memory<Vector, VRAM_CUDA>& m1,
    const Memory<Vector, VRAM_CUDA>& m2)
{
    Memory<Matrix3x3, VRAM_CUDA> Cs(m1.size());
    multNxNTransposed(m1, m2, Cs);
    return Cs;
}

void multNxNTransposed(
    const Memory<Vector, VRAM_CUDA>& m1,
    const Memory<Vector, VRAM_CUDA>& m2,
    const Memory<bool, VRAM_CUDA>& mask,
    Memory<Matrix3x3, VRAM_CUDA>& Cs)
{
    constexpr unsigned int blockSize = 64;
    const unsigned int gridSize = (m1.size() + blockSize - 1) / blockSize;
    covParts_kernel<<<gridSize, blockSize>>>(m1.raw(), m2.raw(), mask.raw(), Cs.raw(), m1.size());
}
    
Memory<Matrix3x3, VRAM_CUDA> multNxNTransposed(
    const Memory<Vector, VRAM_CUDA>& m1,
    const Memory<Vector, VRAM_CUDA>& m2,
    const Memory<bool, VRAM_CUDA>& mask)
{
    Memory<Matrix3x3, VRAM_CUDA> Cs(m1.size());
    multNxNTransposed(m1, m2, mask, Cs);
    return Cs;
}

//////////
// #sum
// TODO: check perfomance of sum_kernel
void sum(
    const Memory<Vector, VRAM_CUDA>& data,
    Memory<Vector, VRAM_CUDA>& s)
{
    sum_kernel<1024> <<<1, 1024>>>(data.raw(), s.raw(), data.size() );
}

Memory<Vector, VRAM_CUDA> sum(
    const Memory<Vector, VRAM_CUDA>& data)
{
    Memory<Vector, VRAM_CUDA> s(1);
    sum(data, s);
    return s;
}

//////////
// #mean
void mean(
    const Memory<Vector, VRAM_CUDA>& X,
    Memory<Vector, VRAM_CUDA>& res)
{
    sum(X, res);
    divNx1Inplace(res, X.size());
}

Memory<Vector, VRAM_CUDA> mean(
    const Memory<Vector, VRAM_CUDA>& X)
{
    Memory<Vector, VRAM_CUDA> res(1);
    mean(X, res);
    return res;
}

//////////
// #cov
void cov(
    const Memory<Vector, VRAM_CUDA>& v1,
    const Memory<Vector, VRAM_CUDA>& v2,
    Memory<Matrix3x3, VRAM_CUDA>& C)
{
    cov_kernel<1024> <<<1, 1024>>>(v1.raw(), v2.raw(), C.raw(), v1.size() );
}

Memory<Matrix3x3, VRAM_CUDA> cov(
    const Memory<Vector, VRAM_CUDA>& v1,
    const Memory<Vector, VRAM_CUDA>& v2
)
{
    Memory<Matrix3x3, VRAM_CUDA> C(1);
    cov(v1, v2, C);
    return C;
}


} // namespace rmagine