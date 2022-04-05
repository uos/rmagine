#include <iostream>
#include <rmagine/math/math.cuh>
#include <rmagine/math/math.h>
#include <rmagine/util/StopWatch.hpp>

using namespace rmagine;

void print(Matrix3x3 M)
{
    for(size_t i=0; i<3; i++)
    {
        for(size_t j=0; j<3; j++)
        {
            std::cout << M[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

void print(Vector v)
{
    std::cout << v.x << " " << v.y << " " << v.z << std::endl;
}

void print(Quaternion q)
{
    std::cout << q.x << " " << q.y << " " << q.z << " " << q.w << std::endl;
}

void print(Transform T)
{
    print(T.R);
    print(T.t);
}

int main(int argc, char** argv)
{
    std::cout << "Rmagine Test: Cuda Math" << std::endl;

    StopWatch sw;
    double el;


    Memory<Vector, RAM_CUDA> v1(1);
    Memory<Quaternion, RAM_CUDA> Q1(1);
    Memory<Transform, RAM_CUDA> T1(1);
    Memory<Matrix3x3, RAM_CUDA> M1(1);

    v1[0].setZeros();
    Q1[0].setIdentity();
    T1[0].setIdentity();
    M1[0].setIdentity();

    size_t N = 100000;
    Memory<Vector, RAM_CUDA> v(N);
    Memory<Quaternion, RAM_CUDA> Q(N);
    Memory<Transform, RAM_CUDA> T(N);
    Memory<Matrix3x3, RAM_CUDA> M(N);

    for(size_t i=0; i<N; i++)
    {
        v[i].x = 1.0;
        v[i].y = 2.0;
        v[i].z = 3.0;

        Q[i].x = 0.0;
        Q[i].y = 0.0;
        Q[i].z = 0.0;
        Q[i].w = 1.0;

        T[i].R = Q[i];
        T[i].t.x = 1.0;
        T[i].t.y = 0.0;
        T[i].t.z = -1.0;

        for(size_t j=0; j<3; j++)
        {
            for(size_t k=0; k<3; k++)
            {
                if(j == k)
                {
                    M[i][j][k] = 1.0;
                } else {
                    M[i][j][k] = 0.0;
                }
            }
        }
    }

    std::cout << "Test Data:" << std::endl;

    std::cout << "- Vector: " << std::endl;
    print(v[N-1]);

    std::cout << "- Quaternion: " << std::endl;
    print(Q[N-1]);

    std::cout << "- Transform: " << std::endl;
    print(T[N-1]);

    std::cout << "- Matrix3x3: " << std::endl;
    print(M[N-1]);

    // Upload
    Memory<Vector, VRAM_CUDA> d_v, d_v1;
    Memory<Quaternion, VRAM_CUDA> d_Q, d_Q1;
    Memory<Transform, VRAM_CUDA> d_T, d_T1;
    Memory<Matrix3x3, VRAM_CUDA> d_M, d_M1;
    d_v = v;
    d_Q = Q;
    d_T = T;
    d_M = M;
    d_v1 = v1;
    d_Q1 = Q1;
    d_T1 = T1;
    d_M1 = M1;



    std::cout << std::endl;
    std::cout << "Testing returning NxN functions with N = " << N << std::endl;

    // Math and download
    Memory<Vector, RAM_CUDA> res(N);
    Memory<Vector, VRAM_CUDA> res_gpu(N);
    // run once without measuring time. Cudas first run seems to be slow
    res_gpu = multNxN(d_Q, d_v);
    sw();
    res_gpu = multNxN(d_Q, d_v);
    el = sw();
    res = res_gpu;
    std::cout << "1. multNxN Quaternion x Vector:" << std::endl; 
    std::cout << "- " << "Runtime: " << el << "s" << std::endl;
    std::cout << "- Result: "<< res[N-1].x << " " << res[N-1].y << " " << res[N-1].z << std::endl;

    sw();
    res_gpu = multNxN(d_T, d_v);
    el = sw();
    res = res_gpu;
    std::cout << "2. multNxN Transform x Vector:" << std::endl;
    std::cout << "- " << "Runtime: " << el << "s" << std::endl;
    std::cout << "- Result: "<< res[N-1].x << " " << res[N-1].y << " " << res[N-1].z << std::endl;

    sw();
    res_gpu = multNxN(d_M, d_v);
    el = sw();
    res = res_gpu;

    std::cout << "3. multNxN Matrix3x3 x Vector:" << std::endl;
    std::cout << "- " << "Runtime: " << el << "s" << std::endl;
    std::cout << "- Result: "<< res[N-1].x << " " << res[N-1].y << " " << res[N-1].z << std::endl;

    std::cout << std::endl;
    std::cout << "Testing none-returning NxN functions with N = " << N << std::endl;
    sw();
    multNxN(d_Q, d_v, res_gpu);
    el = sw();
    res = res_gpu;
    std::cout << "1. multNxN Quaternion x Vector:" << std::endl; 
    std::cout << "- " << "Runtime: " << el << "s" << std::endl;
    std::cout << "- Result: "<< res[N-1].x << " " << res[N-1].y << " " << res[N-1].z << std::endl;


    sw();
    multNxN(d_T, d_v, res_gpu);
    el = sw();
    res = res_gpu;
    std::cout << "2. multNxN Transform x Vector:" << std::endl; 
    std::cout << "- " << "Runtime: " << el << "s" << std::endl;
    std::cout << "- Result: "<< res[N-1].x << " " << res[N-1].y << " " << res[N-1].z << std::endl;


    sw();
    multNxN(d_M, d_v, res_gpu);
    el = sw();
    res = res_gpu;
    std::cout << "3. multNxN Matrix3x3 x Vector:" << std::endl; 
    std::cout << "- " << "Runtime: " << el << "s" << std::endl;
    std::cout << "- Result: "<< res[N-1].x << " " << res[N-1].y << " " << res[N-1].z << std::endl;

    // other functions
    
    std::cout << "Testing multNx1 functions" << std::endl;
    multNx1(d_Q, d_Q1, d_Q);
    d_Q = multNx1(d_Q, d_Q1);
    multNx1(d_Q, d_v1, d_v);
    d_v = multNx1(d_Q, d_v1);
    multNx1(d_T, d_v1, d_v);
    





    return 0;
}