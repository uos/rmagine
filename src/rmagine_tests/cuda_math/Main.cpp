#include <iostream>
#include <rmagine/math/math.cuh>
#include <rmagine/math/math.h>
#include <rmagine/util/StopWatch.hpp>
#include <rmagine/util/prints.h>

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

void cpu_math()
{
    StopWatch sw;
    double el;

    Memory<Vector, RAM> v1(1);
    Memory<Quaternion, RAM> Q1(1);
    Memory<Transform, RAM> T1(1);
    Memory<Matrix3x3, RAM> M1(1);

    v1[0].setZeros();
    Q1[0].setIdentity();
    T1[0].setIdentity();
    M1[0].setIdentity();

    size_t N = 100000;
    Memory<Vector, RAM> v(N);
    Memory<Quaternion, RAM> Q(N);
    Memory<Transform, RAM> T(N);
    Memory<Matrix3x3, RAM> M(N);

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


    std::cout << std::endl;
    std::cout << "Testing returning NxN functions with N = " << N << std::endl;

    // Math and download
    Memory<Vector, RAM> res(N);
    // run once without measuring time. Cudas first run seems to be slow
    
    sw();
    res = multNxN(Q, v);
    el = sw();
    std::cout << "1. multNxN Quaternion x Vector:" << std::endl; 
    std::cout << "- " << "Runtime: " << el << "s" << std::endl;
    std::cout << "- Result: "<< res[N-1].x << " " << res[N-1].y << " " << res[N-1].z << std::endl;

    sw();
    res = multNxN(T, v);
    el = sw();
    std::cout << "2. multNxN Transform x Vector:" << std::endl;
    std::cout << "- " << "Runtime: " << el << "s" << std::endl;
    std::cout << "- Result: "<< res[N-1].x << " " << res[N-1].y << " " << res[N-1].z << std::endl;

    sw();
    res = multNxN(M, v);
    el = sw();

    std::cout << "3. multNxN Matrix3x3 x Vector:" << std::endl;
    std::cout << "- " << "Runtime: " << el << "s" << std::endl;
    std::cout << "- Result: "<< res[N-1].x << " " << res[N-1].y << " " << res[N-1].z << std::endl;

    std::cout << std::endl;
    std::cout << "Testing none-returning NxN functions with N = " << N << std::endl;
    sw();
    multNxN(Q, v, res);
    el = sw();
    std::cout << "1. multNxN Quaternion x Vector:" << std::endl; 
    std::cout << "- " << "Runtime: " << el << "s" << std::endl;
    std::cout << "- Result: "<< res[N-1].x << " " << res[N-1].y << " " << res[N-1].z << std::endl;


    sw();
    multNxN(T, v, res);
    el = sw();
    std::cout << "2. multNxN Transform x Vector:" << std::endl; 
    std::cout << "- " << "Runtime: " << el << "s" << std::endl;
    std::cout << "- Result: "<< res[N-1].x << " " << res[N-1].y << " " << res[N-1].z << std::endl;


    sw();
    multNxN(M, v, res);
    el = sw();
    std::cout << "3. multNxN Matrix3x3 x Vector:" << std::endl; 
    std::cout << "- " << "Runtime: " << el << "s" << std::endl;
    std::cout << "- Result: "<< res[N-1].x << " " << res[N-1].y << " " << res[N-1].z << std::endl;

    // other functions
    
    std::cout << "Testing multNx1 functions" << std::endl;
    multNx1(Q, Q1, Q);
    Q = multNx1(Q, Q1);
    multNx1(Q, v1, v);
    v = multNx1(Q, v1);
    
    multNx1(T, T1, T);
    T = multNx1(T, T1);
    multNx1(T, v1, v);
    v = multNx1(T, v1);
    
    multNx1(M, M1, M);
    M = multNx1(M, M1);
    multNx1(M, v1, v);
    v = multNx1(M, v1);


    std::cout << "Testing mult1xN functions" << std::endl;
    
    mult1xN(Q1, Q, Q);
    Q = mult1xN(Q1, Q);
    mult1xN(Q1, v, v);
    v = mult1xN(Q1, v);

    mult1xN(T1, T, T);
    T = mult1xN(T1, T);
    mult1xN(T1, v, v);
    v = mult1xN(T1, v);
    
    mult1xN(M1, M, M);
    M = mult1xN(M1, M);
    mult1xN(M1, v, v);
    v = mult1xN(M1, v);
}

void cuda_math()
{
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

    std::cout << "- Vector: " << v[N-1] << std::endl;

    std::cout << "- Quaternion: " << Q[N-1] << std::endl;
    

    std::cout << "- Transform: " << T[N-1] << std::endl;
    

    std::cout << "- Matrix3x3: " << std::endl;
    // std::cout << M[N-1] << std::endl;
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
    
    multNx1(d_T, d_T1, d_T);
    d_T = multNx1(d_T, d_T1);
    multNx1(d_T, d_v1, d_v);
    d_v = multNx1(d_T, d_v1);
    
    multNx1(d_M, d_M1, d_M);
    d_M = multNx1(d_M, d_M1);
    multNx1(d_M, d_v1, d_v);
    d_v = multNx1(d_M, d_v1);


    std::cout << "Testing mult1xN functions" << std::endl;
    
    mult1xN(d_Q1, d_Q, d_Q);
    d_Q = mult1xN(d_Q1, d_Q);
    mult1xN(d_Q1, d_v, d_v);
    d_v = mult1xN(d_Q1, d_v);

    mult1xN(d_T1, d_T, d_T);
    d_T = mult1xN(d_T1, d_T);
    mult1xN(d_T1, d_v, d_v);
    d_v = mult1xN(d_T1, d_v);
    
    mult1xN(d_M1, d_M, d_M);
    d_M = mult1xN(d_M1, d_M);
    mult1xN(d_M1, d_v, d_v);
    d_v = mult1xN(d_M1, d_v);

    // mean

    { // VECTOR
        std::cout << "mean Vector " << std::endl;
        Memory<Vector, RAM> v(10000000);

        for(size_t i=0; i<v.size(); i++)
        {
            float i_f = static_cast<float>(i);
            v[i] = {i_f, i_f*2, i_f*3};
        }

        Memory<Vector, VRAM_CUDA> v_d;
        v_d = v;

        StopWatch sw;
        double el;

        sw();
        auto m_d = mean(v_d);
        el = sw();

        Memory<Vector, RAM> m;
        m = m_d;

        std::cout << "- result: " << m[0] << std::endl;
        std::cout << "- runtime: " << v.size() << " in " << el << "s" << std::endl;
    }
}

void math_cuda_batched()
{
    unsigned int batches = 10000;
    unsigned int batchSize = 1000;

    std::cout << "CUDA Batched functions test. Settings:" << std::endl;
    std::cout << "- batches: " << batches << std::endl;
    std::cout << "- batch size: " << batchSize << std::endl;


    { // SCALAR
        std::cout << "sumBatched Scalar" << std::endl;

        Memory<float, RAM> s(batches * batchSize);

        for(size_t i=0; i<s.size(); i++)
        {
            float i_f = static_cast<float>(i);
            s[i] = i_f; 
        }

        Memory<float, VRAM_CUDA> s_d;
        s_d = s;
        
        Memory<float, VRAM_CUDA> sums_d(batches);
        StopWatch sw;
        double el;
        sw();
        sumBatched(s_d, sums_d);
        el = sw();

        Memory<float, RAM> sums;
        sums = sums_d;

        std::cout << "- runtime: " << s.size() << " in " << el << "s" << std::endl;
        std::cout << "- res: " << sums[0] << std::endl;

    }
    { // VECTOR

        std::cout << "sumBatched Vector" << std::endl;
        Memory<Vector, RAM> v(batches * batchSize);

        for(size_t i=0; i<v.size(); i++)
        {
            float i_f = static_cast<float>(i);
            v[i] = {i_f, i_f*2, i_f*3}; 
        }

        Memory<Vector, VRAM_CUDA> v_d;
        v_d = v;

        Memory<Vector, VRAM_CUDA> sums_d(batches);

        StopWatch sw;
        double el;
        sw();
        sumBatched(v_d, sums_d);
        el = sw();

        Memory<Vector, RAM> sums;
        sums = sums_d;
        
        std::cout << "- runtime: " << v.size() << " in " << el << "s" << std::endl;
        std::cout << "- res: " << sums[0] << std::endl;

    }

    { // MATRIX
        std::cout << "sumBatched Matrix3x3" << std::endl;

        Memory<Matrix3x3, RAM> M(batches * batchSize);


        for(size_t i=0; i<batches; i++)
        {
            for(size_t j=0; j<batchSize; j++)
            {
                M[i * batchSize + j].setIdentity();
            }
        }

        Memory<Matrix3x3, VRAM_CUDA> M_d;
        M_d = M;

        Memory<Matrix3x3, VRAM_CUDA> sums_d(batches);
        StopWatch sw;
        double el;
        sw();
        sumBatched(M_d, sums_d);
        el = sw();
        Memory<Matrix3x3, RAM> sums;
        sums = sums_d;

        std::cout << "- runtime: " << M.size() << " in " << el << "s" << std::endl;
        std::cout << "- res: " << sums[sums.size() - 1] << std::endl;
    }

    // mean

    // { // VECTOR
    //     Memory<Vector, RAM> v(10000000);

    //     for(size_t i=0; i<v.size(); i++)
    //     {
    //         float i_f = static_cast<float>(i);
    //         v[i] = {i_f, i_f*2, i_f*3}; 
    //     }

    //     Memory<Vector, VRAM_CUDA> v_d;
    //     v_d = v;

    //     std::cout << "Mean: " << std::endl;
    //     StopWatch sw;
    //     double el;

    //     sw();
    //     auto m_d = mean(v_d);
    //     el = sw();


    //     Memory<Vector, RAM> m;
    //     m = m_d;

    //     std::cout << m[0] << std::endl;
    //     std::cout << "Runtime: " << v.size() << " in " << el << "s" << std::endl;
    // }


}

int main(int argc, char** argv)
{

    std::cout << "Rmagine Test: CPU Math" << std::endl;
    cpu_math();

    std::cout << "Rmagine Test: Cuda Math" << std::endl;
    cuda_math();

    std::cout << "Rmagine Test: Cuda Math Batched" << std::endl;
    math_cuda_batched();

    return 0;
}