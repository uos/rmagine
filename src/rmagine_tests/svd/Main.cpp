#include <iostream>
#include <rmagine/math/math.cuh>
#include <rmagine/math/math.h>
#include <rmagine/util/StopWatch.hpp>
#include <rmagine/math/SVD_cuda.hpp>

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

Memory<Vector, RAM> createPoints()
{
    Memory<Vector, RAM> res(4);

    // Eigen::Matrix<double, 3, -1> res(3, 4);
    res[0].x = 0.0;
    res[0].y = 0.0;
    res[0].z = 0.0,

    res[1].x = 1.0;
    res[1].y = 0.0;
    res[1].z = 0.0;

    res[2].x = 0.0;
    res[2].y = 1.0;
    res[2].z = 0.0;

    res[3].x = 0.0;
    res[3].y = 0.0;
    res[3].z = 1.0;

    return res;
}

Memory<Vector, RAM> createTransformedPoints()
{
    auto points = createPoints();

    Memory<Transform, RAM> Tm(1);
    Transform T;
    T.setIdentity();
    
    EulerAngles e;
    e.roll = 0.0;
    e.pitch = 0.0;
    e.yaw = M_PI / 2.0;
    T.R.set(e);

    Tm[0] = T;

    return mult1xN(Tm, points);
}

int main(int argc, char** argv)
{
    std::cout << "Rmagine Test: SVD" << std::endl;

    SVD_cuda svd_gpu;

    // two point sets: data set and model

    auto Pfrom = createPoints();
    auto Pto = createTransformedPoints();

    StopWatch sw;
    double el;

    sw();
    for(size_t i=0; i<10000;i++)
    {
        auto from_mean = mean(Pfrom);
        auto to_mean = mean(Pto);
    }
    el = sw();
    std::cout << el << "s" << std::endl;

    sw();
    for(size_t i=0; i<10000;i++)
    {
        auto from_mean = mean2(Pfrom);
        auto to_mean = mean2(Pto);
    }
    el = sw();
    std::cout << el << "s" << std::endl;

    // sw();
    // for(size_t i=0; i<10000;i++)
    // {
    //     auto from_mean = mean3(Pfrom);
    //     auto to_mean = mean3(Pto);
    // }
    // el = sw();
    // std::cout << el << "s" << std::endl;

    return 0;
}