#include <iostream>
#include <imagine/math/math.h>
#include <imagine/math/types.h>
#include <imagine/util/StopWatch.hpp>

#include <Eigen/Dense>

using namespace imagine;

void print(Matrix3x3 M)
{
    for(size_t i=0; i<3; i++)
    {
        for(size_t j=0; j<3; j++)
        {
            std::cout << M(i,j) << " ";
        }
        std::cout << std::endl;
    }
}

void print(Matrix4x4 M)
{
    for(size_t i=0; i<4; i++)
    {
        for(size_t j=0; j<4; j++)
        {
            std::cout << M(i,j) << " ";
        }
        std::cout << std::endl;
    }
}

void print(Eigen::Matrix4f M)
{
    for(size_t i=0; i<4; i++)
    {
        for(size_t j=0; j<4; j++)
        {
            std::cout << M(i,j) << " ";
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
    std::cout << "Imagine Test: Basic Math" << std::endl;
    Transform T;
    T.setIdentity();

    Matrix4x4 M2;
    M2.setIdentity();
    M2.setTranslation({2.0, 1.0, 0.5});

    Matrix3x3 R = M2.rotation();
    
    print(R);
    print(M2);

    Eigen::Matrix4f Meig;
    Meig.setIdentity();
    Meig(0,3) = 2.0;
    Meig(1,3) = 1.0;
    Meig(2,3) = 0.5;

    Eigen::Vector4f xeig(1.0, 2.0, 2.5, 1.0);
    Vector x{1.0, 2.0, 2.5};

    std::cout << Meig << std::endl;
    std::cout << (Meig * xeig).transpose() << std::endl;


    print(M2 * x);


    return 0;
}