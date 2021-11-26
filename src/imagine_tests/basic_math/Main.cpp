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

void rotationConversionTest()
{
    std::cout << std::endl;
    std::cout << "--------------------------------" << std::endl;
    std::cout << "---- rotationConversionTest ----" << std::endl;
    std::cout << "--------------------------------" << std::endl;
    std::cout << std::endl;

    EulerAngles e;
    Quaternion q;
    Matrix3x3 R;

    e.roll = 0.0;
    e.pitch = 0.0;
    e.yaw = M_PI / 2.0;
    
    Vector x1{1.0, 0.0, 0.0};
    Vector x2{0.0, 1.0, 0.0};
    Vector x3{0.0, 0.0, 1.0};

    std::cout << "Euler -> Quat" << std::endl;
    q = e;
    print(q * x1);
    print(q * x2);
    print(q * x3);
    std::cout << std::endl;

    std::cout << "Euler -> Matrix" << std::endl;
    R = e;
    print(R * x1);
    print(R * x2);
    print(R * x3);
    std::cout << std::endl;


    std::cout << "Quat -> Matrix" << std::endl;
    R = q;
    print(R * x1);
    print(R * x2);
    print(R * x3);
    std::cout << std::endl;

    std::cout << "Matrix -> Quat" << std::endl;
    q = R;
    print(q * x1);
    print(q * x2);
    print(q * x3);
    std::cout << std::endl;


}

int main(int argc, char** argv)
{
    std::cout << "Imagine Test: Basic Math" << std::endl;
    rotationConversionTest();

    return 0;
}