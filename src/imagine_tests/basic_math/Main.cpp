#include <iostream>
#include <imagine/math/math.h>
#include <imagine/math/types.h>
#include <imagine/util/StopWatch.hpp>

using namespace imagine;

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
    std::cout << "Imagine Test: Basic Math" << std::endl;
    Transform T;
    Matrix3x3 M;

    return 0;
}