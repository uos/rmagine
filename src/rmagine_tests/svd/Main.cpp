#include <iostream>
#include <rmagine/math/math.cuh>
#include <rmagine/math/math.h>
#include <rmagine/util/StopWatch.hpp>
#include <rmagine/math/SVD_cuda.hpp>
#include <rmagine/util/prints.h>

#include <Eigen/Dense>

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

void print(const Memory<Vector, RAM>& points)
{
    for(size_t i=0; i<points.size(); i++)
    {
        std::cout << points[i].x << " ";
    }
    std::cout << std::endl;

    for(size_t i=0; i<points.size(); i++)
    {
        std::cout << points[i].y << " ";
    }
    std::cout << std::endl;

    for(size_t i=0; i<points.size(); i++)
    {
        std::cout << points[i].z << " ";
    }
    std::cout << std::endl;
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
    std::cout << Tm[0] << std::endl;
    return mult1xN(Tm, points);
}


Eigen::Matrix<float, 3, -1> createPointsEigen()
{
    Eigen::Matrix<float, 3, -1> res(3, 4);

    res(0,0) = 0.0;
    res(1,0) = 0.0;
    res(2,0) = 0.0;

    res(0,1) = 1.0;
    res(1,1) = 0.0;
    res(2,1) = 0.0;

    res(0,2) = 0.0;
    res(1,2) = 1.0;
    res(2,2) = 0.0;

    res(0,3) = 0.0;
    res(1,3) = 0.0;
    res(2,3) = 1.0;

    return res;
}

Eigen::Matrix<float, 3, -1> createTransformedPointsEigen()
{
    auto from = createPointsEigen();


    float roll = 0;
    float pitch = 0;
    float yaw = M_PI / 2.0;

    Eigen::Quaternionf q;
    q = Eigen::AngleAxisf(roll, Eigen::Vector3f::UnitX())
        * Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY())
        * Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ());

    Eigen::Affine3f T;
    T.setIdentity();
    T.linear() = q.matrix();
    return T * from;
}



void rmagine_icp()
{
    auto Pfrom = createPoints();
    print(Pfrom);
    auto Pto = createTransformedPoints();
    print(Pto);

    auto from_mean = mean(Pfrom);
    auto to_mean = mean(Pto);

    Memory<Vector, VRAM_CUDA> from_centered_d;
    Memory<Vector, VRAM_CUDA> to_centered_d;

    from_centered_d = subNx1(Pfrom, from_mean);
    to_centered_d = subNx1(Pto, to_mean);

    auto covs_d = covBatched(from_centered_d, to_centered_d, 4);

    Memory<Matrix3x3, RAM> covs;
    covs = covs_d;

    for(size_t i=0; i<covs.size(); i++)
    {
        print(covs[i]);
    }

    SVD_cuda svd_gpu;
}

void eigen_icp()
{
    auto Pfrom = createPointsEigen();
    std::cout << Pfrom.rows() << "x" << Pfrom.cols() << std::endl;
    std::cout << Pfrom << std::endl;
    auto Pto = createTransformedPointsEigen();
    std::cout << Pto.rows() << "x" << Pto.cols() << std::endl;
    std::cout << Pto << std::endl;

    const size_t N = Pto.cols();
    double N_d = Pfrom.cols();

    const Eigen::Vector3f from_mean = Pfrom.rowwise().mean();
    const Eigen::Vector3f to_mean = Pto.rowwise().mean();
    const auto from_centered = Pfrom.colwise() - from_mean;
    const auto to_centered = Pto.colwise() - to_mean;
    const Eigen::Matrix3f C = to_centered * from_centered.transpose() / N_d;

    std::cout << C << std::endl;

    
}

int main(int argc, char** argv)
{
    std::cout << "Rmagine Test: SVD" << std::endl;
    
    std::cout << "Eigen ICP" << std::endl;
    eigen_icp();

    std::cout << "Rmagine ICP" << std::endl;
    rmagine_icp();



    return 0;
}