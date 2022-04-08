#include <iostream>
#include <rmagine/math/math.cuh>
#include <rmagine/math/math.h>
#include <rmagine/util/StopWatch.hpp>
#include <rmagine/math/SVD_cuda.hpp>
#include <rmagine/util/prints.h>

#include <cblas.h>

#include <Eigen/Dense>

using namespace rmagine;

float roll = 0.0;
float pitch = -M_PI/2.0;
float yaw = M_PI/2.0;

Vector trans = {0.0, 0.0, 1.0};

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

void print(Eigen::Matrix3f M)
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
    T.t = trans;
    
    EulerAngles e;
    e.roll = roll;
    e.pitch = pitch;
    e.yaw = yaw;
    T.R.set(e);

    // std::cout << "- rotate with: " << e << std::endl;
    // std::cout << "- rotate with: " << T.R << std::endl;

    Tm[0] = T;
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

    Eigen::Quaternionf q;

    { // rmagine and ros version
        Quaternion q_;
        EulerAngles e;
        e.roll = roll;
        e.pitch = pitch;
        e.yaw = yaw;
        q_.set(e);
        q.x() = q_.x;
        q.y() = q_.y;
        q.z() = q_.z;
        q.w() = q_.w;
    }

    // rmagine has same results as tf.transformations.quaternion_from_euler
    // eigen gives different results. why?
    // { // eigen version
    // q = Eigen::AngleAxisf(roll, Eigen::Vector3f::UnitX())
    //     * Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY())
    //     * Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ());
    // }
    // std::cout << "- rotate with: " << q.x() << ", " << q.y() << ", " << q.z() << ", " << q.w() << std::endl;

    Eigen::Affine3f T;
    T.setIdentity();
    T.linear() = q.matrix();
    T.translation() = Eigen::Vector3f(trans.x, trans.y, trans.z);
    return T * from;
}

float mean_dist_error(
    const Memory<Vector, RAM>& v1,
    const Memory<Vector, RAM>& v2)
{
    float error = 0.0;

    for(size_t i=0; i<v1.size(); i++)
    {
        float distance = (v1[i] - v2[i]).l2norm();
        error += distance;
    }

    error /= static_cast<float>(v1.size());
    return error;
}

float mean_dist_error(
    const Eigen::Matrix<float, 3, -1>& v1, 
    const Eigen::Matrix<float, 3, -1>& v2 )
{
    float error = 0.0;

    for(size_t i=0; i<v1.cols(); i++)
    {
        float distance = sqrt(
            (v1(0, i) - v2(0, i)) * (v1(0, i) - v2(0, i))
            + (v1(1, i) - v2(1, i)) * (v1(1, i) - v2(1, i))
            + (v1(2, i) - v2(2, i)) * (v1(2, i) - v2(2, i))
        );
        error += distance;
    }

    error /= static_cast<float>(v1.cols());
    return error;
}

void rmagine_icp_gpu()
{
    auto Pfrom = createPoints();
    std::cout << "- From:" << std::endl;
    print(Pfrom);
    auto Pto = createTransformedPoints();
    std::cout << "- To:" << std::endl;
    print(Pto);

    Memory<Vector, VRAM_CUDA> Pfrom_d;
    Memory<Vector, VRAM_CUDA> Pto_d;
    Pfrom_d = Pfrom;
    Pto_d = Pto;

    auto from_mean_d = mean(Pfrom_d);
    auto to_mean_d = mean(Pto_d);

    Memory<Vector, VRAM_CUDA> from_centered_d;
    Memory<Vector, VRAM_CUDA> to_centered_d;

    from_centered_d = subNx1(Pfrom_d, from_mean_d);
    to_centered_d = subNx1(Pto_d, to_mean_d);
    auto C_d = cov(from_centered_d, to_centered_d);

    Memory<Matrix3x3, RAM> C;
    C = C_d;

    std::cout << "- C:" << std::endl;
    // print(C[0]);
    std::cout << C[0] << std::endl;
    // return;
    

    Memory<Matrix3x3, VRAM_CUDA> U_d(C_d.size()), V_d(C_d.size());

    SVD_cuda svd_gpu;
    svd_gpu.calcUV(C_d, U_d, V_d);

    Memory<Matrix3x3, RAM> R;
    Memory<Vector, RAM> t;
    auto R_d = multNxN(U_d, transpose(V_d));
    R = R_d;
    // std::cout << "R: " << R[0] << std::endl;
    
    auto t_d = subNxN(to_mean_d, multNxN(R_d, from_mean_d));
    t = t_d;
    std::cout << "- res t: " << t[0] << std::endl;

    Memory<Transform, VRAM_CUDA> T_d(R_d.size());
    pack(R_d, t_d, T_d);

    Memory<Transform, RAM> T;
    T = T_d;

    std::cout << "- res Q: " << T[0].R << std::endl;

    std::cout << "- res T: " << T[0] << std::endl;
    EulerAngles e;
    e.set(T[0].R);
    std::cout << "- res euler: " << e << std::endl;
    // auto Rs = multNxN(U_d, transpose(V_d));


    auto res_d = mult1xN(T_d, Pfrom_d);
    Memory<Vector, RAM> res;
    res = res_d;

    std::cout << "- From transformed:" << std::endl;
    print(res);

    std::cout << "- ICP error: " << mean_dist_error(res, Pto) << std::endl;

    std::cout << std::endl;
}

void eigen_icp()
{
    auto Pfrom = createPointsEigen();
    std::cout << "- From: " << std::endl;
    // std::cout << Pfrom.rows() << "x" << Pfrom.cols() << std::endl;
    std::cout << Pfrom << std::endl;
    auto Pto = createTransformedPointsEigen();
    // std::cout << Pto.rows() << "x" << Pto.cols() << std::endl;
    std::cout << "- To: " << std::endl;
    std::cout << Pto << std::endl;
    

    const size_t N = Pto.cols();
    double N_d = Pfrom.cols();

    const Eigen::Vector3f from_mean = Pfrom.rowwise().mean();
    const Eigen::Vector3f to_mean = Pto.rowwise().mean();
    const auto from_centered = Pfrom.colwise() - from_mean;
    const auto to_centered = Pto.colwise() - to_mean;
    const Eigen::Matrix3f C = (to_centered * from_centered.transpose()) / N_d;

    std::cout << "- C:" << std::endl;
    std::cout << C << std::endl;
    // print(C);
    // return;
    
    
    // std::cout << C << std::endl;
    // return;

    Eigen::JacobiSVD<Eigen::Matrix3f> svd(C, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3f U = svd.matrixU();
    Eigen::Matrix3f V = svd.matrixV();
    auto sing = svd.singularValues();
    std::cout << "Singular values: " << sing.transpose() << std::endl;

    // why?
    Eigen::Vector3f S = Eigen::Vector3f::Ones(3);
    if( U.determinant() * V.determinant() < 0 )
    {
        std::cout << "my_umeyama special case occurred !!! TODO find out why" << std::endl;
        S(2) = -1;
    }

    Eigen::Matrix4f T = Eigen::Matrix4f::Identity(); 

    // rotational part
    Eigen::Matrix3f R = U * V.transpose();

    // std::cout << "- R:" << std::endl;
    // std::cout << R << std::endl;
    T.block<3,3>(0,0).noalias() = R;
    // translational part
    T.block<3,1>(0,3).noalias() = to_mean - R * from_mean;
    // std::cout << "- res mat" << std::endl;
    // std::cout << T << std::endl;

    Eigen::Affine3f Tr = Eigen::Affine3f::Identity();
    Tr.matrix() = T;
    Eigen::Quaternionf q;
    q = Tr.rotation();
    
    
    Quaternion q_ = {q.x(), q.y(), q.z(), q.w()};
    EulerAngles e;
    e.set(q_);

    std::cout << "- res t: " << Tr.translation().transpose() << std::endl;
    std::cout << "- res Q: " << q.x() << ", " << q.y() << ", " << q.z() << ", " << q.w() << std::endl;
    std::cout << "- res euler: " << e << std::endl;

    auto res = Tr * Pfrom;
    std::cout << "- From transformed: " << std::endl;
    std::cout << res << std::endl;
    std::cout << "- ICP error: " << mean_dist_error(res, Pto) << std::endl;

    std::cout << std::endl;
}

int main(int argc, char** argv)
{
    std::cout << "Rmagine Test: SVD" << std::endl;
    
    std::cout << "Eigen ICP" << std::endl;
    eigen_icp();

    std::cout << "Rmagine ICP" << std::endl;
    rmagine_icp_gpu();



    return 0;
}