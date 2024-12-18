#include <iostream>


#include "rmagine/math/types.h"

#include <rmagine/math/math.h>


#include <rmagine/util/StopWatch.hpp>

#include <rmagine/util/exceptions.h>
#include <rmagine/util/prints.h>

#include <Eigen/Dense>

// #include <cblas.h>

#include <stdint.h>
#include <string.h>


using namespace rmagine;
namespace rm = rmagine;

void rotation_init_test()
{
    std::cout << std::endl;
    std::cout << "--------------------------" << std::endl;
    std::cout << "---- rotation_init_test ----" << std::endl;
    std::cout << "--------------------------" << std::endl;
    std::cout << std::endl;

    // EulerAngles
    rm::EulerAngles e1;
    e1.roll = 0.0;
    e1.pitch = 0.0;
    e1.yaw = M_PI / 2.0;
    rm::EulerAngles e2 = {0.0, 0.0, M_PI / 2.0};
    rm::EulerAngles eI = rm::EulerAngles::Identity();
    
    // Quaternion
    rm::Quaternion q1;
    q1.x = 0.0;
    q1.y = 0.0;
    q1.z = 0.7071068;
    q1.w = 0.7071068;
    rm::Quaternion q2 = {0.0, 0.0, 0.7071068, 0.7071068};
    rm::Quaternion qI = rm::Quaternion::Identity();

    // Matrix3x3
    rm::Matrix3x3 M1;
    M1(0,0) =  0.0; M1(0,1) = -1.0; M1(0,2) =  0.0;
    M1(1,0) =  1.0; M1(1,1) =  0.0; M1(1,2) =  0.0;
    M1(2,0) =  0.0; M1(2,1) =  0.0; M1(2,2) =  1.0;
    rm::Matrix3x3 M2 = {{
        0.0, 1.0, 0.0,
        -1.0, 0.0, 0.0,
        0.0, 0.0, 1.0
    }};
    rm::Matrix3x3 MI = rm::Matrix3x3::Identity();
}

bool rotation_conv_test()
{
    std::cout << std::endl;
    std::cout << "--------------------------------" << std::endl;
    std::cout << "---- rotation_conv_test ----" << std::endl;
    std::cout << "--------------------------------" << std::endl;
    std::cout << std::endl;

    EulerAngles e0;
    e0.roll = -0.1;
    e0.pitch = 0.1;
    e0.yaw = M_PI / 2.0;

    EulerAngles e;
    Quaternion q, q0;
    Matrix3x3 R;

    Vector x1{1.0, 0.0, 0.0};
    Vector x2{0.0, 1.0, 0.0};
    Vector x3{0.0, 0.0, 1.0};

    std::cout << "Euler -> Quat" << std::endl;
    // Euler -> Quat
    q = e0;

    // Quat -> Euler
    e = q;

    if(    fabs(e.roll - e0.roll) > 0.0001 
        || fabs(e.pitch - e0.pitch) > 0.0001 
        || fabs(e.yaw - e0.yaw) > 0.0001)
    {
        RM_THROW(Exception, "Euler -> Quat -> Euler error.");
        return false;
    }

    std::cout << "Euler -> Matrix" << std::endl;
    R = e0;
    

    std::cout << "Matrix -> Euler" << std::endl;
    e = R;
    
    if(    fabs(e.roll - e0.roll) > 0.0001 
        || fabs(e.pitch - e0.pitch) > 0.0001 
        || fabs(e.yaw - e0.yaw) > 0.0001)
    {
        RM_THROW(Exception, "Euler -> Matrix -> Euler error.");
        return false;
    }


    R = q;
    q0 = R;

    if(    fabs(q.x - q0.x) > 0.0001 
        || fabs(q.y - q0.y) > 0.0001 
        || fabs(q.z - q0.z) > 0.0001
        || fabs(q.w - q0.w) > 0.0001)
    {
        RM_THROW(Exception, "Quaternion -> Matrix -> Quaternion error.");
        return false;
    }


    return true;
}

void rotation_conv_test_single(
    rm::EulerAngles e = {0.1, 0.2, 0.3})
{
    { // E -> Q -> E
        rm::Quaternion q = e;
        rm::EulerAngles e1 = q; // correct

        if(fabs(e.roll - e1.roll) > 0.0001 
        || fabs(e.pitch - e1.pitch) > 0.0001 
        || fabs(e.yaw - e1.yaw) > 0.0001)
        {
            RM_THROW(Exception, "E <-> Q failed.");
        }
    }

    { // E -> M -> E
        rm::Matrix3x3 M = e;
        rm::EulerAngles e1 = M; // correct

        if(fabs(e.roll - e1.roll) > 0.0001 
        || fabs(e.pitch - e1.pitch) > 0.0001 
        || fabs(e.yaw - e1.yaw) > 0.0001)
        {
            RM_THROW(Exception, "E <-> M failed.");
        }
    }

    { // M -> Q -> M
        rm::Matrix3x3 M = e;
        rm::Quaternion q = M;
        rm::Matrix3x3 M1 = q; // correct
    
        float tot_error = 0.0;

        auto Mdiff = M1 - M;
        for(size_t i=0; i<3; i++)
        {
            for(size_t j=0; j<3; j++)
            {
                tot_error += abs(Mdiff(i, j));
            }
        }
        tot_error /= 9.0;

        if(tot_error > 0.0001)
        {
            RM_THROW(Exception, "M <-> Q failed.");
        }
    }
}

void rotation_conv_test_double(
    rm::EulerAngles e = {0.1, 0.2, 0.3})
{
    { // E -> Q -> M -> E
        rm::Quaternion q = e;
        rm::Matrix3x3 M = q;
        rm::EulerAngles e1 = M; // correct

        if(fabs(e.roll - e1.roll) > 0.0001 
        || fabs(e.pitch - e1.pitch) > 0.0001 
        || fabs(e.yaw - e1.yaw) > 0.0001)
        {
            RM_THROW(Exception, "E -> Q -> M -> E failed.");
        }
    }

    { // E -> M -> Q -> E
        rm::Matrix3x3 M = e;
        rm::Quaternion q = M;
        rm::EulerAngles e1 = q; // correct

        if(fabs(e.roll - e1.roll) > 0.0001 
        || fabs(e.pitch - e1.pitch) > 0.0001 
        || fabs(e.yaw - e1.yaw) > 0.0001)
        {
            RM_THROW(Exception, "E -> Q -> M -> E failed.");
        }
    }


    { // E -> M -> Q -> M -> E
        rm::Matrix3x3 M = e;
        rm::Quaternion q = M;
        rm::Matrix3x3 M1 = q;
        rm::EulerAngles e1 = M1; // correct

        if(fabs(e.roll - e1.roll) > 0.0001 
        || fabs(e.pitch - e1.pitch) > 0.0001 
        || fabs(e.yaw - e1.yaw) > 0.0001)
        {
            RM_THROW(Exception, "E -> Q -> M -> E failed.");
        }
    }

}

void rotation_conv_test_2()
{

    for(int i=-3; i<4; i++)
    {
        for(int j=-3; j<4; j++)
        {
            for(int k=-3; k<4; k++)
            {
                rm::EulerAngles e = {
                    static_cast<float>(i) / 2.0f,
                    static_cast<float>(j) / 2.0f,
                    static_cast<float>(k) / 2.0f
                };

                rotation_conv_test_single(e);
                rotation_conv_test_double(e);
            }
        }
    }
}


void rotation_apply_test()
{
    rm::Vector3 px = {1.0, 0.0, 0.0};

    rm::EulerAngles e = {0.1, 0.2, M_PI / 2.0};

    rm::EulerAngles ex = {e.roll, 0.0, 0.0};
    rm::EulerAngles ey = {0.0, e.pitch, 0.0};
    rm::EulerAngles ez = {0.0, 0.0, e.yaw};

    rm::Quaternion q = e;
    rm::Quaternion qx = ex;
    rm::Quaternion qy = ey;
    rm::Quaternion qz = ez;

    rm::Matrix3x3  M = e;
    rm::Matrix3x3  Mx = ex;
    rm::Matrix3x3  My = ey;
    rm::Matrix3x3  Mz = ez;

    auto p_q = q * px;
    auto p_M = M * px;

    if( (p_q - p_M).l2norm() > 0.0001 )
    {
        RM_THROW(Exception, "M * p != Q * p");
    }
}


Eigen::Vector3f& eigen_view(Vector3& v)
{
    return *reinterpret_cast<Eigen::Vector3f*>( &v );
}

Eigen::Matrix3f& eigen_view(Matrix3x3& M)
{
    return *reinterpret_cast<Eigen::Matrix3f*>( &M );
}

Eigen::Matrix4f& eigen_view(Matrix4x4& M)
{
    return *reinterpret_cast<Eigen::Matrix4f*>( &M );
}

bool check_Matrix3x3()
{
    std::cout << "---------- checkMatrix3x3" << std::endl;
    EulerAngles e{-0.1, 0.1, M_PI / 2.0};

    Matrix3x3 M;
    M = e;
    
    // shallow copy. 
    Eigen::Matrix3f& Meig_shallow = eigen_view(M);

    // setting this, should set Meig_shallow data as well
    M(0,1) = 10.0;

    if( fabs(Meig_shallow(0,1) - 10.0) > 0.00001 )
    {
        RM_THROW(Exception, "rm Mat3x3 Eigen view error.");
    }

    Eigen::Map<Eigen::Matrix3f> Meig_shallow2(&M(0,0));
    M(0,1) = 0.0;
    if( fabs(Meig_shallow2(0,1)) > 0.00001 )
    {
        RM_THROW(Exception, "rm Mat3x3 Eigen map error.");
    }

    // std::cout << Meig_shallow << std::endl;

    // deep copy
    Eigen::Matrix3f Meig(&M(0,0));

    // Eigen::Matrix3f Meig_inv = Meig.inverse();
    Matrix3x3 M_inv = ~M;
    // std::cout << M_inv << std::endl;

    Eigen::Matrix3f Meig_inv = Meig.inverse();

    for(size_t i=0; i<3; i++)
    {
        for(size_t j=0; j<3; j++)
        {
            if( fabs(M_inv(i, j) - Meig_inv(i,j) ) > 0.00001 )
            {
                RM_THROW(Exception, "rm Mat3x3 inverse is not the same as Eigen Mat inverse.");
            }
        }
    }

    Matrix3x3 I = M_inv * M;
    Eigen::Matrix3f Ieig = Meig_inv * Meig;

    for(size_t i=0; i<3; i++)
    {
        for(size_t j=0; j<3; j++)
        {
            if( fabs(I(i, j) - Ieig(i,j) ) > 0.00001 )
            {
                RM_THROW(Exception, "Mat multiplication of rmagine and Eigen differs too much.");
            }
        }
    }

    if(fabs(Meig.determinant() - M.det() ) > 0.00001)
    {
        RM_THROW(Exception, "Determinant of rmagine and Eigen differs too much.");
    }


    if(fabs(Meig.trace() - M.trace() ) > 0.00001)
    {
        RM_THROW(Exception, "Trace of rmagine and Eigen differs too much.");
    }

    return true;
}

bool check_Matrix4x4()
{
    std::cout << "---------- checkMatrix4x4" << std::endl;
    EulerAngles e{-0.1, 0.1, M_PI / 2.0};

    Matrix4x4 M;
    M.setIdentity();
    M.setRotation(e);

    Eigen::Matrix4f& Meig_shallow = eigen_view(M);
    
    
    M(0,1) = 10.0;

    Vector trans{0.0, 0.0, 1.0};
    M.setTranslation(trans);

    if(fabs( Meig_shallow(0,1) - 10.0) > 0.00001)
    {
        RM_THROW(Exception, "rm Mat4x4 Eigen view error.");
    }

    // std::cout << Meig_shallow << std::endl;

    Matrix4x4 M_inv = M.inv();

    // std::cout << "M = " << std::endl;
    // print(M);
    // std::cout << "M_inv = " << std::endl;
    // print(M_inv);
    // std::cout << "(invRigid)=" << std::endl;
    // print(M.invRigid());
    
    // std::cout << "M_inv * M = " << std::endl;
    // print(I);

    Eigen::Matrix4f Meig(&M(0,0));
    Eigen::Matrix4f Meig_inv = Meig.inverse();

    for(size_t i=0; i<4; i++)
    {
        for(size_t j=0; j<4; j++)
        {
            if( fabs(M_inv(i, j) - Meig_inv(i,j) ) > 0.00001 )
            {
                RM_THROW(Exception, "rm Mat4x4 inverse is not the same as Eigen Mat inverse.");
            }
        }
    }

    Matrix4x4 I = M_inv * M;
    Eigen::Matrix4f Ieig = Meig_inv * Meig;

    for(size_t i=0; i<4; i++)
    {
        for(size_t j=0; j<4; j++)
        {
            if( fabs(I(i, j) - Ieig(i,j) ) > 0.00001 )
            {
                RM_THROW(Exception, "Mat multiplication of rmagine and Eigen differs too much.");
            }
        }
    }

    if(fabs(Meig.determinant() - M.det() ) > 0.00001)
    {
        RM_THROW(Exception, "Determinant of rmagine and Eigen differs too much.");
    }


    if(fabs(Meig.trace() - M.trace() ) > 0.00001)
    {
        RM_THROW(Exception, "Trace of rmagine and Eigen differs too much.");
    }

    return true;
}

void init_test()
{
    Vector2 v2 = {1, 2};
    Vector3 v3 = {1, 2, 3};

    EulerAngles e = {0.0, 1.0, 2.0};
    Quaternion q = {0.0, 1.0, 2.0, 3.0};
    Matrix3x3 M = Matrix3x3::Identity();
    e.set(M);

    Transform T = {q, v3};
    AABB bb;

    Quaternion q2 = EulerAngles{0.0, 0.0, 0.0};
    
    std::cout << q2 << std::endl;
    q2 = {0.0, 0.0, 1.0, 1.0};
    std::cout << q2 << std::endl;

    EulerAngles e2 = {0.0, 0.0, 0.0};
    q2.set(e2);
    std::cout << q2 << std::endl;

    Matrix_<float, 3, 3> R = q2;

    std::cout << R << std::endl;

    // int bla = M;
    // std::cout << bla << std::endl;
}

void math_new()
{
    Matrix4x4 M = Matrix4x4::Identity();
    M = M.mult(3.0);
    M(0, 3) = 1.0;
    std::cout << M << std::endl;

    Vector v = {1.0, 2.0, 3.0};
    std::cout << v << std::endl;

    auto vres = M.rotation().mult(v);
    std::cout << vres << std::endl;

    Vector2 bla = {1, 2};

    Matrix_<float, 2, 2> A;

    bla.l2norm();

    float yaw = 1.0;

    auto R = rm::yaw_to_rot_mat_2d(yaw);

    std::cout << R << std::endl;
}

int main(int argc, char** argv)
{
    std::cout << "Rmagine Test: Basic Math" << std::endl;

    rotation_init_test();
    rotation_conv_test();
    rotation_conv_test_2();

    rotation_apply_test();

    check_Matrix3x3();
    check_Matrix4x4();
    init_test();
    math_new();

    return 0;
}