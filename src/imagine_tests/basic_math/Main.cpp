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

bool rotationConversionTest()
{
    std::cout << std::endl;
    std::cout << "--------------------------------" << std::endl;
    std::cout << "---- rotationConversionTest ----" << std::endl;
    std::cout << "--------------------------------" << std::endl;
    std::cout << std::endl;

    EulerAngles e0;
    e0.roll = -0.1;
    e0.pitch = 0.1;
    e0.yaw = M_PI / 2.0;

    EulerAngles e;
    Quaternion q;
    Matrix3x3 R;

    Vector x1{1.0, 0.0, 0.0};
    Vector x2{0.0, 1.0, 0.0};
    Vector x3{0.0, 0.0, 1.0};

    std::cout << "Euler -> Quat" << std::endl;
    q = e0;
    print(q * x1);
    print(q * x2);
    print(q * x3);
    std::cout << std::endl;

    std::cout << "Quat -> Euler" << std::endl;
    e = q;
    std::cout << e.roll << " " << e.pitch << " " << e.yaw << std::endl;
    std::cout << std::endl;

    if(    fabs(e.roll - e0.roll) > 0.0001 
        || fabs(e.pitch - e0.pitch) > 0.0001 
        || fabs(e.yaw - e0.yaw) > 0.0001)
    {
        std::cout << "Euler -> Quat -> Euler error." << std::endl;
        return false;
    }

    std::cout << "Euler -> Matrix" << std::endl;
    R = e0;
    print(R * x1);
    print(R * x2);
    print(R * x3);
    std::cout << std::endl;

    std::cout << "Matrix -> Euler" << std::endl;
    e = R;
    std::cout << e.roll << " " << e.pitch << " " << e.yaw << std::endl;
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

    return true;
}

Eigen::Vector3f& eigenView(Vector3& v)
{
    return *reinterpret_cast<Eigen::Vector3f*>( &v );
}

Eigen::Matrix3f& eigenView(Matrix3x3& M)
{
    return *reinterpret_cast<Eigen::Matrix3f*>( &M );
}

Eigen::Matrix4f& eigenView(Matrix4x4& M)
{
    return *reinterpret_cast<Eigen::Matrix4f*>( &M );
}

bool checkMatrix3x3()
{
    std::cout << "---------- checkMatrix3x3" << std::endl;
    EulerAngles e{-0.1, 0.1, M_PI / 2.0};

    Matrix3x3 M;
    M = e;
    M(0,1) = 10.0;

    // shallow copy. 
    Eigen::Matrix3f& Meig_shallow = eigenView(M);
    std::cout << Meig_shallow << std::endl;

    // deep copy
    Eigen::Matrix3f Meig(&M(0,0));


    // Eigen::Matrix3f Meig_inv = Meig.inverse();
    Matrix3x3 M_inv = ~M;
    Eigen::Matrix3f Meig_inv = Meig.inverse();

    // std::cout << Meig_inv << std::endl;
    // print(M_inv);
    Matrix3x3 I = M_inv * M;
    Eigen::Matrix3f Ieig = Meig_inv * Meig;

    std::cout << "M = " << std::endl;
    print(M);

    std::cout << "M_inv = " << std::endl;
    print(M_inv);

    std::cout << "M_inv * M = " << std::endl;
    print(I);

    std::cout << "Meig = " << std::endl;
    std::cout << Meig << std::endl;

    std::cout << "Meig_inv = " << std::endl;
    std::cout << Meig_inv << std::endl;

    std::cout << "Meig_inv * Meig =" << std::endl;
    std::cout << Meig_inv * Meig <<  std::endl;
    

    std::cout << "Eigen::Matrix3f stats: " << std::endl;
    std::cout << "- det: " << Meig.determinant() << std::endl;
    std::cout << "- trace: " << Meig.trace() << std::endl;
    

    std::cout << "Matrix3x3 stats:" << std::endl;
    std::cout << "- det: " << M.det() << std::endl;
    std::cout << "- trace: " << M.trace() << std::endl;

    return true;
}

bool checkMatrix4x4()
{
    std::cout << "------- checkMatrix4x4" << std::endl;
    EulerAngles e{-0.1, 0.1, M_PI / 2.0};

    Matrix4x4 M;
    M.setIdentity();
    M.setRotation(e);

    Eigen::Matrix4f& Meig_shallow = eigenView(M);
    
    
    // M(0,1) = 10.0;

    Vector trans{0.0, 0.0, 1.0};
    M.setTranslation(trans);

    std::cout << Meig_shallow << std::endl;

    Matrix4x4 M_inv = M.inv();
    Matrix4x4 I = M_inv * M;


    std::cout << "M = " << std::endl;
    print(M);
    std::cout << "M_inv = " << std::endl;
    print(M_inv);
    std::cout << "(invRigid)=" << std::endl;
    print(M.invRigid());
    
    std::cout << "M_inv * M = " << std::endl;
    print(I);

    Eigen::Matrix4f Meig(&M(0,0));
    Eigen::Matrix4f Meig_inv = Meig.inverse();

    std::cout << "Meig = " << std::endl;
    std::cout << Meig << std::endl;

    std::cout << "Meig_inv = " << std::endl;
    std::cout << Meig_inv << std::endl;

    std::cout << "Meig_inv * Meig =" << std::endl;
    std::cout << Meig_inv * Meig <<  std::endl;


    std::cout << "Eigen::Matrix4f stats: " << std::endl;
    std::cout << "- det: " << Meig.determinant() << std::endl;
    std::cout << "- trace: " << Meig.trace() << std::endl;
    
    std::cout << "Matrix4x4 stats:" << std::endl;
    std::cout << "- det: " << M.det() << std::endl;
    std::cout << "- trace: " << M.trace() << std::endl;

    return true;
}


int main(int argc, char** argv)
{
    std::cout << "Imagine Test: Basic Math" << std::endl;
    // rotationConversionTest();


    // checkMatrix3x3();
    // checkMatrix4x4();

    Vector ab{1.0, 2.0, 3.0};
    Vector ac{4.0, 5.0, 6.0};
    Vector n{7.0, 8.0, 9.0};


    Eigen::Matrix3f R;
    R.col(0) = Eigen::Vector3f(ab.x, ab.y, ab.z);
    R.col(1) = Eigen::Vector3f(ac.x, ac.y, ac.z);
    R.col(2) = Eigen::Vector3f(n.x, n.y, n.z);

    std::cout << R << std::endl;

    R(0,0) = ab.x;
    R(1,0) = ab.y;
    R(2,0) = ab.z;

    R(0,1) = ac.x;
    R(1,1) = ac.y;
    R(2,1) = ac.z;

    R(0,2) = n.x;
    R(1,2) = n.y;
    R(2,2) = n.z;

    std::cout << R << std::endl;


    return 0;
}