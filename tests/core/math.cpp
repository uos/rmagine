#include <iostream>

#include "rmagine/math/types.h"

#include <rmagine/math/math.h>
#include <rmagine/math/Matrix.hpp>

#include <rmagine/util/StopWatch.hpp>

#include <rmagine/util/exceptions.h>
#include <rmagine/util/prints.h>

#include <Eigen/Dense>

// #include <cblas.h>

#include <stdint.h>
#include <string.h>


using namespace rmagine;
namespace rm = rmagine;

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

void rotationInitTest()
{
    std::cout << std::endl;
    std::cout << "--------------------------" << std::endl;
    std::cout << "---- rotationInitTest ----" << std::endl;
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
        {0.0, 1.0, 0.0},
        {-1.0, 0.0, 0.0},
        {0.0, 0.0, 1.0}
    }};
    rm::Matrix3x3 MI = rm::Matrix3x3::Identity();
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
    
    // shallow copy. 
    Eigen::Matrix3f& Meig_shallow = eigenView(M);

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

bool checkMatrix4x4()
{
    std::cout << "---------- checkMatrix4x4" << std::endl;
    EulerAngles e{-0.1, 0.1, M_PI / 2.0};

    Matrix4x4 M;
    M.setIdentity();
    M.setRotation(e);

    Eigen::Matrix4f& Meig_shallow = eigenView(M);
    
    
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


template<typename DataT, unsigned int Rows, unsigned int Cols>
inline std::ostream& operator<<(std::ostream& os, const rm::Matrix<DataT, Rows, Cols>& M)
{
    os << "M" << M.rows() << "x" << M.cols() << "[\n";
    for(unsigned int i=0; i<M.rows(); i++)
    {
        for(unsigned int j=0; j<M.cols(); j++)
        {
            os << " " << M(i, j);
        }
        os << "\n";
    }
    os << "]";
    return os;
}



void mathNew()
{
    {
        Matrix<float, 3, 4000> A;
        Eigen::Matrix<float, A.rows(), A.cols()> Aeig;
        // Eigen::Matrix<float, A.rows(), A.cols()>& Aeig = *reinterpret_cast<
        //     Eigen::Matrix<float, A.rows(), A.cols()>* >(&A);

        for(unsigned int i = 0; i < A.rows(); i++)
        {
            for(unsigned int j = 0; j < A.cols(); j++)
            {
                if(j % 2) {
                    A(i, j) =   static_cast<float>(i * A.cols() + j) / static_cast<float>(A.cols() * A.rows());
                } else {
                    A(i, j) = - static_cast<float>(i * A.cols() + j) / static_cast<float>(A.cols() * A.rows());
                }
                Aeig(i, j) = A(i, j);
            }
        }

        std::cout << "A: " << std::endl;
        // std::cout << A << std::endl;
        // std::cout << Aeig << std::endl;

        Matrix<float, A.cols(), 2> B;

        // Eigen::Matrix<float, B.rows(), B.cols()>& Beig = *reinterpret_cast<
        //     Eigen::Matrix<float, B.rows(), B.cols()>* >(&B);
        
        Eigen::Matrix<float, B.rows(), B.cols()> Beig;

        for(unsigned int i=0; i < B.rows(); i++)
        {
            for(unsigned int j=0; j < B.cols(); j++)
            {
                if(j % 2) {
                    B(i, j) =   static_cast<float>(i * B.cols() + j) / static_cast<float>(B.cols() * B.rows());
                } else {
                    B(i, j) = - static_cast<float>(i * B.cols() + j) / static_cast<float>(B.cols() * B.rows());
                }
                Beig(i, j) = B(i, j);
            }
        }

        std::cout << "B: " << std::endl;
        // std::cout << B << std::endl;
        // std::cout << Beig << std::endl;
        
        StopWatchHR sw;
        double el1, el2;

        sw();
        auto C = A * B;
        el1 = sw();

        sw();
        Eigen::Matrix<float, C.rows(), C.cols()> Ceig = Aeig * Beig;
        el2 = sw();

        std::cout << "C: " << std::endl;
        std::cout << C << std::endl;
        std::cout << el1 << "s" << std::endl;

        std::cout << Ceig << std::endl;
        std::cout << el2 << "s" << std::endl;

        using ResT = decltype(A)::MultResultType<decltype(B)>;

        std::cout << ResT::rows() << "x" << ResT::cols() << std::endl;
    }



    {
        std::cout << "Matrix Vector Test" << std::endl;
        Matrix<float, 3, 3> M;
        M.setIdentity();
        std::cout << M << std::endl;

        auto Minv = M.invRigid();

        Matrix<float, 3, 1> v;
        v(0, 0) = 1.0;
        v(1, 0) = 2.0;
        v(2, 0) = 3.0;

        std::cout << v << std::endl;

        auto res = M * v;
        std::cout << res.rows() << "x" << res.cols() << std::endl;

        std::cout << res << std::endl;

        Eigen::Matrix<float, 3, 3> Meig = Eigen::Matrix<float, 3, 3>::Identity();
        Eigen::Matrix<float, 3, 1> veig;
        veig(0) = 1.0;
        veig(1) = 2.0;
        veig(2) = 3.0;

        Eigen::Matrix<float, 3, 1> reseig = Meig * veig;
        std::cout << veig << std::endl;


        // Meig(0) = 1.0;
    }


    {
        Matrix<float, 3, 3> A = {{
            {0.0, 3.0, 6.0},
            {1.0, 4.0, 7.0},
            {2.0, 5.0, 8.0}
        }};

        std::cout << A << std::endl;

        Matrix<float, 3, 3> B;
        B.setIdentity();
        B = B.mult(5.0f);

        std::cout << A * B << std::endl;

        A.multInplace(B);

        std::cout << A << std::endl;
    }
}

int main(int argc, char** argv)
{
    std::cout << "Rmagine Test: Basic Math" << std::endl;
    // rotationInitTest();
    // rotationConversionTest();

    // checkMatrix3x3();
    // checkMatrix4x4();

    mathNew();

    return 0;
}