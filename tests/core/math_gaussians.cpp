#include <iostream>


#include <rmagine/math/types.h>

#include <rmagine/math/memory_math.h>
#include <rmagine/util/StopWatch.hpp>

#include <rmagine/util/exceptions.h>
#include <rmagine/util/prints.h>

#include <rmagine/math/statistics.h>

#include <cassert>

#include <rmagine/math/omp.h>

#include <algorithm>
#include <random>

namespace rm = rmagine;

template<typename T>
bool is_valid(T a)
{
    return a == a;
}

template<typename DataT>
void printStats(rm::CrossStatistics_<DataT> stats)
{
    std::cout << "CrossStatistics: " << std::endl;
    std::cout << "- dataset mean: " << stats.dataset_mean << std::endl;
    std::cout << "- model mean: " << stats.model_mean << std::endl;
    std::cout << "- cov: " << stats.covariance << std::endl;
    std::cout << "- n meas: " << stats.n_meas << std::endl; 
}

rm::Gaussian1D compute_gaussian_twopass(const std::vector<float>& values)
{
    rm::Gaussian1D ret;

    // first pass
    ret.mean = 0.0;
    for(const float& elem : values)
    {
        ret.mean += elem;
    }
    ret.mean /= static_cast<float>(values.size());

    // second pass
    ret.sigma = 0.0;
    for(const float& elem : values)
    {
        ret.sigma += (elem - ret.mean) * (elem - ret.mean);
    }
    ret.sigma /= static_cast<float>(values.size());
    ret.n_meas = values.size();
    return ret;
}

rm::Gaussian2D compute_gaussian_twopass(const std::vector<rm::Vector2f>& values)
{
    rm::Gaussian2D ret;

    // first pass
    ret.mean = {0.0, 0.0};
    for(const rm::Vector2f& elem : values)
    {
        ret.mean += elem;
    }
    ret.mean /= static_cast<float>(values.size());

    // second pass
    ret.sigma.setZeros();
    for(const rm::Vector2f& elem : values)
    {
        ret.sigma += (elem - ret.mean).multT(elem - ret.mean);
    }
    ret.sigma /= static_cast<float>(values.size());
    ret.n_meas = values.size();
    return ret;
}

rm::Gaussian3D compute_gaussian_twopass(const std::vector<rm::Vector3f>& values)
{
    rm::Gaussian3D ret;

    // first pass
    ret.mean = {0.0, 0.0, 0.0};
    for(const rm::Vector3f& elem : values)
    {
        ret.mean += elem;
    }
    ret.mean /= static_cast<float>(values.size());

    // second pass
    ret.sigma.setZeros();
    for(const rm::Vector3f& elem : values)
    {
        ret.sigma += (elem - ret.mean).multT(elem - ret.mean);
    }
    ret.sigma /= static_cast<float>(values.size());
    ret.n_meas = values.size();
    return ret;
}

rm::Gaussian1D compute_gaussian(const std::vector<float>& values)
{
    rm::Gaussian1D ret = rm::Gaussian1D::Identity();

    for(const float& elem : values)
    {
        ret += rm::Gaussian1D::Init(elem);
    }
    
    return ret;
}

rm::Gaussian2D compute_gaussian(const std::vector<rm::Vector2f>& values)
{
    rm::Gaussian2D ret = rm::Gaussian2D::Identity();

    for(const rm::Vector2f& elem : values)
    {
        ret += rm::Gaussian2D::Init(elem);
    }
    
    return ret;
}

rm::Gaussian3D compute_gaussian(const std::vector<rm::Vector3f>& values)
{
    rm::Gaussian3D ret = rm::Gaussian3D::Identity();

    for(const rm::Vector3f& elem : values)
    {
        ret += rm::Gaussian3D::Init(elem);
    }
    
    return ret;
}

bool equal(rm::Gaussian1D g1, rm::Gaussian1D g2)
{
    if(fabs(g1.mean - g2.mean) > 0.00001)
    {
        return false;
    }

    if(fabs(g1.sigma - g2.sigma) > 0.00001)
    {
        return false;
    }

    if(g1.n_meas != g2.n_meas)
    {
        return false;
    }

    return true;
}

bool equal(rm::Gaussian2D g1, rm::Gaussian2D g2)
{
    if((g1.mean - g2.mean).l2norm() > 0.00001)
    {
        return false;
    }

    const auto diff = g1.sigma - g2.sigma;

    float error = 0.0;
    for(size_t i=0; i<2; i++)
    {
      for(size_t j=0; j<2; j++)
      {
        error += diff(i, j);
      }
    }
    error /= 4.0;


    if(error > 0.00001)
    {
        return false;
    }

    if(g1.n_meas != g2.n_meas)
    {
        return false;
    }

    return true;
}

bool equal(rm::Gaussian3D g1, rm::Gaussian3D g2)
{
    if((g1.mean - g2.mean).l2norm() > 0.00001)
    {
        return false;
    }

    const auto diff = g1.sigma - g2.sigma;

    float error = 0.0;
    for(size_t i=0; i<3; i++)
    {
      for(size_t j=0; j<3; j++)
      {
        error += diff(i, j);
      }
    }
    error /= 9.0;

    if(error > 0.00001)
    {
        return false;
    }

    if(g1.n_meas != g2.n_meas)
    {
        return false;
    }

    return true;
}

void test_gaussians_1d_basic()
{
    std::cout << std::endl;
    std::cout << "-- test_gaussians_1d_basic" << std::endl;
    std::vector<float> values(100);
    for(int i=0; i<values.size(); i++)
    {
        values[i] = static_cast<float>(i % 10 - 5);
    }

    rm::Gaussian1D g1 = compute_gaussian_twopass(values);
    rm::Gaussian1D g2 = compute_gaussian(values);

    std::cout << g1 << std::endl;
    std::cout << g2 << std::endl;

    if(!equal(g1, g2))
    {
      RM_THROW(rm::Exception, "test_gaussians_1d_basic -- fail: g1 != g2.");
    }
}

void test_gaussians_2d_basic()
{
    std::cout << std::endl;
    std::cout << "-- test_gaussians_2d_basic" << std::endl;

    std::vector<rm::Vector2f> values(100);
    for(int i=0; i<values.size(); i++)
    {
        values[i] = {
          static_cast<float>(i % 10 - 5),
          static_cast<float>(i % 30 - 40)
        };
    }

    rm::Gaussian2D g1 = compute_gaussian_twopass(values);
    rm::Gaussian2D g2 = compute_gaussian(values);

    std::cout << g1 << std::endl;
    std::cout << g2 << std::endl;

    if(!equal(g1, g2))
    {
      RM_THROW(rm::Exception, "test_gaussians_2d_basic -- fail: g1 != g2.");
    }
}

void test_gaussians_3d_basic()
{
    std::cout << std::endl;
    std::cout << "-- test_gaussians_3d_basic" << std::endl;

    std::vector<rm::Vector3f> values(100);
    for(int i=0; i<values.size(); i++)
    {
        values[i] = {
          static_cast<float>(i % 10 - 5),
          static_cast<float>(i % 30 - 40),
          static_cast<float>(i % 3 + i % 20 - 10)
        };
    }

    rm::Gaussian3D g1 = compute_gaussian_twopass(values);
    rm::Gaussian3D g2 = compute_gaussian(values);

    std::cout << g1 << std::endl;
    std::cout << g2 << std::endl;

    if(!equal(g1, g2))
    {
      RM_THROW(rm::Exception, "test_gaussians_3d_basic -- fail: g1 != g2.");
    }
}

int main(int argc, char** argv)
{
    srand((unsigned int) time(0));

    std::cout << "GAUSSIANS TEST" << std::endl;

    test_gaussians_1d_basic();
    test_gaussians_2d_basic();
    test_gaussians_3d_basic();



    return 0;
}

