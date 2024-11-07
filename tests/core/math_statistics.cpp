#include <iostream>


#include <rmagine/math/types.h>

#include <rmagine/math/math.h>
#include <rmagine/util/StopWatch.hpp>

#include <rmagine/util/exceptions.h>
#include <rmagine/util/prints.h>

#include <rmagine/math/statistics.h>

#include <cassert>

#include <rmagine/math/omp.h>

namespace rm = rmagine;



void test_incremental()
{
    rm::StopWatch sw;
    double el;
    rm::CrossStatistics total = rm::CrossStatistics::Identity();


    size_t n_points = 10000000;

    sw();

    for(size_t i=0; i<n_points; i++)
    {
        rm::Vector3 d = {1.0, 0.0, 1.0};
        rm::Vector3 m = {0.0, 1.0, 0.0};
        total += rm::CrossStatistics::Init(d, m);
    }

    el = sw();

    std::cout << "Incremental: " << el << std::endl;

    std::cout << "Result: " << std::endl;
    std::cout << "- dataset mean: " << total.dataset_mean << std::endl;
    std::cout << "- model mean: " << total.model_mean << std::endl;
    std::cout << "- cov: " << total.covariance << std::endl;
    std::cout << "- n meas: " << total.n_meas << std::endl; 

    assert(total.n_meas == n_points);
}

void test_parallel_reduce()
{
    rm::StopWatch sw;
    double el;
    rm::CrossStatistics total = rm::CrossStatistics::Identity();

    size_t n_points = 10000000;

    sw();

    #pragma omp parallel for reduction(+: total)
    for(size_t i=0; i<n_points; i++)
    {
        rm::Vector3 d = {1.0, 0.0, 1.0};
        rm::Vector3 m = {0.0, 1.0, 0.0};
        total += rm::CrossStatistics::Init(d, m);
    }

    el = sw();

    std::cout << "Parallel Reduce: " << el << std::endl;

    std::cout << "Result: " << std::endl;
    std::cout << "- dataset mean: " << total.dataset_mean << std::endl;
    std::cout << "- model mean: " << total.model_mean << std::endl;
    std::cout << "- cov: " << total.covariance << std::endl;
    std::cout << "- n meas: " << total.n_meas << std::endl; 

    assert(total.n_meas == n_points);
}


int main(int argc, char** argv)
{
    srand((unsigned int) time(0));

    std::cout << "STATS TEST" << std::endl;


    test_incremental();

    test_parallel_reduce();


    



    return 0;
}

