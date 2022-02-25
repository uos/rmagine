#include "rmagine/util/StopWatch.hpp"

namespace rmagine {

StopWatch::StopWatch()
{
    m_old = std::chrono::steady_clock::now();
}

double StopWatch::tic()
{
    return toc();
}

double StopWatch::toc()
{
    auto t = std::chrono::steady_clock::now();

    double elapsed_seconds = std::chrono::duration_cast<
        std::chrono::duration<double> >(t - m_old).count();

    m_old = t;

    return elapsed_seconds;
}

double StopWatch::operator()()
{   
    return toc();
}

} // namespace mamcl