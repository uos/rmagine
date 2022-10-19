#include "StopWatch.hpp"

namespace rmagine
{

template<typename ClockT>
StopWatch_<ClockT>::StopWatch_()
{
    m_old = ClockT::now();
}

template<typename ClockT>
inline double StopWatch_<ClockT>::toc()
{
    auto t = ClockT::now();

    double elapsed_seconds = std::chrono::duration_cast<
        std::chrono::duration<double> >(t - m_old).count();

    m_old = t;
    return elapsed_seconds;
}

template<typename ClockT>
inline double StopWatch_<ClockT>::tic()
{
    return toc();
}

template<typename ClockT>
inline double StopWatch_<ClockT>::operator()()
{
    return toc();
}

} // namespace rmagine