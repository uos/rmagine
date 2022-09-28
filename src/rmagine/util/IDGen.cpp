#include "rmagine/util/IDGen.hpp"

namespace rmagine
{

unsigned int IDGen::get()
{
    if(m_removed.empty())
    {
        return m_last++;
    } else {
        unsigned int ret = m_removed.front();
        m_removed.pop(); 
        return ret;
    }
}

unsigned int IDGen::operator()()
{
    return get();
}

void IDGen::give_back(unsigned int number)
{
    m_removed.push(number);
}

} // namespace rmagine