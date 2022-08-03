#ifndef RMAGINE_UTIL_IDGEN_HPP
#define RMAGINE_UTIL_IDGEN_HPP

#include <memory>
#include <queue>

namespace rmagine
{

class IDGen
{
public:
    unsigned int get();
    unsigned int operator()();

    void give_back(unsigned int number);

private:
    unsigned int m_last = 0;
    std::queue<unsigned int> m_removed;
};

using IDGenPtr = std::shared_ptr<IDGen>;

} // namespace rmagine

#endif // RMAGINE_UTIL_IDGEN_HPP