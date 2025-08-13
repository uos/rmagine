#include "rmagine/types/MemoryVulkan.hpp"



namespace rmagine
{

size_t MemoryData::memIDcounter = 0;

size_t MemoryData::getNewMemID()
{
    if(memIDcounter == SIZE_MAX)
    {
        throw std::runtime_error("You created way too many memory objects!"); 
    }
    return ++memIDcounter;
}

} // namespace rmagine