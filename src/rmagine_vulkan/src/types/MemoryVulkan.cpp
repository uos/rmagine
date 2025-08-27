#include "rmagine/types/MemoryVulkan.hpp"



namespace rmagine
{

size_t MemoryHelper::MemIDcounter = 0;

size_t MemoryHelper::GetNewMemID()
{
    if(MemIDcounter == SIZE_MAX)
    {
        throw std::runtime_error("You created way too many memory objects!"); 
    }
    return ++MemIDcounter;
}

} // namespace rmagine