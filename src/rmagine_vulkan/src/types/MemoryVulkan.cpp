#include "rmagine/types/MemoryVulkan.hpp"



namespace rmagine
{

size_t MemoryHelper::MemIDcounter = 0;

size_t MemoryHelper::GetNewMemID()
{
    if(MemIDcounter == SIZE_MAX)
    {
        #ifdef VDEBUG
            std::cout << "[MemoryHelper::GetNewMemID()] DEBUG WARNING - created too many MemIDs, restarting at 1!" << std::endl;
        #endif
        ++MemIDcounter;//skip 0 - it is supposed to be an invalid value
    }
    return ++MemIDcounter;
}

} // namespace rmagine