
#include "imagine/util/optix/OptixUtil.hpp"
#include <optix_stubs.h>
#include <algorithm>


namespace imagine {

OptixResult optixUtilAccumulateStackSizes( 
        OptixProgramGroup programGroup, OptixStackSizes* stackSizes )
{
    if( !stackSizes )
        return OPTIX_ERROR_INVALID_VALUE;

    OptixStackSizes localStackSizes;
    OptixResult     result = optixProgramGroupGetStackSize( programGroup, &localStackSizes );
    if( result != OPTIX_SUCCESS )
        return result;

    stackSizes->cssRG = std::max( stackSizes->cssRG, localStackSizes.cssRG );
    stackSizes->cssMS = std::max( stackSizes->cssMS, localStackSizes.cssMS );
    stackSizes->cssCH = std::max( stackSizes->cssCH, localStackSizes.cssCH );
    stackSizes->cssAH = std::max( stackSizes->cssAH, localStackSizes.cssAH );
    stackSizes->cssIS = std::max( stackSizes->cssIS, localStackSizes.cssIS );
    stackSizes->cssCC = std::max( stackSizes->cssCC, localStackSizes.cssCC );
    stackSizes->dssDC = std::max( stackSizes->dssDC, localStackSizes.dssDC );

    return OPTIX_SUCCESS;
}

OptixResult optixUtilComputeStackSizes( 
        const OptixStackSizes* stackSizes,
        unsigned int           maxTraceDepth,
        unsigned int           maxCCDepth,
        unsigned int           maxDCDepth,
        unsigned int*          directCallableStackSizeFromTraversal,
        unsigned int*          directCallableStackSizeFromState,
        unsigned int*          continuationStackSize )
{
    if( !stackSizes )
        return OPTIX_ERROR_INVALID_VALUE;

    const unsigned int cssRG = stackSizes->cssRG;
    const unsigned int cssMS = stackSizes->cssMS;
    const unsigned int cssCH = stackSizes->cssCH;
    const unsigned int cssAH = stackSizes->cssAH;
    const unsigned int cssIS = stackSizes->cssIS;
    const unsigned int cssCC = stackSizes->cssCC;
    const unsigned int dssDC = stackSizes->dssDC;

    if( directCallableStackSizeFromTraversal )
        *directCallableStackSizeFromTraversal = maxDCDepth * dssDC;
    if( directCallableStackSizeFromState )
        *directCallableStackSizeFromState = maxDCDepth * dssDC;

    // upper bound on continuation stack used by call trees of continuation callables
    unsigned int cssCCTree = maxCCDepth * cssCC;

    // upper bound on continuation stack used by CH or MS programs including the call tree of
    // continuation callables
    unsigned int cssCHOrMSPlusCCTree = std::max( cssCH, cssMS ) + cssCCTree;

    // clang-format off
    if( continuationStackSize )
        *continuationStackSize
            = cssRG + cssCCTree
            + ( std::max( maxTraceDepth, 1u ) - 1 ) * cssCHOrMSPlusCCTree
            + std::min( maxTraceDepth, 1u ) * std::max( cssCHOrMSPlusCCTree, cssIS + cssAH );
    // clang-format on

    return OPTIX_SUCCESS;
}

} // namespace imagine