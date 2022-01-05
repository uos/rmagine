
#include "imagine/util/optix/OptixUtil.hpp"
#include <optix_stubs.h>
#include <algorithm>

#include <iostream>
#include <fstream>
#include <sstream>

#if OPTIX_VERSION < 70300

// include header as actual code
#include "optix_stack_size.h"

#endif // OPTIX_VERSION < 70300
namespace imagine {

std::string loadProgramPtx(const std::string& program_name)
{
    std::string ptx("");

    // TODO: how to do this properly?
    std::string optix_ptx_dir(IMAGINE_OPTIX_PTX_DIR);
    std::string optix_ptx_glob_dir(IMAGINE_OPTIX_PTX_GLOB_DIR);

    std::string filename = optix_ptx_dir + "/cuda_compile_ptx_1_generated_" + program_name + ".cu.ptx";

    std::ifstream file( filename.c_str() );
    if( file.good() )
    {
        // Found usable source file
        std::stringstream source_buffer;
        source_buffer << file.rdbuf();
        ptx = source_buffer.str();
    } else {
        std::string filename_glob = optix_ptx_glob_dir + "/cuda_compile_ptx_1_generated_" + program_name + ".cu.ptx";
        std::ifstream file_glob( filename_glob.c_str() );
        if(file_glob.good())
        {
            std::stringstream source_buffer;
            source_buffer << file_glob.rdbuf();
            ptx = source_buffer.str();
        } else {
            std::stringstream error_msg;
            error_msg << "ScanProgramRanges could not find its PTX part. Searched at locations: \n";
            error_msg << "- " << filename << "\n";
            error_msg << "- " << filename_glob << "\n";
            throw std::runtime_error(error_msg.str());
        }
    }

    return ptx;
}


} // namespace imagine