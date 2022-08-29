#include "rmagine/util/exceptions.h"

namespace rmagine
{

Exception::Exception(const std::string& msg)
: std::runtime_error(msg.c_str())
{

}

Exception::Exception(
    const std::string& msg, 
    const char* file, 
    const char* func, 
    int line)
: std::runtime_error(msg.c_str())
{
    std::ostringstream ss;
    ss << "\t" << msg << "\n";
    ss << "- File:\t\t'" << file << "'\n";
    ss << "- Location:\t" << func << "\n";
    ss << "- Line:\t\t" << line;
    m_msg = ss.str();
}

const char* Exception::what() const throw()
{
    return m_msg.c_str();
}

Exception::~Exception()
{

}



EmbreeException::EmbreeException(const std::string& msg)
:Exception(msg)
{

}

EmbreeException::EmbreeException(const std::string& msg, const char* file, const char* func, int line)
:Exception(msg, file, func, line)
{

}

EmbreeException::~EmbreeException()
{

}



CudaException::CudaException(const std::string& msg)
:Exception(msg)
{

}

CudaException::CudaException(const std::string& msg, const char* file, const char* func, int line)
:Exception(msg, file, func, line)
{

}

CudaException::~CudaException()
{

}


OptixException::OptixException(const std::string& msg)
:Exception(msg)
{

}

OptixException::OptixException(const std::string& msg, const char* file, const char* func, int line)
:Exception(msg, file, func, line)
{

}

OptixException::~OptixException()
{

}




} // namespace rmagine