#ifndef RMAGINE_UTIL_EXCEPTIONS_H
#define RMAGINE_UTIL_EXCEPTIONS_H



#include <stdexcept>
#include <sstream>


#define RM_THROW(Type, arg) throw Type(arg, __FILE__, __PRETTY_FUNCTION__, __LINE__)


namespace rmagine
{

class Exception
: public std::runtime_error
{
public:
    Exception(const std::string& msg);
    Exception(const std::string& msg, const char* file, const char* func, int line);
    const char* what() const throw();
    ~Exception() throw();
private:
    std::string m_msg;
};


class EmbreeException 
: public Exception
{
public:
    EmbreeException(const std::string& msg);
    EmbreeException(const std::string& msg, const char* file, const char* func, int line);
    ~EmbreeException() throw();
};



class CudaException 
: public Exception
{
public:
    CudaException(const std::string& msg);
    CudaException(const std::string& msg, const char* file, const char* func, int line);
    ~CudaException() throw();
};


class OptixException 
: public Exception
{
public:
    OptixException(const std::string& msg);
    OptixException(const std::string& msg, const char* file, const char* func, int line);
    ~OptixException() throw();
};



} // namespace rmagine


#endif // RMAGINE_UTIL_EXCEPTIONS_H
