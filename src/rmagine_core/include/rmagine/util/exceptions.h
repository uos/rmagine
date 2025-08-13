/*
 * Copyright (c) 2022, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * @file
 * 
 * @brief Rmagine Exceptions
 *
 * @date 03.10.2022
 * @author Alexander Mock
 * 
 * @copyright Copyright (c) 2022, University Osnabrück. All rights reserved.
 * This project is released under the 3-Clause BSD License.
 * 
 */

#ifndef RMAGINE_UTIL_EXCEPTIONS_H
#define RMAGINE_UTIL_EXCEPTIONS_H

#include <stdexcept>
#include <sstream>
#include <rmagine/types/shared_functions.h>

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
