#ifndef RMAGINE_UTIL_DEBUG_HPP
#define RMAGINE_UTIL_DEBUG_HPP

#include <boost/current_function.hpp>
#include <iostream>


#define TODO std::cout << "WARNING - TODO at '" << BOOST_CURRENT_FUNCTION << "'" << std::endl;

#define TODO_TEST_FUNCTION std::cout << "WARNING - Untested Function. \n Send this to amock@uos.de if everything works as expected: \n '" << BOOST_CURRENT_FUNCTION << "'" << std::endl;
#define TODO_NOT_IMPLEMENTED std::cout << "WARNING - Not implemented at '" << BOOST_CURRENT_FUNCTION << "'" << std::endl;




#endif // RMAGINE_UTIL_DEBUG_HPP