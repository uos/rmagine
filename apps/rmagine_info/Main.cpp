
#include <iostream>
#include <rmagine/version.h>

namespace rm = rmagine;

int main(int argc, char** argv)
{
  std::cout << "Rmagine" << std::endl;
  std::cout << "- Header Version:  " << RMAGINE_VERSION << std::endl;
  std::cout << "- Library Version: " << rm::version_string() << std::endl;
  return 0;
}
