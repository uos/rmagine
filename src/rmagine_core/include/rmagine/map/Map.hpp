#ifndef RMAGINE_MAP_HPP
#define RMAGINE_MAP_HPP

#include <memory>
#include <rmagine/types/shared_functions.h>

namespace rmagine
{

// this is just a container for downcasting
class RMAGINE_API Map
{
public:
  virtual ~Map() = default;
};

using MapPtr = std::shared_ptr<Map>;

} // namespace rmagine

#endif // RMAGINE_MAP_HPP