#ifndef RMAGINE_VERSION_H
#define RMAGINE_VERSION_H

#include <cstddef>

namespace rmagine
{

size_t version_major();
size_t version_minor();
size_t version_patch();

const char* version_string();

} // namespace rmagine

#endif // RMAGINE_VERSION_H
