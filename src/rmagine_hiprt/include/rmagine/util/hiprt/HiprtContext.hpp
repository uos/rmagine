/**
 * @file HiprtContext.hpp
 * @brief HIPRT context wrapper
 *
 * Manages hiprtContext creation/destruction and JIT kernel compilation.
 */

#ifndef RMAGINE_UTIL_HIPRT_CONTEXT_HPP
#define RMAGINE_UTIL_HIPRT_CONTEXT_HPP

#include <memory>
#include <string>

// Forward declarations -- HIPRT types
struct _hiprtContext;
typedef _hiprtContext* hiprtContext;

namespace rmagine
{

class HiprtContext;
using HiprtContextPtr = std::shared_ptr<HiprtContext>;

/**
 * @brief HIPRT context wrapper
 *
 * Creates and owns a hiprtContext. Also manages the trace kernel
 * compilation via hiprtBuildTraceKernels.
 */
class HiprtContext
{
public:
    HiprtContext();
    ~HiprtContext();

    // Non-copyable
    HiprtContext(const HiprtContext&) = delete;
    HiprtContext& operator=(const HiprtContext&) = delete;

    /**
     * @brief Get the raw hiprtContext handle
     */
    hiprtContext handle() const { return m_ctx; }

    /**
     * @brief Check if context is valid
     */
    bool valid() const { return m_ctx != nullptr; }

    /**
     * @brief Get the device index this context is on
     */
    int deviceIndex() const { return m_device_index; }

private:
    hiprtContext m_ctx = nullptr;
    int m_device_index = 0;
};

} // namespace rmagine

#endif // RMAGINE_UTIL_HIPRT_CONTEXT_HPP
