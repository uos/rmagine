/**
 * @file
 *
 * @brief Backend-independent settings for simulators.
 *
 * Collects enums and parameters that configure how simulators trace rays.
 * Each backend maps these to its own native flags.
 *
 * @copyright Copyright (c) 2026, University Osnabrück. All rights reserved.
 * This project is released under the 3-Clause BSD License.
 */

#ifndef RMAGINE_SIMULATION_SIMULATOR_SETTINGS_HPP
#define RMAGINE_SIMULATION_SIMULATOR_SETTINGS_HPP

namespace rmagine
{

/**
 * @brief Face culling mode used during ray traversal
 *
 * - None: front and back faces are hit
 * - Backface: back-facing triangles are ignored
 * - Frontface: front-facing triangles are ignored
 */
enum class Culling
{
  None,
  Backface,
  Frontface
};

} // namespace rmagine

#endif // RMAGINE_SIMULATION_SIMULATOR_SETTINGS_HPP
