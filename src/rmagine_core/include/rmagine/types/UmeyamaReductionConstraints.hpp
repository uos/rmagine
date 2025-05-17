#ifndef RMAGINE_TYPES_UMEYAMA_REDUCTION_CONSTRAINTS_HPP
#define RMAGINE_TYPES_UMEYAMA_REDUCTION_CONSTRAINTS_HPP

namespace rmagine
{

struct UmeyamaReductionConstraints 
{
  ///
  // Ignore all correspondences larger than `max_dist`
  float max_dist;
  
  ///
  // Ignore dataset ids except for `dataset_id`
  unsigned int dataset_id;
  
  /// 
  // Ignore model ids except for `model_id`
  unsigned int model_id;
};

} // namespace rmagine

#endif // RMAGINE_TYPES_UMEYAMA_REDUCTION_CONSTRAINTS_HPP