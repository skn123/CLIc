#include "tier0.hpp"
#include "tier1.hpp"

#include "utils.hpp"

#include "cle_add_images_weighted.h"

namespace cle::tier1
{

auto
add_images_weighted_func(const Device::Pointer & device,
                         const Array::Pointer &  src0,
                         const Array::Pointer &  src1,
                         Array::Pointer          dst,
                         float                   factor1,
                         float                   factor2) -> Array::Pointer
{
  tier0::create_or_check_broadcast_dst(src0, src1, dst, promoteType(src0->dtype(), src1->dtype()));
  std::vector<ParameterType> params;
  params.emplace_back(src0);
  params.emplace_back(factor1);
  params.emplace_back(src1);
  params.emplace_back(factor2);
  evaluate(device, "a * x + b * y", params, dst);
  return dst;
}

} // namespace cle::tier1
