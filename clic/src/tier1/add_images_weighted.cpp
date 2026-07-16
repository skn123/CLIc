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
  tier0::create_like(src0, dst, promoteType(src0->dtype(), src1->dtype()));
  evaluate(device, "a * x + b * y", { src0, factor1, src1, factor2 }, dst);
  return dst;
}

} // namespace cle::tier1
