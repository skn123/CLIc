#include "tier0.hpp"
#include "tier1.hpp"
#include "tier2.hpp"
#include "tier3.hpp"
#include "tier4.hpp"

#include "utils.hpp"

namespace cle::tier4
{

auto
variance_of_all_pixels_func(const Device::Pointer & device, const Array::Pointer & src, int ddof) -> float
{
  float mean = tier3::mean_of_all_pixels_func(device, src);
  auto  diff = Array::create(src->width(), src->height(), src->depth(), src->dimension(), dType::FLOAT, src->mtype(), device);
  evaluate(device, "pow(src - mean, 2)", { src, mean }, diff);
  float variance = tier3::mean_of_all_pixels_func(device, diff);
  if (ddof != 0)
  {
    const auto n = static_cast<float>(src->size());
    const auto dof = n - static_cast<float>(ddof);
    variance = (dof > 0) ? (variance * n / dof) : 0.f;
  }
  return variance;
}

auto
standard_deviation_of_all_pixels_func(const Device::Pointer & device, const Array::Pointer & src, int ddof) -> float
{
  return std::sqrt(variance_of_all_pixels_func(device, src, ddof));
}

} // namespace cle::tier4
