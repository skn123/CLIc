#include "tier0.hpp"
#include "tier1.hpp"

#include "utils.hpp"

#include "cle_range.h"

namespace cle::tier1
{

auto
gather_func(const Device::Pointer & device,
            const Array::Pointer &  src,
            Array::Pointer          dst,
            int                     start_x,
            int                     stop_x,
            int                     step_x,
            int                     start_y,
            int                     stop_y,
            int                     step_y,
            int                     start_z,
            int                     stop_z,
            int                     step_z) -> Array::Pointer
{
  const size_t range_width = std::max<size_t>(abs(stop_x - start_x) / std::max(std::abs(step_x), 1), 1);
  const size_t range_height = std::max<size_t>(abs(stop_y - start_y) / std::max(std::abs(step_y), 1), 1);
  const size_t range_depth = std::max<size_t>(abs(stop_z - start_z) / std::max(std::abs(step_z), 1), 1);

  if (dst == nullptr)
  {
    tier0::create_dst(src, dst, range_width, range_height, range_depth, src->dtype());
  }

  correct_range(&start_x, &stop_x, &step_x, static_cast<int>(src->width()));
  correct_range(&start_y, &stop_y, &step_y, static_cast<int>(src->height()));
  correct_range(&start_z, &stop_z, &step_z, static_cast<int>(src->depth()));

  const KernelInfo    kernel = { "range", kernel::range };
  const ParameterList params = { { "src", src },         { "dst", dst },       { "start_x", start_x }, { "step_x", step_x },
                                 { "start_y", start_y }, { "step_y", step_y }, { "start_z", start_z }, { "step_z", step_z } };
  const RangeArray    range = { range_width, range_height, range_depth };
  execute(device, kernel, params, range);
  return dst;
}

auto
range_func(const Device::Pointer & device,
           const Array::Pointer &  src,
           Array::Pointer          dst,
           int                     start_x,
           int                     stop_x,
           int                     step_x,
           int                     start_y,
           int                     stop_y,
           int                     step_y,
           int                     start_z,
           int                     stop_z,
           int                     step_z) -> Array::Pointer
{
  return gather_func(device, src, dst, start_x, stop_x, step_x, start_y, stop_y, step_y, start_z, stop_z, step_z);
}

namespace
{

// Scatter kernel: inverse of the range (gather) kernel. It writes a small src image into a
// strided region of a larger dst image in place, following dst[start + i * step] = src[i].
// The global range matches the src size; dst is never reallocated.
constexpr const char * scatter_kernel_source = R"CLC(
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void scatter(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst,
    const int       start_x,
    const int       step_x,
    const int       start_y,
    const int       step_y,
    const int       start_z,
    const int       step_z
)
{
  const int dx = get_global_id(0);
  const int dy = get_global_id(1);
  const int dz = get_global_id(2);

  const int sx = start_x + dx * step_x;
  const int sy = start_y + dy * step_y;
  const int sz = start_z + dz * step_z;

  const float value = READ_IMAGE(src, sampler, POS_src_INSTANCE(dx, dy, dz, 0)).x;
  WRITE_IMAGE(dst, POS_dst_INSTANCE(sx, sy, sz, 0), CONVERT_dst_PIXEL_TYPE(value));
}
)CLC";

} // namespace

auto
scatter_func(const Device::Pointer & device,
             const Array::Pointer &  src,
             Array::Pointer          dst,
             int                     start_x,
             int                     stop_x,
             int                     step_x,
             int                     start_y,
             int                     stop_y,
             int                     step_y,
             int                     start_z,
             int                     stop_z,
             int                     step_z) -> Array::Pointer
{
  if (dst == nullptr)
  {
    throw std::runtime_error("Error: scatter requires a pre-allocated destination array.");
  }

  const size_t range_width = std::max<size_t>(abs(stop_x - start_x) / std::max(std::abs(step_x), 1), 1);
  const size_t range_height = std::max<size_t>(abs(stop_y - start_y) / std::max(std::abs(step_y), 1), 1);
  const size_t range_depth = std::max<size_t>(abs(stop_z - start_z) / std::max(std::abs(step_z), 1), 1);

  correct_range(&start_x, &stop_x, &step_x, static_cast<int>(dst->width()));
  correct_range(&start_y, &stop_y, &step_y, static_cast<int>(dst->height()));
  correct_range(&start_z, &stop_z, &step_z, static_cast<int>(dst->depth()));

  const KernelInfo    kernel = { "scatter", scatter_kernel_source };
  const ParameterList params = { { "src", src },         { "dst", dst },       { "start_x", start_x }, { "step_x", step_x },
                                 { "start_y", start_y }, { "step_y", step_y }, { "start_z", start_z }, { "step_z", step_z } };
  const RangeArray    range = { range_width, range_height, range_depth };
  execute(device, kernel, params, range);
  return dst;
}

// read_values_from_map_func

} // namespace cle::tier1
