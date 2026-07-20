#include "tier0.hpp"
#include "tier1.hpp"

#include "utils.hpp"


#include "cle_y_position_of_maximum_y_projection.h"
#include "cle_y_position_of_minimum_y_projection.h"
#include "cle_z_position_of_maximum_z_projection.h"
#include "cle_z_position_of_minimum_z_projection.h"
#include "cle_z_position_projection.h"

namespace kernel
{

// Generalized variance / standard-deviation projection kernel for any axis
// (np.var / np.std along the target axis). Uses Welford's online algorithm for
// a single-pass, numerically stable computation. Set compute_std to a non-zero
// value to return the standard deviation (square root of the variance).
constexpr const char * deviation_projection_kernel = R"CLC(
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void deviation_projection(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst,
    int axis,       // 0=X projection, 1=Y projection, 2=Z projection
    int ddof,       // delta degrees of freedom (divisor is n - ddof)
    int compute_std // 0=variance, non-zero=standard deviation
)
{
  const int id0 = get_global_id(0);
  const int id1 = get_global_id(1);

  // Projection length along the target axis
  const int n = (axis == 0) ? GET_IMAGE_WIDTH(src) :
                (axis == 1) ? GET_IMAGE_HEIGHT(src) : GET_IMAGE_DEPTH(src);

  // Welford's online algorithm: single pass, numerically stable
  float mean = 0;
  float m2 = 0;

  for (int i = 0; i < n; i++)
  {
    // Map (id0, id1, i) to (x, y, z) based on projection axis
    const int x = (axis == 0) ? i   : id0;
    const int y = (axis == 0) ? id0 : (axis == 1) ? i : id1;
    const int z = (axis == 2) ? i   : id1;

    const float value = (float) READ_IMAGE(src, sampler, POS_src_INSTANCE(x, y, z, 0)).x;
    const float delta = value - mean;
    mean += delta / (float)(i + 1);
    const float delta2 = value - mean;
    m2 += delta * delta2;
  }

  const int   dof = n - ddof;
  float       result = (dof > 0) ? (m2 / (float)dof) : 0;
  if (compute_std)
  {
    result = sqrt(result);
  }

  WRITE_IMAGE(dst, POS_dst_INSTANCE(id0, id1, 0, 0), CONVERT_dst_PIXEL_TYPE(result));
}
)CLC";

// Generalized product projection kernel for any axis (np.prod along the target axis)
constexpr const char * product_projection_kernel = R"CLC(
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void product_projection(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst,
    int axis  // 0=X projection, 1=Y projection, 2=Z projection
)
{
  const int id0 = get_global_id(0);
  const int id1 = get_global_id(1);

  // Projection length along the target axis
  const int n = (axis == 0) ? GET_IMAGE_WIDTH(src) :
                (axis == 1) ? GET_IMAGE_HEIGHT(src) : GET_IMAGE_DEPTH(src);

  float product = 1.0f;
  for (int i = 0; i < n; i++)
  {
    // Map (id0, id1, i) to (x, y, z) based on projection axis
    const int x = (axis == 0) ? i   : id0;
    const int y = (axis == 0) ? id0 : (axis == 1) ? i : id1;
    const int z = (axis == 2) ? i   : id1;

    product *= (float) READ_IMAGE(src, sampler, POS_src_INSTANCE(x, y, z, 0)).x;
  }

  WRITE_IMAGE(dst, POS_dst_INSTANCE(id0, id1, 0, 0), CONVERT_dst_PIXEL_TYPE(product));
}
)CLC";

constexpr const char * maximum_projection_kernel = R"CLC(
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void maximum_projection(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst,
    int axis  // 0=X projection, 1=Y projection, 2=Z projection
)
{
  const int id0 = get_global_id(0);
  const int id1 = get_global_id(1);

  // Projection length along the target axis
  const int n = (axis == 0) ? GET_IMAGE_WIDTH(src) :
                (axis == 1) ? GET_IMAGE_HEIGHT(src) : GET_IMAGE_DEPTH(src);

  IMAGE_src_PIXEL_TYPE maximum = 0;
  for (int i = 0; i < n; i++)
  {
    // Map (id0, id1, i) to (x, y, z) based on projection axis
    const int x = (axis == 0) ? i   : id0;
    const int y = (axis == 0) ? id0 : (axis == 1) ? i : id1;
    const int z = (axis == 2) ? i   : id1;

    const IMAGE_src_PIXEL_TYPE value = READ_IMAGE(src, sampler, POS_src_INSTANCE(x, y, z, 0)).x;
    maximum = (i == 0 || value > maximum) ? value : maximum;
  }

  WRITE_IMAGE(dst, POS_dst_INSTANCE(id0, id1, 0, 0), CONVERT_dst_PIXEL_TYPE(maximum));
}
)CLC";

constexpr const char * minimum_projection_kernel = R"CLC(
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void minimum_projection(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst,
    int axis  // 0=X projection, 1=Y projection, 2=Z projection
)
{
  const int id0 = get_global_id(0);
  const int id1 = get_global_id(1);

  // Projection length along the target axis
  const int n = (axis == 0) ? GET_IMAGE_WIDTH(src) :
                (axis == 1) ? GET_IMAGE_HEIGHT(src) : GET_IMAGE_DEPTH(src);

  IMAGE_src_PIXEL_TYPE minimum = 0;
  for (int i = 0; i < n; i++)
  {
    // Map (id0, id1, i) to (x, y, z) based on projection axis
    const int x = (axis == 0) ? i   : id0;
    const int y = (axis == 0) ? id0 : (axis == 1) ? i : id1;
    const int z = (axis == 2) ? i   : id1;

    const IMAGE_src_PIXEL_TYPE value = READ_IMAGE(src, sampler, POS_src_INSTANCE(x, y, z, 0)).x;
    minimum = (i == 0 || value < minimum) ? value : minimum;
  }

  WRITE_IMAGE(dst, POS_dst_INSTANCE(id0, id1, 0, 0), CONVERT_dst_PIXEL_TYPE(minimum));
}
)CLC";

constexpr const char * sum_projection_kernel = R"CLC(
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void sum_projection(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst,
    int axis  // 0=X projection, 1=Y projection, 2=Z projection
)
{
  const int id0 = get_global_id(0);
  const int id1 = get_global_id(1);

  // Projection length along the target axis
  const int n = (axis == 0) ? GET_IMAGE_WIDTH(src) :
                (axis == 1) ? GET_IMAGE_HEIGHT(src) : GET_IMAGE_DEPTH(src);

  float sum = 0;
  for (int i = 0; i < n; i++)
  {
    // Map (id0, id1, i) to (x, y, z) based on projection axis
    const int x = (axis == 0) ? i   : id0;
    const int y = (axis == 0) ? id0 : (axis == 1) ? i : id1;
    const int z = (axis == 2) ? i   : id1;

    sum += (float) READ_IMAGE(src, sampler, POS_src_INSTANCE(x, y, z, 0)).x;
  }

  WRITE_IMAGE(dst, POS_dst_INSTANCE(id0, id1, 0, 0), CONVERT_dst_PIXEL_TYPE(sum));
}
)CLC";

constexpr const char * mean_projection_kernel = R"CLC(
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void mean_projection(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst,
    int axis  // 0=X projection, 1=Y projection, 2=Z projection
)
{
  const int id0 = get_global_id(0);
  const int id1 = get_global_id(1);

  // Projection length along the target axis
  const int n = (axis == 0) ? GET_IMAGE_WIDTH(src) :
                (axis == 1) ? GET_IMAGE_HEIGHT(src) : GET_IMAGE_DEPTH(src);

  float sum = 0;
  for (int i = 0; i < n; i++)
  {
    // Map (id0, id1, i) to (x, y, z) based on projection axis
    const int x = (axis == 0) ? i   : id0;
    const int y = (axis == 0) ? id0 : (axis == 1) ? i : id1;
    const int z = (axis == 2) ? i   : id1;

    sum += (float) READ_IMAGE(src, sampler, POS_src_INSTANCE(x, y, z, 0)).x;
  }

  WRITE_IMAGE(dst, POS_dst_INSTANCE(id0, id1, 0, 0), CONVERT_dst_PIXEL_TYPE(sum / (float) n));
}
)CLC";

constexpr const char * x_position_of_maximum_x_projection = R"CLC(
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void x_position_of_maximum_x_projection(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst
)
{
  const int y = get_global_id(0);
  const int z = get_global_id(1);

  float max = 0;
  int   max_pos = 0;
  for (int x = 0; x < GET_IMAGE_WIDTH(src); ++x)
  {
    float value = READ_IMAGE(src, sampler, POS_src_INSTANCE(x, y, z, 0)).x;
    if (value > max || x == 0)
    {
      max = value;
      max_pos = x;
    }
  }
  WRITE_IMAGE(dst, POS_dst_INSTANCE(y, z, 0, 0), CONVERT_dst_PIXEL_TYPE(max_pos));
}
)CLC";

constexpr const char * x_position_of_minimum_x_projection = R"CLC(
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void x_position_of_minimum_x_projection(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst
)
{
  const int y = get_global_id(0);
  const int z = get_global_id(1);

  float min = 0;
  int   min_pos = 0;
  for (int x = 0; x < GET_IMAGE_WIDTH(src); ++x)
  {
    float value = READ_IMAGE(src, sampler, POS_src_INSTANCE(x, y, z, 0)).x;
    if (value < min || x == 0)
    {
      min = value;
      min_pos = x;
    }
  }
  WRITE_IMAGE(dst, POS_dst_INSTANCE(y, z, 0, 0), CONVERT_dst_PIXEL_TYPE(min_pos));
}
)CLC";

} // namespace kernel

namespace cle::tier1
{

namespace
{
auto
keepdims_y(const Array::Pointer & src, const Array::Pointer & dst) -> Array::Pointer
{
  return dst->reshape(dst->width(), 1, dst->height(), src->dimension());
}
auto
keepdims_x(const Array::Pointer & src, const Array::Pointer & dst) -> Array::Pointer
{
  return dst->reshape(1, dst->width(), dst->height(), src->dimension());
}

// Select the accumulator dtype for the sum / product projections based on the
// source dtype: floating-point (or complex) sources accumulate into FLOAT,
// signed integers into INT32, and unsigned integers into UINT32.
auto
accumulator_type(const Array::Pointer & src) -> dType
{
  switch (src->dtype())
  {
    case dType::INT8:
    case dType::INT16:
    case dType::INT32:
      return dType::INT32;
    case dType::UINT8:
    case dType::UINT16:
    case dType::UINT32:
      return dType::UINT32;
    default:
      return dType::FLOAT;
  }
}

// Shared implementation for the variance / standard-deviation projections along
// a given axis (0=X, 1=Y, 2=Z). Set compute_std to return the standard deviation.
auto
deviation_projection(const Device::Pointer & device,
                     const Array::Pointer &  src,
                     Array::Pointer          dst,
                     int                     axis,
                     int                     ddof,
                     bool                    keep_dims,
                     bool                    compute_std) -> Array::Pointer
{
  switch (axis)
  {
    case 0:
      tier0::create_yz(src, dst, dType::FLOAT, keep_dims);
      break;
    case 1:
      tier0::create_xz(src, dst, dType::FLOAT, keep_dims);
      break;
    default:
      tier0::create_xy(src, dst, dType::FLOAT, keep_dims);
      break;
  }
  const KernelInfo    kernel = { "deviation_projection", kernel::deviation_projection_kernel };
  const ParameterList params = {
    { "src", src }, { "dst", dst }, { "axis", axis }, { "ddof", ddof }, { "compute_std", compute_std ? 1 : 0 }
  };
  const RangeArray range = { dst->width(), dst->height(), 1 };
  execute(device, kernel, params, range);
  if (keep_dims)
  {
    dst = (axis == 0) ? keepdims_x(src, dst) : (axis == 1) ? keepdims_y(src, dst) : dst;
  }
  return dst;
}

// Shared implementation for the simple reduction projections (maximum, minimum,
// sum, mean) along a given axis (0=X, 1=Y, 2=Z). The kernel selects the axis at
// runtime through the "axis" argument.
auto
axis_projection(const Device::Pointer & device,
                const Array::Pointer &  src,
                Array::Pointer          dst,
                int                     axis,
                bool                    keep_dims,
                const KernelInfo &      kernel,
                dType                   type) -> Array::Pointer
{
  switch (axis)
  {
    case 0:
      tier0::create_yz(src, dst, type, keep_dims);
      break;
    case 1:
      tier0::create_xz(src, dst, type, keep_dims);
      break;
    default:
      tier0::create_xy(src, dst, type, keep_dims);
      break;
  }
  const ParameterList params = { { "src", src }, { "dst", dst }, { "axis", axis } };
  const RangeArray    range = { dst->width(), dst->height(), 1 };
  execute(device, kernel, params, range);
  if (keep_dims)
  {
    dst = (axis == 0) ? keepdims_x(src, dst) : (axis == 1) ? keepdims_y(src, dst) : dst;
  }
  return dst;
}
} // namespace

auto
std_x_projection_func(const Device::Pointer & device, const Array::Pointer & src, Array::Pointer dst, int ddof, bool keep_dims)
  -> Array::Pointer
{
  return deviation_projection(device, src, dst, 0, ddof, keep_dims, true);
}

auto
std_y_projection_func(const Device::Pointer & device, const Array::Pointer & src, Array::Pointer dst, int ddof, bool keep_dims)
  -> Array::Pointer
{
  return deviation_projection(device, src, dst, 1, ddof, keep_dims, true);
}

auto
std_z_projection_func(const Device::Pointer & device, const Array::Pointer & src, Array::Pointer dst, int ddof, bool keep_dims)
  -> Array::Pointer
{
  return deviation_projection(device, src, dst, 2, ddof, keep_dims, true);
}

auto
variance_x_projection_func(const Device::Pointer & device, const Array::Pointer & src, Array::Pointer dst, int ddof, bool keep_dims)
  -> Array::Pointer
{
  return deviation_projection(device, src, dst, 0, ddof, keep_dims, false);
}

auto
variance_y_projection_func(const Device::Pointer & device, const Array::Pointer & src, Array::Pointer dst, int ddof, bool keep_dims)
  -> Array::Pointer
{
  return deviation_projection(device, src, dst, 1, ddof, keep_dims, false);
}

auto
variance_z_projection_func(const Device::Pointer & device, const Array::Pointer & src, Array::Pointer dst, int ddof, bool keep_dims)
  -> Array::Pointer
{
  return deviation_projection(device, src, dst, 2, ddof, keep_dims, false);
}

auto
maximum_x_projection_func(const Device::Pointer & device, const Array::Pointer & src, Array::Pointer dst, bool keep_dims) -> Array::Pointer
{
  return axis_projection(device, src, dst, 0, keep_dims, { "maximum_projection", kernel::maximum_projection_kernel }, dType::UNKNOWN);
}

auto
maximum_y_projection_func(const Device::Pointer & device, const Array::Pointer & src, Array::Pointer dst, bool keep_dims) -> Array::Pointer
{
  return axis_projection(device, src, dst, 1, keep_dims, { "maximum_projection", kernel::maximum_projection_kernel }, dType::UNKNOWN);
}

auto
maximum_z_projection_func(const Device::Pointer & device, const Array::Pointer & src, Array::Pointer dst, bool keep_dims) -> Array::Pointer
{
  return axis_projection(device, src, dst, 2, keep_dims, { "maximum_projection", kernel::maximum_projection_kernel }, dType::UNKNOWN);
}

auto
mean_x_projection_func(const Device::Pointer & device, const Array::Pointer & src, Array::Pointer dst, bool keep_dims) -> Array::Pointer
{
  return axis_projection(device, src, dst, 0, keep_dims, { "mean_projection", kernel::mean_projection_kernel }, dType::FLOAT);
}

auto
mean_y_projection_func(const Device::Pointer & device, const Array::Pointer & src, Array::Pointer dst, bool keep_dims) -> Array::Pointer
{
  return axis_projection(device, src, dst, 1, keep_dims, { "mean_projection", kernel::mean_projection_kernel }, dType::FLOAT);
}

auto
mean_z_projection_func(const Device::Pointer & device, const Array::Pointer & src, Array::Pointer dst, bool keep_dims) -> Array::Pointer
{
  return axis_projection(device, src, dst, 2, keep_dims, { "mean_projection", kernel::mean_projection_kernel }, dType::FLOAT);
}

auto
minimum_x_projection_func(const Device::Pointer & device, const Array::Pointer & src, Array::Pointer dst, bool keep_dims) -> Array::Pointer
{
  return axis_projection(device, src, dst, 0, keep_dims, { "minimum_projection", kernel::minimum_projection_kernel }, dType::UNKNOWN);
}

auto
minimum_y_projection_func(const Device::Pointer & device, const Array::Pointer & src, Array::Pointer dst, bool keep_dims) -> Array::Pointer
{
  return axis_projection(device, src, dst, 1, keep_dims, { "minimum_projection", kernel::minimum_projection_kernel }, dType::UNKNOWN);
}

auto
minimum_z_projection_func(const Device::Pointer & device, const Array::Pointer & src, Array::Pointer dst, bool keep_dims) -> Array::Pointer
{
  return axis_projection(device, src, dst, 2, keep_dims, { "minimum_projection", kernel::minimum_projection_kernel }, dType::UNKNOWN);
}

auto
sum_x_projection_func(const Device::Pointer & device, const Array::Pointer & src, Array::Pointer dst, bool keep_dims) -> Array::Pointer
{
  return axis_projection(device, src, dst, 0, keep_dims, { "sum_projection", kernel::sum_projection_kernel }, accumulator_type(src));
}

auto
sum_y_projection_func(const Device::Pointer & device, const Array::Pointer & src, Array::Pointer dst, bool keep_dims) -> Array::Pointer
{
  return axis_projection(device, src, dst, 1, keep_dims, { "sum_projection", kernel::sum_projection_kernel }, accumulator_type(src));
}

auto
sum_z_projection_func(const Device::Pointer & device, const Array::Pointer & src, Array::Pointer dst, bool keep_dims) -> Array::Pointer
{
  return axis_projection(device, src, dst, 2, keep_dims, { "sum_projection", kernel::sum_projection_kernel }, accumulator_type(src));
}

auto
product_x_projection_func(const Device::Pointer & device, const Array::Pointer & src, Array::Pointer dst, bool keep_dims) -> Array::Pointer
{
  return axis_projection(
    device, src, dst, 0, keep_dims, { "product_projection", kernel::product_projection_kernel }, accumulator_type(src));
}

auto
product_y_projection_func(const Device::Pointer & device, const Array::Pointer & src, Array::Pointer dst, bool keep_dims) -> Array::Pointer
{
  return axis_projection(
    device, src, dst, 1, keep_dims, { "product_projection", kernel::product_projection_kernel }, accumulator_type(src));
}

auto
product_z_projection_func(const Device::Pointer & device, const Array::Pointer & src, Array::Pointer dst, bool keep_dims) -> Array::Pointer
{
  return axis_projection(
    device, src, dst, 2, keep_dims, { "product_projection", kernel::product_projection_kernel }, accumulator_type(src));
}

auto
x_position_of_maximum_x_projection_func(const Device::Pointer & device, const Array::Pointer & src, Array::Pointer dst, bool keep_dims)
  -> Array::Pointer
{
  tier0::create_yz(src, dst, dType::INDEX, keep_dims);
  const KernelInfo    kernel = { "x_position_of_maximum_x_projection", kernel::x_position_of_maximum_x_projection };
  const ParameterList params = { { "src", src }, { "dst", dst } };
  const RangeArray    range = { dst->width(), dst->height(), 1 };
  execute(device, kernel, params, range);
  if (keep_dims)
  {
    dst = keepdims_x(src, dst);
  }
  return dst;
}

auto
x_position_of_minimum_x_projection_func(const Device::Pointer & device, const Array::Pointer & src, Array::Pointer dst, bool keep_dims)
  -> Array::Pointer
{
  tier0::create_yz(src, dst, dType::INDEX, keep_dims);
  const KernelInfo    kernel = { "x_position_of_minimum_x_projection", kernel::x_position_of_minimum_x_projection };
  const ParameterList params = { { "src", src }, { "dst", dst } };
  const RangeArray    range = { dst->width(), dst->height(), 1 };
  execute(device, kernel, params, range);
  if (keep_dims)
  {
    dst = keepdims_x(src, dst);
  }
  return dst;
}

auto
y_position_of_maximum_y_projection_func(const Device::Pointer & device, const Array::Pointer & src, Array::Pointer dst, bool keep_dims)
  -> Array::Pointer
{
  tier0::create_xz(src, dst, dType::INDEX, keep_dims);
  const KernelInfo    kernel = { "y_position_of_maximum_y_projection", kernel::y_position_of_maximum_y_projection };
  const ParameterList params = { { "src", src }, { "dst", dst } };
  const RangeArray    range = { dst->width(), dst->height(), dst->depth() };
  execute(device, kernel, params, range);
  if (keep_dims)
  {
    dst = keepdims_y(src, dst);
  }
  return dst;
}

auto
y_position_of_minimum_y_projection_func(const Device::Pointer & device, const Array::Pointer & src, Array::Pointer dst, bool keep_dims)
  -> Array::Pointer
{
  tier0::create_xz(src, dst, dType::INDEX, keep_dims);
  const KernelInfo    kernel = { "y_position_of_minimum_y_projection", kernel::y_position_of_minimum_y_projection };
  const ParameterList params = { { "src", src }, { "dst", dst } };
  const RangeArray    range = { dst->width(), dst->height(), dst->depth() };
  execute(device, kernel, params, range);
  if (keep_dims)
  {
    dst = keepdims_y(src, dst);
  }
  return dst;
}

auto
z_position_of_maximum_z_projection_func(const Device::Pointer & device, const Array::Pointer & src, Array::Pointer dst, bool keep_dims)
  -> Array::Pointer
{
  tier0::create_xy(src, dst, dType::INDEX, keep_dims);
  const KernelInfo    kernel = { "z_position_of_maximum_z_projection", kernel::z_position_of_maximum_z_projection };
  const ParameterList params = { { "src", src }, { "dst", dst } };
  const RangeArray    range = { dst->width(), dst->height(), dst->depth() };
  execute(device, kernel, params, range);
  return dst;
}

auto
z_position_of_minimum_z_projection_func(const Device::Pointer & device, const Array::Pointer & src, Array::Pointer dst, bool keep_dims)
  -> Array::Pointer
{
  tier0::create_xy(src, dst, dType::INDEX, keep_dims);
  const KernelInfo    kernel = { "z_position_of_minimum_z_projection", kernel::z_position_of_minimum_z_projection };
  const ParameterList params = { { "src", src }, { "dst", dst } };
  const RangeArray    range = { dst->width(), dst->height(), dst->depth() };
  execute(device, kernel, params, range);
  return dst;
}

auto
z_position_projection_func(const Device::Pointer & device, const Array::Pointer & src, const Array::Pointer & position, Array::Pointer dst)
  -> Array::Pointer
{
  tier0::create_xy(src, dst);
  const KernelInfo    kernel = { "z_position_projection", kernel::z_position_projection };
  const ParameterList params = { { "src", src }, { "position", position }, { "dst", dst } };
  const RangeArray    range = { dst->width(), dst->height(), dst->depth() };
  execute(device, kernel, params, range);
  return dst;
}

} // namespace cle::tier1
