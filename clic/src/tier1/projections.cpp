#include "tier0.hpp"
#include "tier1.hpp"

#include "utils.hpp"


#include "cle_maximum_projection.h"
#include "cle_mean_projection.h"
#include "cle_minimum_projection.h"
#include "cle_sum_projection.h"

#include "cle_y_position_of_maximum_y_projection.h"
#include "cle_y_position_of_minimum_y_projection.h"
#include "cle_z_position_of_maximum_z_projection.h"
#include "cle_z_position_of_minimum_z_projection.h"
#include "cle_z_position_projection.h"

namespace kernel
{

// Generalized standard deviation projection kernel for any axis
// Uses Welford's online algorithm for single-pass, numerically stable computation
constexpr const char * std_projection_kernel = R"CLC(
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void std_projection(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst,
    int axis, // 0=X projection, 1=Y projection, 2=Z projection
    int ddof  // delta degrees of freedom (divisor is n - ddof)
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
  const float std_value = (dof > 0) ? sqrt(m2 / (float)dof) : 0;

  const int ox = id0;
  const int oy = id1;
  const int oz = 0;

  WRITE_IMAGE(dst, POS_dst_INSTANCE(ox, oy, oz, 0), CONVERT_dst_PIXEL_TYPE(std_value));
}
)CLC";

// Generalized variance projection kernel for any axis (np.var along the target axis)
// Uses Welford's online algorithm for single-pass, numerically stable computation
constexpr const char * variance_projection_kernel = R"CLC(
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void variance_projection(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst,
    int axis, // 0=X projection, 1=Y projection, 2=Z projection
    int ddof  // delta degrees of freedom (divisor is n - ddof)
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
  const float variance_value = (dof > 0) ? (m2 / (float)dof) : 0;

  const int ox = id0;
  const int oy = id1;
  const int oz = 0;

  WRITE_IMAGE(dst, POS_dst_INSTANCE(ox, oy, oz, 0), CONVERT_dst_PIXEL_TYPE(variance_value));
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

constexpr const char * maximum_x_projection = R"CLC(
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void maximum_x_projection(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst
)
{
  const int y = get_global_id(0);
  const int z = get_global_id(1);

  IMAGE_src_PIXEL_TYPE maximum = READ_IMAGE(src, sampler, POS_src_INSTANCE(0, y, z, 0)).x;
  for (int x = 1; x < GET_IMAGE_WIDTH(src); ++x)
  {
    const IMAGE_src_PIXEL_TYPE value = READ_IMAGE(src, sampler, POS_src_INSTANCE(x, y, z, 0)).x;
    maximum = (maximum > value) ? maximum : value;
  }
  WRITE_IMAGE(dst, POS_dst_INSTANCE(y, z, 0, 0), CONVERT_dst_PIXEL_TYPE(maximum));
}
)CLC";

constexpr const char * minimum_x_projection = R"CLC(
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void minimum_x_projection(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst
)
{
  const int y = get_global_id(0);
  const int z = get_global_id(1);

  IMAGE_src_PIXEL_TYPE minimum = READ_IMAGE(src, sampler, POS_src_INSTANCE(0, y, z, 0)).x;
  for (int x = 1; x < GET_IMAGE_WIDTH(src); ++x)
  {
    const IMAGE_src_PIXEL_TYPE value = READ_IMAGE(src, sampler, POS_src_INSTANCE(x, y, z, 0)).x;
    minimum = (minimum < value) ? minimum : value;
  }
  WRITE_IMAGE(dst, POS_dst_INSTANCE(y, z, 0, 0), CONVERT_dst_PIXEL_TYPE(minimum));
}
)CLC";

constexpr const char * mean_x_projection = R"CLC(
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void mean_x_projection(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst
)
{
  const int y = get_global_id(0);
  const int z = get_global_id(1);

  float sum = (float) READ_IMAGE(src, sampler, POS_src_INSTANCE(0, y, z, 0)).x;
  for (int x = 1; x < GET_IMAGE_WIDTH(src); ++x)
  {
    sum += (float) READ_IMAGE(src, sampler, POS_src_INSTANCE(x, y, z, 0)).x;
  }
  float mean = sum / GET_IMAGE_WIDTH(src);
  WRITE_IMAGE(dst, POS_dst_INSTANCE(y, z, 0, 0), CONVERT_dst_PIXEL_TYPE(mean));
}
)CLC";

constexpr const char * sum_x_projection = R"CLC(
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void sum_x_projection(
    IMAGE_src_TYPE  src,
    IMAGE_dst_TYPE  dst
)
{
  const int y = get_global_id(0);
  const int z = get_global_id(1);

  float sum = (float) READ_IMAGE(src, sampler, POS_src_INSTANCE(0, y, z, 0)).x;
  for (int x = 1; x < GET_IMAGE_WIDTH(src); ++x)
  {
    sum += (float) READ_IMAGE(src, sampler, POS_src_INSTANCE(x, y, z, 0)).x;
  }
  WRITE_IMAGE(dst, POS_dst_INSTANCE(y, z, 0, 0), CONVERT_dst_PIXEL_TYPE(sum));
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
} // namespace

auto
std_x_projection_func(const Device::Pointer & device, const Array::Pointer & src, Array::Pointer dst, int ddof, bool keep_dims) -> Array::Pointer
{
  tier0::create_yz(src, dst, dType::FLOAT, keep_dims);
  const KernelInfo    kernel = { "std_projection", kernel::std_projection_kernel };
  const ParameterList params = { { "src", src }, { "dst", dst }, { "axis", 0 }, { "ddof", ddof } };
  const RangeArray    range = { dst->width(), dst->height(), 1 };
  execute(device, kernel, params, range);
  if (keep_dims)
  {
    dst = keepdims_x(src, dst);
  }
  return dst;
}

auto
std_y_projection_func(const Device::Pointer & device, const Array::Pointer & src, Array::Pointer dst, int ddof, bool keep_dims) -> Array::Pointer
{
  tier0::create_xz(src, dst, dType::FLOAT, keep_dims);
  const KernelInfo    kernel = { "std_projection", kernel::std_projection_kernel };
  const ParameterList params = { { "src", src }, { "dst", dst }, { "axis", 1 }, { "ddof", ddof } };
  const RangeArray    range = { dst->width(), dst->height(), 1 };
  execute(device, kernel, params, range);
  if (keep_dims)
  {
    dst = keepdims_y(src, dst);
  }
  return dst;
}

auto
std_z_projection_func(const Device::Pointer & device, const Array::Pointer & src, Array::Pointer dst, int ddof, bool keep_dims) -> Array::Pointer
{
  tier0::create_xy(src, dst, dType::FLOAT, keep_dims);
  const KernelInfo    kernel = { "std_projection", kernel::std_projection_kernel };
  const ParameterList params = { { "src", src }, { "dst", dst }, { "axis", 2 }, { "ddof", ddof } };
  const RangeArray    range = { dst->width(), dst->height(), 1 };
  execute(device, kernel, params, range);
  return dst;
}

auto
variance_x_projection_func(const Device::Pointer & device, const Array::Pointer & src, Array::Pointer dst, int ddof, bool keep_dims) -> Array::Pointer
{
  tier0::create_yz(src, dst, dType::FLOAT, keep_dims);
  const KernelInfo    kernel = { "variance_projection", kernel::variance_projection_kernel };
  const ParameterList params = { { "src", src }, { "dst", dst }, { "axis", 0 }, { "ddof", ddof } };
  const RangeArray    range = { dst->width(), dst->height(), 1 };
  execute(device, kernel, params, range);
  if (keep_dims)
  {
    dst = keepdims_x(src, dst);
  }
  return dst;
}

auto
variance_y_projection_func(const Device::Pointer & device, const Array::Pointer & src, Array::Pointer dst, int ddof, bool keep_dims) -> Array::Pointer
{
  tier0::create_xz(src, dst, dType::FLOAT, keep_dims);
  const KernelInfo    kernel = { "variance_projection", kernel::variance_projection_kernel };
  const ParameterList params = { { "src", src }, { "dst", dst }, { "axis", 1 }, { "ddof", ddof } };
  const RangeArray    range = { dst->width(), dst->height(), 1 };
  execute(device, kernel, params, range);
  if (keep_dims)
  {
    dst = keepdims_y(src, dst);
  }
  return dst;
}

auto
variance_z_projection_func(const Device::Pointer & device, const Array::Pointer & src, Array::Pointer dst, int ddof, bool keep_dims) -> Array::Pointer
{
  tier0::create_xy(src, dst, dType::FLOAT, keep_dims);
  const KernelInfo    kernel = { "variance_projection", kernel::variance_projection_kernel };
  const ParameterList params = { { "src", src }, { "dst", dst }, { "axis", 2 }, { "ddof", ddof } };
  const RangeArray    range = { dst->width(), dst->height(), 1 };
  execute(device, kernel, params, range);
  return dst;
}

auto
maximum_x_projection_func(const Device::Pointer & device, const Array::Pointer & src, Array::Pointer dst, bool keep_dims) -> Array::Pointer
{
  tier0::create_yz(src, dst, dType::UNKNOWN, keep_dims);
  const KernelInfo    kernel = { "maximum_x_projection", kernel::maximum_x_projection };
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
maximum_y_projection_func(const Device::Pointer & device, const Array::Pointer & src, Array::Pointer dst, bool keep_dims) -> Array::Pointer
{
  tier0::create_xz(src, dst, dType::UNKNOWN, keep_dims);
  const KernelInfo    kernel = { "maximum_projection", kernel::maximum_projection };
  const ParameterList params = { { "src", src }, { "dst", dst } };
  const RangeArray    range = { dst->width(), dst->height(), dst->depth() };
  const RangeArray    local = { 0, 0, 0 };
  const ConstantList  constants = { { "PROJECTION_AXIS", 1 } }; // 1 for Y axis
  execute(device, kernel, params, range, local, constants);
  if (keep_dims)
  {
    dst = keepdims_y(src, dst);
  }
  return dst;
}

auto
maximum_z_projection_func(const Device::Pointer & device, const Array::Pointer & src, Array::Pointer dst, bool keep_dims) -> Array::Pointer
{
  tier0::create_xy(src, dst, dType::UNKNOWN, keep_dims);
  const KernelInfo    kernel = { "maximum_projection", kernel::maximum_projection };
  const ParameterList params = { { "src", src }, { "dst", dst } };
  const RangeArray    range = { dst->width(), dst->height(), dst->depth() };
  const RangeArray    local = { 0, 0, 0 };
  const ConstantList  constants = { { "PROJECTION_AXIS", 2 } }; // 2 for Z axis
  execute(device, kernel, params, range, local, constants);
  return dst;
}

auto
mean_x_projection_func(const Device::Pointer & device, const Array::Pointer & src, Array::Pointer dst, bool keep_dims) -> Array::Pointer
{
  tier0::create_yz(src, dst, dType::UNKNOWN, keep_dims);
  const KernelInfo    kernel = { "mean_x_projection", kernel::mean_x_projection };
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
mean_y_projection_func(const Device::Pointer & device, const Array::Pointer & src, Array::Pointer dst, bool keep_dims) -> Array::Pointer
{
  tier0::create_xz(src, dst, dType::UNKNOWN, keep_dims);
  const KernelInfo    kernel = { "mean_projection", kernel::mean_projection };
  const ParameterList params = { { "src", src }, { "dst", dst } };
  const RangeArray    range = { dst->width(), dst->height(), dst->depth() };
  const RangeArray    local = { 0, 0, 0 };
  const ConstantList  constants = { { "PROJECTION_AXIS", 1 } }; // 1 for Y axis
  execute(device, kernel, params, range, local, constants);
  if (keep_dims)
  {
    dst = keepdims_y(src, dst);
  }
  return dst;
}

auto
mean_z_projection_func(const Device::Pointer & device, const Array::Pointer & src, Array::Pointer dst, bool keep_dims) -> Array::Pointer
{
  tier0::create_xy(src, dst, dType::UNKNOWN, keep_dims);
  const KernelInfo    kernel = { "mean_projection", kernel::mean_projection };
  const ParameterList params = { { "src", src }, { "dst", dst } };
  const RangeArray    range = { dst->width(), dst->height(), dst->depth() };
  const RangeArray    local = { 0, 0, 0 };
  const ConstantList  constants = { { "PROJECTION_AXIS", 2 } }; // 2 for Z axis
  execute(device, kernel, params, range, local, constants);
  return dst;
}

auto
minimum_x_projection_func(const Device::Pointer & device, const Array::Pointer & src, Array::Pointer dst, bool keep_dims) -> Array::Pointer
{
  tier0::create_yz(src, dst, dType::UNKNOWN, keep_dims);
  const KernelInfo    kernel = { "minimum_x_projection", kernel::minimum_x_projection };
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
minimum_y_projection_func(const Device::Pointer & device, const Array::Pointer & src, Array::Pointer dst, bool keep_dims) -> Array::Pointer
{
  tier0::create_xz(src, dst, dType::UNKNOWN, keep_dims);
  const KernelInfo    kernel = { "minimum_projection", kernel::minimum_projection };
  const ParameterList params = { { "src", src }, { "dst", dst } };
  const RangeArray    range = { dst->width(), dst->height(), dst->depth() };
  const RangeArray    local = { 0, 0, 0 };
  const ConstantList  constants = { { "PROJECTION_AXIS", 1 } }; // 1 for Y axis
  execute(device, kernel, params, range, local, constants);
  if (keep_dims)
  {
    dst = keepdims_y(src, dst);
  }
  return dst;
}

auto
minimum_z_projection_func(const Device::Pointer & device, const Array::Pointer & src, Array::Pointer dst, bool keep_dims) -> Array::Pointer
{
  tier0::create_xy(src, dst, dType::UNKNOWN, keep_dims);
  const KernelInfo    kernel = { "minimum_projection", kernel::minimum_projection };
  const ParameterList params = { { "src", src }, { "dst", dst } };
  const RangeArray    range = { dst->width(), dst->height(), dst->depth() };
  const RangeArray    local = { 0, 0, 0 };
  const ConstantList  constants = { { "PROJECTION_AXIS", 2 } }; // 2 for Z axis
  execute(device, kernel, params, range, local, constants);
  return dst;
}

auto
sum_x_projection_func(const Device::Pointer & device, const Array::Pointer & src, Array::Pointer dst, bool keep_dims) -> Array::Pointer
{
  tier0::create_yz(src, dst, dType::FLOAT, keep_dims);
  const KernelInfo    kernel = { "sum_x_projection", kernel::sum_x_projection };
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
sum_y_projection_func(const Device::Pointer & device, const Array::Pointer & src, Array::Pointer dst, bool keep_dims) -> Array::Pointer
{
  tier0::create_xz(src, dst, dType::FLOAT, keep_dims);
  const KernelInfo    kernel = { "sum_projection", kernel::sum_projection };
  const ParameterList params = { { "src", src }, { "dst", dst } };
  const RangeArray    range = { dst->width(), dst->height(), dst->depth() };
  const RangeArray    local = { 0, 0, 0 };
  const ConstantList  constants = { { "PROJECTION_AXIS", 1 } }; // 1 for Y axis
  execute(device, kernel, params, range, local, constants);
  if (keep_dims)
  {
    dst = keepdims_y(src, dst);
  }
  return dst;
}

auto
sum_z_projection_func(const Device::Pointer & device, const Array::Pointer & src, Array::Pointer dst, bool keep_dims) -> Array::Pointer
{
  tier0::create_xy(src, dst, dType::FLOAT, keep_dims);
  const KernelInfo    kernel = { "sum_projection", kernel::sum_projection };
  const ParameterList params = { { "src", src }, { "dst", dst } };
  const RangeArray    range = { dst->width(), dst->height(), dst->depth() };
  const RangeArray    local = { 0, 0, 0 };
  const ConstantList  constants = { { "PROJECTION_AXIS", 2 } }; // 2 for Z axis
  execute(device, kernel, params, range, local, constants);
  return dst;
}

auto
product_x_projection_func(const Device::Pointer & device, const Array::Pointer & src, Array::Pointer dst, bool keep_dims) -> Array::Pointer
{
  tier0::create_yz(src, dst, dType::FLOAT, keep_dims);
  const KernelInfo    kernel = { "product_projection", kernel::product_projection_kernel };
  const ParameterList params = { { "src", src }, { "dst", dst }, { "axis", 0 } };
  const RangeArray    range = { dst->width(), dst->height(), 1 };
  execute(device, kernel, params, range);
  if (keep_dims)
  {
    dst = keepdims_x(src, dst);
  }
  return dst;
}

auto
product_y_projection_func(const Device::Pointer & device, const Array::Pointer & src, Array::Pointer dst, bool keep_dims) -> Array::Pointer
{
  tier0::create_xz(src, dst, dType::FLOAT, keep_dims);
  const KernelInfo    kernel = { "product_projection", kernel::product_projection_kernel };
  const ParameterList params = { { "src", src }, { "dst", dst }, { "axis", 1 } };
  const RangeArray    range = { dst->width(), dst->height(), 1 };
  execute(device, kernel, params, range);
  if (keep_dims)
  {
    dst = keepdims_y(src, dst);
  }
  return dst;
}

auto
product_z_projection_func(const Device::Pointer & device, const Array::Pointer & src, Array::Pointer dst, bool keep_dims) -> Array::Pointer
{
  tier0::create_xy(src, dst, dType::FLOAT, keep_dims);
  const KernelInfo    kernel = { "product_projection", kernel::product_projection_kernel };
  const ParameterList params = { { "src", src }, { "dst", dst }, { "axis", 2 } };
  const RangeArray    range = { dst->width(), dst->height(), 1 };
  execute(device, kernel, params, range);
  return dst;
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
