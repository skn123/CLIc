#include "tier0.hpp"
#include "tier1.hpp"

#include "utils.hpp"

#include <stdexcept>

namespace kernel
{

constexpr const char * cumulative_axis_kernel = R"CLC(
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void cumulative_axis(
		IMAGE_src_TYPE  src,
		IMAGE_dst_TYPE  dst,
		int axis, // 0=X, 1=Y, 2=Z
		int op    // 0=sum, 1=min, 2=max, 3=product
)
{
	const int id0 = get_global_id(0);
	const int id1 = get_global_id(1);

	const int n = (axis == 0) ? GET_IMAGE_WIDTH(src) :
								(axis == 1) ? GET_IMAGE_HEIGHT(src) : GET_IMAGE_DEPTH(src);

	float cumulative = 0.0f;
	for (int i = 0; i < n; ++i)
	{
		const int x = (axis == 0) ? i : id0;
		const int y = (axis == 1) ? i : (axis == 0) ? id0 : id1;
		const int z = (axis == 2) ? i : id1;

		const float value = (float) READ_IMAGE(src, sampler, POS_src_INSTANCE(x, y, z, 0)).x;
		if (i == 0)
		{
			cumulative = value;
		}
		else if (op == 0)
		{
			cumulative += value;
		}
		else if (op == 1)
		{
			cumulative = fmin(cumulative, value);
		}
		else if (op == 2)
		{
			cumulative = fmax(cumulative, value);
		}
		else
		{
			cumulative *= value;
		}

		WRITE_IMAGE(dst, POS_dst_INSTANCE(x, y, z, 0), CONVERT_dst_PIXEL_TYPE(cumulative));
	}
}
)CLC";

} // namespace kernel

namespace cle::tier1
{

namespace
{
enum class CumulativeOperation : int
{
  SUM = 0,
  MIN = 1,
  MAX = 2,
  PRODUCT = 3,
};

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

auto
cumulative_axis(const Device::Pointer & device,
                const Array::Pointer &  src,
                Array::Pointer          dst,
                int                     axis,
                bool                    keep_dims,
                CumulativeOperation     operation,
                dType                   type) -> Array::Pointer
{
  if (axis < 0 || axis > 2)
  {
    throw std::invalid_argument("Error: axis must be 0 (X), 1 (Y), or 2 (Z).");
  }

  tier0::create_dst(src, dst, src->width(), src->height(), src->depth(), type, keep_dims);

  const KernelInfo    kernel = { "cumulative_axis", kernel::cumulative_axis_kernel };
  const int           op_value = static_cast<int>(operation);
  const ParameterList params = { { "src", src }, { "dst", dst }, { "axis", axis }, { "op", op_value } };

  RangeArray range;
  if (axis == 0)
  {
    range = { src->height(), src->depth(), 1 };
  }
  else if (axis == 1)
  {
    range = { src->width(), src->depth(), 1 };
  }
  else
  {
    range = { src->width(), src->height(), 1 };
  }

  execute(device, kernel, params, range);
  return dst;
}
} // namespace

auto
cumulative_sum_func(const Device::Pointer & device, const Array::Pointer & src, Array::Pointer dst, int axis, bool keep_dims)
  -> Array::Pointer
{
  return cumulative_axis(device, src, dst, axis, keep_dims, CumulativeOperation::SUM, accumulator_type(src));
}

auto
cumulative_min_func(const Device::Pointer & device, const Array::Pointer & src, Array::Pointer dst, int axis, bool keep_dims)
  -> Array::Pointer
{
  return cumulative_axis(device, src, dst, axis, keep_dims, CumulativeOperation::MIN, dType::UNKNOWN);
}

auto
cumulative_max_func(const Device::Pointer & device, const Array::Pointer & src, Array::Pointer dst, int axis, bool keep_dims)
  -> Array::Pointer
{
  return cumulative_axis(device, src, dst, axis, keep_dims, CumulativeOperation::MAX, dType::UNKNOWN);
}

auto
cumulative_product_func(const Device::Pointer & device, const Array::Pointer & src, Array::Pointer dst, int axis, bool keep_dims)
  -> Array::Pointer
{
  return cumulative_axis(device, src, dst, axis, keep_dims, CumulativeOperation::PRODUCT, accumulator_type(src));
}

} // namespace cle::tier1
