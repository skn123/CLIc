#include "tier0.hpp"

namespace cle::tier0
{

/**
 * @brief Check ptr and set type if unknown
 * @param src source array pointer
 * @param dst destination array pointer
 * @param type data type
 * @return true if dst is not null
 */
auto
check_and_set(const Array::Pointer & src, Array::Pointer & dst, dType & type) -> bool
{
  if (dst != nullptr)
  {
    return true;
  }
  if (src == nullptr)
  {
    throw std::invalid_argument("Error: Cannot generate output Array because the provided 'src' is null.");
  }
  if (type == dType::UNKNOWN)
  {
    type = src->dtype();
  }
  return false;
}

/**
 * @brief Validate a user-provided destination array against the expected extents and dimension
 * @param dst destination array pointer (assumed non-null)
 * @param width expected width
 * @param height expected height
 * @param depth expected depth
 * @param dimension expected dimension (0 to skip the dimension check)
 * @return void
 */
static auto
check_dst_shape(const Array::Pointer & dst, size_t width, size_t height, size_t depth, size_t dimension = 0) -> void
{
  if (dst->width() != width || dst->height() != height || dst->depth() != depth)
  {
    throw std::invalid_argument("Error: provided 'dst' extents (" + std::to_string(dst->width()) + "," + std::to_string(dst->height()) +
                                "," + std::to_string(dst->depth()) + ") do not match the expected (" + std::to_string(width) + "," +
                                std::to_string(height) + "," + std::to_string(depth) + ").");
  }
  if (dimension != 0 && dst->dimension() != dimension)
  {
    throw std::invalid_argument("Error: provided 'dst' dimension (" + std::to_string(dst->dimension()) + ") does not match the expected (" +
                                std::to_string(dimension) + ").");
  }
}

/**
 * @brief Create a destination array with the provided parameters
 * @param src source array pointer
 * @param dst destination array pointer
 * @param width width of the array
 * @param height height of the array
 * @param depth depth of the array
 * @param type data type
 * @param keep_dims keep the source dimension instead of inferring it from the extents
 * @return void
 */
auto
create_dst(const Array::Pointer & src, Array::Pointer & dst, size_t width, size_t height, size_t depth, dType type, bool keep_dims) -> void
{
  if (check_and_set(src, dst, type))
  {
    check_dst_shape(dst, width, height, depth, keep_dims ? src->dimension() : 0);
    return;
  }
  auto dim = keep_dims ? src->dimension() : shape_to_dimension(width, height, depth);
  dst = Array::create(width, height, depth, dim, type, src->mtype(), src->device());
}

/**
 * @brief Create a destination array identical to the source array
 * @param src source array pointer
 * @param dst destination array pointer
 * @param type data type
 * @return void
 */
auto
create_like(const Array::Pointer & src, Array::Pointer & dst, dType type) -> void
{
  if (check_and_set(src, dst, type))
  {
    return;
  }
  dst = Array::create(src->width(), src->height(), src->depth(), src->dimension(), type, src->mtype(), src->device());
}

/**
 * @brief Create a destination array with a single element
 * @param src source array pointer
 * @param dst destination array pointer
 * @param type data type
 * @return void
 */
auto
create_one(const Array::Pointer & src, Array::Pointer & dst, dType type) -> void
{
  if (check_and_set(src, dst, type))
  {
    return;
  }
  dst = Array::create(1, 1, 1, 1, type, mType::BUFFER, src->device());
}

/**
 * @brief Create a destination array of dimension 1
 * @param src source array pointer
 * @param dst destination array pointer
 * @param size size of the array
 * @param type data type
 * @return void
 */
auto
create_vector(const Array::Pointer & src, Array::Pointer & dst, const size_t & size, dType type) -> void
{
  if (check_and_set(src, dst, type))
  {
    return;
  }
  dst = Array::create(size, 1, 1, 1, type, mType::BUFFER, src->device());
}

/**
 * @brief Create a destination array with the (x,y,1) as the source array
 * @param src source array pointer
 * @param dst destination array pointer
 * @param type data type
 * @param keep_dims keep the source dimension instead of inferring it from the extents
 * @return void
 */
auto
create_xy(const Array::Pointer & src, Array::Pointer & dst, dType type, bool keep_dims) -> void
{
  if (check_and_set(src, dst, type))
  {
    check_dst_shape(dst, src->width(), src->height(), 1, keep_dims ? src->dimension() : 0);
    return;
  }
  auto dim = keep_dims ? src->dimension() : shape_to_dimension(src->width(), src->height(), 1);
  dst = Array::create(src->width(), src->height(), 1, dim, type, src->mtype(), src->device());
}

/**
 * @brief Create a destination array with the (y,x,1) as the source array
 * @param src source array pointer
 * @param dst destination array pointer
 * @param type data type
 * @param keep_dims keep the source dimension instead of inferring it from the extents
 * @return void
 */
auto
create_yx(const Array::Pointer & src, Array::Pointer & dst, dType type, bool keep_dims) -> void
{
  if (check_and_set(src, dst, type))
  {
    check_dst_shape(dst, src->height(), src->width(), 1, keep_dims ? src->dimension() : 0);
    return;
  }
  auto dim = keep_dims ? src->dimension() : shape_to_dimension(src->height(), src->width(), 1);
  dst = Array::create(src->height(), src->width(), 1, dim, type, src->mtype(), src->device());
}

/**
 * @brief Create a destination array with the (z,y,1) as the source array
 * @param src source array pointer
 * @param dst destination array pointer
 * @param type data type
 * @param keep_dims keep the source dimension instead of inferring it from the extents
 * @return void
 */
auto
create_zy(const Array::Pointer & src, Array::Pointer & dst, dType type, bool keep_dims) -> void
{
  if (check_and_set(src, dst, type))
  {
    check_dst_shape(dst, src->depth(), src->height(), 1, keep_dims ? src->dimension() : 0);
    return;
  }
  auto dim = keep_dims ? src->dimension() : shape_to_dimension(src->depth(), src->height(), 1);
  dst = Array::create(src->depth(), src->height(), 1, dim, type, src->mtype(), src->device());
}

/**
 * @brief Create a destination array with the (y,z,1) as the source array
 * @param src source array pointer
 * @param dst destination array pointer
 * @param type data type
 * @param keep_dims keep the source dimension instead of inferring it from the extents
 * @return void
 */
auto
create_yz(const Array::Pointer & src, Array::Pointer & dst, dType type, bool keep_dims) -> void
{
  if (check_and_set(src, dst, type))
  {
    check_dst_shape(dst, src->height(), src->depth(), 1, keep_dims ? src->dimension() : 0);
    return;
  }
  auto dim = keep_dims ? src->dimension() : shape_to_dimension(src->height(), src->depth(), 1);
  dst = Array::create(src->height(), src->depth(), 1, dim, type, src->mtype(), src->device());
}

/**
 * @brief Create a destination array with the (x,z,1) as the source array
 * @param src source array pointer
 * @param dst destination array pointer
 * @param type data type
 * @param keep_dims keep the source dimension instead of inferring it from the extents
 * @return void
 */
auto
create_xz(const Array::Pointer & src, Array::Pointer & dst, dType type, bool keep_dims) -> void
{
  if (check_and_set(src, dst, type))
  {
    check_dst_shape(dst, src->width(), src->depth(), 1, keep_dims ? src->dimension() : 0);
    return;
  }
  auto dim = keep_dims ? src->dimension() : shape_to_dimension(src->width(), src->depth(), 1);
  dst = Array::create(src->width(), src->depth(), 1, dim, type, src->mtype(), src->device());
}

/**
 * @brief Create a destination array with the (z,x,1) as the source array
 * @param src source array pointer
 * @param dst destination array pointer
 * @param type data type
 * @param keep_dims keep the source dimension instead of inferring it from the extents
 * @return void
 */
auto
create_zx(const Array::Pointer & src, Array::Pointer & dst, dType type, bool keep_dims) -> void
{
  if (check_and_set(src, dst, type))
  {
    check_dst_shape(dst, src->depth(), src->width(), 1, keep_dims ? src->dimension() : 0);
    return;
  }
  auto dim = keep_dims ? src->dimension() : shape_to_dimension(src->depth(), src->width(), 1);
  dst = Array::create(src->depth(), src->width(), 1, dim, type, src->mtype(), src->device());
}

auto
infer_broadcast_shape(const Array::Pointer & src0, const Array::Pointer & src1) -> std::array<size_t, 3>
{
  if (src0 == nullptr || src1 == nullptr)
  {
    throw std::invalid_argument("Error: source Array is null");
  }

  const std::array<size_t, 3> shape0 = { src0->width(), src0->height(), src0->depth() };
  const std::array<size_t, 3> shape1 = { src1->width(), src1->height(), src1->depth() };
  std::array<size_t, 3>       out_shape = { 1, 1, 1 };

  for (size_t axis = 0; axis < 3; ++axis)
  {
    const size_t lhs = shape0[axis];
    const size_t rhs = shape1[axis];
    if (lhs == rhs)
    {
      out_shape[axis] = lhs;
    }
    else if (lhs == 1)
    {
      out_shape[axis] = rhs;
    }
    else if (rhs == 1)
    {
      out_shape[axis] = lhs;
    }
    else
    {
      throw std::invalid_argument("Error: Arrays cannot be broadcast together. src0 shape is (" + std::to_string(shape0[0]) + "," +
                                  std::to_string(shape0[1]) + "," + std::to_string(shape0[2]) + "), src1 shape is (" +
                                  std::to_string(shape1[0]) + "," + std::to_string(shape1[1]) + "," + std::to_string(shape1[2]) + ").");
    }
  }

  return out_shape;
}

auto
infer_broadcast_shape(const std::vector<Array::Pointer> & arrays) -> std::array<size_t, 3>
{
  if (arrays.empty())
  {
    throw std::invalid_argument("Error: array list is empty.");
  }

  std::array<size_t, 3> out_shape = { 1, 1, 1 };
  bool                  has_shape = false;

  for (const auto & arr : arrays)
  {
    if (arr == nullptr)
    {
      throw std::invalid_argument("Error: source Array is null");
    }

    const std::array<size_t, 3> shape = { arr->width(), arr->height(), arr->depth() };
    if (!has_shape)
    {
      out_shape = shape;
      has_shape = true;
      continue;
    }

    for (size_t axis = 0; axis < 3; ++axis)
    {
      const size_t lhs = out_shape[axis];
      const size_t rhs = shape[axis];
      if (lhs == rhs)
      {
        continue;
      }
      if (lhs == 1)
      {
        out_shape[axis] = rhs;
        continue;
      }
      if (rhs == 1)
      {
        continue;
      }
      throw std::invalid_argument("Error: Arrays cannot be broadcast together.");
    }
  }

  return out_shape;
}

auto
create_or_check_broadcast_dst(const Array::Pointer & src0, const Array::Pointer & src1, Array::Pointer & dst, dType output_type) -> void
{
  if (src0 == nullptr || src1 == nullptr)
  {
    throw std::invalid_argument("Error: source Array is null");
  }

  if (src0->device() != src1->device())
  {
    throw std::invalid_argument("Error: source Arrays are on different devices.");
  }

  const auto out_shape = infer_broadcast_shape(src0, src1);
  const auto out_dim = shape_to_dimension(out_shape[0], out_shape[1], out_shape[2]);

  if (dst == nullptr)
  {
    dst = Array::create(out_shape[0], out_shape[1], out_shape[2], out_dim, output_type, src0->mtype(), src0->device());
    return;
  }

  check_dst_shape(dst, out_shape[0], out_shape[1], out_shape[2], out_dim);
}


} // namespace cle::tier0
