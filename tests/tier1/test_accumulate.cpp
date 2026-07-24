#include "cle.hpp"

#include "test_utils.hpp"
#include <gtest/gtest.h>
#include <vector>

namespace
{
enum class CpuOp
{
  Sum,
  Min,
  Max,
  Product,
};

auto
idx(int x, int y, int z, int width, int height) -> int
{
  return x + y * width + z * width * height;
}

auto
cpu_cumulative(const std::vector<float> & input, int width, int height, int depth, int axis, CpuOp op) -> std::vector<float>
{
  std::vector<float> output(input.size(), 0.0f);

  if (axis == 0)
  {
    for (int z = 0; z < depth; ++z)
    {
      for (int y = 0; y < height; ++y)
      {
        float cumulative = 0.0f;
        for (int x = 0; x < width; ++x)
        {
          const float value = input[idx(x, y, z, width, height)];
          if (x == 0)
          {
            cumulative = value;
          }
          else if (op == CpuOp::Sum)
          {
            cumulative += value;
          }
          else if (op == CpuOp::Min)
          {
            cumulative = (cumulative < value) ? cumulative : value;
          }
          else if (op == CpuOp::Max)
          {
            cumulative = (cumulative > value) ? cumulative : value;
          }
          else
          {
            cumulative *= value;
          }
          output[idx(x, y, z, width, height)] = cumulative;
        }
      }
    }
  }
  else if (axis == 1)
  {
    for (int z = 0; z < depth; ++z)
    {
      for (int x = 0; x < width; ++x)
      {
        float cumulative = 0.0f;
        for (int y = 0; y < height; ++y)
        {
          const float value = input[idx(x, y, z, width, height)];
          if (y == 0)
          {
            cumulative = value;
          }
          else if (op == CpuOp::Sum)
          {
            cumulative += value;
          }
          else if (op == CpuOp::Min)
          {
            cumulative = (cumulative < value) ? cumulative : value;
          }
          else if (op == CpuOp::Max)
          {
            cumulative = (cumulative > value) ? cumulative : value;
          }
          else
          {
            cumulative *= value;
          }
          output[idx(x, y, z, width, height)] = cumulative;
        }
      }
    }
  }
  else
  {
    for (int y = 0; y < height; ++y)
    {
      for (int x = 0; x < width; ++x)
      {
        float cumulative = 0.0f;
        for (int z = 0; z < depth; ++z)
        {
          const float value = input[idx(x, y, z, width, height)];
          if (z == 0)
          {
            cumulative = value;
          }
          else if (op == CpuOp::Sum)
          {
            cumulative += value;
          }
          else if (op == CpuOp::Min)
          {
            cumulative = (cumulative < value) ? cumulative : value;
          }
          else if (op == CpuOp::Max)
          {
            cumulative = (cumulative > value) ? cumulative : value;
          }
          else
          {
            cumulative *= value;
          }
          output[idx(x, y, z, width, height)] = cumulative;
        }
      }
    }
  }

  return output;
}

} // namespace

class TestAccumulate : public ::testing::TestWithParam<std::string>
{
protected:
  std::string          backend;
  cle::Device::Pointer device;

  void
  SetUp() override
  {
    backend = GetParam();
    cle::BackendManager::getInstance().setBackend(backend);
    device = cle::BackendManager::getInstance().getBackend().getDevice("", "gpu");
    device->setWaitToFinish(true);
  }
};

TEST_P(TestAccumulate, cumulative_sum_all_axes)
{
  constexpr int             width = 3;
  constexpr int             height = 2;
  constexpr int             depth = 2;
  const std::vector<float>  input = { 3, 1, 4, 2, 5, 0, 6, 2, 7, 1, 8, 3 };
  std::vector<float>        output(input.size());
  const auto                gpu_input = cle::Array::create(width, height, depth, 3, cle::dType::FLOAT, cle::mType::BUFFER, device);
  gpu_input->writeFrom(input.data());

  for (int axis = 0; axis < 3; ++axis)
  {
    const auto expected = cpu_cumulative(input, width, height, depth, axis, CpuOp::Sum);
    auto       gpu_out = cle::tier1::cumulative_sum_func(device, gpu_input, nullptr, axis, false);
    gpu_out->readTo(output.data());

    for (size_t i = 0; i < output.size(); ++i)
    {
      EXPECT_NEAR(output[i], expected[i], 1e-5f);
    }
  }
}

TEST_P(TestAccumulate, cumulative_min_all_axes)
{
  constexpr int             width = 3;
  constexpr int             height = 2;
  constexpr int             depth = 2;
  const std::vector<float>  input = { 3, 1, 4, 2, 5, 0, 6, 2, 7, 1, 8, 3 };
  std::vector<float>        output(input.size());
  const auto                gpu_input = cle::Array::create(width, height, depth, 3, cle::dType::FLOAT, cle::mType::BUFFER, device);
  gpu_input->writeFrom(input.data());

  for (int axis = 0; axis < 3; ++axis)
  {
    const auto expected = cpu_cumulative(input, width, height, depth, axis, CpuOp::Min);
    auto       gpu_out = cle::tier1::cumulative_min_func(device, gpu_input, nullptr, axis, false);
    gpu_out->readTo(output.data());

    for (size_t i = 0; i < output.size(); ++i)
    {
      EXPECT_NEAR(output[i], expected[i], 1e-5f);
    }
  }
}

TEST_P(TestAccumulate, cumulative_max_all_axes)
{
  constexpr int             width = 3;
  constexpr int             height = 2;
  constexpr int             depth = 2;
  const std::vector<float>  input = { 3, 1, 4, 2, 5, 0, 6, 2, 7, 1, 8, 3 };
  std::vector<float>        output(input.size());
  const auto                gpu_input = cle::Array::create(width, height, depth, 3, cle::dType::FLOAT, cle::mType::BUFFER, device);
  gpu_input->writeFrom(input.data());

  for (int axis = 0; axis < 3; ++axis)
  {
    const auto expected = cpu_cumulative(input, width, height, depth, axis, CpuOp::Max);
    auto       gpu_out = cle::tier1::cumulative_max_func(device, gpu_input, nullptr, axis, false);
    gpu_out->readTo(output.data());

    for (size_t i = 0; i < output.size(); ++i)
    {
      EXPECT_NEAR(output[i], expected[i], 1e-5f);
    }
  }
}

TEST_P(TestAccumulate, cumulative_product_all_axes)
{
  constexpr int             width = 3;
  constexpr int             height = 2;
  constexpr int             depth = 2;
  const std::vector<float>  input = { 3, 1, 4, 2, 5, 2, 6, 2, 7, 1, 8, 3 };
  std::vector<float>        output(input.size());
  const auto                gpu_input = cle::Array::create(width, height, depth, 3, cle::dType::FLOAT, cle::mType::BUFFER, device);
  gpu_input->writeFrom(input.data());

  for (int axis = 0; axis < 3; ++axis)
  {
    const auto expected = cpu_cumulative(input, width, height, depth, axis, CpuOp::Product);
    auto       gpu_out = cle::tier1::cumulative_product_func(device, gpu_input, nullptr, axis, false);
    gpu_out->readTo(output.data());

    for (size_t i = 0; i < output.size(); ++i)
    {
      EXPECT_NEAR(output[i], expected[i], 1e-5f);
    }
  }
}

TEST_P(TestAccumulate, cumulative_keep_dims_true)
{
  constexpr int            width = 3;
  constexpr int            height = 2;
  constexpr int            depth = 2;
  const std::vector<float> input = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
  const auto               gpu_input = cle::Array::create(width, height, depth, 3, cle::dType::FLOAT, cle::mType::BUFFER, device);
  gpu_input->writeFrom(input.data());

  auto gpu_out = cle::tier1::cumulative_sum_func(device, gpu_input, nullptr, 1, true);

  EXPECT_EQ(gpu_out->width(), width);
  EXPECT_EQ(gpu_out->height(), height);
  EXPECT_EQ(gpu_out->depth(), depth);
  EXPECT_EQ(gpu_out->dimension(), gpu_input->dimension());
}

TEST_P(TestAccumulate, cumulative_invalid_axis_throws)
{
  constexpr int            width = 3;
  constexpr int            height = 2;
  constexpr int            depth = 2;
  const std::vector<float> input = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
  const auto               gpu_input = cle::Array::create(width, height, depth, 3, cle::dType::FLOAT, cle::mType::BUFFER, device);
  gpu_input->writeFrom(input.data());

  EXPECT_THROW(cle::tier1::cumulative_sum_func(device, gpu_input, nullptr, 3, false), std::invalid_argument);
}

INSTANTIATE_TEST_SUITE_P(InstantiationName, TestAccumulate, ::testing::ValuesIn(getParameters()));
