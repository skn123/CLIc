#include "cle.hpp"

#include "test_utils.hpp"
#include <array>
#include <gtest/gtest.h>

class TestAddImagesWeighted : public ::testing::TestWithParam<std::string>
{
protected:
  std::array<float, 10 * 5 * 3> output;
  std::array<float, 10 * 5 * 3> input1;
  std::array<float, 10 * 5 * 3> input2;
  std::array<float, 10 * 5 * 3> valid;

  const float value_1 = 25;
  const float value_2 = 75;
  const float factor1 = 0.5;
  const float factor2 = 0.25;

  virtual void
  SetUp()
  {
    std::fill(input1.begin(), input1.end(), value_1);
    std::fill(input2.begin(), input2.end(), value_2);
    std::fill(valid.begin(), valid.end(), static_cast<float>(value_1 * factor1 + value_2 * factor2));
  }
};

TEST_P(TestAddImagesWeighted, execute)
{
  std::string param = GetParam();
  cle::BackendManager::getInstance().setBackend(param);
  auto device = cle::BackendManager::getInstance().getBackend().getDevice("", "gpu");
  device->setWaitToFinish(true);

  auto gpu_input1 = cle::Array::create(10, 5, 3, 3, cle::dType::FLOAT, cle::mType::BUFFER, device);
  auto gpu_input2 = cle::Array::create(gpu_input1);
  gpu_input1->writeFrom(input1.data());
  gpu_input2->writeFrom(input2.data());
  auto gpu_output = cle::tier1::add_images_weighted_func(device, gpu_input1, gpu_input2, nullptr, factor1, factor2);

  gpu_output->readTo(output.data());
  for (int i = 0; i < output.size(); i++)
  {
    EXPECT_EQ(output[i], valid[i]);
  }
}

TEST_P(TestAddImagesWeighted, broadcast_src1_over_width_and_depth)
{
  std::string param = GetParam();
  cle::BackendManager::getInstance().setBackend(param);
  auto device = cle::BackendManager::getInstance().getBackend().getDevice("", "gpu");
  device->setWaitToFinish(true);

  constexpr size_t width = 4;
  constexpr size_t height = 2;
  constexpr size_t depth = 2;
  constexpr float  f1 = 2.0f;
  constexpr float  f2 = 0.5f;

  std::array<float, width * height * depth> src0_data = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
  std::array<float, height>                 src1_data = { 10.0f, 20.0f };

  auto src0 = cle::Array::create(width, height, depth, 3, cle::dType::FLOAT, cle::mType::BUFFER, device);
  auto src1 = cle::Array::create(1, height, 1, 2, cle::dType::FLOAT, cle::mType::BUFFER, device);
  src0->writeFrom(src0_data.data());
  src1->writeFrom(src1_data.data());

  auto out = cle::tier1::add_images_weighted_func(device, src0, src1, nullptr, f1, f2);

  EXPECT_EQ(out->width(), width);
  EXPECT_EQ(out->height(), height);
  EXPECT_EQ(out->depth(), depth);

  std::array<float, width * height * depth> output_data = { 0 };
  out->readTo(output_data.data());

  for (size_t z = 0; z < depth; ++z)
  {
    for (size_t y = 0; y < height; ++y)
    {
      for (size_t x = 0; x < width; ++x)
      {
        const size_t idx = z * width * height + y * width + x;
        EXPECT_FLOAT_EQ(output_data[idx], src0_data[idx] * f1 + src1_data[y] * f2);
      }
    }
  }
}

TEST_P(TestAddImagesWeighted, broadcast_src0_over_width_and_depth)
{
  std::string param = GetParam();
  cle::BackendManager::getInstance().setBackend(param);
  auto device = cle::BackendManager::getInstance().getBackend().getDevice("", "gpu");
  device->setWaitToFinish(true);

  constexpr size_t width = 4;
  constexpr size_t height = 2;
  constexpr size_t depth = 2;
  constexpr float  f1 = 2.0f;
  constexpr float  f2 = 0.5f;

  std::array<float, height>                 src0_data = { 1.0f, 3.0f };
  std::array<float, width * height * depth> src1_data = { 2, 4, 6, 8, 1, 3, 5, 7, 10, 20, 30, 40, 9, 8, 7, 6 };

  auto src0 = cle::Array::create(1, height, 1, 2, cle::dType::FLOAT, cle::mType::BUFFER, device);
  auto src1 = cle::Array::create(width, height, depth, 3, cle::dType::FLOAT, cle::mType::BUFFER, device);
  src0->writeFrom(src0_data.data());
  src1->writeFrom(src1_data.data());

  auto out = cle::tier1::add_images_weighted_func(device, src0, src1, nullptr, f1, f2);

  EXPECT_EQ(out->width(), width);
  EXPECT_EQ(out->height(), height);
  EXPECT_EQ(out->depth(), depth);

  std::array<float, width * height * depth> output_data = { 0 };
  out->readTo(output_data.data());

  for (size_t z = 0; z < depth; ++z)
  {
    for (size_t y = 0; y < height; ++y)
    {
      for (size_t x = 0; x < width; ++x)
      {
        const size_t idx = z * width * height + y * width + x;
        EXPECT_FLOAT_EQ(output_data[idx], src0_data[y] * f1 + src1_data[idx] * f2);
      }
    }
  }
}

TEST_P(TestAddImagesWeighted, broadcast_src1_into_preallocated_output)
{
  std::string param = GetParam();
  cle::BackendManager::getInstance().setBackend(param);
  auto device = cle::BackendManager::getInstance().getBackend().getDevice("", "gpu");
  device->setWaitToFinish(true);

  constexpr size_t width = 4;
  constexpr size_t height = 2;
  constexpr size_t depth = 2;
  constexpr float  f1 = 2.0f;
  constexpr float  f2 = 0.5f;

  std::array<float, width * height * depth> src0_data = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
  std::array<float, height>                 src1_data = { 10.0f, 20.0f };

  auto src0 = cle::Array::create(width, height, depth, 3, cle::dType::FLOAT, cle::mType::BUFFER, device);
  auto src1 = cle::Array::create(1, height, 1, 2, cle::dType::FLOAT, cle::mType::BUFFER, device);
  auto out = cle::Array::create(width, height, depth, 3, cle::dType::FLOAT, cle::mType::BUFFER, device);
  src0->writeFrom(src0_data.data());
  src1->writeFrom(src1_data.data());

  auto returned = cle::tier1::add_images_weighted_func(device, src0, src1, out, f1, f2);

  EXPECT_EQ(returned, out);

  std::array<float, width * height * depth> output_data = { 0 };
  out->readTo(output_data.data());

  for (size_t z = 0; z < depth; ++z)
  {
    for (size_t y = 0; y < height; ++y)
    {
      for (size_t x = 0; x < width; ++x)
      {
        const size_t idx = z * width * height + y * width + x;
        EXPECT_FLOAT_EQ(output_data[idx], src0_data[idx] * f1 + src1_data[y] * f2);
      }
    }
  }
}

TEST_P(TestAddImagesWeighted, incompatible_shapes_throw)
{
  std::string param = GetParam();
  cle::BackendManager::getInstance().setBackend(param);
  auto device = cle::BackendManager::getInstance().getBackend().getDevice("", "gpu");
  device->setWaitToFinish(true);

  auto src0 = cle::Array::create(4, 2, 1, 2, cle::dType::FLOAT, cle::mType::BUFFER, device);
  auto src1 = cle::Array::create(3, 2, 1, 2, cle::dType::FLOAT, cle::mType::BUFFER, device);

  EXPECT_THROW(
    {
      auto out = cle::tier1::add_images_weighted_func(device, src0, src1, nullptr, 1.0f, 1.0f);
      (void)out;
    },
    std::invalid_argument);
}

INSTANTIATE_TEST_SUITE_P(InstantiationName, TestAddImagesWeighted, ::testing::ValuesIn(getParameters()));
