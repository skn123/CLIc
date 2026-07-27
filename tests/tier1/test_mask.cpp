#include "cle.hpp"

#include "test_utils.hpp"
#include <array>
#include <gtest/gtest.h>

class TestMask : public ::testing::TestWithParam<std::string>
{
protected:
  std::array<float, 10 * 5 * 3> output;
  std::array<float, 10 * 5 * 3> input;
  std::array<float, 10 * 5 * 3> mask;
  std::array<float, 10 * 5 * 3> valid;

  virtual void
  SetUp()
  {
    for (size_t i = 0; i < input.size(); ++i)
    {
      input[i] = static_cast<float>(rand() % 10);
    }
    std::fill(mask.begin(), mask.end(), static_cast<float>(0));
    std::fill(valid.begin(), valid.end(), static_cast<float>(0));
    const int center = (10 / 2) + (5 / 2) * 10 + (3 / 2) * 10 * 5;
    valid[center] = input[center];
    mask[center] = static_cast<float>(1);
  }
};

TEST_P(TestMask, execute)
{
  std::string param = GetParam();
  cle::BackendManager::getInstance().setBackend(param);
  auto device = cle::BackendManager::getInstance().getBackend().getDevice("", "gpu");
  device->setWaitToFinish(true);

  auto gpu_input = cle::Array::create(10, 5, 3, 3, cle::dType::FLOAT, cle::mType::BUFFER, device);
  auto gpu_mask = cle::Array::create(gpu_input);
  gpu_input->writeFrom(input.data());
  gpu_mask->writeFrom(mask.data());

  auto gpu_output = cle::tier1::mask_func(device, gpu_input, gpu_mask, nullptr);

  gpu_output->readTo(output.data());
  for (int i = 0; i < output.size(); i++)
  {
    EXPECT_EQ(output[i], valid[i]);
  }
}

TEST_P(TestMask, broadcast_src1_over_width_and_depth)
{
  std::string param = GetParam();
  cle::BackendManager::getInstance().setBackend(param);
  auto device = cle::BackendManager::getInstance().getBackend().getDevice("", "gpu");
  device->setWaitToFinish(true);

  constexpr size_t                          width = 3;
  constexpr size_t                          height = 2;
  constexpr size_t                          depth = 2;
  std::array<float, width * height * depth> src_data = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
  std::array<float, height>                 mask_data = { 1.0f, 0.0f };

  auto src = cle::Array::create(width, height, depth, 3, cle::dType::FLOAT, cle::mType::BUFFER, device);
  auto mask = cle::Array::create(1, height, 1, 2, cle::dType::FLOAT, cle::mType::BUFFER, device);
  src->writeFrom(src_data.data());
  mask->writeFrom(mask_data.data());

  auto out = cle::tier1::mask_func(device, src, mask, nullptr);

  EXPECT_EQ(out->width(), width);
  EXPECT_EQ(out->height(), height);
  EXPECT_EQ(out->depth(), depth);

  std::array<float, width * height * depth> out_data = { 0 };
  out->readTo(out_data.data());

  for (size_t z = 0; z < depth; ++z)
  {
    for (size_t y = 0; y < height; ++y)
    {
      for (size_t x = 0; x < width; ++x)
      {
        const size_t idx = z * width * height + y * width + x;
        EXPECT_FLOAT_EQ(out_data[idx], (mask_data[y] != 0.0f) ? src_data[idx] : 0.0f);
      }
    }
  }
}

TEST_P(TestMask, broadcast_src0_over_width_and_depth)
{
  std::string param = GetParam();
  cle::BackendManager::getInstance().setBackend(param);
  auto device = cle::BackendManager::getInstance().getBackend().getDevice("", "gpu");
  device->setWaitToFinish(true);

  constexpr size_t                          width = 3;
  constexpr size_t                          height = 2;
  constexpr size_t                          depth = 2;
  std::array<float, height>                 src_data = { 2.0f, 5.0f };
  std::array<float, width * height * depth> mask_data = { 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0 };

  auto src = cle::Array::create(1, height, 1, 2, cle::dType::FLOAT, cle::mType::BUFFER, device);
  auto mask = cle::Array::create(width, height, depth, 3, cle::dType::FLOAT, cle::mType::BUFFER, device);
  src->writeFrom(src_data.data());
  mask->writeFrom(mask_data.data());

  auto out = cle::tier1::mask_func(device, src, mask, nullptr);

  EXPECT_EQ(out->width(), width);
  EXPECT_EQ(out->height(), height);
  EXPECT_EQ(out->depth(), depth);

  std::array<float, width * height * depth> out_data = { 0 };
  out->readTo(out_data.data());

  for (size_t z = 0; z < depth; ++z)
  {
    for (size_t y = 0; y < height; ++y)
    {
      for (size_t x = 0; x < width; ++x)
      {
        const size_t idx = z * width * height + y * width + x;
        EXPECT_FLOAT_EQ(out_data[idx], (mask_data[idx] != 0.0f) ? src_data[y] : 0.0f);
      }
    }
  }
}

TEST_P(TestMask, incompatible_shapes_throw)
{
  std::string param = GetParam();
  cle::BackendManager::getInstance().setBackend(param);
  auto device = cle::BackendManager::getInstance().getBackend().getDevice("", "gpu");
  device->setWaitToFinish(true);

  auto src = cle::Array::create(4, 2, 1, 2, cle::dType::FLOAT, cle::mType::BUFFER, device);
  auto mask = cle::Array::create(3, 2, 1, 2, cle::dType::FLOAT, cle::mType::BUFFER, device);

  EXPECT_THROW(
    {
      auto out = cle::tier1::mask_func(device, src, mask, nullptr);
      (void)out;
    },
    std::invalid_argument);
}

INSTANTIATE_TEST_SUITE_P(InstantiationName, TestMask, ::testing::ValuesIn(getParameters()));
