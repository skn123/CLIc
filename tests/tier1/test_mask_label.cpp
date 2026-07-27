#include "cle.hpp"

#include "test_utils.hpp"
#include <array>
#include <gtest/gtest.h>

class TestMaskLabel : public ::testing::TestWithParam<std::string>
{
protected:
  std::array<float, 5 * 5 * 1> output;
  std::array<float, 5 * 5 * 1> input = { 0, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 3, 3, 4, 0, 0, 4, 4, 5, 0, 0, 0, 0, 0, 0 };
  std::array<float, 5 * 5 * 1> mask = { 0, 0, 0, 0, 0, 0, 2, 1, 3, 0, 0, 2, 1, 3, 0, 0, 1, 1, 3, 0, 0, 0, 0, 0, 0 };
  std::array<float, 5 * 5 * 1> valid = { 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0, 0 };
};

TEST_P(TestMaskLabel, execute)
{
  std::string param = GetParam();
  cle::BackendManager::getInstance().setBackend(param);
  auto device = cle::BackendManager::getInstance().getBackend().getDevice("", "gpu");
  device->setWaitToFinish(true);

  auto gpu_input = cle::Array::create(5, 5, 1, 3, cle::dType::FLOAT, cle::mType::BUFFER, device);
  auto gpu_mask = cle::Array::create(gpu_input);
  gpu_input->writeFrom(input.data());
  gpu_mask->writeFrom(mask.data());

  auto gpu_output = cle::tier1::mask_label_func(device, gpu_input, gpu_mask, nullptr, 1);

  gpu_output->readTo(output.data());
  for (int i = 0; i < output.size(); i++)
  {
    EXPECT_EQ(output[i], valid[i]);
  }
}

TEST_P(TestMaskLabel, broadcast_src1_over_width_and_depth)
{
  std::string param = GetParam();
  cle::BackendManager::getInstance().setBackend(param);
  auto device = cle::BackendManager::getInstance().getBackend().getDevice("", "gpu");
  device->setWaitToFinish(true);

  constexpr size_t width = 3;
  constexpr size_t height = 2;
  constexpr size_t depth = 2;
  std::array<float, width * height * depth> src_data = {
    1, 2, 3,
    4, 5, 6,
    7, 8, 9,
    10, 11, 12
  };
  std::array<float, height> labels = { 1.0f, 2.0f };

  auto src = cle::Array::create(width, height, depth, 3, cle::dType::FLOAT, cle::mType::BUFFER, device);
  auto lbl = cle::Array::create(1, height, 1, 2, cle::dType::FLOAT, cle::mType::BUFFER, device);
  src->writeFrom(src_data.data());
  lbl->writeFrom(labels.data());

  auto out = cle::tier1::mask_label_func(device, src, lbl, nullptr, 1.0f);

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
        EXPECT_FLOAT_EQ(out_data[idx], (labels[y] == 1.0f) ? src_data[idx] : 0.0f);
      }
    }
  }
}

TEST_P(TestMaskLabel, broadcast_src0_over_width_and_depth)
{
  std::string param = GetParam();
  cle::BackendManager::getInstance().setBackend(param);
  auto device = cle::BackendManager::getInstance().getBackend().getDevice("", "gpu");
  device->setWaitToFinish(true);

  constexpr size_t width = 3;
  constexpr size_t height = 2;
  constexpr size_t depth = 2;
  std::array<float, height> src_data = { 2.0f, 5.0f };
  std::array<float, width * height * depth> labels = {
    1, 2, 1,
    2, 1, 2,
    2, 2, 1,
    1, 1, 2
  };

  auto src = cle::Array::create(1, height, 1, 2, cle::dType::FLOAT, cle::mType::BUFFER, device);
  auto lbl = cle::Array::create(width, height, depth, 3, cle::dType::FLOAT, cle::mType::BUFFER, device);
  src->writeFrom(src_data.data());
  lbl->writeFrom(labels.data());

  auto out = cle::tier1::mask_label_func(device, src, lbl, nullptr, 1.0f);

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
        EXPECT_FLOAT_EQ(out_data[idx], (labels[idx] == 1.0f) ? src_data[y] : 0.0f);
      }
    }
  }
}

TEST_P(TestMaskLabel, incompatible_shapes_throw)
{
  std::string param = GetParam();
  cle::BackendManager::getInstance().setBackend(param);
  auto device = cle::BackendManager::getInstance().getBackend().getDevice("", "gpu");
  device->setWaitToFinish(true);

  auto src0 = cle::Array::create(4, 2, 1, 2, cle::dType::FLOAT, cle::mType::BUFFER, device);
  auto src1 = cle::Array::create(3, 2, 1, 2, cle::dType::FLOAT, cle::mType::BUFFER, device);

  EXPECT_THROW({
    auto out = cle::tier1::mask_label_func(device, src0, src1, nullptr, 1.0f);
    (void) out;
  }, std::invalid_argument);
}

INSTANTIATE_TEST_SUITE_P(InstantiationName, TestMaskLabel, ::testing::ValuesIn(getParameters()));
