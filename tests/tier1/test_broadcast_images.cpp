#include "cle.hpp"

#include "test_utils.hpp"
#include <array>
#include <gtest/gtest.h>

class TestBroadcastImages : public ::testing::TestWithParam<std::string>
{
protected:
  std::string          backend;
  cle::Device::Pointer device;

  virtual void
  SetUp()
  {
    backend = GetParam();
    cle::BackendManager::getInstance().setBackend(backend);
    device = cle::BackendManager::getInstance().getBackend().getDevice("", "gpu");
    device->setWaitToFinish(true);
  }
};

TEST_P(TestBroadcastImages, maximum_images_broadcast_row_and_depth)
{
  constexpr size_t width = 4;
  constexpr size_t height = 3;
  constexpr size_t depth = 2;

  std::array<float, width * height * depth> src0_data = { 0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6, 4, 5, 6, 7, 5, 6, 7, 8 };

  // Shape (1, 3, 1): broadcasts over width and depth.
  std::array<float, 3> src1_data = { 2, 4, 6 };

  auto gpu_src0 = cle::Array::create(width, height, depth, 3, cle::dType::FLOAT, cle::mType::BUFFER, device);
  auto gpu_src1 = cle::Array::create(1, height, 1, 2, cle::dType::FLOAT, cle::mType::BUFFER, device);

  gpu_src0->writeFrom(src0_data.data());
  gpu_src1->writeFrom(src1_data.data());

  auto gpu_out = cle::tier1::maximum_images_func(device, gpu_src0, gpu_src1, nullptr);

  EXPECT_EQ(gpu_out->width(), width);
  EXPECT_EQ(gpu_out->height(), height);
  EXPECT_EQ(gpu_out->depth(), depth);

  std::array<float, width * height * depth> out_data = { 0 };
  gpu_out->readTo(out_data.data());

  for (size_t z = 0; z < depth; ++z)
  {
    for (size_t y = 0; y < height; ++y)
    {
      for (size_t x = 0; x < width; ++x)
      {
        const size_t idx = z * width * height + y * width + x;
        EXPECT_EQ(out_data[idx], std::max(src0_data[idx], src1_data[y]));
      }
    }
  }
}

TEST_P(TestBroadcastImages, greater_func_broadcast_depth_vector)
{
  constexpr size_t width = 3;
  constexpr size_t height = 2;
  constexpr size_t depth = 2;

  std::array<int8_t, width * height * depth> src0_data = { 1, 2, 3, 4, 5, 6, 3, 2, 1, 6, 5, 4 };

  // Shape (1, 1, 2): broadcasts over width and height.
  std::array<int8_t, depth> src1_data = { 2, 4 };

  auto gpu_src0 = cle::Array::create(width, height, depth, 3, cle::dType::INT8, cle::mType::BUFFER, device);
  auto gpu_src1 = cle::Array::create(1, 1, depth, 3, cle::dType::INT8, cle::mType::BUFFER, device);

  gpu_src0->writeFrom(src0_data.data());
  gpu_src1->writeFrom(src1_data.data());

  auto gpu_out = cle::tier1::greater_func(device, gpu_src0, gpu_src1, nullptr);

  EXPECT_EQ(gpu_out->width(), width);
  EXPECT_EQ(gpu_out->height(), height);
  EXPECT_EQ(gpu_out->depth(), depth);

  std::array<int8_t, width * height * depth> out_data = { 0 };
  gpu_out->readTo(out_data.data());

  for (size_t z = 0; z < depth; ++z)
  {
    for (size_t y = 0; y < height; ++y)
    {
      for (size_t x = 0; x < width; ++x)
      {
        const size_t idx = z * width * height + y * width + x;
        EXPECT_EQ(out_data[idx], src0_data[idx] > src1_data[z] ? 1 : 0);
      }
    }
  }
}

TEST_P(TestBroadcastImages, incompatible_shapes_throw)
{
  auto gpu_src0 = cle::Array::create(4, 3, 1, 2, cle::dType::FLOAT, cle::mType::BUFFER, device);
  auto gpu_src1 = cle::Array::create(5, 3, 1, 2, cle::dType::FLOAT, cle::mType::BUFFER, device);

  EXPECT_THROW(
    {
      auto out = cle::tier1::maximum_images_func(device, gpu_src0, gpu_src1, nullptr);
      (void)out;
    },
    std::invalid_argument);
}

INSTANTIATE_TEST_SUITE_P(InstantiationName, TestBroadcastImages, ::testing::ValuesIn(getParameters()));
