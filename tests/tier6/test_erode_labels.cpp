#include "cle.hpp"

#include "test_utils.hpp"
#include <array>
#include <gtest/gtest.h>

class TestErodeLabels : public ::testing::TestWithParam<std::string>
{
protected:
  std::string           backend;
  cle::Device::Pointer  device;

  virtual void
  SetUp()
  {
    backend = GetParam();
    cle::BackendManager::getInstance().setBackend(backend);
    device = cle::BackendManager::getInstance().getBackend().getDevice("", "gpu");
    device->setWaitToFinish(true);
  }
};

TEST_P(TestErodeLabels, erode2d)
{
  const std::array<uint32_t, 6 * 6 * 1> input = { 1, 1, 0, 0, 2, 2, 1, 1, 0, 0, 2, 2, 0, 0, 4, 4, 4, 0,
                                                   0, 0, 4, 4, 4, 0, 5, 5, 4, 4, 4, 3, 5, 5, 0, 0, 3, 3 };

  const std::array<uint32_t, 6 * 6 * 1> valid = { 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                   0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 3 };

  std::array<uint32_t, 6 * 6 * 1> output;

  auto gpu_input = cle::Array::create(6, 6, 1, 2, cle::dType::UINT32, cle::mType::BUFFER, device);
  gpu_input->writeFrom(input.data());
  auto gpu_output = cle::tier6::erode_labels_func(device, gpu_input, nullptr, 1, false);
  gpu_output->readTo(output.data());

  for (size_t i = 0; i < output.size(); i++)
  {
    EXPECT_EQ(output[i], valid[i]);
  }
}

TEST_P(TestErodeLabels, erode3d)
{
  const std::array<uint32_t, 6 * 6 * 5> input = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2, 0, 0, 1, 1, 4, 0, 0, 0, 0, 4, 3, 3, 3, 5, 5, 4, 3, 3, 3, 5, 5, 0, 3, 3, 3,
    0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2, 0, 0, 1, 1, 4, 0, 0, 0, 0, 4, 3, 3, 3, 5, 5, 4, 3, 3, 3, 5, 5, 0, 3, 3, 3,
    0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2, 0, 0, 1, 1, 4, 0, 0, 0, 0, 4, 3, 3, 3, 5, 5, 4, 3, 3, 3, 5, 5, 0, 3, 3, 3,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  };

  const std::array<uint32_t, 6 * 6 * 5> valid = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 0, 0, 0, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  };

  std::array<uint32_t, 6 * 6 * 5> output;

  auto gpu_input = cle::Array::create(6, 6, 5, 3, cle::dType::UINT32, cle::mType::BUFFER, device);
  gpu_input->writeFrom(input.data());
  auto gpu_output = cle::tier6::erode_labels_func(device, gpu_input, nullptr, 1, false);
  gpu_output->readTo(output.data());

  for (size_t i = 0; i < output.size(); i++)
  {
    EXPECT_EQ(output[i], valid[i]);
  }
}

TEST_P(TestErodeLabels, noSaturationAt255)
{
  // Two separate 3x3 blobs: label 1 (top-left) and label 257 (bottom-right, >255).
  // With the saturation bug, UINT8 intermediates truncate 257 to 1, merging both blobs
  // into a single label. The fix (UINT32 intermediates) preserves both as distinct labels.
  const std::array<uint32_t, 9 * 9 * 1> input = {
    0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   1,   1,   1,   0,   0,   0,   0,   0,
    0,   1,   1,   1,   0,   0,   0,   0,   0,
    0,   1,   1,   1,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0, 257, 257, 257,   0,
    0,   0,   0,   0,   0, 257, 257, 257,   0,
    0,   0,   0,   0,   0, 257, 257, 257,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0
  };

  std::array<uint32_t, 9 * 9 * 1> output;

  auto gpu_input = cle::Array::create(9, 9, 1, 2, cle::dType::UINT32, cle::mType::BUFFER, device);
  gpu_input->writeFrom(input.data());
  auto gpu_output = cle::tier6::erode_labels_func(device, gpu_input, nullptr, 1, false);
  gpu_output->readTo(output.data());

  const uint32_t center_label1   = output[2 * 9 + 2];
  const uint32_t center_label257 = output[6 * 9 + 6];

  EXPECT_GT(center_label1, 0)        << "Label 1 blob center should survive erosion";
  EXPECT_GT(center_label257, 0)      << "Label 257 blob center should survive erosion (saturation bug would zero it out)";
  EXPECT_NE(center_label1, center_label257) << "The two blobs must remain distinct labels (saturation bug merges them)";
}

INSTANTIATE_TEST_SUITE_P(InstantiationName, TestErodeLabels, ::testing::ValuesIn(getParameters()));
