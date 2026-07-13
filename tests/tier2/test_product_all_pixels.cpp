#include "cle.hpp"

#include "test_utils.hpp"
#include <array>
#include <gtest/gtest.h>

class TestProductAllPixel : public ::testing::TestWithParam<std::string>
{
protected:
  std::string                  backend;
  cle::Device::Pointer         device;
  std::array<float, 2 * 2 * 2> input;
  float                        valid;

  virtual void
  SetUp()
  {
    backend = GetParam();
    cle::BackendManager::getInstance().setBackend(backend);
    device = cle::BackendManager::getInstance().getBackend().getDevice("", "gpu");
    device->setWaitToFinish(true);
    std::fill(input.begin(), input.end(), 2.0f);
    valid = 256.0f; // 2^8
  }
};

TEST_P(TestProductAllPixel, execute)
{
  auto array = cle::Array::create(2, 2, 2, 3, cle::dType::FLOAT, cle::mType::BUFFER, device);
  array->writeFrom(input.data());

  auto output = cle::tier2::product_of_all_pixels_func(device, array);

  EXPECT_EQ(output, valid);
}
INSTANTIATE_TEST_SUITE_P(InstantiationName, TestProductAllPixel, ::testing::ValuesIn(getParameters()));
