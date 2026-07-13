#include "cle.hpp"

#include "test_utils.hpp"
#include <array>
#include <gtest/gtest.h>

class TestScatter : public ::testing::TestWithParam<std::string>
{
protected:
  std::array<float, 6 * 1 * 1> dst_init = { 0, 0, 0, 0, 0, 0 };
  std::array<float, 3 * 1 * 1> src_data = { 7, 8, 9 };
};


TEST_P(TestScatter, executeStridedX)
{
  std::array<float, 6 * 1 * 1> output;
  std::array<float, 6 * 1 * 1> valid = { 7, 0, 8, 0, 9, 0 };

  std::string param = GetParam();
  cle::BackendManager::getInstance().setBackend(param);
  auto device = cle::BackendManager::getInstance().getBackend().getDevice("", "gpu");
  device->setWaitToFinish(true);

  auto gpu_dst = cle::Array::create(6, 1, 1, 1, cle::dType::FLOAT, cle::mType::BUFFER, device);
  gpu_dst->writeFrom(dst_init.data());
  auto gpu_src = cle::Array::create(3, 1, 1, 1, cle::dType::FLOAT, cle::mType::BUFFER, device);
  gpu_src->writeFrom(src_data.data());

  cle::tier1::scatter_func(device, gpu_src, gpu_dst, 0, 6, 2, 0, 1, 1, 0, 1, 1);

  gpu_dst->readTo(output.data());
  for (int i = 0; i < output.size(); i++)
  {
    EXPECT_EQ(output[i], valid[i]);
  }
}

TEST_P(TestScatter, executeNonDivisibleStride)
{
  // x in [1, 6) step 2 -> positions 1, 3, 5, i.e. ceil(5 / 2) = 3 elements
  std::array<float, 6 * 1 * 1> output;
  std::array<float, 6 * 1 * 1> valid = { 0, 7, 0, 8, 0, 9 };

  std::string param = GetParam();
  cle::BackendManager::getInstance().setBackend(param);
  auto device = cle::BackendManager::getInstance().getBackend().getDevice("", "gpu");
  device->setWaitToFinish(true);

  auto gpu_dst = cle::Array::create(6, 1, 1, 1, cle::dType::FLOAT, cle::mType::BUFFER, device);
  gpu_dst->writeFrom(dst_init.data());
  auto gpu_src = cle::Array::create(3, 1, 1, 1, cle::dType::FLOAT, cle::mType::BUFFER, device);
  gpu_src->writeFrom(src_data.data());

  cle::tier1::scatter_func(device, gpu_src, gpu_dst, 1, 6, 2, 0, 1, 1, 0, 1, 1);

  gpu_dst->readTo(output.data());
  for (int i = 0; i < output.size(); i++)
  {
    EXPECT_EQ(output[i], valid[i]);
  }
}
INSTANTIATE_TEST_SUITE_P(InstantiationName, TestScatter, ::testing::ValuesIn(getParameters()));
