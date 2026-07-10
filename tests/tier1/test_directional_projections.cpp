#include "cle.hpp"

#include "test_utils.hpp"
#include <array>
#include <gtest/gtest.h>
#include <random>

// Parameterized test for all directional projections
// Tests: std, mean, minimum, maximum, sum projections
// Directions: x, y, z axes

class TestDirectionalProjections : public ::testing::TestWithParam<std::string>
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

// Test STD projections with static data
TEST_P(TestDirectionalProjections, std_x_projection)
{
  std::array<float, 5 * 5 * 1> output;
  std::array<float, 5 * 5 * 5> input = { 1, 0, 0, 0, 9, 0, 2, 0, 8, 0,  3, 0, 1, 0, 10, 0, 4, 0, 7, 0,  5, 0, 6, 0, 10,
                                         0, 2, 0, 8, 0, 1, 0, 0, 0, 9,  3, 0, 1, 0, 10, 0, 4, 0, 7, 0,  5, 0, 6, 0, 10,
                                         0, 2, 0, 8, 0, 3, 0, 1, 0, 10, 0, 4, 0, 7, 0,  1, 0, 0, 0, 9,  5, 0, 6, 0, 10,
                                         0, 2, 0, 8, 0, 1, 0, 0, 0, 9,  0, 4, 0, 7, 0,  3, 0, 1, 0, 10, 5, 0, 6, 0, 10,
                                         1, 0, 0, 0, 9, 0, 4, 0, 7, 0,  3, 0, 1, 0, 10, 0, 2, 0, 8, 0,  5, 0, 6, 0, 10 };
  std::array<float, 5 * 5 * 1> valid = { 3.52, 3.10, 3.76, 2.86, 3.82, 3.10, 3.52, 3.76, 2.86, 3.82, 3.10, 3.76, 2.86,
                                         3.52, 3.82, 3.10, 3.52, 2.86, 3.76, 3.82, 3.52, 2.86, 3.76, 3.10, 3.82 };

  auto gpu_input = cle::Array::create(5, 5, 5, 3, cle::dType::FLOAT, cle::mType::BUFFER, device);
  gpu_input->writeFrom(input.data());
  auto gpu_output = cle::tier1::std_x_projection_func(device, gpu_input, nullptr);
  gpu_output->readTo(output.data());

  for (size_t i = 0; i < output.size(); i++)
  {
    EXPECT_NEAR(output[i], valid[i], 0.01);
  }
}

TEST_P(TestDirectionalProjections, std_y_projection)
{
  std::array<float, 5 * 5 * 1> output;
  std::array<float, 5 * 5 * 5> input = { 1, 0, 0, 0, 9, 0, 2, 0, 8, 0,  3, 0, 1, 0, 10, 0, 4, 0, 7, 0,  5, 0, 6, 0, 10,
                                         0, 2, 0, 8, 0, 1, 0, 0, 0, 9,  3, 0, 1, 0, 10, 0, 4, 0, 7, 0,  5, 0, 6, 0, 10,
                                         0, 2, 0, 8, 0, 3, 0, 1, 0, 10, 0, 4, 0, 7, 0,  1, 0, 0, 0, 9,  5, 0, 6, 0, 10,
                                         0, 2, 0, 8, 0, 1, 0, 0, 0, 9,  0, 4, 0, 7, 0,  3, 0, 1, 0, 10, 5, 0, 6, 0, 10,
                                         1, 0, 0, 0, 9, 0, 4, 0, 7, 0,  3, 0, 1, 0, 10, 0, 2, 0, 8, 0,  5, 0, 6, 0, 10 };
  std::array<float, 5 * 5 * 1> valid = { 1.94, 1.60, 2.33, 3.69, 4.75, 1.94, 1.60, 2.33, 3.69, 4.75, 1.94, 1.60, 2.33,
                                         3.69, 4.75, 1.94, 1.60, 2.33, 3.69, 4.75, 1.94, 1.60, 2.33, 3.69, 4.75 };

  auto gpu_input = cle::Array::create(5, 5, 5, 3, cle::dType::FLOAT, cle::mType::BUFFER, device);
  gpu_input->writeFrom(input.data());
  auto gpu_output = cle::tier1::std_y_projection_func(device, gpu_input, nullptr);
  gpu_output->readTo(output.data());

  for (size_t i = 0; i < output.size(); i++)
  {
    EXPECT_NEAR(output[i], valid[i], 0.01);
  }
}

TEST_P(TestDirectionalProjections, std_z_projection)
{
  std::array<float, 5 * 5 * 1> output;
  std::array<float, 5 * 5 * 5> input = { 1, 0, 0, 0, 9, 0, 2, 0, 8, 0,  3, 0, 1, 0, 10, 0, 4, 0, 7, 0,  5, 0, 6, 0, 10,
                                         0, 2, 0, 8, 0, 1, 0, 0, 0, 9,  3, 0, 1, 0, 10, 0, 4, 0, 7, 0,  5, 0, 6, 0, 10,
                                         0, 2, 0, 8, 0, 3, 0, 1, 0, 10, 0, 4, 0, 7, 0,  1, 0, 0, 0, 9,  5, 0, 6, 0, 10,
                                         0, 2, 0, 8, 0, 1, 0, 0, 0, 9,  0, 4, 0, 7, 0,  3, 0, 1, 0, 10, 5, 0, 6, 0, 10,
                                         1, 0, 0, 0, 9, 0, 4, 0, 7, 0,  3, 0, 1, 0, 10, 0, 2, 0, 8, 0,  5, 0, 6, 0, 10 };
  std::array<float, 5 * 5 * 1> valid = { 0.49, 0.98, 0,    3.92, 4.41, 1.10, 1.60, 0.40, 3.69, 4.59, 1.47, 1.96, 0.49,
                                         3.43, 4.90, 1.17, 1.79, 0.40, 3.61, 4.66, 0,    0,    0,    0,    0 };

  auto gpu_input = cle::Array::create(5, 5, 5, 3, cle::dType::FLOAT, cle::mType::BUFFER, device);
  gpu_input->writeFrom(input.data());
  auto gpu_output = cle::tier1::std_z_projection_func(device, gpu_input, nullptr);
  gpu_output->readTo(output.data());

  for (size_t i = 0; i < output.size(); i++)
  {
    EXPECT_NEAR(output[i], valid[i], 0.01);
  }
}

// Test MEAN projections with static data
TEST_P(TestDirectionalProjections, mean_x_projection)
{
  std::array<float, 1 * 10 * 2>  output;
  std::array<float, 10 * 10 * 2> input = {
    1, 2, 3, 4, 5,  6, 7, 8, 9,  10, 2, 3, 4, 5,  6, 7, 8, 9,  10, 1, 3, 4, 5,  6, 7, 8, 9,  10, 1, 2, 4, 5,  6, 7, 8, 9, 10, 1, 2, 3, 5,
    6, 7, 8, 9, 10, 1, 2, 3, 4,  6,  7, 8, 9, 10, 1, 2, 3, 4,  5,  7, 8, 9, 10, 1, 2, 3, 4,  5,  6, 8, 9, 10, 1, 2, 3, 4, 5,  6, 7, 9, 10,
    1, 2, 3, 4, 5,  6, 7, 8, 10, 1,  2, 3, 4, 5,  6, 7, 8, 9,  1,  2, 3, 4, 5,  6, 7, 8, 9,  10, 2, 3, 4, 5,  6, 7, 8, 9, 10, 1, 3, 4, 5,
    6, 7, 8, 9, 10, 1, 2, 4, 5,  6,  7, 8, 9, 10, 1, 2, 3, 5,  6,  7, 8, 9, 10, 1, 2, 3, 4,  6,  7, 8, 9, 10, 1, 2, 3, 4, 5,  7, 8, 9, 10,
    1, 2, 3, 4, 5,  6, 8, 9, 10, 1,  2, 3, 4, 5,  6, 7, 9, 10, 1,  2, 3, 4, 5,  6, 7, 8, 10, 1,  2, 3, 4, 5,  6, 7, 8, 9,
  };
  std::array<float, 1 * 10 * 2> valid = {
    5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5
  };

  auto gpu_input = cle::Array::create(10, 10, 2, 3, cle::dType::FLOAT, cle::mType::BUFFER, device);
  gpu_input->writeFrom(input.data());
  auto gpu_output = cle::tier1::mean_x_projection_func(device, gpu_input, nullptr);
  gpu_output->readTo(output.data());

  for (size_t i = 0; i < output.size(); i++)
  {
    EXPECT_EQ(output[i], valid[i]);
  }
}

TEST_P(TestDirectionalProjections, mean_y_projection)
{
  std::array<float, 10 * 1 * 2>  output;
  std::array<float, 10 * 10 * 2> input = {
    1, 2, 3, 4, 5,  6, 7, 8, 9,  10, 2, 3, 4, 5,  6, 7, 8, 9,  10, 1, 3, 4, 5,  6, 7, 8, 9,  10, 1, 2, 4, 5,  6, 7, 8, 9, 10, 1, 2, 3, 5,
    6, 7, 8, 9, 10, 1, 2, 3, 4,  6,  7, 8, 9, 10, 1, 2, 3, 4,  5,  7, 8, 9, 10, 1, 2, 3, 4,  5,  6, 8, 9, 10, 1, 2, 3, 4, 5,  6, 7, 9, 10,
    1, 2, 3, 4, 5,  6, 7, 8, 10, 1,  2, 3, 4, 5,  6, 7, 8, 9,  1,  2, 3, 4, 5,  6, 7, 8, 9,  10, 2, 3, 4, 5,  6, 7, 8, 9, 10, 1, 3, 4, 5,
    6, 7, 8, 9, 10, 1, 2, 4, 5,  6,  7, 8, 9, 10, 1, 2, 3, 5,  6,  7, 8, 9, 10, 1, 2, 3, 4,  6,  7, 8, 9, 10, 1, 2, 3, 4, 5,  7, 8, 9, 10,
    1, 2, 3, 4, 5,  6, 8, 9, 10, 1,  2, 3, 4, 5,  6, 7, 9, 10, 1,  2, 3, 4, 5,  6, 7, 8, 10, 1,  2, 3, 4, 5,  6, 7, 8, 9,
  };
  std::array<float, 10 * 1 * 2> valid = {
    5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5
  };

  auto gpu_input = cle::Array::create(10, 10, 2, 3, cle::dType::FLOAT, cle::mType::BUFFER, device);
  gpu_input->writeFrom(input.data());
  auto gpu_output = cle::tier1::mean_y_projection_func(device, gpu_input, nullptr);
  gpu_output->readTo(output.data());

  for (size_t i = 0; i < output.size(); i++)
  {
    EXPECT_EQ(output[i], valid[i]);
  }
}

TEST_P(TestDirectionalProjections, mean_z_projection)
{
  std::array<float, 10 * 10 * 1> output;
  std::array<float, 10 * 10 * 2> input = {
    1, 2, 3, 4, 5,  6, 7, 8, 9,  10, 2, 3, 4, 5,  6, 7, 8, 9,  10, 1, 3, 4, 5,  6, 7, 8, 9,  10, 1, 2, 4, 5,  6, 7, 8, 9, 10, 1, 2, 3, 5,
    6, 7, 8, 9, 10, 1, 2, 3, 4,  6,  7, 8, 9, 10, 1, 2, 3, 4,  5,  7, 8, 9, 10, 1, 2, 3, 4,  5,  6, 8, 9, 10, 1, 2, 3, 4, 5,  6, 7, 9, 10,
    1, 2, 3, 4, 5,  6, 7, 8, 10, 1,  2, 3, 4, 5,  6, 7, 8, 9,  1,  2, 3, 4, 5,  6, 7, 8, 9,  10, 2, 3, 4, 5,  6, 7, 8, 9, 10, 1, 3, 4, 5,
    6, 7, 8, 9, 10, 1, 2, 4, 5,  6,  7, 8, 9, 10, 1, 2, 3, 5,  6,  7, 8, 9, 10, 1, 2, 3, 4,  6,  7, 8, 9, 10, 1, 2, 3, 4, 5,  7, 8, 9, 10,
    1, 2, 3, 4, 5,  6, 8, 9, 10, 1,  2, 3, 4, 5,  6, 7, 9, 10, 1,  2, 3, 4, 5,  6, 7, 8, 10, 1,  2, 3, 4, 5,  6, 7, 8, 9,
  };
  std::array<float, 10 * 10 * 1> valid = { 1, 2, 3,  4, 5,  6, 7,  8, 9, 10, 2, 3,  4, 5,  6, 7,  8, 9, 10, 1, 3,  4, 5,  6, 7,
                                           8, 9, 10, 1, 2,  4, 5,  6, 7, 8,  9, 10, 1, 2,  3, 5,  6, 7, 8,  9, 10, 1, 2,  3, 4,
                                           6, 7, 8,  9, 10, 1, 2,  3, 4, 5,  7, 8,  9, 10, 1, 2,  3, 4, 5,  6, 8,  9, 10, 1, 2,
                                           3, 4, 5,  6, 7,  9, 10, 1, 2, 3,  4, 5,  6, 7,  8, 10, 1, 2, 3,  4, 5,  6, 7,  8, 9 };

  auto gpu_input = cle::Array::create(10, 10, 2, 3, cle::dType::FLOAT, cle::mType::BUFFER, device);
  gpu_input->writeFrom(input.data());
  auto gpu_output = cle::tier1::mean_z_projection_func(device, gpu_input, nullptr);
  gpu_output->readTo(output.data());

  for (size_t i = 0; i < output.size(); i++)
  {
    EXPECT_EQ(output[i], valid[i]);
  }
}

// Test MINIMUM projections with dynamic data
TEST_P(TestDirectionalProjections, minimum_x_projection)
{
  std::array<float, 1 * 5 * 3>  output;
  std::array<float, 10 * 5 * 3> input;
  std::array<float, 1 * 5 * 3>  valid;

  std::fill(input.begin(), input.end(), static_cast<float>(10));
  std::fill(valid.begin(), valid.end(), static_cast<float>(1));
  for (auto it = input.begin(); it != input.end(); std::advance(it, 10))
  {
    int idx = (it - input.begin()) + (rand() % 10);
    input[idx] = static_cast<float>(1);
  }

  auto gpu_input = cle::Array::create(10, 5, 3, 3, cle::dType::FLOAT, cle::mType::BUFFER, device);
  gpu_input->writeFrom(input.data());
  auto gpu_output = cle::tier1::minimum_x_projection_func(device, gpu_input, nullptr);
  gpu_output->readTo(output.data());

  for (size_t i = 0; i < output.size(); i++)
  {
    EXPECT_EQ(output[i], valid[i]);
  }
}

TEST_P(TestDirectionalProjections, minimum_y_projection)
{
  std::array<float, 10 * 1 * 3> output;
  std::array<float, 10 * 5 * 3> input;
  std::array<float, 10 * 1 * 3> valid;

  std::fill(input.begin(), input.end(), static_cast<float>(10));
  std::fill(valid.begin(), valid.end(), static_cast<float>(1));
  for (auto it = input.begin(); it != input.end(); std::advance(it, 10 * 5))
  {
    for (size_t j = 0; j < 10; ++j)
    {
      int idx = (it - input.begin() + j) + (rand() % 5) * 10;
      input[idx] = static_cast<float>(1);
    }
  }

  auto gpu_input = cle::Array::create(10, 5, 3, 3, cle::dType::FLOAT, cle::mType::BUFFER, device);
  gpu_input->writeFrom(input.data());
  auto gpu_output = cle::tier1::minimum_y_projection_func(device, gpu_input, nullptr);
  gpu_output->readTo(output.data());

  for (size_t i = 0; i < output.size(); i++)
  {
    EXPECT_EQ(output[i], valid[i]);
  }
}

TEST_P(TestDirectionalProjections, minimum_z_projection)
{
  std::array<float, 10 * 5 * 1> output;
  std::array<float, 10 * 5 * 3> input;
  std::array<float, 10 * 5 * 1> valid;

  std::fill(input.begin(), input.end(), static_cast<float>(10));
  std::fill(valid.begin(), valid.end(), static_cast<float>(1));
  for (int i = 0; i < int(5 * 10); i++)
  {
    int idx = i + (rand() % 3) * 10 * 5;
    input[idx] = static_cast<float>(1);
  }

  auto gpu_input = cle::Array::create(10, 5, 3, 3, cle::dType::FLOAT, cle::mType::BUFFER, device);
  gpu_input->writeFrom(input.data());
  auto gpu_output = cle::tier1::minimum_z_projection_func(device, gpu_input, nullptr);
  gpu_output->readTo(output.data());

  for (size_t i = 0; i < output.size(); i++)
  {
    EXPECT_EQ(output[i], valid[i]);
  }
}

// Test MAXIMUM projections with dynamic data
TEST_P(TestDirectionalProjections, maximum_x_projection)
{
  std::array<float, 1 * 5 * 3>  output;
  std::array<float, 10 * 5 * 3> input;
  std::array<float, 1 * 5 * 3>  valid;

  std::fill(input.begin(), input.end(), static_cast<float>(0));
  std::fill(valid.begin(), valid.end(), static_cast<float>(10));
  for (auto it = input.begin(); it != input.end(); std::advance(it, 10))
  {
    int idx = (it - input.begin()) + (rand() % 10);
    input[idx] = static_cast<float>(10);
  }

  auto gpu_input = cle::Array::create(10, 5, 3, 3, cle::dType::FLOAT, cle::mType::BUFFER, device);
  gpu_input->writeFrom(input.data());
  auto gpu_output = cle::tier1::maximum_x_projection_func(device, gpu_input, nullptr);
  gpu_output->readTo(output.data());

  for (size_t i = 0; i < output.size(); i++)
  {
    EXPECT_EQ(output[i], valid[i]);
  }
}

TEST_P(TestDirectionalProjections, maximum_y_projection)
{
  std::array<float, 10 * 1 * 3> output;
  std::array<float, 10 * 5 * 3> input;
  std::array<float, 10 * 1 * 3> valid;

  std::fill(input.begin(), input.end(), static_cast<float>(0));
  std::fill(valid.begin(), valid.end(), static_cast<float>(10));
  for (auto it = input.begin(); it != input.end(); std::advance(it, 10 * 5))
  {
    for (size_t j = 0; j < 10; j++)
    {
      int idx = (it - input.begin() + j) + (rand() % 5) * 10;
      input[idx] = static_cast<float>(10);
    }
  }

  auto gpu_input = cle::Array::create(10, 5, 3, 3, cle::dType::FLOAT, cle::mType::BUFFER, device);
  gpu_input->writeFrom(input.data());
  auto gpu_output = cle::tier1::maximum_y_projection_func(device, gpu_input, nullptr);
  gpu_output->readTo(output.data());

  for (size_t i = 0; i < output.size(); i++)
  {
    EXPECT_EQ(output[i], valid[i]);
  }
}

TEST_P(TestDirectionalProjections, maximum_z_projection)
{
  std::array<float, 10 * 5 * 1> output;
  std::array<float, 10 * 5 * 3> input;
  std::array<float, 10 * 5 * 1> valid;

  std::fill(input.begin(), input.end(), static_cast<float>(0));
  std::fill(valid.begin(), valid.end(), static_cast<float>(10));
  for (int i = 0; i < int(5 * 10); i++)
  {
    int idx = i + (rand() % 3) * 10 * 5;
    input[idx] = static_cast<float>(10);
  }

  auto gpu_input = cle::Array::create(10, 5, 3, 3, cle::dType::FLOAT, cle::mType::BUFFER, device);
  gpu_input->writeFrom(input.data());
  auto gpu_output = cle::tier1::maximum_z_projection_func(device, gpu_input, nullptr);
  gpu_output->readTo(output.data());

  for (size_t i = 0; i < output.size(); i++)
  {
    EXPECT_EQ(output[i], valid[i]);
  }
}

// Test SUM projections with dynamic data
TEST_P(TestDirectionalProjections, sum_x_projection)
{
  std::array<float, 1 * 5 * 3>  output;
  std::array<float, 10 * 5 * 3> input;
  std::array<float, 1 * 5 * 3>  valid;

  std::fill(input.begin(), input.end(), static_cast<float>(1));
  std::fill(valid.begin(), valid.end(), static_cast<float>(10));

  auto gpu_input = cle::Array::create(10, 5, 3, 3, cle::dType::FLOAT, cle::mType::BUFFER, device);
  gpu_input->writeFrom(input.data());
  auto gpu_output = cle::tier1::sum_x_projection_func(device, gpu_input, nullptr);
  gpu_output->readTo(output.data());

  for (size_t i = 0; i < output.size(); i++)
  {
    EXPECT_EQ(output[i], valid[i]);
  }
}

TEST_P(TestDirectionalProjections, sum_y_projection)
{
  std::array<float, 10 * 1 * 3> output;
  std::array<float, 10 * 5 * 3> input;
  std::array<float, 10 * 1 * 3> valid;

  std::fill(input.begin(), input.end(), static_cast<float>(1));
  std::fill(valid.begin(), valid.end(), static_cast<float>(5));

  auto gpu_input = cle::Array::create(10, 5, 3, 3, cle::dType::FLOAT, cle::mType::BUFFER, device);
  gpu_input->writeFrom(input.data());
  auto gpu_output = cle::tier1::sum_y_projection_func(device, gpu_input, nullptr);
  gpu_output->readTo(output.data());

  for (size_t i = 0; i < output.size(); i++)
  {
    EXPECT_EQ(output[i], valid[i]);
  }
}

TEST_P(TestDirectionalProjections, sum_z_projection)
{
  std::array<float, 10 * 5 * 1> output;
  std::array<float, 10 * 5 * 3> input;
  std::array<float, 10 * 5 * 1> valid;

  std::fill(input.begin(), input.end(), static_cast<float>(1));
  std::fill(valid.begin(), valid.end(), static_cast<float>(3));

  auto gpu_input = cle::Array::create(10, 5, 3, 3, cle::dType::FLOAT, cle::mType::BUFFER, device);
  gpu_input->writeFrom(input.data());
  auto gpu_output = cle::tier1::sum_z_projection_func(device, gpu_input, nullptr);
  gpu_output->readTo(output.data());

  for (size_t i = 0; i < output.size(); i++)
  {
    EXPECT_EQ(output[i], valid[i]);
  }
}

TEST_P(TestDirectionalProjections, keep_dims_y_projection)
{
  auto gpu_input = cle::Array::create(4, 3, 2, 3, cle::dType::FLOAT, cle::mType::BUFFER, device);
  gpu_input->fill(1);

  auto gpu_output = cle::tier1::sum_y_projection_func(device, gpu_input, nullptr, true);
  EXPECT_EQ(gpu_output->width(), 4);
  EXPECT_EQ(gpu_output->height(), 1);
  EXPECT_EQ(gpu_output->depth(), 2);
  EXPECT_EQ(gpu_output->dimension(), 3);
}

TEST_P(TestDirectionalProjections, keep_dims_z_projection)
{
  auto gpu_input = cle::Array::create(4, 3, 2, 3, cle::dType::FLOAT, cle::mType::BUFFER, device);
  gpu_input->fill(1);

  auto gpu_output = cle::tier1::sum_z_projection_func(device, gpu_input, nullptr, true);
  EXPECT_EQ(gpu_output->width(), 4);
  EXPECT_EQ(gpu_output->height(), 3);
  EXPECT_EQ(gpu_output->depth(), 1);
  EXPECT_EQ(gpu_output->dimension(), 3);
}

TEST_P(TestDirectionalProjections, keep_dims_position_y_projection)
{
  auto gpu_input = cle::Array::create(4, 3, 2, 3, cle::dType::FLOAT, cle::mType::BUFFER, device);
  gpu_input->fill(1);

  auto gpu_output = cle::tier1::y_position_of_maximum_y_projection_func(device, gpu_input, nullptr, true);
  EXPECT_EQ(gpu_output->width(), 4);
  EXPECT_EQ(gpu_output->height(), 1);
  EXPECT_EQ(gpu_output->depth(), 2);
  EXPECT_EQ(gpu_output->dimension(), 3);
}

INSTANTIATE_TEST_SUITE_P(InstantiationName, TestDirectionalProjections, ::testing::ValuesIn(getParameters()));
