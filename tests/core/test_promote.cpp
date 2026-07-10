#include "cle.hpp"

#include <gtest/gtest.h>

TEST(TestPromoteType, sameType)
{
  EXPECT_EQ(cle::promoteType(cle::dType::FLOAT, cle::dType::FLOAT), cle::dType::FLOAT);
  EXPECT_EQ(cle::promoteType(cle::dType::INT16, cle::dType::INT16), cle::dType::INT16);
  EXPECT_EQ(cle::promoteType(cle::dType::UINT8, cle::dType::UINT8), cle::dType::UINT8);
}

TEST(TestPromoteType, floatAndComplexWin)
{
  EXPECT_EQ(cle::promoteType(cle::dType::FLOAT, cle::dType::INT32), cle::dType::FLOAT);
  EXPECT_EQ(cle::promoteType(cle::dType::UINT8, cle::dType::FLOAT), cle::dType::FLOAT);
  EXPECT_EQ(cle::promoteType(cle::dType::COMPLEX, cle::dType::FLOAT), cle::dType::COMPLEX);
  EXPECT_EQ(cle::promoteType(cle::dType::INT32, cle::dType::COMPLEX), cle::dType::COMPLEX);
}

TEST(TestPromoteType, sameSignedness)
{
  EXPECT_EQ(cle::promoteType(cle::dType::INT8, cle::dType::INT32), cle::dType::INT32);
  EXPECT_EQ(cle::promoteType(cle::dType::UINT16, cle::dType::UINT8), cle::dType::UINT16);
  EXPECT_EQ(cle::promoteType(cle::dType::INT16, cle::dType::INT8), cle::dType::INT16);
}

TEST(TestPromoteType, mixedSignedness)
{
  EXPECT_EQ(cle::promoteType(cle::dType::INT8, cle::dType::UINT8), cle::dType::INT16);
  EXPECT_EQ(cle::promoteType(cle::dType::INT8, cle::dType::UINT16), cle::dType::INT32);
  EXPECT_EQ(cle::promoteType(cle::dType::INT16, cle::dType::UINT8), cle::dType::INT16);
  EXPECT_EQ(cle::promoteType(cle::dType::INT16, cle::dType::UINT16), cle::dType::INT32);
  EXPECT_EQ(cle::promoteType(cle::dType::INT32, cle::dType::UINT8), cle::dType::INT32);
  EXPECT_EQ(cle::promoteType(cle::dType::INT32, cle::dType::UINT16), cle::dType::INT32);
}

TEST(TestPromoteType, clampedToFloat)
{
  EXPECT_EQ(cle::promoteType(cle::dType::INT32, cle::dType::UINT32), cle::dType::FLOAT);
  EXPECT_EQ(cle::promoteType(cle::dType::INT8, cle::dType::UINT32), cle::dType::FLOAT);
  EXPECT_EQ(cle::promoteType(cle::dType::INT16, cle::dType::UINT32), cle::dType::FLOAT);
}

TEST(TestPromoteType, unknownThrows)
{
  EXPECT_ANY_THROW(cle::promoteType(cle::dType::UNKNOWN, cle::dType::FLOAT));
  EXPECT_ANY_THROW(cle::promoteType(cle::dType::INT8, cle::dType::UNKNOWN));
}
