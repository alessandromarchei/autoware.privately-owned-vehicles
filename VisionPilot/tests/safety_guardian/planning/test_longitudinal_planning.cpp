#include <gtest/gtest.h>

#include <planning/longitudinal_planning.hpp>

#include <cmath>
#include <memory>

class LongitudinalPlannerTest : public ::testing::Test
{
protected:

    LongitudinalPlanner::Config cfg;
    std::unique_ptr<LongitudinalPlanner> planner;

    void SetUp() override
    {
        cfg.speed_limit = 30.0;   // m/s

        planner = std::make_unique<LongitudinalPlanner>(cfg);
    }
};

// ------------------------------------------------------------
// Constructor
// ------------------------------------------------------------

TEST_F(LongitudinalPlannerTest, Constructor)
{
    ASSERT_NE(planner, nullptr);
}

// ------------------------------------------------------------
// Free road
// ------------------------------------------------------------

TEST_F(LongitudinalPlannerTest, AcceleratesOnFreeRoad)
{
    double accel =
        planner->compute_acceleration(
            0.0,
            5.0,
            false,
            cfg.speed_limit,
            9999.0);

    EXPECT_GT(accel, 0.0);
}

// ------------------------------------------------------------
// Desired speed
// ------------------------------------------------------------

TEST_F(LongitudinalPlannerTest, ZeroAccelerationNearSpeedLimit)
{
    double accel =
        planner->compute_acceleration(
            0.0,
            cfg.speed_limit,
            false,
            cfg.speed_limit,
            9999.0);

    EXPECT_NEAR(accel, 0.0, 0.2);
}

// ------------------------------------------------------------
// Above speed limit
// ------------------------------------------------------------

TEST_F(LongitudinalPlannerTest, BrakesAboveSpeedLimit)
{
    double accel =
        planner->compute_acceleration(
            0.0,
            35.0,
            false,
            cfg.speed_limit,
            9999.0);

    EXPECT_LT(accel, 0.0);
}

// ------------------------------------------------------------
// Curvature should reduce speed
// ------------------------------------------------------------

TEST_F(LongitudinalPlannerTest, TightCurveProducesBraking)
{
    double accel =
        planner->compute_acceleration(
            0.20,
            20.0,
            false,
            cfg.speed_limit,
            9999.0);

    EXPECT_LT(accel, 0.0);
}

// ------------------------------------------------------------
// Lead vehicle close
// ------------------------------------------------------------

TEST_F(LongitudinalPlannerTest, CloseVehicleProducesBraking)
{
    double accel =
        planner->compute_acceleration(
            0.0,
            20.0,
            true,
            10.0,
            5.0);

    EXPECT_LT(accel, 0.0);
}

// ------------------------------------------------------------
// Lead vehicle far away
// ------------------------------------------------------------

TEST_F(LongitudinalPlannerTest, FarVehicleHasSmallInfluence)
{
    double accel =
        planner->compute_acceleration(
            0.0,
            20.0,
            true,
            20.0,
            1000.0);

    EXPECT_GT(accel, -0.5);
}

// ------------------------------------------------------------
// Standstill
// ------------------------------------------------------------

TEST_F(LongitudinalPlannerTest, AcceleratesFromStandstill)
{
    double accel =
        planner->compute_acceleration(
            0.0,
            0.0,
            false,
            cfg.speed_limit,
            9999.0);

    EXPECT_GT(accel, 0.0);
}

// ------------------------------------------------------------
// Gap equal to zero
// ------------------------------------------------------------

TEST_F(LongitudinalPlannerTest, ZeroGapDoesNotCrash)
{
    EXPECT_NO_THROW(
    {
        planner->compute_acceleration(
            0.0,
            20.0,
            true,
            10.0,
            0.0);
    });
}

// ------------------------------------------------------------
// Negative gap
// ------------------------------------------------------------

TEST_F(LongitudinalPlannerTest, NegativeGapDoesNotCrash)
{
    EXPECT_NO_THROW(
    {
        planner->compute_acceleration(
            0.0,
            20.0,
            true,
            10.0,
            -10.0);
    });
}

// ------------------------------------------------------------
// Output should always be finite
// ------------------------------------------------------------

TEST_F(LongitudinalPlannerTest, OutputIsFinite)
{
    double accel =
        planner->compute_acceleration(
            0.0,
            15.0,
            false,
            cfg.speed_limit,
            9999.0);

    EXPECT_TRUE(std::isfinite(accel));
}

