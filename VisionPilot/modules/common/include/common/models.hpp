#ifndef VISIONPILOT_MODELS_HPP
#define VISIONPILOT_MODELS_HPP

#include <array>
#include <vector>
#include <string>
#include <cstdint>
#include <common/types.hpp>

namespace visionpilot::common {

/*
AUTODRIVE COMMON STRUCT
*/

// ─── Output ───────────────────────────────────────────────────────────────────
// Raw scalars straight from the model. Domain conversion is the caller's job:
//   distance_m    = D_MAX_M * (1.0f - dist_normalized)   // D_MAX_M = 150.0
//   curvature_1pm = curvature_raw * CURV_SCALE
struct AutoDriveOutput {
    float dist_normalized = 0.f;  // normalised distance  [0, 1]
    float curvature_raw   = 0.f;  // raw curvature output
    float flag_prob       = 0.f;  // sigmoid(flag_logit), CIPO probability [0, 1]
    bool  valid           = false;
};


/*
AUTOSPEED COMMON STRUCT
*/

// ─── Output ───────────────────────────────────────────────────────────────────
// Bounding boxes in model-input pixel space (1024 × 512) after NMS.
// Coordinate mapping back to original image coordinates is the caller's job
// (reverse the letterbox: subtract pad, divide by scale).
struct Detection {
    float x1 = 0.f, y1 = 0.f;  // top-left
    float x2 = 0.f, y2 = 0.f;  // bottom-right
    float score    = 0.f;
    int   class_id = 0;
};

struct AutoSpeedOutput {
    std::vector<Detection> detections;
    bool valid = false;
};


/*
AUTOSTEER COMMON STRUCT
*/

// ─── Output ───────────────────────────────────────────────────────────────────
// Both tensors are (1, 64) in the ONNX model, flattened here:
//   xp[i] = lateral x at fixed image row i (normalized [0,1], ×1024 for px)
//   NOT (u,v) pairs — v comes from linspace(0, H-1, 64) in the visualizer.
struct AutoSteerOutput {
    std::array<float, 64> xp{};        // (1, 64) ego-path waypoints
    std::array<float, 64> h_vector{};  // (1, 64) waypoint confidence/mask
    bool                   valid = false;
};


struct LateralFusionEstimate {
    bool valid = false;

    // ── Particle-filter tracked outputs (ready for planning) ──────────────────
    float cte_m          = 0.f;   // cross-track error [m]; +ve = ego right of path
    float cte_rate_mps   = 0.f;   // d(cte)/dt [m/s]
    float yaw_rad        = 0.f;   // yaw error [rad];  +ve = path heading left
    float yaw_rate_rps   = 0.f;   // d(yaw)/dt [rad/s]
    float cte_stddev_m   = 0.f;
    float yaw_stddev_rad = 0.f;

    float curvature      = 0.f;   // fused curvature [1/m]; +ve = left turn
    float curv_stddev    = 0.f;

    // ── Raw intermediates for debug / downstream use ───────────────────────────
    bool  path_valid         = false;  // RANSAC polynomial fit succeeded
    float raw_cte_m          = 0.f;   // CTE direct from polynomial (= c-coeff)
    float raw_yaw_rad        = 0.f;   // yaw direct from polynomial (= atan(b))
    float raw_path_curvature = 0.f;   // κ sampled along fitted path (median)
    float raw_ad_curvature   = 0.f;   // curvature_raw from AutoDrive (scaled)
    int   path_inliers       = 0;     // RANSAC inlier count
    int   path_points        = 0;     // world points projected from waypoints
    // Fitted polynomial y = path_a·x² + path_b·x + path_c  (world frame)
    float path_a = 0.f, path_b = 0.f, path_c = 0.f;
    // Forward extent of RANSAC inliers [m] — cap path visualization / MPC samples
    float path_x_min_m = 0.f;
    float path_x_max_m = 0.f;
};


struct CIPOFusionEstimate {
    bool  valid             = false;

    // Particle-filter fused posterior
    float distance_m        = 0.f;
    float velocity_ms       = 0.f;   // negative = approaching; from particle ensemble
    float distance_stddev_m = 0.f;

    // Raw CIPO distance from AutoSpeed bboxes via homography (no tracking state)
    bool  cipo_raw_found    = false;
    float cipo_raw_dist_m   = 0.f;
    bool  cut_in_detected   = false; // Level 2 is closer than Level 1
};


struct InferenceFrameResult {
    uint64_t    frame_id = 0;
    double      wall_ms  = 0;
    double      pre_ms   = 0;
    // double      ad_ms    = 0;
    // double      as_ms    = 0;
    // double      asp_ms   = 0;
    double      visionpilot_ms = 0;

    //common::VisionPilotOutput              visionpilot;
    AutoDriveOutput              auto_drive;
    AutoSteerOutput              auto_steer;
    AutoSpeedOutput              auto_speed;
    CIPOFusionEstimate   cipo;
    LateralFusionEstimate  lateral;
};

struct Plan {
    double                acceleration;
    std::vector<double>   steering;
    std::vector<Warning>  warnings;
};

struct VisionPilotOutput {
    InferenceFrameResult    inference;
    Plan                    plan;
};


}

#endif //VISIONPILOT_MODELS_HPP

