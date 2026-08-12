#ifndef VISIONPILOT_MODELS_HPP
#define VISIONPILOT_MODELS_HPP

#include <array>
#include <vector>
#include <string>

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

}

#endif //VISIONPILOT_MODELS_HPP

