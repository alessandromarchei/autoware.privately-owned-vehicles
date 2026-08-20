#pragma once

#include <fusion/lateral_fusion.hpp>
#include <fusion/longitudinal_fusion.hpp>
#include <models/visionpilot_single.hpp>
#include <opencv2/core.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <filesystem>
#include <vector>

namespace visionpilot::models {

struct Config {
    std::string precision    = "fp32";
    std::string core         = "core0";

    std::string visionpilot_model_path;
    
    bool        fusion_debug = false;
    float       cte_bias_m   = 0.0f;  // camera mounting offset [m] — subtracted from raw CTE before filter
};

struct LatencyStats {
    double pre{0}, visionpilot{0}, wall{0};

    void update(double pre_, double visionpilot_);
    void print() const;
    void reset();
};

// Two-frame buffer → parallel ONNX → longitudinal + lateral fusion.
class InferencePipeline {
public:
    InferencePipeline(const Config& cfg);

    // nullopt until two frames collected (AutoDrive needs t-1 and t).
    // warped  : BEV 1024×512 image → AutoDrive only.
    // resized : plain-resized 1024×512 image → AutoSteer + AutoSpeed.
    //           If empty, falls back to warped for all networks (legacy behaviour).
    std::optional<visionpilot::common::InferenceFrameResult> process(const cv::Mat& warped,
                                                                     const cv::Mat& resized = {});

    // Compute and apply H_resized to both fusion modules so that AutoSteer /
    // AutoSpeed outputs are projected correctly when they run on a resized
    // (non-BEV) image.  Call once after the first preprocess() when using
    // the resized routing.
    //   C        : raw-camera → warped-BEV homography (from ImagePreprocessor)
    //   raw_size : original frame dimensions before top-crop + resize
    void set_H_resized(const cv::Mat& H, cv::Size raw_size);

    // H that maps resized-image pixel → world (set after set_H_resized()).
    // Empty until set_H_resized() is called.
    const cv::Mat& H_resized() const { return H_resized_; }

    // Inverse of H_resized: world → resized-image pixel.
    // Used by both debug and production visualizers for path projection.
    const cv::Mat& H_world2resized() const { return H_world2resized_; }

    void reset();
    const LatencyStats& latency() const { return stats_; }

private:
    cv::Mat H_resized_;
    cv::Mat H_world2resized_;
    // models::AutoDrive          auto_drive_;
    // models::AutoSteer          auto_steer_;
    // models::AutoSpeed          auto_speed_;
    models::VisionPilot          visionpilot_;
    fusion::LongitudinalFusion long_fusion_;
    fusion::LateralFusion      lat_fusion_;
    LatencyStats       stats_;
    uint64_t           frame_count_ = 0;

    std::vector<float> prev_warped_imn_;
    int     frame_buf_count_ = 0;
};

}  // namespace visionpilot::models
