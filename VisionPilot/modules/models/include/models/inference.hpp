#pragma once

#include <common/models.hpp>
#include <fusion/lateral_fusion.hpp>
#include <fusion/longitudinal_fusion.hpp>
#include <models/visionpilot.hpp>
#include <opencv2/core.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace visionpilot::models {

struct Config {
    std::string precision = "fp32";
    std::string core = "core0";
    std::string auto_drive_model_path;
    std::string auto_steer_model_path;
    std::string auto_speed_model_path;
    bool fusion_debug = false;
    float cte_bias_m = 0.0f;
};

struct LatencyStats {
    double pre{0.0};
    double ad{0.0};
    double as{0.0};
    double asp{0.0};
    double inference{0.0};
    double wall{0.0};

    void update(
        double pre_ms,
        double autodrive_ms,
        double autosteer_ms,
        double autospeed_ms,
        double inference_ms,
        double wall_ms);
    void print() const;
    void reset();
};

class InferencePipeline {
public:
    explicit InferencePipeline(const Config& cfg);

    // AutoDrive always uses warped(t-1) and warped(t).
    // AutoSteer/AutoSpeed share resized(t), or warped(t) when resized is empty.
    std::optional<visionpilot::common::InferenceFrameResult> process(
        const cv::Mat& warped,
        const cv::Mat& resized = {});

    void set_H_resized(const cv::Mat& H, cv::Size raw_size);

    [[nodiscard]] const cv::Mat& H_resized() const
    {
        return H_resized_;
    }

    [[nodiscard]] const cv::Mat& H_world2resized() const
    {
        return H_world2resized_;
    }

    void reset();

    [[nodiscard]] const LatencyStats& latency() const
    {
        return stats_;
    }

private:
    cv::Mat H_resized_;
    cv::Mat H_world2resized_;
    models::VisionPilot visionpilot_;
    fusion::LongitudinalFusion long_fusion_;
    fusion::LateralFusion lat_fusion_;
    LatencyStats stats_;
    std::uint64_t frame_count_{0};
    std::vector<float> previous_warped_imagenet_;
};

}  // namespace visionpilot::models
