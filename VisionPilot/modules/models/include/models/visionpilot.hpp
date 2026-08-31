#pragma once

#include <common/models.hpp>
#include <engine/v4m_engine.hpp>

#include <memory>
#include <string>
#include <vector>

namespace visionpilot::models {

struct VisionPilotTimings {
    // Complete load + copy + run + postprocess + unload time per model.
    double autodrive_ms{0.0};
    double autosteer_ms{0.0};
    double autospeed_ms{0.0};
    double total_ms{0.0};

    // H2D + hardware execution + D2H only.
    double autodrive_run_ms{0.0};
    double autosteer_run_ms{0.0};
    double autospeed_run_ms{0.0};
};

class VisionPilot {
public:
    static constexpr int NET_H = 512;
    static constexpr int NET_W = 1024;
    static constexpr int CHW_SIZE = 3 * NET_H * NET_W;

    VisionPilot(
        const std::string& autodrive_model_path,
        const std::string& autosteer_model_path,
        const std::string& autospeed_model_path);

    visionpilot::common::VisionPilotOutput infer(
        const float* previous_warped_chw,
        const float* current_warped_chw,
        const float* current_steer_speed_chw);

    [[nodiscard]] const VisionPilotTimings& last_timings() const noexcept
    {
        return last_timings_;
    }

private:
    void open_model(
        const std::string& path,
        std::size_t expected_inputs,
        std::size_t expected_outputs,
        const char* model_name);

    void validate_frame_input(
        std::size_t input_index,
        const char* input_name) const;

    visionpilot::common::AutoDriveOutput run_autodrive(
        const float* previous_warped_chw,
        const float* current_warped_chw);

    visionpilot::common::AutoSteerOutput run_autosteer(
        const float* current_chw);

    visionpilot::common::AutoSpeedOutput run_autospeed(
        const float* current_chw);

    visionpilot::common::AutoDriveOutput postprocess_autodrive(
        const float* distance,
        const float* curvature,
        const float* flag) const;

    visionpilot::common::AutoSteerOutput postprocess_autosteer(
        const float* lane,
        const float* height) const;

    visionpilot::common::AutoSpeedOutput postprocess_autospeed(
        const float* data,
        float confidence_threshold = 0.6f,
        float iou_threshold = 0.45f) const;

    static float iou(
        const visionpilot::common::Detection& a,
        const visionpilot::common::Detection& b);
    static std::vector<visionpilot::common::Detection> nms(
        std::vector<visionpilot::common::Detection> detections,
        float threshold);

    std::unique_ptr<engine::V4MEngine> engine_;
    std::string autodrive_model_path_;
    std::string autosteer_model_path_;
    std::string autospeed_model_path_;
    VisionPilotTimings last_timings_{};
};

}  // namespace visionpilot::models
