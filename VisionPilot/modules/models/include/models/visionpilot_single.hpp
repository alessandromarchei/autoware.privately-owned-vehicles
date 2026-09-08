#pragma once

#include <engine/v4m_engine.hpp>
#include <common/models.hpp>
#include <logging/logger.hpp>
// #include <onnxruntime_cxx_api.h>

#include <memory>
#include <string>
#include <vector>
#include <cstring>
namespace visionpilot::models {

// ─── Model ────────────────────────────────────────────────────────────────────
class VisionPilot {
public:
    static constexpr int NET_H    = 512;
    static constexpr int NET_W    = 1024;
    static constexpr int CHW_SIZE = 3 * NET_H * NET_W;
    static constexpr std::size_t AUTODRIVE_FEATURE_SIZE = 256 * 16 * 32;

    // model_path — path to the VisionPilot .msgpack file
    VisionPilot(const std::string& model_path);

    visionpilot::common::VisionPilotOutput infer(const float * prev_features, const float * curr_autodrive_input, const float * curr_autosteer_input);

    visionpilot::common::AutoDriveOutput postprocess_autodrive(float * dist_normalized, float * curvature_raw, float * flag_logit);

    visionpilot::common::AutoSteerOutput postprocess_autosteer(float * lane_value, float * height);

    visionpilot::common::AutoSpeedOutput postprocess_autospeed(const float * data, float conf_thres = 0.6f, float iou_thres = 0.45f);

    float* get_curr_features_autodrive() const { return curr_features_autodrive_; }

  private:
    //std::unique_ptr<Ort::Session> session_;
    // Ort::MemoryInfo               mem_info_;
    std::unique_ptr<engine::V4MEngine> engine_;

    float iou(const visionpilot::common::Detection & a, const visionpilot::common::Detection & b);

    std::vector<visionpilot::common::Detection> nms(std::vector<visionpilot::common::Detection> dets, float iou_thres);
    float* curr_features_autodrive_ = nullptr;  //autodrive features from previous frame
};  

}  // namespace visionpilot::models
