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
// YOLO-style object detection model.
//
// Preprocessing contract (caller, before infer()):
//   • Letterbox-resize frame to NET_W × NET_H (1024 × 512),
//     preserving aspect ratio, padding with (114, 114, 114)
//   • Convert BGR → RGB
//   • Scale to [0, 1] — NO ImageNet normalisation
//   • Layout: CHW float32, CHW_SIZE elements
//
// Raw model output shape: [1, C, N]
//   C = 4 + num_classes   (cx, cy, w, h, logit_0 … logit_{K-1})
//   Post-processing (sigmoid + threshold + NMS) is done inside infer().
class AutoSpeed {
public:
    static constexpr int NET_H    = 512;
    static constexpr int NET_W    = 1024;
    static constexpr int CHW_SIZE = 3 * NET_H * NET_W;

    AutoSpeed(const std::string& model_path);

    // image_chw  : float32 CHW buffer, CHW_SIZE elements, RGB [0, 1]
    // conf_thres : sigmoid class probability threshold
    // iou_thres  : IoU threshold for NMS
    visionpilot::common::AutoSpeedOutput infer(const float* image_chw,
                          float conf_thres = 0.6f,
                          float iou_thres  = 0.45f);

private:
    visionpilot::common::AutoSpeedOutput post_process(const float* data, float conf_thres, float iou_thres) const;


    static float iou(const visionpilot::common::Detection& a, const visionpilot::common::Detection& b);
    static std::vector<visionpilot::common::Detection> nms(std::vector<visionpilot::common::Detection> dets, float iou_thres);

    std::unique_ptr<engine::V4MEngine> engine_;
    // std::unique_ptr<Ort::Session> session_;
    // Ort::MemoryInfo               mem_info_;

    // std::vector<std::string> in_name_strs_;
    // std::vector<const char*> in_names_;
    // std::vector<std::string> out_name_strs_;
    // std::vector<const char*> out_names_;

    // std::vector<int64_t> input_shape_;  // {1, 3, NET_H, NET_W}
    // std::string arena_shrink_;
};

}  // namespace visionpilot::models
