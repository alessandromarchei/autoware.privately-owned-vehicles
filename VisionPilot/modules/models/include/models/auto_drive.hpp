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
// Two-frame unified model.
//
// Preprocessing contract (caller, before infer()):
//   • Resize both frames to NET_W × NET_H  (1024 × 512)
//   • Convert BGR → RGB
//   • Apply ImageNet normalisation:
//       mean = [0.485, 0.456, 0.406]  std = [0.229, 0.224, 0.225]
//   • Layout: CHW float32, CHW_SIZE elements each frame
class AutoDrive {
public:
    static constexpr int NET_H    = 512;
    static constexpr int NET_W    = 1024;
    static constexpr int CHW_SIZE = 3 * NET_H * NET_W;

    // model_path — path to the AutoDrive .msgpack file
    AutoDrive(const std::string& model_path);

    // prev_chw, curr_chw : float32 CHW buffers, CHW_SIZE elements each
    visionpilot::common::AutoDriveOutput infer(const float* prev_chw, const float* curr_chw);

private:
    //std::unique_ptr<Ort::Session> session_;
    // Ort::MemoryInfo               mem_info_;
    std::unique_ptr<engine::V4MEngine> engine_;
};

}  // namespace visionpilot::models
