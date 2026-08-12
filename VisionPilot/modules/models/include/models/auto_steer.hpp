#pragma once

#include <engine/onnx_engine.hpp>
#include <onnxruntime_cxx_api.h>

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace visionpilot::models {


// ─── Model ────────────────────────────────────────────────────────────────────
// Single-frame path prediction model.
//
// Preprocessing contract (caller, before infer()):
//   • Resize frame to NET_W × NET_H  (1024 × 512)
//   • Convert BGR → RGB
//   • Scale to [0, 1]  — NO ImageNet normalisation (commented out in Python)
//   • Layout: CHW float32, CHW_SIZE elements
class AutoSteer {
public:
    static constexpr int NET_H    = 512;
    static constexpr int NET_W    = 1024;
    static constexpr int CHW_SIZE = 3 * NET_H * NET_W;

    AutoSteer(engine::OnnxEngine& engine, const std::string& model_path);

    // image_chw : float32 CHW buffer, CHW_SIZE elements, RGB [0, 1]
    AutoSteerOutput infer(const float* image_chw);

private:
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo               mem_info_;

    std::vector<std::string> in_name_strs_;
    std::vector<const char*> in_names_;
    std::vector<std::string> out_name_strs_;
    std::vector<const char*> out_names_;

    std::vector<int64_t> input_shape_;  // {1, 3, NET_H, NET_W}
    std::string arena_shrink_;
};

}  // namespace visionpilot::models
