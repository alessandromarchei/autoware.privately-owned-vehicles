#include "models/auto_speed.hpp"

// #include <onnxruntime_run_options_config_keys.h>

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace visionpilot::models {

// ─── Constructor ─────────────────────────────────────────────────────────────

AutoSpeed::AutoSpeed(const std::string& model_path)
{

    //instantiate underlying v4m engine for this model
    engine_ = std::make_unique<engine::V4MEngine>(model_path);

    if (engine_->num_inputs() != 1) {
        throw std::runtime_error(
            "AutoSpeed expects exactly 1 model inputs"
        );
    }

    if (engine_->num_outputs() != 1) {
        throw std::runtime_error(
            "AutoSpeed expects exactly 1 model output"
        );
    }
}

// ─── Inference ───────────────────────────────────────────────────────────────

visionpilot::common::AutoSpeedOutput AutoSpeed::infer(
    const float* image_chw, float conf_thres, float iou_thres)
{
    /*
        AUTOSPEED : 
            - input : image_chw (float32 CHW buffer, CHW_SIZE elements, RGB [0, 1])

            // Raw model output shape: [1, C, N]
            //   C = 4 + num_classes   (cx, cy, w, h, logit_0 … logit_{K-1})
            //   Post-processing (sigmoid + threshold + NMS) is done inside infer().
    */

    const std::size_t expected_frame_bytes = CHW_SIZE * sizeof(float);

    if (engine_->input_size(0) != expected_frame_bytes) {
        throw std::runtime_error(
            "AutoSpeed input size mismatch"
        );
    }

    //send image to model buffer
    std::memcpy(engine_->input_ptr(0), image_chw, expected_frame_bytes);

    //execute inference
    if (engine_->run() != 0) {
        throw std::runtime_error(
            "AutoSpeed V4M inference failed"
        );
    }

    //retrieve output buffer
    float* output0 = engine_->output<float>(0);

    //post process output (NMS, thresholding)
    visionpilot::common::AutoSpeedOutput result = post_process(output0, conf_thres, iou_thres);

    return result;
}

// ─── Post-processing ─────────────────────────────────────────────────────────

visionpilot::common::AutoSpeedOutput AutoSpeed::post_process(const float* data, float conf_thres, float iou_thres) const
{
    visionpilot::common::AutoSpeedOutput out;

    //analyze shape
    //get output descriptor data for the output 0 (only output for this model)
    std::vector<int> shape = engine_->output_desc(0).shape;

    const int64_t C           = shape[1];
    const int64_t N           = shape[2];
    const int     num_classes = static_cast<int>(C) - 4;

    if (num_classes <= 0) {
        printf("[AutoSpeed] Invalid channel count C=%lld\n",
               static_cast<long long>(C));
        return out;
    }

    std::vector<visionpilot::common::Detection> candidates;
    candidates.reserve(256);

    for (int64_t n = 0; n < N; ++n) {
        const float cx = data[0 * N + n];
        const float cy = data[1 * N + n];
        const float w  = data[2 * N + n];
        const float h  = data[3 * N + n];

        float best_prob = -1.f;
        int   best_cls  =  0;
        for (int c = 0; c < num_classes; ++c) {
            const float prob = 1.f / (1.f + std::exp(-data[(4 + c) * N + n]));
            if (prob > best_prob) { best_prob = prob; best_cls = c; }
        }

        if (best_prob < conf_thres) continue;

        visionpilot::common::Detection d;
        d.x1       = cx - w * 0.5f;
        d.y1       = cy - h * 0.5f;
        d.x2       = cx + w * 0.5f;
        d.y2       = cy + h * 0.5f;
        d.score    = best_prob;
        d.class_id = best_cls;
        candidates.push_back(d);
    }

    out.detections = nms(std::move(candidates), iou_thres);
    out.valid      = true;
    return out;
}

// ─── NMS helpers ─────────────────────────────────────────────────────────────

float AutoSpeed::iou(const visionpilot::common::Detection& a, const visionpilot::common::Detection& b)
{
    const float ix1   = std::max(a.x1, b.x1);
    const float iy1   = std::max(a.y1, b.y1);
    const float ix2   = std::min(a.x2, b.x2);
    const float iy2   = std::min(a.y2, b.y2);
    const float inter = std::max(0.f, ix2 - ix1) * std::max(0.f, iy2 - iy1);
    const float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    const float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter / (area_a + area_b - inter + 1e-6f);
}

std::vector<visionpilot::common::Detection> AutoSpeed::nms(
    std::vector<visionpilot::common::Detection> dets, float iou_thres)
{
    std::sort(dets.begin(), dets.end(),
              [](const visionpilot::common::Detection& a, const visionpilot::common::Detection& b) {
                  return a.score > b.score;
              });

    std::vector<bool>      suppressed(dets.size(), false);
    std::vector<visionpilot::common::Detection> keep;
    keep.reserve(dets.size());

    for (size_t i = 0; i < dets.size(); ++i) {
        if (suppressed[i]) continue;
        keep.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (!suppressed[j] && iou(dets[i], dets[j]) > iou_thres)
                suppressed[j] = true;
        }
    }
    return keep;
}

}  // namespace visionpilot::models
