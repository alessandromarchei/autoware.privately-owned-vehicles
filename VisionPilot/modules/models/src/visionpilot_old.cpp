#include <array>
#include <cmath>
#include <cstdio>
#include <models/visionpilot.hpp>

namespace visionpilot::models {


VisionPilot::VisionPilot(const std::string& autodrive_model_path, const std::string& autosteer_model_path, const std::string& autospeed_model_path)
{
    //instantiate underlying v4m engine for this model
    engine_ = std::make_unique<engine::V4MEngine>(autodrive_model_path, autosteer_model_path, autospeed_model_path);

    /*
        VISION PILOT INPUTS:
        input[0] previous_frame [1, 3, 512, 1024]   (WARPED)
        input[1] current_frame  [1, 3, 512, 1024]   (WARPED)
        input[2] resized_frame  [1, 3, 512, 1024]   (RESIZED)
    */
    if (engine_->num_inputs() != 3) {
        throw std::runtime_error(
            "VisionPilot expects exactly 3 model inputs, got " + std::to_string(engine_->num_inputs())
        );
    }

    /*
        VISIONPILOT comprises :
        - AutoDrive model (3 outputs)
        - AutoSteer model (2 outputs)
        - AutoSpeed model (1 output)
        Total outputs = 3 + 2 + 1 = 6
    */
    if (engine_->num_outputs() != 6) {
        throw std::runtime_error(
            "VisionPilot expects exactly 6 model outputs, got " + std::to_string(engine_->num_outputs())
        );
    }

}

visionpilot::common::VisionPilotOutput VisionPilot::infer(const float* prev_chw_warped,const float* curr_chw_warped,const float* resized_chw)
{
    const std::size_t expected_frame_bytes = CHW_SIZE * sizeof(float);

    if (engine_->input_size(0) != expected_frame_bytes || engine_->input_size(1) != expected_frame_bytes || engine_->input_size(2) != expected_frame_bytes) {
        throw std::runtime_error(
            "VisionPilot input size mismatch"
        );
    }

    //copy input data to engine buffers
    std::memcpy(engine_->input_ptr(0), prev_chw_warped, expected_frame_bytes);

    std::memcpy(engine_->input_ptr(1), curr_chw_warped, expected_frame_bytes);

    std::memcpy(engine_->input_ptr(2), resized_chw, expected_frame_bytes);

    if (engine_->run() != 0) {
        throw std::runtime_error(
            "VisionPilot V4M inference failed"
        );
    }

    // retrieve output from buffers
    /*
        output1 = autodrive(dist_normalized)
        output2 = autodrive(curvature_raw)
        output3 = autodrive(flag_logit)
        output4 = autosteer(lane_value)
        output5 = autosteer(height)
        output6 = autospeed(detections)
    */
    float* autodrive_dist_normalized = engine_->output<float>(0);
    float* autodrive_curvature_raw = engine_->output<float>(1);
    float* autodrive_flag_logit = engine_->output<float>(2);
    float* autosteer_lane_value = engine_->output<float>(3);
    float* autosteer_height = engine_->output<float>(4);
    float* autospeed_detections = engine_->output<float>(5);

    visionpilot::common::VisionPilotOutput result{};

    //postprocess the outputs and fill the result struct
    result.inference.auto_drive = postprocess_autodrive(autodrive_dist_normalized, autodrive_curvature_raw, autodrive_flag_logit);
    result.inference.auto_steer = postprocess_autosteer(autosteer_lane_value, autosteer_height);
    result.inference.auto_speed = postprocess_autospeed(autospeed_detections);    //default conf_thres :0.6f, default iou_thres : 0.45f

    //return the complete struct containing all the 3 outpouts and a valid flag
    result.inference.auto_speed.valid     = true;
    result.inference.auto_steer.valid     = true;
    result.inference.auto_drive.valid     = true;

    return result;
}


visionpilot::common::AutoDriveOutput VisionPilot::postprocess_autodrive(float* dist_normalized, float* curvature_raw, float* flag_logit)
{
    visionpilot::common::AutoDriveOutput autodrive_output{};
    autodrive_output.dist_normalized = *dist_normalized;
    autodrive_output.curvature_raw = *curvature_raw;
    autodrive_output.flag_prob = 1.f / (1.f + std::exp(-*flag_logit));
    autodrive_output.valid = true;

    return autodrive_output;

}

visionpilot::common::AutoSteerOutput VisionPilot::postprocess_autosteer(float* lane_value, float* height)
{
    visionpilot::common::AutoSteerOutput autosteer_output{};
    // Assuming lane_value and height are arrays of size 64
    std::memcpy(autosteer_output.xp.data(), lane_value, 64 * sizeof(float));
    std::memcpy(autosteer_output.h_vector.data(), height, 64 * sizeof(float));
    autosteer_output.valid = true;

    return autosteer_output;
}


// ─── AutoSpeed post processing ─────────────────────────────────────────────────────────

visionpilot::common::AutoSpeedOutput VisionPilot::postprocess_autospeed(const float* data, float conf_thres, float iou_thres)
{
    visionpilot::common::AutoSpeedOutput out;

    //analyze shape
    //get output descriptor data for the output 0 (only output for this model)
    std::vector<int> shape = engine_->output_desc(0).shape;

    const int64_t C           = shape[1];
    const int64_t N           = shape[2];
    const int     num_classes = static_cast<int>(C) - 4;

    if (num_classes <= 0) {
        throw std::runtime_error(
            "AutoSpeed invalid channel count"
        );
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

float VisionPilot::iou(const visionpilot::common::Detection& a, const visionpilot::common::Detection& b)
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

std::vector<visionpilot::common::Detection> VisionPilot::nms(
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


}// namespace visionpilot::models
