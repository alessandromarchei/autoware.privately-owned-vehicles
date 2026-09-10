#include "models/visionpilot_single.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <models/visionpilot_single.hpp>

#include <limits>
#include <algorithm>
static void dump_tensor_stats(
    const char* name,
    const float* data,
    std::size_t n,
    std::size_t print_n = 16)
{
    if (!data || n == 0) {
        VP_INFO("[RAW] %s: EMPTY", name);
        return;
    }

    double sum = 0.0;
    double sum_abs = 0.0;
    float min_v = std::numeric_limits<float>::infinity();
    float max_v = -std::numeric_limits<float>::infinity();

    std::size_t zeros = 0;
    std::size_t nan_count = 0;
    std::size_t inf_count = 0;

    for (std::size_t i = 0; i < n; ++i) {
        const float v = data[i];

        if (std::isnan(v)) {
            ++nan_count;
            continue;
        }

        if (!std::isfinite(v)) {
            ++inf_count;
            continue;
        }

        min_v = std::min(min_v, v);
        max_v = std::max(max_v, v);

        sum += v;
        sum_abs += std::abs(v);

        if (v == 0.0f)
            ++zeros;
    }

    VP_INFO(
        "[RAW] %s n=%zu min=%g max=%g mean=%g mean_abs=%g "
        "zeros=%zu nan=%zu inf=%zu",
        name,
        n,
        min_v,
        max_v,
        sum / static_cast<double>(n),
        sum_abs / static_cast<double>(n),
        zeros,
        nan_count,
        inf_count
    );

    std::printf("[RAW] %s first:", name);

    for (std::size_t i = 0; i < std::min(n, print_n); ++i) {
        std::printf(" %.7g", data[i]);
    }

    std::printf("\n");
}


namespace visionpilot::models {


VisionPilot::VisionPilot(const std::string& model_path)
{
    //instantiate underlying v4m engine for this model
    engine_ = std::make_unique<engine::V4MEngine>(model_path);

    /*
        Inputs: [0] autosteer_image [1, 3, 512, 1024] 
                [1] autospeed_image [1, 3, 512, 1024]
                [2] autodrive_image [1, 3, 512, 1024]
                [3] autodrive_prev_features [1, 256, 16, 32]
        Outputs: [0] autodrive__distance [1, 1]
                [1] autodrive__curvature [1, 1]
                [2] autodrive__flag_logit [1, 1]
                [3] autodrive__feature_curr [1, 256, 16, 32]
                [4] autosteer__lane_value [1, 1, 64, 1]
                [5] autosteer__height [1, 1, 64, 1]
                [6] autospeed__output [1, 8, 10752]
    */
    if (engine_->num_inputs() != 4) {
        throw std::runtime_error(
            "VisionPilot expects exactly 4 model inputs, got " + std::to_string(engine_->num_inputs())
        );
    }

    if (engine_->num_outputs() != 7) {
        throw std::runtime_error(
            "VisionPilot expects exactly 7 model outputs, got " + std::to_string(engine_->num_outputs())
        );
    }

    VP_INFO("[VisionPilot] Created V4MEngine for model: %s\n", model_path.c_str());
}

visionpilot::common::VisionPilotOutput VisionPilot::infer(const float* prev_features,const float* curr_autodrive_input,const float* curr_autosteer_input)
{
    const std::size_t expected_frame_bytes = CHW_SIZE * sizeof(float);
    const std::size_t expected_features_bytes = 256 * 16 * 32 * sizeof(float);

    if (engine_->input_size(0) != expected_frame_bytes || engine_->input_size(1) != expected_frame_bytes 
    || engine_->input_size(2) != expected_frame_bytes || engine_->input_size(3) != expected_features_bytes) {
        throw std::runtime_error(
            "VisionPilot input size mismatch"
        );
    }


    //AUTOSTEER AND AUTOSPEED INPUTS
    std::memcpy(engine_->input_ptr(0), curr_autosteer_input, expected_frame_bytes);
    std::memcpy(engine_->input_ptr(1), curr_autosteer_input, expected_frame_bytes);


    //AUTODRIVE INPUTS
    std::memcpy(engine_->input_ptr(2), curr_autodrive_input, expected_frame_bytes);
    std::memcpy(engine_->input_ptr(3), prev_features, expected_features_bytes);

    if (engine_->run() != 0) {
        throw std::runtime_error(
            "VisionPilot V4M inference failed"
        );
    }

    // retrieve output from buffers
    /*
        Outputs: [0] autodrive__distance [1, 1]
                [1] autodrive__curvature [1, 1]
                [2] autodrive__flag_logit [1, 1]
                [3] autodrive__feature_curr [1, 256, 16, 32]
                [4] autosteer__lane_value [1, 1, 64, 1]
                [5] autosteer__height [1, 1, 64, 1]
                [6] autospeed__output [1, 8, 10752]
    */
   
    float* autodrive_dist_normalized = engine_->output<float>(0);
    float* autodrive_curvature_raw = engine_->output<float>(1);
    float* autodrive_flag_logit = engine_->output<float>(2);
    float* autodrive_feature_curr = engine_->output<float>(3);
    float* autosteer_lane_value = engine_->output<float>(4);
    float* autosteer_height = engine_->output<float>(5);
    float* autospeed_detections = engine_->output<float>(6);


    dump_tensor_stats(
        "AutoDrive distance",
        autodrive_dist_normalized,
        1
    );

    dump_tensor_stats(
        "AutoDrive curvature",
        autodrive_curvature_raw,
        1
    );

    dump_tensor_stats(
        "AutoDrive flag_logit",
        autodrive_flag_logit,
        1
    );

    dump_tensor_stats(
        "AutoDrive features",
        autodrive_feature_curr,
        256 * 16 * 32
    );

    dump_tensor_stats(
        "AutoSteer lane",
        autosteer_lane_value,
        64
    );

    dump_tensor_stats(
        "AutoSteer height",
        autosteer_height,
        64
    );

    dump_tensor_stats(
        "AutoSpeed raw",
        autospeed_detections,
        8 * 10752
    );


    visionpilot::common::VisionPilotOutput result{};

    //postprocess the outputs and fill the result struct
    result.inference.auto_drive = postprocess_autodrive(autodrive_dist_normalized, autodrive_curvature_raw, autodrive_flag_logit);
    result.inference.auto_steer = postprocess_autosteer(autosteer_lane_value, autosteer_height);
    result.inference.auto_speed = postprocess_autospeed(autospeed_detections);    //default conf_thres :0.6f, default iou_thres : 0.45f

    //return the complete struct containing all the 3 outpouts and a valid flag
    result.inference.auto_speed.valid     = true;
    result.inference.auto_steer.valid     = true;
    result.inference.auto_drive.valid     = true;

    //save the current features for the next iteration. it will be retrieved from the inference class
    curr_features_autodrive_ = autodrive_feature_curr;

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
    //get output descriptor data for the output 6 (only output for this model)
    std::vector<int> shape = engine_->output_desc(6).shape;

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
