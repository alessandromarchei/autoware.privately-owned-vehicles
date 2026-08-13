#include "models/auto_drive.hpp"

// #include <onnxruntime_run_options_config_keys.h>


#include <array>
#include <cmath>
#include <cstdio>

namespace visionpilot::models {


AutoDrive::AutoDrive(const std::string& model_path)
{
    //instantiate underlying v4m engine for this model
    engine_ = std::make_unique<engine::V4MEngine>(model_path);

    if (engine_->num_inputs() != 2) {
        throw std::runtime_error(
            "AutoDrive expects exactly 2 model inputs"
        );
    }

    if (engine_->num_outputs() != 3) {
        throw std::runtime_error(
            "AutoDrive expects exactly 3 model outputs"
        );
    }

    VP_INFO("[AutoDrive] Created V4MEngine for model: %s\n", model_path.c_str());
}

visionpilot::common::AutoDriveOutput AutoDrive::infer(const float* prev_chw,const float* curr_chw)
{
    const std::size_t expected_frame_bytes = CHW_SIZE * sizeof(float);

    if (engine_->input_size(0) != expected_frame_bytes || engine_->input_size(1) != expected_frame_bytes) {
        throw std::runtime_error(
            "AutoDrive input size mismatch"
        );
    }

    std::memcpy(engine_->input_ptr(0), prev_chw, expected_frame_bytes);

    std::memcpy(engine_->input_ptr(1), curr_chw, expected_frame_bytes);

    if (engine_->run() != 0) {
        throw std::runtime_error(
            "AutoDrive V4M inference failed"
        );
    }

    // retrieve output from buffers
    float* output0 = engine_->output<float>(0);
    float* output1 = engine_->output<float>(1);
    float* output2 = engine_->output<float>(2);

    visionpilot::common::AutoDriveOutput result{};

    // Extract outputs
    // results[0] : dist_normalized (float32 scalar)
    // results[1] : curvature_raw   (float32 scalar)
    // results[2] : flag_prob       (float32 scalar)
    result.dist_normalized = *output0;
    result.curvature_raw   = *output1;
    result.flag_prob = 1.f / (1.f + std::exp(-*output2));
    result.valid     = true;

    return result;
}


}  // namespace visionpilot::models
