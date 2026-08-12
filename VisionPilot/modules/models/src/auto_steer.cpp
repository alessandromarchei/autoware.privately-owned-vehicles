#include "models/auto_steer.hpp"

// #include <onnxruntime_run_options_config_keys.h>

#include <cstdio>
#include <cstring>

namespace visionpilot::models {

AutoSteer::AutoSteer(const std::string& model_path)
{
    //instantiate underlying v4m engine for this model
    engine_ = std::make_unique<engine::V4MEngine>(model_path);

    //AUTOSTEER : expects 1 input (input image), 2 outputs (lane_value, height)
    if (engine_->num_inputs() != 1) {
        throw std::runtime_error(
            "AutoSteer expects exactly 1 model inputs"
        );
    }

    if (engine_->num_outputs() != 2) {
        throw std::runtime_error(
            "AutoSteer expects exactly 2 model outputs"
        );
    }

}

visionpilot::common::AutoSteerOutput AutoSteer::infer(const float* image_chw)
{
    /*
        1 input , 2 outputs
    */
    visionpilot::common::AutoSteerOutput out;
    const std::size_t expected_frame_bytes = CHW_SIZE * sizeof(float);

    if (engine_->input_size(0) != expected_frame_bytes) {
        throw std::runtime_error(
            "AutoSteer input size mismatch"
        );
    }

    // Copy input image to engine input buffer
    std::memcpy(engine_->input_ptr(0), image_chw, expected_frame_bytes);


    //run inference
    if (engine_->run() != 0) {
        throw std::runtime_error(
            "AutoSteer V4M inference failed"
        );
    }

    // retrieve output from buffers
    float* output0 = engine_->output<float>(0);
    float* output1 = engine_->output<float>(1);


    // Output 0 — xp [1, 1, 64, 1] → 64 floats
    std::memcpy(out.xp.data(), output0, out.xp.size() * sizeof(float));

    // Output 1 — h_vector [1, 1, 64, 1] → 64 floats
    std::memcpy(out.h_vector.data(), output1, out.h_vector.size() * sizeof(float));


    out.valid = true;
    return out;
}

}  // namespace visionpilot::models
