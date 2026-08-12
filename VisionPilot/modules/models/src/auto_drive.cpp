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
}


// AutoDrive::AutoDrive(const std::string& model_path)
//     : model_path_(model_path),
//     frame_shape_{1, 3, NET_H, NET_W}        
// {
//     //instantiate underlying v4m engine for this model
//     engine_ = std::make_unique<engine::V4MEngine>();

//     printf("[AutoDrive] Creating V4MEngine for model: %s\n", model_path.c_str());
//     printf("[AutoDrive] frame shape: [1, 3, %d, %d]\n", NET_H, NET_W);

//     // Ort::AllocatorWithDefaultOptions alloc;
//     // const size_t n_in  = session_->GetInputCount();
//     // const size_t n_out = session_->GetOutputCount();

//     // in_name_strs_.resize(n_in);
//     // in_names_.resize(n_in);
//     // for (size_t i = 0; i < n_in; ++i) {
//     //     in_name_strs_[i] = session_->GetInputNameAllocated(i, alloc).get();
//     //     in_names_[i]     = in_name_strs_[i].c_str();
//     //     printf("[AutoDrive] input[%zu]  = %s\n", i, in_names_[i]);
//     // }

//     // out_name_strs_.resize(n_out);
//     // out_names_.resize(n_out);
//     // for (size_t i = 0; i < n_out; ++i) {
//     //     out_name_strs_[i] = session_->GetOutputNameAllocated(i, alloc).get();
//     //     out_names_[i]     = out_name_strs_[i].c_str();
//     //     printf("[AutoDrive] output[%zu] = %s\n", i, out_names_[i]);
//     // }

//     // printf("[AutoDrive] Ready — %zu inputs, %zu outputs | "
//     //        "frame [1, 3, %d, %d]\n", n_in, n_out, NET_H, NET_W);
// }


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


// visionpilot::common::AutoDriveOutput AutoDrive::infer(const float* prev_chw, const float* curr_chw)
// {
//     visionpilot::common::AutoDriveOutput out;

//     std::array<Ort::Value, 2> inputs{
//         Ort::Value::CreateTensor<float>(
//             mem_info_,
//             const_cast<float*>(prev_chw), CHW_SIZE,
//             frame_shape_.data(), frame_shape_.size()),
//         Ort::Value::CreateTensor<float>(
//             mem_info_,
//             const_cast<float*>(curr_chw), CHW_SIZE,
//             frame_shape_.data(), frame_shape_.size()),
//     };

//     std::vector<Ort::Value> results;
//     try {
//         Ort::RunOptions run_options;
//         run_options.AddConfigEntry(kOrtRunOptionsConfigEnableMemoryArenaShrinkage, arena_shrink_.c_str());
//         results = session_->Run(
//             run_options,
//             in_names_.data(),  inputs.data(),    inputs.size(),
//             out_names_.data(), out_names_.size());
//     } catch (const Ort::Exception& e) {
//         printf("[AutoDrive] Inference error: %s\n", e.what());
//         return out;
//     }

//     if (results.size() < 3) {
//         printf("[AutoDrive] Expected 3 outputs, got %zu\n", results.size());
//         return out;
//     }

//     // Extract outputs
//     // results[0] : dist_normalized (float32 scalar)
//     // results[1] : curvature_raw   (float32 scalar)
//     // results[2] : flag_prob       (float32 scalar)
//     out.dist_normalized = results[0].GetTensorData<float>()[0];
//     out.curvature_raw   = results[1].GetTensorData<float>()[0];
//     out.flag_prob = 1.f / (1.f + std::exp(-results[2].GetTensorData<float>()[0]));
//     out.valid     = true;
//     return out;
// }

}  // namespace visionpilot::models
