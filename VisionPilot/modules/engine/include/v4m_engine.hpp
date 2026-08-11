#pragma once

#include <memory>
#include <string>


#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>
#include <unordered_map>

#include "common.hpp"
#include "rcar-xos/hycoah/r_hycoah.hpp"
#include "rcar-xos/hycoah/r_hycoah_io_desc.hpp"
#include "rcar-xos/hycoah/r_hycoah_types.hpp"
#include "rcar-xos/osal/r_osal.h"
#include "vsfwk_utils.hpp"

using namespace hycoah;


namespace visionpilot::engine {

// Configuration that governs how the engine creates sessions.
// One EngineConfig instance is typically shared across all models in main().
struct Config {
    // Execution provider:
    std::string provider     = "npu";

    // Used only when provider == "tensorrt"
    std::string precision    = "int8";   // "fp32" | "fp16"
    std::string cache_dir    = "";
    double      workspace_gb = 1.0;

    // not used in case of NPU
    int device_id = 0;
};
class V4MEngine {
public:
    explicit V4MEngine(const Config& cfg);
    ~V4MEngine();

    int create_session(const std::string& model_path);

    const Config& config() const
    {
        return cfg_;
    }

private:
    Config cfg_;

    std::string model_path_;

    bool osal_initialized_ = false;
    bool helper_initialized_ = false;
    bool input_buffer_created_ = false;
    bool output_buffer_created_ = false;

    e_osal_return_t osal_ret = OSAL_RETURN_OK;

    BufMgr_BufferManager* buffer_manager = nullptr;
    R_EXFWK* exfwk = nullptr;

    int input_container_id = 0;
    int output_container_id = 0;

    std::unique_ptr<Network> network;
    std::unique_ptr<JobContainer> job_container;

    std::unordered_map<PipelineId, std::vector<InputMemory>>
        user_input_memories;

    std::unordered_map<PipelineId, std::vector<OutputMemory>>
        user_output_memories;

    ArtifactHelper helper;

    std::vector<InputMemory> input_memories;
    std::vector<OutputMemory> output_memories;

    std::vector<JobId> job_dependency;
    std::vector<JobId> job_ids;
};

}  // namespace visionpilot::engine
