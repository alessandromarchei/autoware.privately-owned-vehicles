#pragma once

#include <memory>
#include <string>


#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>

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

    void create_session(const std::string& model_path) const;

    // Read-only access to config (models may inspect provider, etc.)
    const Config& config() const { return cfg_; }

private:

    Config     cfg_;

    std::string model_path;

    //data for the engine to run the model
    e_osal_return_t osal_ret;
    BufMgr_BufferManager *buffer_manager;

    std::unique_ptr<Network> network;

    R_EXFWK *exfwk;
    std::unique_ptr<JobContainer> job_container;

    std::unordered_map<PipelineId, std::vector<InputMemory>> user_input_memories;
    std::unordered_map<PipelineId, std::vector<OutputMemory>> user_output_memories;

    ArtifactHelper helper;
    

    //input/output memories
    std::vector<InputMemory> input_memories;
    std::vector<OutputMemory> output_memories;

    std::vector<JobId> job_dependency;
    std::vector<JobId> job_ids;

};

}  // namespace visionpilot::engine
