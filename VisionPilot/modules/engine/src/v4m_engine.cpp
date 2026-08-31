#include <engine/v4m_engine.hpp>

#include <chrono>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <unordered_map>

namespace visionpilot::engine {
namespace {

using Clock = std::chrono::steady_clock;
using Milliseconds = std::chrono::duration<double, std::milli>;

}  // namespace

V4MEngine::V4MEngine()
{
    std::cout << "[V4MEngine] Creating reusable single-session engine"
              << std::endl;
}

V4MEngine::V4MEngine(const std::string& model_path)
    : V4MEngine()
{
    if (create_session(model_path) != 0) {
        throw std::runtime_error(
            "Failed to create V4M session for: " + model_path);
    }
}

int V4MEngine::create_session(const std::string& model_path)
{
    if (session_initialized_ || network_ != nullptr || exfwk_ != nullptr) {
        std::cerr << "[V4MEngine] A model session is already active: "
                  << model_path_ << std::endl;
        return -1;
    }

    model_path_ = model_path;
    last_run_ms_ = 0.0;

    auto osal_ret = R_OSAL_Initialize();
    if (osal_ret != OSAL_RETURN_OK) {
        std::cerr << "[V4MEngine] OSAL initialization failed: "
                  << static_cast<int>(osal_ret) << std::endl;
        cleanup();
        return -1;
    }
    osal_initialized_ = true;

    buffer_manager_ = createBufferManager();
    if (buffer_manager_ == nullptr) {
        std::cerr << "[V4MEngine] Failed to create BufferManager" << std::endl;
        cleanup();
        return -1;
    }

    try {
        network_ = std::make_unique<hycoah::Network>(model_path_.c_str());
    } catch (const std::exception& exception) {
        std::cerr << "[V4MEngine] Failed to load " << model_path_ << ": "
                  << exception.what() << std::endl;
        cleanup();
        return -1;
    }

    // Store owned copies. Some HyCoAH SDK versions return descriptor vectors
    // by value, so references obtained directly from the getter can dangle.
    input_descs_ = network_->getInputDesc();
    output_descs_ = network_->getOutputDesc();

    const auto& input_descs = input_descs_;
    const auto& output_descs = output_descs_;
    if (input_descs.empty() || output_descs.empty()) {
        std::cerr << "[V4MEngine] Model has no inputs or outputs: "
                  << model_path_ << std::endl;
        cleanup();
        return -1;
    }

    input_container_ids_.assign(input_descs.size(), 0);
    output_container_ids_.assign(output_descs.size(), 0);

    std::unordered_map<hycoah::PipelineId,
                       std::vector<hycoah::InputMemory>> user_inputs;
    std::unordered_map<hycoah::PipelineId,
                       std::vector<hycoah::OutputMemory>> user_outputs;

    auto& pipeline_inputs = user_inputs[PIPELINE_ID];
    auto& pipeline_outputs = user_outputs[PIPELINE_ID];
    pipeline_inputs.reserve(input_descs.size());
    pipeline_outputs.reserve(output_descs.size());

    // Same explicit descriptor-driven allocation used by the working dummy app.
    for (std::size_t index = 0; index < input_descs.size(); ++index) {
        const auto bytes =
            static_cast<std::size_t>(input_descs[index].size_bytes);
        createBuffer(buffer_manager_, input_container_ids_[index], bytes);
        pipeline_inputs.push_back({
            buffer_manager_,
            input_container_ids_[index],
            0,
            static_cast<std::int64_t>(bytes),
            0
        });
    }

    for (std::size_t index = 0; index < output_descs.size(); ++index) {
        const auto bytes =
            static_cast<std::size_t>(output_descs[index].size_bytes);
        createBuffer(buffer_manager_, output_container_ids_[index], bytes);
        pipeline_outputs.push_back({
            buffer_manager_,
            output_container_ids_[index],
            0,
            static_cast<std::int64_t>(bytes),
            0
        });
    }

    exfwk_ = createExfwk(buffer_manager_);
    if (exfwk_ == nullptr) {
        std::cerr << "[V4MEngine] Failed to create ExecFWK" << std::endl;
        cleanup();
        return -1;
    }
    exfwk_initialized_ = true;

    helper_ = std::make_unique<hycoah::ArtifactHelper>();
    const auto config_ret = helper_->config(
        std::vector<hycoah::Network>{*network_},
        hycoah::st_hycoah_config_t{});
    if (config_ret != hycoah::e_hycoah_return_t::RETURN_HYCOAH_OK) {
        std::cerr << "[V4MEngine] ArtifactHelper config failed for "
                  << model_path_ << ": " << static_cast<int>(config_ret)
                  << std::endl;
        cleanup();
        return -1;
    }

    const auto init_ret = helper_->init(
        *network_,
        buffer_manager_,
        exfwk_,
        user_inputs,
        user_outputs);
    if (init_ret != hycoah::e_hycoah_return_t::RETURN_HYCOAH_OK) {
        std::cerr << "[V4MEngine] ArtifactHelper init failed for "
                  << model_path_ << ": " << static_cast<int>(init_ret)
                  << std::endl;
        cleanup();
        return -1;
    }
    helper_initialized_ = true;

    input_memories_ = network_->getInputMemory(PIPELINE_ID);
    output_memories_ = network_->getOutputMemory(PIPELINE_ID);
    if (input_memories_.size() != input_descs.size() ||
        output_memories_.size() != output_descs.size()) {
        std::cerr << "[V4MEngine] Runtime I/O count mismatch for "
                  << model_path_ << std::endl;
        cleanup();
        return -1;
    }

    job_container_ = std::make_unique<JobContainer>();
    std::vector<JobId> dependencies;
    const auto jobs_ret = network_->addJobs(
        job_container_.get(),
        PIPELINE_ID,
        dependencies,
        job_ids_);
    if (jobs_ret != hycoah::e_hycoah_return_t::RETURN_HYCOAH_OK) {
        std::cerr << "[V4MEngine] addJobs failed for " << model_path_
                  << ": " << static_cast<int>(jobs_ret) << std::endl;
        cleanup();
        return -1;
    }

    session_initialized_ = true;
    std::cout << "[V4MEngine] Active model: " << model_path_ << std::endl;
    return 0;
}

int V4MEngine::run()
{
    if (!session_initialized_ || network_ == nullptr ||
        exfwk_ == nullptr || job_container_ == nullptr) {
        std::cerr << "[V4MEngine] run() called without an active session"
                  << std::endl;
        return -1;
    }

    const auto start = Clock::now();

    auto ret = network_->syncIO(
        hycoah::e_sync_direction_t::H2D, PIPELINE_ID);
    if (ret != hycoah::e_hycoah_return_t::RETURN_HYCOAH_OK) {
        std::cerr << "[V4MEngine] H2D sync failed: "
                  << static_cast<int>(ret) << std::endl;
        return -1;
    }

    const auto execute_ret = execute(exfwk_, job_container_.get());
    if (execute_ret != hycoah::e_hycoah_return_t::RETURN_HYCOAH_OK) {
        std::cerr << "[V4MEngine] Execution failed: "
                  << static_cast<int>(execute_ret) << std::endl;
        return -1;
    }

    ret = network_->syncIO(
        hycoah::e_sync_direction_t::D2H, PIPELINE_ID);
    if (ret != hycoah::e_hycoah_return_t::RETURN_HYCOAH_OK) {
        std::cerr << "[V4MEngine] D2H sync failed: "
                  << static_cast<int>(ret) << std::endl;
        return -1;
    }

    last_run_ms_ = Milliseconds(Clock::now() - start).count();
    return 0;
}

std::size_t V4MEngine::num_inputs() const noexcept
{
    return input_memories_.size();
}

std::size_t V4MEngine::num_outputs() const noexcept
{
    return output_memories_.size();
}

const hycoah::InputDesc& V4MEngine::input_desc(std::size_t index) const
{
    if (!session_initialized_) {
        throw std::runtime_error("No active V4M model session");
    }
    return input_descs_.at(index);
}

const hycoah::OutputDesc& V4MEngine::output_desc(std::size_t index) const
{
    if (!session_initialized_) {
        throw std::runtime_error("No active V4M model session");
    }
    return output_descs_.at(index);
}

void* V4MEngine::input_ptr(std::size_t index)
{
    auto& memory = input_memories_.at(index);
    return static_cast<std::uint8_t*>(memory.cpuPtr()) + memory.offset;
}

const void* V4MEngine::input_ptr(std::size_t index) const
{
    const auto& memory = input_memories_.at(index);
    return static_cast<const std::uint8_t*>(memory.cpuPtr()) + memory.offset;
}

void* V4MEngine::output_ptr(std::size_t index)
{
    auto& memory = output_memories_.at(index);
    return static_cast<std::uint8_t*>(memory.cpuPtr()) + memory.offset;
}

const void* V4MEngine::output_ptr(std::size_t index) const
{
    const auto& memory = output_memories_.at(index);
    return static_cast<const std::uint8_t*>(memory.cpuPtr()) + memory.offset;
}

std::size_t V4MEngine::input_size(std::size_t index) const
{
    return static_cast<std::size_t>(input_memories_.at(index).size_bytes);
}

std::size_t V4MEngine::output_size(std::size_t index) const
{
    return static_cast<std::size_t>(output_memories_.at(index).size_bytes);
}

void V4MEngine::close_session() noexcept
{
    cleanup();
}

void V4MEngine::cleanup() noexcept
{
    session_initialized_ = false;
    job_container_.reset();
    job_ids_.clear();
    input_memories_.clear();
    output_memories_.clear();

    // Exact successful dummy-app teardown order.
    if (exfwk_ != nullptr) {
        if (exfwk_initialized_) {
            const auto ret = exfwk_->exfwk_quit();
            if (ret != RETURN_EXFWK_OK) {
                std::fprintf(stderr, "[V4MEngine] ExecFWK quit failed: %d\n",
                             static_cast<int>(ret));
            }
        }
        delete exfwk_;
        exfwk_ = nullptr;
        exfwk_initialized_ = false;
    }

    if (helper_ != nullptr && helper_initialized_) {
        const auto ret = helper_->deinit();
        if (ret != hycoah::e_hycoah_return_t::RETURN_HYCOAH_OK) {
            std::fprintf(stderr, "[V4MEngine] Helper deinit failed: %d\n",
                         static_cast<int>(ret));
        }
        helper_initialized_ = false;
    }
    helper_.reset();
    network_.reset();

    if (buffer_manager_ != nullptr) {
        for (const int container_id : input_container_ids_) {
            if (buffer_manager_->R_BufMgr_DeleteContainer(container_id) !=
                RETURN_BUFMNGR_OK) {
                std::fprintf(stderr,
                             "[V4MEngine] Failed deleting input container %d\n",
                             container_id);
            }
        }
        for (const int container_id : output_container_ids_) {
            if (buffer_manager_->R_BufMgr_DeleteContainer(container_id) !=
                RETURN_BUFMNGR_OK) {
                std::fprintf(stderr,
                             "[V4MEngine] Failed deleting output container %d\n",
                             container_id);
            }
        }
        input_container_ids_.clear();
        output_container_ids_.clear();

        const auto ret = buffer_manager_->R_BufMgr_Close();
        if (ret != RETURN_BUFMNGR_OK) {
            std::fprintf(stderr, "[V4MEngine] BufferManager close failed: %d\n",
                         static_cast<int>(ret));
        }
        delete buffer_manager_;
        buffer_manager_ = nullptr;
    }

    if (osal_initialized_) {
        const auto ret = R_OSAL_Deinitialize();
        if (ret != OSAL_RETURN_OK) {
            std::fprintf(stderr, "[V4MEngine] OSAL deinit failed: %d\n",
                         static_cast<int>(ret));
        }
        osal_initialized_ = false;
    }

    input_descs_.clear();
    output_descs_.clear();
    model_path_.clear();
}

V4MEngine::~V4MEngine()
{
    cleanup();
}

}  // namespace visionpilot::engine
