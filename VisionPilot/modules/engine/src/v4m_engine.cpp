#include <engine/v4m_engine.hpp>

#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <unordered_map>

namespace visionpilot::engine {

V4MEngine::V4MEngine()
{
    std::cout << "[V4MEngine] Creating engine" << std::endl;
}

V4MEngine::V4MEngine(const std::string& model_path)
    : V4MEngine()
{
    if (create_session(model_path) != 0) {
        throw std::runtime_error(
            "Failed to create V4M session for model: " + model_path
        );
    }
}

int V4MEngine::create_session(const std::string& model_path)
{
    if (session_initialized_) {
        std::cerr
            << "[V4MEngine] Session already initialized for model: "
            << model_path_
            << std::endl;
        return -1;
    }

    model_path_ = model_path;

    // -------------------------------------------------------------------------
    // 1. Initialize OSAL
    // -------------------------------------------------------------------------

    const auto osal_ret = R_OSAL_Initialize();

    if (osal_ret != OSAL_RETURN_OK) {
        std::cerr
            << "[V4MEngine] OSAL initialization failed: "
            << static_cast<int>(osal_ret)
            << std::endl;
        return -1;
    }

    osal_initialized_ = true;

    // -------------------------------------------------------------------------
    // 2. Create Buffer Manager
    // -------------------------------------------------------------------------

    buffer_manager_ = createBufferManager();

    if (buffer_manager_ == nullptr) {
        std::cerr
            << "[V4MEngine] Failed to create BufferManager"
            << std::endl;

        cleanup();
        return -1;
    }

    // -------------------------------------------------------------------------
    // 3. Create execution framework
    // -------------------------------------------------------------------------

    exfwk_ = createExfwk(buffer_manager_);

    if (exfwk_ == nullptr) {
        std::cerr
            << "[V4MEngine] Failed to create ExecFWK"
            << std::endl;

        cleanup();
        return -1;
    }

    exfwk_initialized_ = true;

    // -------------------------------------------------------------------------
    // 4. Load compiled HyCo network
    // -------------------------------------------------------------------------

    try {
        network_ = std::make_unique<hycoah::Network>(
            model_path_.c_str()
        );
    }
    catch (const std::exception& e) {
        std::cerr
            << "[V4MEngine] Failed to create Network: "
            << e.what()
            << std::endl;

        cleanup();
        return -1;
    }

    // -------------------------------------------------------------------------
    // 5. Read model I/O descriptors
    // -------------------------------------------------------------------------

    const auto& input_descs = network_->getInputDesc();

    const auto& output_descs = network_->getOutputDesc();

    std::cout
        << "[V4MEngine] Model: "
        << model_path_
        << std::endl;

    std::cout
        << "[V4MEngine] Inputs: "
        << input_descs.size()
        << std::endl;

    for (std::size_t i = 0; i < input_descs.size(); ++i) {
        const auto& desc = input_descs[i];

        std::cout
            << "  input[" << i << "]"
            << " name=" << desc.name
            << " size=" << desc.size_bytes
            << " bytes"
            << std::endl;
    }

    std::cout
        << "[V4MEngine] Outputs: "
        << output_descs.size()
        << std::endl;

    for (std::size_t i = 0; i < output_descs.size(); ++i) {
        const auto& desc = output_descs[i];

        std::cout
            << "  output[" << i << "]"
            << " name=" << desc.name
            << " size=" << desc.size_bytes
            << " bytes"
            << std::endl;
    }

    if (input_descs.empty()) {
        std::cerr
            << "[V4MEngine] Network has no inputs"
            << std::endl;

        cleanup();
        return -1;
    }

    if (output_descs.empty()) {
        std::cerr
            << "[V4MEngine] Network has no outputs"
            << std::endl;

        cleanup();
        return -1;
    }

    // -------------------------------------------------------------------------
    // 6. Create JobContainer
    // -------------------------------------------------------------------------

    job_container_ = std::make_unique<JobContainer>();

    // -------------------------------------------------------------------------
    // 7. Automatic I/O memory management
    //
    // IMPORTANT:
    //
    // These maps are intentionally EMPTY.
    //
    // This tells ArtifactHelper:
    //
    //     "Allocate all input and output memories automatically."
    //
    // No assumptions are made about:
    //   - number of inputs
    //   - number of outputs
    //   - sizes
    //   - offsets
    //   - physical buffer layout
    // -------------------------------------------------------------------------

    std::unordered_map<
        hycoah::PipelineId,
        std::vector<hycoah::InputMemory>
    > user_input_memories;

    std::unordered_map<
        hycoah::PipelineId,
        std::vector<hycoah::OutputMemory>
    > user_output_memories;

    // -------------------------------------------------------------------------
    // 8. Configure ArtifactHelper
    // -------------------------------------------------------------------------

    const auto config_ret = helper_.config(
        std::vector<hycoah::Network>{*network_},
        hycoah::st_hycoah_config_t{}
    );

    if (config_ret !=
        hycoah::e_hycoah_return_t::RETURN_HYCOAH_OK) {

        std::cerr
            << "[V4MEngine] ArtifactHelper config failed: "
            << static_cast<int>(config_ret)
            << std::endl;

        cleanup();
        return -1;
    }

    // -------------------------------------------------------------------------
    // 9. Initialize ArtifactHelper
    //
    // Empty maps => automatic I/O buffer allocation.
    // -------------------------------------------------------------------------

    const auto init_ret = helper_.init(
        *network_,
        buffer_manager_,
        exfwk_,
        user_input_memories,
        user_output_memories
    );

    if (init_ret !=
        hycoah::e_hycoah_return_t::RETURN_HYCOAH_OK) {

        std::cerr
            << "[V4MEngine] ArtifactHelper init failed: "
            << static_cast<int>(init_ret)
            << std::endl;

        cleanup();
        return -1;
    }

    helper_initialized_ = true;

    // -------------------------------------------------------------------------
    // 10. Retrieve ACTUAL buffers allocated by ArtifactHelper
    // -------------------------------------------------------------------------

    input_memories_ =
        network_->getInputMemory(PIPELINE_ID);

    output_memories_ =
        network_->getOutputMemory(PIPELINE_ID);

    if (input_memories_.size() != input_descs.size()) {
        std::cerr
            << "[V4MEngine] Input memory count mismatch. "
            << "Descriptors=" << input_descs.size()
            << ", memories=" << input_memories_.size()
            << std::endl;

        cleanup();
        return -1;
    }

    if (output_memories_.size() != output_descs.size()) {
        std::cerr
            << "[V4MEngine] Output memory count mismatch. "
            << "Descriptors=" << output_descs.size()
            << ", memories=" << output_memories_.size()
            << std::endl;

        cleanup();
        return -1;
    }

    std::cout
        << "[V4MEngine] I/O buffers allocated successfully"
        << std::endl;

    for (std::size_t i = 0; i < input_memories_.size(); ++i) {
        const auto& mem = input_memories_[i];

        std::cout
            << "  input_memory[" << i << "]"
            << " ptr=" << mem.cpuPtr()
            << " size=" << mem.size_bytes
            << " container=" << mem.container_id
            << " buffer=" << mem.buffer_id
            << " offset=" << mem.offset
            << std::endl;
    }

    for (std::size_t i = 0; i < output_memories_.size(); ++i) {
        const auto& mem = output_memories_[i];

        std::cout
            << "  output_memory[" << i << "]"
            << " ptr=" << mem.cpuPtr()
            << " size=" << mem.size_bytes
            << " container=" << mem.container_id
            << " buffer=" << mem.buffer_id
            << " offset=" << mem.offset
            << std::endl;
    }

    // -------------------------------------------------------------------------
    // 11. Build execution jobs once
    // -------------------------------------------------------------------------

    std::vector<JobId> dependencies;

    const auto job_ret = network_->addJobs(
        job_container_.get(),
        PIPELINE_ID,
        dependencies,
        job_ids_
    );

    if (job_ret !=
        hycoah::e_hycoah_return_t::RETURN_HYCOAH_OK) {

        std::cerr
            << "[V4MEngine] addJobs failed: "
            << static_cast<int>(job_ret)
            << std::endl;

        cleanup();
        return -1;
    }

    std::cout
        << "[V4MEngine] Created "
        << job_ids_.size()
        << " execution jobs"
        << std::endl;

    for (const auto job_id : job_ids_) {
        std::cout
            << "  job_id="
            << static_cast<int>(job_id)
            << std::endl;
    }

    session_initialized_ = true;

    std::cout
        << "[V4MEngine] Session ready"
        << std::endl;

    return 0;
}


// =============================================================================
// Inference
// =============================================================================

int V4MEngine::run()
{
    if (!session_initialized_ ||
        network_ == nullptr ||
        exfwk_ == nullptr ||
        job_container_ == nullptr) {

        std::cerr
            << "[V4MEngine] run() called before session initialization"
            << std::endl;

        return -1;
    }

    // CPU-written inputs -> hardware-visible data.
    auto ret = network_->syncIO(
        hycoah::e_sync_direction_t::H2D,
        PIPELINE_ID
    );

    if (ret != hycoah::e_hycoah_return_t::RETURN_HYCOAH_OK) {
        std::cerr
            << "[V4MEngine] H2D sync failed: "
            << static_cast<int>(ret)
            << std::endl;

        return -1;
    }

    // Execute already prepared JobContainer.
    const auto exec_ret = execute(
        exfwk_,
        job_container_.get()
    );

    if (exec_ret !=
        hycoah::e_hycoah_return_t::RETURN_HYCOAH_OK) {

        std::cerr
            << "[V4MEngine] Execution failed: "
            << static_cast<int>(exec_ret)
            << std::endl;

        return -1;
    }

    // Hardware-written outputs -> CPU-visible data.
    ret = network_->syncIO(
        hycoah::e_sync_direction_t::D2H,
        PIPELINE_ID
    );

    if (ret != hycoah::e_hycoah_return_t::RETURN_HYCOAH_OK) {
        std::cerr
            << "[V4MEngine] D2H sync failed: "
            << static_cast<int>(ret)
            << std::endl;

        return -1;
    }

    return 0;
}


// =============================================================================
// I/O descriptors
// =============================================================================

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
    if (network_ == nullptr) {
        throw std::runtime_error(
            "V4MEngine session is not initialized"
        );
    }

    return network_->getInputDesc().at(index);
}

const hycoah::OutputDesc&
V4MEngine::output_desc(std::size_t index) const
{
    if (network_ == nullptr) {
        throw std::runtime_error(
            "V4MEngine session is not initialized"
        );
    }

    return network_->getOutputDesc().at(index);
}


// =============================================================================
// Direct I/O access
// =============================================================================

void* V4MEngine::input_ptr(std::size_t index)
{
    return input_memories_.at(index).cpuPtr();
}

const void* V4MEngine::input_ptr(std::size_t index) const
{
    return input_memories_.at(index).cpuPtr();
}

void* V4MEngine::output_ptr(std::size_t index)
{
    return output_memories_.at(index).cpuPtr();
}

const void* V4MEngine::output_ptr(std::size_t index) const
{
    return output_memories_.at(index).cpuPtr();
}

std::size_t V4MEngine::input_size(std::size_t index) const
{
    return static_cast<std::size_t>(
        input_memories_.at(index).size_bytes
    );
}

std::size_t V4MEngine::output_size(std::size_t index) const
{
    return static_cast<std::size_t>(
        output_memories_.at(index).size_bytes
    );
}


// =============================================================================
// Cleanup
// =============================================================================

void V4MEngine::cleanup() noexcept
{
    session_initialized_ = false;

    // JobContainer belongs to the application.
    job_container_.reset();
    job_ids_.clear();

    input_memories_.clear();
    output_memories_.clear();

    // Follow the same teardown ordering used by the Renesas sample:
    // ExecFWK quit -> ArtifactHelper deinit -> BufferManager close -> OSAL deinit.

    if (exfwk_ != nullptr) {
        if (exfwk_initialized_) {
            const auto ret = exfwk_->exfwk_quit();

            if (ret != RETURN_EXFWK_OK) {
                std::fprintf(
                    stderr,
                    "[V4MEngine] ExecFWK quit failed: %d\n",
                    static_cast<int>(ret)
                );
            }
        }

        delete exfwk_;
        exfwk_ = nullptr;
        exfwk_initialized_ = false;
    }

    if (helper_initialized_) {
        const auto ret = helper_.deinit();

        if (ret !=
            hycoah::e_hycoah_return_t::RETURN_HYCOAH_OK) {

            std::fprintf(
                stderr,
                "[V4MEngine] ArtifactHelper deinit failed: %d\n",
                static_cast<int>(ret)
            );
        }

        helper_initialized_ = false;
    }

    network_.reset();

    if (buffer_manager_ != nullptr) {
        const auto ret = buffer_manager_->R_BufMgr_Close();

        if (ret != RETURN_BUFMNGR_OK) {
            std::fprintf(
                stderr,
                "[V4MEngine] BufferManager close failed: %d\n",
                static_cast<int>(ret)
            );
        }

        delete buffer_manager_;
        buffer_manager_ = nullptr;
    }

    if (osal_initialized_) {
        const auto ret = R_OSAL_Deinitialize();

        if (ret != OSAL_RETURN_OK) {
            std::fprintf(
                stderr,
                "[V4MEngine] OSAL deinitialization failed: %d\n",
                static_cast<int>(ret)
            );
        }

        osal_initialized_ = false;
    }
}


V4MEngine::~V4MEngine()
{
    cleanup();
}

}  // namespace visionpilot::engine