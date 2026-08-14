#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rcar-xos/hycoah/r_hycoah.hpp"
#include "rcar-xos/hycoah/r_hycoah_io_desc.hpp"
#include "rcar-xos/hycoah/r_hycoah_types.hpp"
#include "rcar-xos/osal/r_osal.h"

#include "vsfwk_utils.hpp"

using namespace hycoah;


namespace visionpilot::engine {

class V4MEngine {
public:
    V4MEngine();
    explicit V4MEngine(const std::string& model_path);

    ~V4MEngine();

    V4MEngine(const V4MEngine&) = delete;
    V4MEngine& operator=(const V4MEngine&) = delete;

    V4MEngine(V4MEngine&&) = delete;
    V4MEngine& operator=(V4MEngine&&) = delete;

    // Load and initialize a .msgpack network.
    int create_session(const std::string& model_path);

    // Execute one inference.
    // Input buffers must already contain valid model input data.
    int run();

    // -------------------------------------------------------------------------
    // Input / output information
    // -------------------------------------------------------------------------

    [[nodiscard]] std::size_t num_inputs() const noexcept;
    [[nodiscard]] std::size_t num_outputs() const noexcept;

    [[nodiscard]] const hycoah::InputDesc& input_desc(std::size_t index) const;

    [[nodiscard]] const hycoah::OutputDesc& output_desc(std::size_t index) const;

    // -------------------------------------------------------------------------
    // Direct CPU access to model I/O buffers
    // -------------------------------------------------------------------------

    [[nodiscard]] void* input_ptr(std::size_t index);
    [[nodiscard]] const void* input_ptr(std::size_t index) const;

    [[nodiscard]] void* output_ptr(std::size_t index);
    [[nodiscard]] const void* output_ptr(std::size_t index) const;

    [[nodiscard]] std::size_t input_size(std::size_t index) const;
    [[nodiscard]] std::size_t output_size(std::size_t index) const;

    // Convenience typed access.
    template <typename T>
    [[nodiscard]] T* input(std::size_t index)
    {
        return static_cast<T*>(input_ptr(index));
    }

    template <typename T>
    [[nodiscard]] const T* input(std::size_t index) const
    {
        return static_cast<const T*>(input_ptr(index));
    }

    template <typename T>
    [[nodiscard]] T* output(std::size_t index)
    {
        return static_cast<T*>(output_ptr(index));
    }

    template <typename T>
    [[nodiscard]] const T* output(std::size_t index) const
    {
        return static_cast<const T*>(output_ptr(index));
    }

    [[nodiscard]] bool initialized() const noexcept
    {
        return session_initialized_;
    }

    [[nodiscard]] const std::string& model_path() const noexcept
    {
        return model_path_;
    }

private:
    // static constexpr hycoah::PipelineId PIPELINE_ID = 0;

    void cleanup() noexcept;

private:
    std::string model_path_;

    bool osal_initialized_   = false;
    bool helper_initialized_ = false;
    bool exfwk_initialized_  = false;
    bool session_initialized_ = false;

    BufMgr_BufferManager* buffer_manager_ = nullptr;
    R_EXFWK* exfwk_ = nullptr;

    std::unique_ptr<hycoah::Network> network_;
    std::unique_ptr<JobContainer> job_container_;

    hycoah::ArtifactHelper helper_;

    // These are the ACTUAL buffers associated with the network after
    // ArtifactHelper::init().
    //
    // ArtifactHelper allocates them automatically because we pass empty
    // user_input_memories/user_output_memories maps during initialization.
    std::vector<hycoah::InputMemory> input_memories_;
    std::vector<hycoah::OutputMemory> output_memories_;

    std::vector<JobId> job_ids_;
};

}  // namespace visionpilot::engine