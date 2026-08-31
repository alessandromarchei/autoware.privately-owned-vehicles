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

// One reusable engine object, but at most one active HyCoAH session/network.
class V4MEngine {
public:
    V4MEngine();
    explicit V4MEngine(const std::string& model_path);
    ~V4MEngine();

    V4MEngine(const V4MEngine&) = delete;
    V4MEngine& operator=(const V4MEngine&) = delete;
    V4MEngine(V4MEngine&&) = delete;
    V4MEngine& operator=(V4MEngine&&) = delete;

    // Full activation: OSAL -> buffers -> one Network -> ExecFWK -> Helper.
    int create_session(const std::string& model_path);

    // H2D -> execute -> D2H for the currently active model.
    int run();

    // Full teardown. After this returns another model may be activated safely.
    void close_session() noexcept;

    [[nodiscard]] std::size_t num_inputs() const noexcept;
    [[nodiscard]] std::size_t num_outputs() const noexcept;
    [[nodiscard]] const hycoah::InputDesc& input_desc(std::size_t index) const;
    [[nodiscard]] const hycoah::OutputDesc& output_desc(std::size_t index) const;

    [[nodiscard]] void* input_ptr(std::size_t index);
    [[nodiscard]] const void* input_ptr(std::size_t index) const;
    [[nodiscard]] void* output_ptr(std::size_t index);
    [[nodiscard]] const void* output_ptr(std::size_t index) const;
    [[nodiscard]] std::size_t input_size(std::size_t index) const;
    [[nodiscard]] std::size_t output_size(std::size_t index) const;

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

    // Time of the latest H2D + execute + D2H operation.
    [[nodiscard]] double last_run_ms() const noexcept
    {
        return last_run_ms_;
    }

private:
    static constexpr hycoah::PipelineId PIPELINE_ID = 0;
    void cleanup() noexcept;

    std::string model_path_;
    bool osal_initialized_{false};
    bool helper_initialized_{false};
    bool exfwk_initialized_{false};
    bool session_initialized_{false};

    BufMgr_BufferManager* buffer_manager_{nullptr};
    R_EXFWK* exfwk_{nullptr};
    std::unique_ptr<hycoah::Network> network_;
    std::unique_ptr<JobContainer> job_container_;
    std::unique_ptr<hycoah::ArtifactHelper> helper_;

    // Keep owned descriptor copies. Returning a reference directly from
    // network_->get*Desc().at() is unsafe if the SDK getter returns by value.
    std::vector<hycoah::InputDesc> input_descs_;
    std::vector<hycoah::OutputDesc> output_descs_;
    std::vector<hycoah::InputMemory> input_memories_;
    std::vector<hycoah::OutputMemory> output_memories_;
    std::vector<int> input_container_ids_;
    std::vector<int> output_container_ids_;
    std::vector<JobId> job_ids_;
    double last_run_ms_{0.0};
};

}  // namespace visionpilot::engine
