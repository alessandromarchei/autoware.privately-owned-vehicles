#include <models/visionpilot.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace visionpilot::models {
namespace {

using Clock = std::chrono::steady_clock;
using Milliseconds = std::chrono::duration<double, std::milli>;

double elapsedMilliseconds(const Clock::time_point start)
{
    return Milliseconds(Clock::now() - start).count();
}

class SessionGuard {
public:
    explicit SessionGuard(engine::V4MEngine& engine) : engine_(engine) {}
    ~SessionGuard() { engine_.close_session(); }

    SessionGuard(const SessionGuard&) = delete;
    SessionGuard& operator=(const SessionGuard&) = delete;

private:
    engine::V4MEngine& engine_;
};

}  // namespace

VisionPilot::VisionPilot(
    const std::string& autodrive_model_path,
    const std::string& autosteer_model_path,
    const std::string& autospeed_model_path)
    : engine_(std::make_unique<engine::V4MEngine>()),
      autodrive_model_path_(autodrive_model_path),
      autosteer_model_path_(autosteer_model_path),
      autospeed_model_path_(autospeed_model_path)
{
    if (autodrive_model_path_.empty() || autosteer_model_path_.empty() ||
        autospeed_model_path_.empty()) {
        throw std::invalid_argument("VisionPilot model paths must not be empty");
    }
}

void VisionPilot::open_model(
    const std::string& path,
    std::size_t expected_inputs,
    std::size_t expected_outputs,
    const char* model_name)
{
    if (engine_->create_session(path) != 0) {
        throw std::runtime_error(
            std::string("Failed to activate ") + model_name + ": " + path);
    }

    if (engine_->num_inputs() != expected_inputs ||
        engine_->num_outputs() != expected_outputs) {
        engine_->close_session();
        throw std::runtime_error(
            std::string(model_name) + " I/O count mismatch: expected " +
            std::to_string(expected_inputs) + "/" +
            std::to_string(expected_outputs));
    }
}

void VisionPilot::validate_frame_input(
    std::size_t input_index,
    const char* input_name) const
{
    constexpr std::size_t expected_bytes = CHW_SIZE * sizeof(float);
    if (engine_->input_size(input_index) != expected_bytes) {
        throw std::runtime_error(
            std::string(input_name) + " has an unexpected byte size");
    }
}

visionpilot::common::VisionPilotOutput VisionPilot::infer(
    const float* previous_warped_chw,
    const float* current_warped_chw,
    const float* current_steer_speed_chw)
{
    if (previous_warped_chw == nullptr || current_warped_chw == nullptr ||
        current_steer_speed_chw == nullptr) {
        throw std::invalid_argument("VisionPilot received a null tensor");
    }

    const auto total_start = Clock::now();
    visionpilot::common::VisionPilotOutput result{};

    auto model_start = Clock::now();
    result.inference.auto_drive =
        run_autodrive(previous_warped_chw, current_warped_chw);
    last_timings_.autodrive_ms = elapsedMilliseconds(model_start);

    model_start = Clock::now();
    result.inference.auto_steer = run_autosteer(current_steer_speed_chw);
    last_timings_.autosteer_ms = elapsedMilliseconds(model_start);

    model_start = Clock::now();
    result.inference.auto_speed = run_autospeed(current_steer_speed_chw);
    last_timings_.autospeed_ms = elapsedMilliseconds(model_start);

    last_timings_.total_ms = elapsedMilliseconds(total_start);
    return result;
}

visionpilot::common::AutoDriveOutput VisionPilot::run_autodrive(
    const float* previous_warped_chw,
    const float* current_warped_chw)
{
    open_model(autodrive_model_path_, 2, 3, "AutoDrive");
    SessionGuard guard(*engine_);

    validate_frame_input(0, "AutoDrive previous input");
    validate_frame_input(1, "AutoDrive current input");
    constexpr std::size_t bytes = CHW_SIZE * sizeof(float);
    std::memcpy(engine_->input_ptr(0), previous_warped_chw, bytes);
    std::memcpy(engine_->input_ptr(1), current_warped_chw, bytes);

    if (engine_->run() != 0) {
        throw std::runtime_error("AutoDrive inference failed");
    }
    last_timings_.autodrive_run_ms = engine_->last_run_ms();

    if (engine_->output_size(0) < sizeof(float) ||
        engine_->output_size(1) < sizeof(float) ||
        engine_->output_size(2) < sizeof(float)) {
        throw std::runtime_error("AutoDrive output size mismatch");
    }

    return postprocess_autodrive(
        engine_->output<float>(0),
        engine_->output<float>(1),
        engine_->output<float>(2));
}

visionpilot::common::AutoSteerOutput VisionPilot::run_autosteer(
    const float* current_chw)
{
    open_model(autosteer_model_path_, 1, 2, "AutoSteer");
    SessionGuard guard(*engine_);

    validate_frame_input(0, "AutoSteer current input");
    constexpr std::size_t bytes = CHW_SIZE * sizeof(float);
    std::memcpy(engine_->input_ptr(0), current_chw, bytes);

    if (engine_->run() != 0) {
        throw std::runtime_error("AutoSteer inference failed");
    }
    last_timings_.autosteer_run_ms = engine_->last_run_ms();

    if (engine_->output_size(0) < 64 * sizeof(float) ||
        engine_->output_size(1) < 64 * sizeof(float)) {
        throw std::runtime_error("AutoSteer output size mismatch");
    }
    return postprocess_autosteer(
        engine_->output<float>(0), engine_->output<float>(1));
}

visionpilot::common::AutoSpeedOutput VisionPilot::run_autospeed(
    const float* current_chw)
{
    open_model(autospeed_model_path_, 1, 1, "AutoSpeed");
    SessionGuard guard(*engine_);

    validate_frame_input(0, "AutoSpeed current input");
    constexpr std::size_t bytes = CHW_SIZE * sizeof(float);
    std::memcpy(engine_->input_ptr(0), current_chw, bytes);

    if (engine_->run() != 0) {
        throw std::runtime_error("AutoSpeed inference failed");
    }
    last_timings_.autospeed_run_ms = engine_->last_run_ms();
    return postprocess_autospeed(engine_->output<float>(0));
}

visionpilot::common::AutoDriveOutput VisionPilot::postprocess_autodrive(
    const float* distance,
    const float* curvature,
    const float* flag) const
{
    visionpilot::common::AutoDriveOutput output{};
    output.dist_normalized = *distance;
    output.curvature_raw = *curvature;
    output.flag_prob = 1.0f / (1.0f + std::exp(-*flag));
    output.valid = true;
    return output;
}

visionpilot::common::AutoSteerOutput VisionPilot::postprocess_autosteer(
    const float* lane,
    const float* height) const
{
    visionpilot::common::AutoSteerOutput output{};
    std::memcpy(output.xp.data(), lane, 64 * sizeof(float));
    std::memcpy(output.h_vector.data(), height, 64 * sizeof(float));
    output.valid = true;
    return output;
}

visionpilot::common::AutoSpeedOutput VisionPilot::postprocess_autospeed(
    const float* data,
    float confidence_threshold,
    float iou_threshold) const
{
    visionpilot::common::AutoSpeedOutput output{};
    // Copy the shape while the AutoSpeed session is active. Do not retain a
    // reference into an SDK descriptor object whose lifetime may be temporary.
    const std::vector<int> shape = engine_->output_desc(0).shape;
    if (shape.size() < 3) {
        throw std::runtime_error("AutoSpeed output rank is smaller than 3");
    }

    const std::int64_t channels = shape[1];
    const std::int64_t predictions = shape[2];
    const int classes = static_cast<int>(channels) - 4;
    if (classes <= 0 || predictions <= 0) {
        throw std::runtime_error("AutoSpeed output shape is invalid");
    }

    std::vector<visionpilot::common::Detection> candidates;
    candidates.reserve(256);
    for (std::int64_t index = 0; index < predictions; ++index) {
        const float cx = data[index];
        const float cy = data[predictions + index];
        const float width = data[2 * predictions + index];
        const float height = data[3 * predictions + index];

        float best_probability = -1.0f;
        int best_class = 0;
        for (int class_index = 0; class_index < classes; ++class_index) {
            const float logit =
                data[(4 + class_index) * predictions + index];
            const float probability = 1.0f / (1.0f + std::exp(-logit));
            if (probability > best_probability) {
                best_probability = probability;
                best_class = class_index;
            }
        }

        if (best_probability < confidence_threshold) {
            continue;
        }

        visionpilot::common::Detection detection{};
        detection.x1 = cx - width * 0.5f;
        detection.y1 = cy - height * 0.5f;
        detection.x2 = cx + width * 0.5f;
        detection.y2 = cy + height * 0.5f;
        detection.score = best_probability;
        detection.class_id = best_class;
        candidates.push_back(detection);
    }

    output.detections = nms(std::move(candidates), iou_threshold);
    output.valid = true;
    return output;
}

float VisionPilot::iou(
    const visionpilot::common::Detection& a,
    const visionpilot::common::Detection& b)
{
    const float x1 = std::max(a.x1, b.x1);
    const float y1 = std::max(a.y1, b.y1);
    const float x2 = std::min(a.x2, b.x2);
    const float y2 = std::min(a.y2, b.y2);
    const float intersection =
        std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    const float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    const float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    return intersection / (area_a + area_b - intersection + 1e-6f);
}

std::vector<visionpilot::common::Detection> VisionPilot::nms(
    std::vector<visionpilot::common::Detection> detections,
    float threshold)
{
    std::sort(
        detections.begin(), detections.end(),
        [](const auto& left, const auto& right) {
            return left.score > right.score;
        });

    std::vector<bool> suppressed(detections.size(), false);
    std::vector<visionpilot::common::Detection> kept;
    kept.reserve(detections.size());

    for (std::size_t i = 0; i < detections.size(); ++i) {
        if (suppressed[i]) {
            continue;
        }
        kept.push_back(detections[i]);
        for (std::size_t j = i + 1; j < detections.size(); ++j) {
            if (!suppressed[j] && iou(detections[i], detections[j]) > threshold) {
                suppressed[j] = true;
            }
        }
    }
    return kept;
}

}  // namespace visionpilot::models
