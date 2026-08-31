#include <models/inference.hpp>

#include <common/utils.hpp>
#include <logging/logger.hpp>

#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

namespace visionpilot::models {
namespace {

constexpr int NET_W = VisionPilot::NET_W;
constexpr int NET_H = VisionPilot::NET_H;
constexpr int CHW_SIZE = VisionPilot::CHW_SIZE;

constexpr float MEAN[3] = {0.485f, 0.456f, 0.406f};
constexpr float STD[3] = {0.229f, 0.224f, 0.225f};

using Clock = std::chrono::steady_clock;
using Milliseconds = std::chrono::duration<double, std::milli>;

double elapsedMilliseconds(const Clock::time_point start)
{
    return Milliseconds(Clock::now() - start).count();
}

std::vector<float> chw_imagenet(const cv::Mat& bgr)
{
    cv::Mat rgb;
    cv::Mat float_image;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(float_image, CV_32FC3, 1.0 / 255.0);

    std::vector<cv::Mat> channels(3);
    cv::split(float_image, channels);

    std::vector<float> output(CHW_SIZE);
    for (int channel = 0; channel < 3; ++channel) {
        float* destination = output.data() + channel * NET_H * NET_W;
        const float* source =
            reinterpret_cast<const float*>(channels[channel].data);

        for (int index = 0; index < NET_H * NET_W; ++index) {
            destination[index] =
                (source[index] - MEAN[channel]) / STD[channel];
        }
    }

    return output;
}

std::vector<float> chw_01(const cv::Mat& bgr)
{
    cv::Mat rgb;
    cv::Mat float_image;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(float_image, CV_32FC3, 1.0 / 255.0);

    std::vector<cv::Mat> channels(3);
    cv::split(float_image, channels);

    std::vector<float> output(CHW_SIZE);
    for (int channel = 0; channel < 3; ++channel) {
        std::memcpy(
            output.data() + channel * NET_H * NET_W,
            channels[channel].data,
            static_cast<std::size_t>(NET_H * NET_W) * sizeof(float));
    }

    return output;
}

}  // namespace

void LatencyStats::update(
    double pre_ms,
    double autodrive_ms,
    double autosteer_ms,
    double autospeed_ms,
    double inference_ms,
    double wall_ms)
{
    pre = pre_ms;
    ad = autodrive_ms;
    as = autosteer_ms;
    asp = autospeed_ms;
    inference = inference_ms;
    wall = wall_ms;
}

void LatencyStats::print() const
{
    VP_INFO(
        "Latency pre=%.2f ms AutoDrive=%.2f ms AutoSteer=%.2f ms "
        "AutoSpeed=%.2f ms inference=%.2f ms wall=%.2f ms",
        pre,
        ad,
        as,
        asp,
        inference,
        wall);
}

void LatencyStats::reset()
{
    *this = {};
}

InferencePipeline::InferencePipeline(const Config& cfg)
    : visionpilot_(
          cfg.auto_drive_model_path,
          cfg.auto_steer_model_path,
          cfg.auto_speed_model_path)
{
    fusion::LongitudinalFusion::Config longitudinal_config;
    longitudinal_config.debug = cfg.fusion_debug;
    long_fusion_ = fusion::LongitudinalFusion{longitudinal_config};

    fusion::LateralFusion::Config lateral_config;
    lateral_config.debug = cfg.fusion_debug;
    lateral_config.cte_bias_m = cfg.cte_bias_m;
    lat_fusion_ = fusion::LateralFusion{lateral_config};
}

void InferencePipeline::set_H_resized(const cv::Mat& H, cv::Size raw_size)
{
    cv::Mat H64;
    H.convertTo(H64, CV_64F);

    const int crop_top =
        compute_top_crop_2_1(raw_size.height, raw_size.width);
    const double crop_height =
        static_cast<double>(raw_size.height - crop_top);
    const double scale_x = static_cast<double>(raw_size.width) / NET_W;
    const double scale_y = crop_height / NET_H;

    const cv::Matx33d transform(
        scale_x, 0.0, 0.0,
        0.0, scale_y, static_cast<double>(crop_top),
        0.0, 0.0, 1.0);

    H_resized_ = H64 * cv::Mat(transform);
    cv::Mat inverse = H_resized_.inv();
    inverse.convertTo(H_world2resized_, CV_32F);

    lat_fusion_.set_H(H_resized_);
    long_fusion_.set_H(H_resized_);

    VP_INFO(
        "[Pipeline] H_resized set raw=%dx%d top_crop=%d sx=%.4f sy=%.4f",
        raw_size.width,
        raw_size.height,
        crop_top,
        scale_x,
        scale_y);
}

std::optional<visionpilot::common::InferenceFrameResult>
InferencePipeline::process(const cv::Mat& warped, const cv::Mat& resized)
{
    if (warped.empty()) {
        throw std::invalid_argument("InferencePipeline received an empty warped frame");
    }

    const auto wall_start = Clock::now();
    ++frame_count_;

    // This is the single current image shared by AutoSteer and AutoSpeed.
    // Pass resized when those models expect the non-BEV view; leave it empty
    // when they should consume the warped view.
    const cv::Mat& steer_speed_frame = resized.empty() ? warped : resized;

    const auto preprocess_start = Clock::now();
    auto current_warped_imagenet = chw_imagenet(warped);
    auto current_steer_speed_01 = chw_01(steer_speed_frame);
    const double preprocess_ms = elapsedMilliseconds(preprocess_start);

    // AutoDrive needs t-1. The first frame only initializes history.
    if (previous_warped_imagenet_.empty()) {
        previous_warped_imagenet_ = std::move(current_warped_imagenet);
        return std::nullopt;
    }

    const auto inference_start = Clock::now();
    const auto visionpilot_result = visionpilot_.infer(
        previous_warped_imagenet_.data(),
        current_warped_imagenet.data(),
        current_steer_speed_01.data());
    const double measured_inference_ms = elapsedMilliseconds(inference_start);

    const auto& timings = visionpilot_.last_timings();

    visionpilot::common::InferenceFrameResult output{};
    output.frame_id = frame_count_;
    output.pre_ms = preprocess_ms;
    output.ad_ms = timings.autodrive_ms;
    output.as_ms = timings.autosteer_ms;
    output.asp_ms = timings.autospeed_ms;
    output.auto_drive = visionpilot_result.inference.auto_drive;
    output.auto_steer = visionpilot_result.inference.auto_steer;
    output.auto_speed = visionpilot_result.inference.auto_speed;
    output.cipo = long_fusion_.update(output.auto_drive, output.auto_speed, warped);
    output.lateral = lat_fusion_.update(output.auto_steer, output.auto_drive);
    output.wall_ms = elapsedMilliseconds(wall_start);

    previous_warped_imagenet_ = std::move(current_warped_imagenet);

    stats_.update(
        preprocess_ms,
        timings.autodrive_ms,
        timings.autosteer_ms,
        timings.autospeed_ms,
        measured_inference_ms,
        output.wall_ms);

    return output;
}

void InferencePipeline::reset()
{
    previous_warped_imagenet_.clear();
    frame_count_ = 0;
    stats_.reset();
    long_fusion_.reset();
    lat_fusion_.reset();
}

}  // namespace visionpilot::models
