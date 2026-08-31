#include <models/inference.hpp>

#include <common/utils.hpp>
#include <logging/logger.hpp>
#include <common/models.hpp>

#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cstring>
#include <future>
#include <utility>
#include <vector>

namespace visionpilot::models {

namespace {

constexpr int NET_W    = VisionPilot::NET_W;
constexpr int NET_H    = VisionPilot::NET_H;
constexpr int CHW_SIZE = VisionPilot::CHW_SIZE;

constexpr float MEAN[3] = {0.485f, 0.456f, 0.406f};
constexpr float STD[3]  = {0.229f, 0.224f, 0.225f};

std::vector<float> chw_imagenet(const cv::Mat& bgr)
{
    cv::Mat rgb, f32;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(f32, CV_32FC3, 1.0 / 255.0);
    std::vector<cv::Mat> ch(3);
    cv::split(f32, ch);
    std::vector<float> out(CHW_SIZE);
    for (int c = 0; c < 3; ++c) {
        float* dst = out.data() + c * NET_H * NET_W;
        const float* src = reinterpret_cast<const float*>(ch[c].data);
        for (int i = 0; i < NET_H * NET_W; ++i)
            dst[i] = (src[i] - MEAN[c]) / STD[c];
    }
    return out;
}

std::vector<float> chw_01(const cv::Mat& bgr)
{
    cv::Mat rgb, f32;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(f32, CV_32FC3, 1.0 / 255.0);
    std::vector<cv::Mat> ch(3);
    cv::split(f32, ch);
    std::vector<float> out(CHW_SIZE);
    for (int c = 0; c < 3; ++c)
        std::memcpy(out.data() + c * NET_H * NET_W, ch[c].data,
                    static_cast<std::size_t>(NET_H * NET_W) * sizeof(float));
    return out;
}


}  // namespace

void LatencyStats:: void update(double pre_, double ad_, double as_, double asp_, double wall_)
{
    pre = pre_; ad = ad_; as = as_; asp = asp_; wall = wall_;
}

void LatencyStats::print() const
{
    const double total = pre + ad + as + asp + wall;
    VP_INFO("Latency  pre=%.1f ms  AutoDrive=%.1f ms  AutoSteer=%.1f ms  AutoSpeed=%.1f ms  Wall=%.1f ms",
            pre, ad, as, asp, wall);
}

void LatencyStats::reset() { *this = {}; }

InferencePipeline::InferencePipeline(const Config& cfg)
    : auto_drive_(cfg.auto_drive_model_path),
      auto_steer_(cfg.auto_steer_model_path),
      auto_speed_(cfg.auto_speed_model_path)
{
    fusion::LongitudinalFusion::Config lc;
    lc.debug           = cfg.fusion_debug;
    long_fusion_ = fusion::LongitudinalFusion{lc};

    fusion::LateralFusion::Config latc;
    latc.debug      = cfg.fusion_debug;
    latc.cte_bias_m = cfg.cte_bias_m;
    lat_fusion_ = fusion::LateralFusion{latc};
}

// V matrix — warped BEV 1024×512 → world.  Matches lateral/longitudinal fusion H_.
// DO NOT MODIFY — must stay in sync with the hardcoded H_ in both fusion modules.
static const cv::Matx33d kV(
     0.00209514907, -0.000941721466, -9.24906396,
     0.00662758637, -0.000352940531, -3.33396502,
     0.000120077371, -0.00411343505,  1.0);

void InferencePipeline::set_H_resized(const cv::Mat& H, cv::Size raw_size)
{
    // H_resized: resized_px → world  (AutoSteer / AutoSpeed path)
    // Preprocessor: top-crop to 2:1, then resize → 1024×512.
    //   u_raw = u_r · (raw_w / 1024)
    //   v_raw = v_r · (crop_h / 512) + crop_top
    //   world = H × raw_px  ⟹  H_resized = H × T
    cv::Mat H64;
    H.convertTo(H64, CV_64F);

    const int crop_top = compute_top_crop_2_1(raw_size.height, raw_size.width);
    const double crop_h = static_cast<double>(raw_size.height - crop_top);
    const double sx = static_cast<double>(raw_size.width) / 1024.0;
    const double sy = crop_h / 512.0;

    const cv::Matx33d T(sx, 0,  0,
                        0,  sy, static_cast<double>(crop_top),
                        0,   0, 1);

    const cv::Mat H_resized = H64 * cv::Mat(T);

    H_resized_ = H_resized.clone();
    cv::Mat H64_inv = H_resized.inv();   // MatExpr → cv::Mat
    H64_inv.convertTo(H_world2resized_, CV_32F);
    lat_fusion_.set_H(H_resized_);
    long_fusion_.set_H(H_resized_);
    VP_INFO("[Pipeline] H_resized set — raw=%dx%d  top_crop=%d  sx=%.4f sy=%.4f",
            raw_size.width, raw_size.height, crop_top, sx, sy);
}

std::optional<visionpilot::common::InferenceFrameResult> InferencePipeline::process(const cv::Mat& warped,
                           const cv::Mat& resized)
{
    using Clock = std::chrono::steady_clock;
    using Ms = std::chrono::duration<double, std::milli>;

    ++frame_count_;

    const cv::Mat& current_resized = !resized.empty() ? resized : warped;

    auto t0 = Clock::now();

    // preprocess the current frame for AutoSteer / AutoSpeed
    auto curr_warped_imn = chw_imagenet(warped);
    auto curr_resized_01 = chw_01(current_resized);

    const double ms_pre = Ms(Clock::now() - t0).count();

    if (prev_warped_imn_.empty()) {
        prev_warped_imn_ = std::move(curr_warped_imn);
        return std::nullopt;
    }

    auto t_wall = Clock::now();
    auto f_drive = std::async(std::launch::async, [&] {
        auto t = Clock::now();
        auto r = auto_drive_.infer(prev_imn.data(), curr_imn.data());
        return std::make_pair(std::move(r), Ms(Clock::now() - t).count());
    });
    auto f_steer = std::async(std::launch::async, [&] {
        auto t = Clock::now();
        auto r = auto_steer_.infer(curr_01_as.data());
        return std::make_pair(std::move(r), Ms(Clock::now() - t).count());
    });
    auto f_speed = std::async(std::launch::async, [&] {
        auto t = Clock::now();
        auto r = auto_speed_.infer(curr_01_asp.data());
        return std::make_pair(std::move(r), Ms(Clock::now() - t).count());
    });

    auto [res_drive, ms_drive] = f_drive.get();
    auto [res_steer, ms_steer] = f_steer.get();
    auto [res_speed, ms_speed] = f_speed.get();
    const double ms_wall = Ms(Clock::now() - t_wall).count();

    visionpilot::common::InferenceFrameResult out;
    out.frame_id   = frame_count_;
    out.wall_ms    = ms_wall;
    out.pre_ms     = ms_pre;
    out.ad_ms      = ms_drive;
    out.as_ms      = ms_steer;
    out.asp_ms     = ms_speed;
    out.auto_drive = res_drive;
    out.auto_steer = res_steer;
    out.auto_speed = res_speed;
    out.cipo       = long_fusion_.update(res_drive, res_speed, warped);
    out.lateral    = lat_fusion_.update(res_steer, res_drive);

    //move the current frame to previous frame for the next iteration
    prev_warped_imn_ = std::move(curr_warped_imn);

    stats_.update(ms_pre, ms_drive, ms_steer, ms_speed, ms_wall);
    return out;
}

void InferencePipeline::reset()
{
    prev_warped_imn_.clear();
    frame_buf_count_ = 0;
    frame_count_ = 0;
    stats_.reset();
    long_fusion_.reset();
    lat_fusion_.reset();
}

}  // namespace visionpilot::models
