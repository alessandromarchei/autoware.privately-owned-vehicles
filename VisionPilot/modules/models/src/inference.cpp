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

void LatencyStats::update(double pre_, double visionpilot_)
{
    pre = pre_; visionpilot = visionpilot_;
}

void LatencyStats::print() const
{
    const double total = pre + visionpilot;
    VP_INFO("Latency  pre=%.1f ms  VisionPilot=%.1f ms  %.0f fps",
            pre, visionpilot, total > 0 ? 1000.0 / total : 0.0);
}

void LatencyStats::reset() { *this = {}; }

InferencePipeline::InferencePipeline(const Config& cfg)
    : visionpilot_(cfg.visionpilot_model_path)
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

std::optional<visionpilot::common::InferenceFrameResult> InferencePipeline::process(
    const cv::Mat& in_autodrive_curr, const cv::Mat& in_autosteer_curr)
{
    // warped  -> AutoDrive
    // resized -> AutoSteer / AutoSpeed

    using Clock = std::chrono::steady_clock;
    using Ms = std::chrono::duration<double, std::milli>;

    ++frame_count_;

    const cv::Mat& current_resized = !in_autosteer_curr.empty() ? in_autosteer_curr : in_autodrive_curr;

    auto t0 = Clock::now();

    // preprocess the current frame for AutoSteer / AutoSpeed
    auto curr_warped_imn = chw_imagenet(in_autodrive_curr);
    auto curr_resized_01 = chw_01(current_resized);

    const double ms_pre = Ms(Clock::now() - t0).count();

    // first frame is not valid, as autodrive needs the previous frame to compute features.
    //execute inference but set outputs as not valid. this is to retrieve the current features for next iteration
    const bool output_valid = !prev_features_autodrive.empty();

    if (prev_features_autodrive.empty()) {
        prev_features_autodrive.resize(
            VisionPilot::AUTODRIVE_FEATURE_SIZE,
            0.0f
        );
    }

    auto t = Clock::now();

    //execute inference on the current frame and previous frame
    auto result = visionpilot_.infer(
        prev_features_autodrive.data(),
        curr_warped_imn.data(),
        curr_resized_01.data()
    );

    const double ms_visionpilot = Ms(Clock::now() - t).count();

    
    //copy the current features from the inference result to the prev_features_autodrive vector for next iteration
    const float* curr_features = visionpilot_.get_curr_features_autodrive();

    if (curr_features == nullptr) {
        throw std::runtime_error(
            "VisionPilot returned null AutoDrive feature pointer"
        );
    }

    std::memcpy(prev_features_autodrive.data(), curr_features, VisionPilot::AUTODRIVE_FEATURE_SIZE * sizeof(float));

    //retrieve results
    visionpilot::common::InferenceFrameResult out;

    //set valid flag for the outputs. if this is the first frame, the outputs are not valid as autodrive needs the previous frame to compute features
    out.auto_drive = result.inference.auto_drive;
    out.auto_steer = result.inference.auto_steer;
    out.auto_speed = result.inference.auto_speed;

    out.auto_drive.valid = output_valid;
    out.auto_steer.valid = true;
    out.auto_speed.valid = true;

    out.frame_id = frame_count_;
    out.pre_ms = ms_pre;
    out.visionpilot_ms = ms_visionpilot;
    out.total_ms = ms_pre + ms_visionpilot;     //total time is considered to be preprocessing + inference time. postprocessing is considered to be negligible

    out.cipo = long_fusion_.update(
        out.auto_drive,
        out.auto_speed
    );

    out.lateral = lat_fusion_.update(
        out.auto_steer,
        out.auto_drive
    );

    stats_.update(ms_pre, ms_visionpilot);
    return out;
}

void InferencePipeline::reset()
{
    prev_features_autodrive.clear();
    frame_buf_count_ = 0;
    frame_count_ = 0;
    stats_.reset();
    long_fusion_.reset();
    lat_fusion_.reset();
}

}  // namespace visionpilot::models
