#include "camera_interface/frames_interface.hpp"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <thread>

#include <opencv2/imgcodecs.hpp>

namespace camera_interface {

namespace {

bool is_supported_image(const std::filesystem::path& path)
{
    if (!path.has_extension()) {
        return false;
    }

    std::string extension = path.extension().string();

    std::transform(
        extension.begin(),
        extension.end(),
        extension.begin(),
        [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });

    return extension == ".png" ||
           extension == ".bmp" ||
           extension == ".ppm";
}

} // namespace

FramesInterface::FramesInterface(
    const std::string& directory,
    bool loop,
    bool realtime,
    double fps)
    : directory_(directory),
      loop_(loop),
      realtime_(realtime),
      fps_(fps)
{
    const std::filesystem::path directory_path(directory_);

    std::error_code error;

    if (!std::filesystem::exists(directory_path, error) ||
        !std::filesystem::is_directory(directory_path, error)) {
        std::cerr
            << "[FramesInterface] Directory does not exist: "
            << directory_
            << '\n';
        return;
    }

    for (const auto& entry :
         std::filesystem::directory_iterator(directory_path, error)) {

        if (error) {
            std::cerr
                << "[FramesInterface] Cannot inspect directory: "
                << directory_
                << ": "
                << error.message()
                << '\n';

            frame_paths_.clear();
            return;
        }

        if (!entry.is_regular_file()) {
            continue;
        }

        if (is_supported_image(entry.path())) {
            frame_paths_.push_back(entry.path());
        }
    }

    // frame_000000.png, frame_000001.png, ... are ordered correctly
    // by lexicographical sorting because the numeric part is zero-padded.
    std::sort(frame_paths_.begin(), frame_paths_.end());

    if (frame_paths_.empty()) {
        std::cerr
            << "[FramesInterface] No supported frames found in: "
            << directory_
            << '\n';
        return;
    }

    if (realtime_ && fps_ > 0.0) {
        frame_period_ = std::chrono::duration<double>(1.0 / fps_);
    }

    std::cout
        << "[FramesInterface] Found "
        << frame_paths_.size()
        << " frames in "
        << directory_
        << '\n';

    std::cout
        << "[FramesInterface] FPS: "
        << fps_
        << ", realtime: "
        << (realtime_ ? "true" : "false")
        << ", loop: "
        << (loop_ ? "true" : "false")
        << '\n';
}

bool FramesInterface::is_device_open() const
{
    return !frame_paths_.empty();
}

std::tuple<bool, cv::Mat> FramesInterface::get_latest_frame()
{
    if (frame_paths_.empty()) {
        return {false, {}};
    }

    if (current_frame_ >= frame_paths_.size()) {
        if (!loop_) {
            return {false, {}};
        }

        current_frame_ = 0;
    }

    const auto start_time = std::chrono::steady_clock::now();
    const auto& frame_path = frame_paths_[current_frame_];

    cv::Mat frame = cv::imread(
        frame_path.string(),
        cv::IMREAD_COLOR);

    if (frame.empty()) {
        std::cerr
            << "[FramesInterface] Cannot read frame "
            << current_frame_
            << ": "
            << frame_path
            << '\n';

        return {false, {}};
    }

    ++current_frame_;

    if (frame_period_.count() > 0.0) {
        const auto elapsed =
            std::chrono::steady_clock::now() - start_time;

        const auto remaining = frame_period_ - elapsed;

        if (remaining.count() > 0.0) {
            std::this_thread::sleep_for(remaining);
        }
    }

    return {true, std::move(frame)};
}

std::vector<std::string> FramesInterface::get_overlay() const
{
    return {
        "frames: " + directory_,
        "frame: " +
            std::to_string(current_frame_) +
            "/" +
            std::to_string(frame_paths_.size())
    };
}

} // namespace camera_interface