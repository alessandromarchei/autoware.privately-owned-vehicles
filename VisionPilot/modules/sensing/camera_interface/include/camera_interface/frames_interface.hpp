#ifndef VISIONPILOT_FRAMES_INTERFACE_HPP
#define VISIONPILOT_FRAMES_INTERFACE_HPP

#include <camera_interface/camera_interface.hpp>

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace camera_interface {

class FramesInterface final : public CameraInterface
{
public:
    FramesInterface(
        const std::string& directory,
        bool loop,
        bool realtime,
        double fps = 30.0);

    bool is_device_open() const override;

    std::tuple<bool, cv::Mat> get_latest_frame() override;

    std::vector<std::string> get_overlay() const override;

private:
    std::string directory_;
    bool loop_ = false;
    bool realtime_ = false;
    double fps_ = 0.0;

    std::vector<std::filesystem::path> frame_paths_;
    std::size_t current_frame_ = 0;

    std::chrono::duration<double> frame_period_{0.0};
};

} // namespace camera_interface

#endif