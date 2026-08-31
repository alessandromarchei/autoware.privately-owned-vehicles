#pragma once

#include <cstdint>
#include <mutex>
#include <string>

#include <opencv2/core/mat.hpp>

#include <tcp/protocol.hpp>
#include <common/models.hpp>

namespace visionpilot::tcp {

class TCPClient
{
public:
    TCPClient();

    TCPClient(
        std::string server_address,
        std::uint16_t frame_port,
        std::uint16_t result_port);

    ~TCPClient();

    TCPClient(const TCPClient&) = delete;
    TCPClient& operator=(const TCPClient&) = delete;

    bool connect_to(
        const std::string& server_address,
        std::uint16_t frame_port,
        std::uint16_t result_port,
        int timeout_ms = 5000);

    bool reconnect(int timeout_ms = 5000);

    void disconnect();

    bool is_connected() const;

    // Riceve dal ServerSender del Predator.
    bool receive_frame(
        cv::Mat& frame,
        ReceivedFrame& metadata,
        int timeout_ms = -1);

    // Invia al ServerReceiver del Predator.
    bool send_result(const visionpilot::common::VisionPilotOutput& result);

    const std::string& server_address() const noexcept;

    std::uint16_t frame_port() const noexcept;
    std::uint16_t result_port() const noexcept;

    const std::string& last_error() const noexcept;

private:
    bool connect_socket_locked(
        int& socket_fd,
        const std::string& address,
        std::uint16_t port,
        int timeout_ms);

    bool fail_socket_locked(
        int& socket_fd,
        const std::string& message,
        bool close_socket = true);

private:
    mutable std::mutex receive_mutex_;
    mutable std::mutex send_mutex_;
    mutable std::mutex state_mutex_;

    int frame_socket_fd_{-1};
    int result_socket_fd_{-1};

    std::string server_address_;
    std::uint16_t frame_port_{0};
    std::uint16_t result_port_{0};

    std::string last_error_;
};

} // namespace visionpilot::tcp