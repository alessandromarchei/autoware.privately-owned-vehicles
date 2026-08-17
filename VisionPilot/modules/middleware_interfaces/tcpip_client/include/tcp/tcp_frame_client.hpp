#pragma once

#include <cstdint>
#include <mutex>
#include <string>

#include <opencv2/core/mat.hpp>
#include <tcp/protocol.hpp>

namespace visionpilot::tcp {

// TCP client intended for the V4M. It connects to an image server, receives
// Image messages as cv::Mat, and can return VisionResult on the same socket.
class TCPClient {
public:
    TCPClient();
    TCPClient(std::string server_address, std::uint16_t server_port);
    ~TCPClient();

    TCPClient(const TCPClient&) = delete;
    TCPClient& operator=(const TCPClient&) = delete;

    bool connect_to(const std::string& server_address,
                    std::uint16_t server_port,
                    int timeout_ms = 5000);

    bool reconnect(int timeout_ms = 5000);
    void disconnect();
    bool is_connected() const;

    // timeout_ms < 0 blocks indefinitely. The returned Mat owns its memory.
    bool receive_frame(cv::Mat& frame,
                       ReceivedFrame& metadata,
                       int timeout_ms = -1);

    bool send_result(const VisionResult& result);

    const std::string& server_address() const noexcept;
    std::uint16_t server_port() const noexcept;
    const std::string& last_error() const noexcept;

private:
    bool fail_locked(const std::string& message, bool close_socket = true);

    mutable std::mutex mutex_;
    int socket_fd_{-1};
    std::string server_address_;
    std::uint16_t server_port_{0};
    std::string last_error_;
};

// Compatibility name for the shorter class name used by VisionPilot.
using TCPClient = TCPClient;

} // namespace visionpilot::tcp
