#include <tcp/tcp_frame_client.hpp>

#include "wire_protocol.hpp"

#include <array>
#include <cerrno>
#include <cstring>
#include <limits>
#include <vector>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <opencv2/core.hpp>

namespace visionpilot::tcp {
namespace {

void close_fd(int& fd)
{
    if (fd >= 0) {
        ::shutdown(fd, SHUT_RDWR);
        ::close(fd);
        fd = -1;
    }
}

bool set_blocking(int fd, bool blocking)
{
    const int flags = ::fcntl(fd, F_GETFL, 0);
    if (flags < 0)
        return false;
    const int next = blocking ? (flags & ~O_NONBLOCK) : (flags | O_NONBLOCK);
    return ::fcntl(fd, F_SETFL, next) == 0;
}

} // namespace

TCPClient::TCPClient() = default;

TCPClient::TCPClient(std::string server_address,
                               std::uint16_t server_port)
    : server_address_(std::move(server_address)), server_port_(server_port)
{
}

TCPClient::~TCPClient()
{
    disconnect();
}

bool TCPClient::connect_to(const std::string& address,
                                std::uint16_t port,
                                int timeout_ms)
{
    std::unique_lock<std::mutex> lock(mutex_);
    close_fd(socket_fd_);
    server_address_ = address;
    server_port_ = port;
    last_error_.clear();

    socket_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (socket_fd_ < 0)
        return fail_locked("socket: " + std::string(std::strerror(errno)));

    int enabled = 1;
    ::setsockopt(socket_fd_, IPPROTO_TCP, TCP_NODELAY, &enabled, sizeof(enabled));

    sockaddr_in remote{};
    remote.sin_family = AF_INET;
    remote.sin_port = htons(port);
    if (::inet_pton(AF_INET, address.c_str(), &remote.sin_addr) != 1)
        return fail_locked("invalid IPv4 address: " + address);

    if (!set_blocking(socket_fd_, false))
        return fail_locked("fcntl: " + std::string(std::strerror(errno)));

    const int result = ::connect(socket_fd_,
        reinterpret_cast<const sockaddr*>(&remote), sizeof(remote));
    if (result != 0 && errno != EINPROGRESS)
        return fail_locked("connect: " + std::string(std::strerror(errno)));

    if (result != 0) {
        pollfd descriptor{socket_fd_, POLLOUT, 0};
        const int poll_result = ::poll(&descriptor, 1, timeout_ms);
        if (poll_result <= 0)
            return fail_locked(poll_result == 0 ? "connect timeout" :
                "poll: " + std::string(std::strerror(errno)));
        int socket_error = 0;
        socklen_t length = sizeof(socket_error);
        if (::getsockopt(socket_fd_, SOL_SOCKET, SO_ERROR,
                         &socket_error, &length) != 0 || socket_error != 0) {
            const int error = socket_error != 0 ? socket_error : errno;
            return fail_locked("connect: " + std::string(std::strerror(error)));
        }
    }

    if (!set_blocking(socket_fd_, true))
        return fail_locked("fcntl: " + std::string(std::strerror(errno)));
    return true;
}

bool TCPClient::reconnect(int timeout_ms)
{
    std::string address;
    std::uint16_t port;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        address = server_address_;
        port = server_port_;
    }
    if (address.empty() || port == 0)
        return false;
    return connect_to(address, port, timeout_ms);
}

void TCPClient::disconnect()
{
    std::lock_guard<std::mutex> lock(mutex_);
    close_fd(socket_fd_);
}

bool TCPClient::is_connected() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return socket_fd_ >= 0;
}

bool TCPClient::receive_frame(cv::Mat& frame,
                                   ReceivedFrame& received,
                                   int timeout_ms)
{
    std::unique_lock<std::mutex> lock(mutex_);
    if (socket_fd_ < 0)
        return fail_locked("not connected", false);
    if (timeout_ms >= 0 && !detail::wait_readable(socket_fd_, timeout_ms))
        return fail_locked("receive timeout", false);

    std::array<std::uint8_t, WIRE_HEADER_SIZE> wire_header{};
    if (!detail::recv_all(socket_fd_, wire_header.data(), wire_header.size()))
        return fail_locked("connection closed while receiving header");

    MessageHeader header;
    if (!detail::decode_header(wire_header.data(), header))
        return fail_locked("invalid protocol header");

    if (header.type == MessageType::Ping) {
        if (header.payload_size > 0 &&
            !detail::discard_bytes(socket_fd_, header.payload_size))
            return fail_locked("connection closed while receiving ping");
        const MessageHeader pong{MessageType::Pong, 0, header.sequence};
        const auto wire_pong = detail::encode_header(pong);
        if (!detail::send_all(socket_fd_, wire_pong.data(), wire_pong.size()))
            return fail_locked("failed to send pong");
        lock.unlock();
        return receive_frame(frame, received, timeout_ms);
    }

    if (header.type != MessageType::Image)
        return fail_locked("expected Image message");
    if (header.payload_size < WIRE_IMAGE_METADATA_SIZE ||
        header.payload_size > WIRE_IMAGE_METADATA_SIZE + MAX_IMAGE_BYTES)
        return fail_locked("invalid image payload size");

    std::array<std::uint8_t, WIRE_IMAGE_METADATA_SIZE> wire_metadata{};
    if (!detail::recv_all(socket_fd_, wire_metadata.data(), wire_metadata.size()))
        return fail_locked("connection closed while receiving metadata");

    ImageMetadata metadata;
    detail::decode_image_metadata(wire_metadata.data(), metadata);
    if (metadata.width == 0 || metadata.height == 0 ||
        metadata.data_size == 0 || metadata.data_size > MAX_IMAGE_BYTES ||
        header.payload_size != WIRE_IMAGE_METADATA_SIZE + metadata.data_size)
        return fail_locked("invalid image metadata");

    int cv_type = -1;
    std::uint32_t channels = 0;
    if (metadata.encoding == ImageEncoding::Bgr8 ||
        metadata.encoding == ImageEncoding::Rgb8) {
        cv_type = CV_8UC3;
        channels = 3;
    } else if (metadata.encoding == ImageEncoding::Gray8) {
        cv_type = CV_8UC1;
        channels = 1;
    } else {
        return fail_locked("unsupported image encoding");
    }

    const std::uint64_t minimum_stride =
        static_cast<std::uint64_t>(metadata.width) * channels;
    const std::uint64_t expected_size =
        static_cast<std::uint64_t>(metadata.stride) * metadata.height;
    if (metadata.stride < minimum_stride || expected_size != metadata.data_size)
        return fail_locked("invalid image stride or data size");

    std::vector<std::uint8_t> payload(metadata.data_size);
    if (!detail::recv_all(socket_fd_, payload.data(), payload.size()))
        return fail_locked("connection closed while receiving image");

    cv::Mat view(static_cast<int>(metadata.height),
                 static_cast<int>(metadata.width), cv_type,
                 payload.data(), metadata.stride);
    frame = view.clone();
    received.frame_id = header.sequence;
    received.timestamp_ns = metadata.timestamp_ns;
    last_error_.clear();
    return true;
}

bool TCPClient::send_result(const VisionResult& result)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (socket_fd_ < 0)
        return fail_locked("not connected", false);
    const auto payload = detail::encode_result(result);
    const MessageHeader header{MessageType::Result,
        static_cast<std::uint32_t>(payload.size()), result.frame_id};
    const auto wire_header = detail::encode_header(header);
    if (!detail::send_all(socket_fd_, wire_header.data(), wire_header.size()) ||
        !detail::send_all(socket_fd_, payload.data(), payload.size()))
        return fail_locked("failed to send result");
    last_error_.clear();
    return true;
}

const std::string& TCPClient::server_address() const noexcept
{
    return server_address_;
}

std::uint16_t TCPClient::server_port() const noexcept
{
    return server_port_;
}

const std::string& TCPClient::last_error() const noexcept
{
    return last_error_;
}

bool TCPClient::fail_locked(const std::string& message, bool close_socket)
{
    last_error_ = message;
    if (close_socket)
        close_fd(socket_fd_);
    return false;
}

} // namespace visionpilot::tcp
