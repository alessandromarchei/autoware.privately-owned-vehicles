#include <tcp/tcp_frame_client.hpp>

#include <tcp/wire_protocol.hpp>

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
#include <opencv2/imgproc.hpp>

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

TCPClient::TCPClient(
    std::string server_address,
    std::uint16_t frame_port,
    std::uint16_t result_port)
    : server_address_(std::move(server_address)),
      frame_port_(frame_port),
      result_port_(result_port)
{
}

TCPClient::~TCPClient()
{
    disconnect();
}

bool TCPClient::connect_socket_locked(
    int& socket_fd,
    const std::string& address,
    std::uint16_t port,
    int timeout_ms)
{
    close_fd(socket_fd);

    socket_fd = ::socket(AF_INET, SOCK_STREAM, 0);

    if (socket_fd < 0)
    {
        last_error_ =
            "socket: " + std::string(std::strerror(errno));

        return false;
    }

    int enabled = 1;

    ::setsockopt(
        socket_fd,
        IPPROTO_TCP,
        TCP_NODELAY,
        &enabled,
        sizeof(enabled));

    sockaddr_in remote{};
    remote.sin_family = AF_INET;
    remote.sin_port = htons(port);

    if (::inet_pton(
            AF_INET,
            address.c_str(),
            &remote.sin_addr) != 1)
    {
        last_error_ = "invalid IPv4 address: " + address;
        close_fd(socket_fd);
        return false;
    }

    if (!set_blocking(socket_fd, false))
    {
        last_error_ =
            "fcntl: " + std::string(std::strerror(errno));

        close_fd(socket_fd);
        return false;
    }

    const int result = ::connect(
        socket_fd,
        reinterpret_cast<const sockaddr*>(&remote),
        sizeof(remote));

    if (result != 0 && errno != EINPROGRESS)
    {
        last_error_ =
            "connect " +
            address + ":" + std::to_string(port) +
            ": " + std::strerror(errno);

        close_fd(socket_fd);
        return false;
    }

    if (result != 0)
    {
        pollfd descriptor{
            socket_fd,
            POLLOUT,
            0
        };

        const int poll_result =
            ::poll(&descriptor, 1, timeout_ms);

        if (poll_result <= 0)
        {
            last_error_ =
                poll_result == 0
                    ? "connection timeout to " +
                        address + ":" + std::to_string(port)
                    : "poll: " +
                        std::string(std::strerror(errno));

            close_fd(socket_fd);
            return false;
        }

        int socket_error = 0;
        socklen_t length = sizeof(socket_error);

        if (::getsockopt(
                socket_fd,
                SOL_SOCKET,
                SO_ERROR,
                &socket_error,
                &length) != 0 ||
            socket_error != 0)
        {
            const int error =
                socket_error != 0
                    ? socket_error
                    : errno;

            last_error_ =
                "connect " +
                address + ":" + std::to_string(port) +
                ": " + std::strerror(error);

            close_fd(socket_fd);
            return false;
        }
    }

    if (!set_blocking(socket_fd, true))
    {
        last_error_ =
            "fcntl: " + std::string(std::strerror(errno));

        close_fd(socket_fd);
        return false;
    }

    return true;
}

bool TCPClient::connect_to(
    const std::string& address,
    std::uint16_t frame_port,
    std::uint16_t result_port,
    int timeout_ms)
{
    std::scoped_lock lock(
        receive_mutex_,
        send_mutex_,
        state_mutex_);

    close_fd(frame_socket_fd_);
    close_fd(result_socket_fd_);

    server_address_ = address;
    frame_port_ = frame_port;
    result_port_ = result_port;
    last_error_.clear();

    if (!connect_socket_locked(
            frame_socket_fd_,
            address,
            frame_port,
            timeout_ms))
    {
        return false;
    }

    if (!connect_socket_locked(
            result_socket_fd_,
            address,
            result_port,
            timeout_ms))
    {
        close_fd(frame_socket_fd_);
        return false;
    }

    return true;
}

bool TCPClient::reconnect(int timeout_ms)
{
    std::string address;
    std::uint16_t frame_port = 0;
    std::uint16_t result_port = 0;

    {
        std::lock_guard lock(state_mutex_);

        address = server_address_;
        frame_port = frame_port_;
        result_port = result_port_;
    }

    if (address.empty() ||
        frame_port == 0 ||
        result_port == 0)
    {
        return false;
    }

    return connect_to(
        address,
        frame_port,
        result_port,
        timeout_ms);
}

void TCPClient::disconnect()
{
    std::scoped_lock lock(
        receive_mutex_,
        send_mutex_);

    close_fd(frame_socket_fd_);
    close_fd(result_socket_fd_);
}

bool TCPClient::is_connected() const
{
    std::scoped_lock lock(
        receive_mutex_,
        send_mutex_);

    return
        frame_socket_fd_ >= 0 &&
        result_socket_fd_ >= 0;
}

bool TCPClient::receive_frame(
    cv::Mat& frame,
    ReceivedFrame& received_metadata,
    int timeout_ms)
{
    std::lock_guard lock(receive_mutex_);

    // Evita che il chiamante riutilizzi accidentalmente il frame precedente
    // quando la ricezione fallisce.
    frame.release();
    received_metadata = ReceivedFrame{};

    if (frame_socket_fd_ < 0)
    {
        return fail_socket_locked(
            frame_socket_fd_,
            "frame socket is not connected",
            false);
    }

    /*
     * Aspetta che sia disponibile almeno l'inizio di un messaggio.
     *
     * timeout_ms < 0:
     *     attesa bloccante tramite recv_all().
     *
     * timeout_ms >= 0:
     *     poll() attende fino al timeout specificato.
     */
    if (timeout_ms >= 0 &&
        !detail::wait_readable(
            frame_socket_fd_,
            timeout_ms))
    {
        return fail_socket_locked(
            frame_socket_fd_,
            "frame receive timeout",
            false);
    }

    // -------------------------------------------------------------------------
    // Receive and decode MessageHeader
    // -------------------------------------------------------------------------

    std::array<std::uint8_t, WIRE_HEADER_SIZE>
        header_wire{};

    if (!detail::recv_all(
            frame_socket_fd_,
            header_wire.data(),
            header_wire.size()))
    {
        return fail_socket_locked(
            frame_socket_fd_,
            "failed to receive image message header");
    }

    MessageHeader header{};

    if (!detail::decode_header(
            header_wire.data(),
            header))
    {
        return fail_socket_locked(
            frame_socket_fd_,
            "invalid protocol magic or version");
    }

    if (header.type != MessageType::Image)
    {
        return fail_socket_locked(
            frame_socket_fd_,
            "received message is not an Image");
    }

    if (header.payload_size < WIRE_IMAGE_METADATA_SIZE)
    {
        return fail_socket_locked(
            frame_socket_fd_,
            "image payload is smaller than image metadata");
    }

    /*
     * payload_size comprende:
     *
     *     WIRE_IMAGE_METADATA_SIZE + image data
     */
    const std::uint32_t image_bytes_from_header =
        header.payload_size -
        static_cast<std::uint32_t>(
            WIRE_IMAGE_METADATA_SIZE);

    if (image_bytes_from_header == 0 ||
        image_bytes_from_header > MAX_IMAGE_BYTES)
    {
        return fail_socket_locked(
            frame_socket_fd_,
            "invalid image payload size");
    }

    // -------------------------------------------------------------------------
    // Receive and decode ImageMetadata
    // -------------------------------------------------------------------------

    std::array<
        std::uint8_t,
        WIRE_IMAGE_METADATA_SIZE>
        metadata_wire{};

    if (!detail::recv_all(
            frame_socket_fd_,
            metadata_wire.data(),
            metadata_wire.size()))
    {
        return fail_socket_locked(
            frame_socket_fd_,
            "failed to receive image metadata");
    }

    ImageMetadata metadata{};

    if (!detail::decode_image_metadata(
            metadata_wire.data(),
            metadata))
    {
        return fail_socket_locked(
            frame_socket_fd_,
            "failed to decode image metadata");
    }

    if (metadata.data_size != image_bytes_from_header)
    {
        return fail_socket_locked(
            frame_socket_fd_,
            "image data size does not match message payload size");
    }

    if (metadata.data_size == 0 ||
        metadata.data_size > MAX_IMAGE_BYTES)
    {
        return fail_socket_locked(
            frame_socket_fd_,
            "image data size is invalid");
    }

    // -------------------------------------------------------------------------
    // Determine OpenCV image type
    // -------------------------------------------------------------------------

    int cv_type = 0;
    std::uint32_t channels = 0;

    switch (metadata.encoding)
    {
        case ImageEncoding::Bgr8:
        {
            cv_type = CV_8UC3;
            channels = 3;
            break;
        }

        case ImageEncoding::Rgb8:
        {
            cv_type = CV_8UC3;
            channels = 3;
            break;
        }

        case ImageEncoding::Gray8:
        {
            cv_type = CV_8UC1;
            channels = 1;
            break;
        }

        default:
        {
            return fail_socket_locked(
                frame_socket_fd_,
                "unsupported image encoding");
        }
    }

    // -------------------------------------------------------------------------
    // Validate dimensions and stride
    // -------------------------------------------------------------------------

    if (metadata.width == 0 ||
        metadata.height == 0)
    {
        return fail_socket_locked(
            frame_socket_fd_,
            "image width or height is zero");
    }

    /*
     * OpenCV usa int per rows/cols: evita overflow durante il cast.
     */
    if (metadata.width >
            static_cast<std::uint32_t>(
                std::numeric_limits<int>::max()) ||
        metadata.height >
            static_cast<std::uint32_t>(
                std::numeric_limits<int>::max()))
    {
        return fail_socket_locked(
            frame_socket_fd_,
            "image dimensions exceed OpenCV limits");
    }

    const std::uint64_t packed_row_bytes =
        static_cast<std::uint64_t>(metadata.width) *
        channels;

    if (metadata.stride < packed_row_bytes)
    {
        return fail_socket_locked(
            frame_socket_fd_,
            "image stride is smaller than packed row size");
    }

    const std::uint64_t expected_data_size =
        static_cast<std::uint64_t>(metadata.stride) *
        metadata.height;

    if (expected_data_size != metadata.data_size)
    {
        return fail_socket_locked(
            frame_socket_fd_,
            "image stride and height do not match data size");
    }

    if (expected_data_size > MAX_IMAGE_BYTES)
    {
        return fail_socket_locked(
            frame_socket_fd_,
            "image exceeds maximum supported size");
    }

    // -------------------------------------------------------------------------
    // Receive image data
    // -------------------------------------------------------------------------

    std::vector<std::uint8_t> image_buffer(
        metadata.data_size);

    if (!detail::recv_all(
            frame_socket_fd_,
            image_buffer.data(),
            image_buffer.size()))
    {
        return fail_socket_locked(
            frame_socket_fd_,
            "failed to receive image data");
    }

    // -------------------------------------------------------------------------
    // Construct an owning cv::Mat
    // -------------------------------------------------------------------------

    frame.create(
        static_cast<int>(metadata.height),
        static_cast<int>(metadata.width),
        cv_type);

    const std::size_t destination_row_bytes =
        static_cast<std::size_t>(metadata.width) *
        channels;

    /*
     * Copia riga per riga per gestire anche immagini con padding alla fine
     * di ciascuna riga.
     */
    for (std::uint32_t row = 0;
         row < metadata.height;
         ++row)
    {
        const std::uint8_t* source_row =
            image_buffer.data() +
            static_cast<std::size_t>(row) *
                metadata.stride;

        std::uint8_t* destination_row =
            frame.ptr<std::uint8_t>(
                static_cast<int>(row));

        std::memcpy(
            destination_row,
            source_row,
            destination_row_bytes);
    }

    /*
     * Tutto il resto di VisionPilot si aspetta BGR.
     */
    if (metadata.encoding == ImageEncoding::Rgb8)
    {
        cv::cvtColor(
            frame,
            frame,
            cv::COLOR_RGB2BGR);
    }

    // -------------------------------------------------------------------------
    // Return metadata associated with this exact frame
    // -------------------------------------------------------------------------

    received_metadata.frame_id =
        header.sequence;

    received_metadata.timestamp_ns =
        metadata.timestamp_ns;

    received_metadata.vehicle_speed_ms =
        metadata.vehicle_speed_ms;

    {
        std::lock_guard state_lock(state_mutex_);
        last_error_.clear();
    }

    return true;
}

bool TCPClient::send_result(
    const VisionResult& result)
{
    std::lock_guard lock(send_mutex_);

    if (result_socket_fd_ < 0)
    {
        return fail_socket_locked(
            result_socket_fd_,
            "result socket is not connected",
            false);
    }

    const auto payload =
        detail::encode_result(result);

    const MessageHeader header{
        MessageType::Result,
        static_cast<std::uint32_t>(payload.size()),
        result.frame_id
    };

    const auto wire_header =
        detail::encode_header(header);

    if (!detail::send_all(
            result_socket_fd_,
            wire_header.data(),
            wire_header.size()) ||
        !detail::send_all(
            result_socket_fd_,
            payload.data(),
            payload.size()))
    {
        return fail_socket_locked(
            result_socket_fd_,
            "failed to send VisionResult");
    }

    return true;
}


const std::string&
TCPClient::server_address() const noexcept
{
    return server_address_;
}

std::uint16_t
TCPClient::frame_port() const noexcept
{
    return frame_port_;
}

std::uint16_t
TCPClient::result_port() const noexcept
{
    return result_port_;
}

const std::string&
TCPClient::last_error() const noexcept
{
    return last_error_;
}

bool TCPClient::fail_socket_locked(
    int& socket_fd,
    const std::string& message,
    bool close_socket)
{
    {
        std::lock_guard state_lock(state_mutex_);
        last_error_ = message;
    }

    if (close_socket)
        close_fd(socket_fd);

    return false;
}

} // namespace visionpilot::tcp
