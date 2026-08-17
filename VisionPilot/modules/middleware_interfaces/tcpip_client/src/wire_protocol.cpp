#include "wire_protocol.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>

#include <arpa/inet.h>
#include <poll.h>
#include <sys/socket.h>

namespace visionpilot::tcp::detail {
namespace {

std::uint64_t swap_u64(std::uint64_t value)
{
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    return __builtin_bswap64(value);
#else
    return value;
#endif
}

void put_u16(std::uint8_t* out, std::uint16_t value)
{
    value = htons(value);
    std::memcpy(out, &value, sizeof(value));
}

void put_u32(std::uint8_t* out, std::uint32_t value)
{
    value = htonl(value);
    std::memcpy(out, &value, sizeof(value));
}

void put_u64(std::uint8_t* out, std::uint64_t value)
{
    value = swap_u64(value);
    std::memcpy(out, &value, sizeof(value));
}

std::uint16_t get_u16(const std::uint8_t* in)
{
    std::uint16_t value;
    std::memcpy(&value, in, sizeof(value));
    return ntohs(value);
}

std::uint32_t get_u32(const std::uint8_t* in)
{
    std::uint32_t value;
    std::memcpy(&value, in, sizeof(value));
    return ntohl(value);
}

std::uint64_t get_u64(const std::uint8_t* in)
{
    std::uint64_t value;
    std::memcpy(&value, in, sizeof(value));
    return swap_u64(value);
}

void put_f32(std::uint8_t* out, float value)
{
    std::uint32_t bits;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    put_u32(out, bits);
}

} // namespace

bool send_all(int fd, const void* data, std::size_t size)
{
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    std::size_t sent = 0;
    while (sent < size) {
        const ssize_t result = ::send(fd, bytes + sent, size - sent, MSG_NOSIGNAL);
        if (result > 0) {
            sent += static_cast<std::size_t>(result);
        } else if (result < 0 && errno == EINTR) {
            continue;
        } else {
            return false;
        }
    }
    return true;
}

bool recv_all(int fd, void* data, std::size_t size)
{
    auto* bytes = static_cast<std::uint8_t*>(data);
    std::size_t received = 0;
    while (received < size) {
        const ssize_t result = ::recv(fd, bytes + received, size - received, 0);
        if (result > 0) {
            received += static_cast<std::size_t>(result);
        } else if (result < 0 && errno == EINTR) {
            continue;
        } else {
            return false;
        }
    }
    return true;
}

bool wait_readable(int fd, int timeout_ms)
{
    pollfd descriptor{fd, POLLIN, 0};
    while (true) {
        const int result = ::poll(&descriptor, 1, timeout_ms);
        if (result > 0)
            return (descriptor.revents & POLLIN) != 0;
        if (result < 0 && errno == EINTR)
            continue;
        return false;
    }
}

std::array<std::uint8_t, WIRE_HEADER_SIZE>
encode_header(const MessageHeader& header)
{
    std::array<std::uint8_t, WIRE_HEADER_SIZE> wire{};
    put_u32(wire.data(), PROTOCOL_MAGIC);
    put_u16(wire.data() + 4, PROTOCOL_VERSION);
    put_u16(wire.data() + 6, static_cast<std::uint16_t>(header.type));
    put_u32(wire.data() + 8, header.payload_size);
    put_u64(wire.data() + 12, header.sequence);
    return wire;
}

bool decode_header(const std::uint8_t* wire, MessageHeader& header)
{
    if (get_u32(wire) != PROTOCOL_MAGIC ||
        get_u16(wire + 4) != PROTOCOL_VERSION)
        return false;
    header.type = static_cast<MessageType>(get_u16(wire + 6));
    header.payload_size = get_u32(wire + 8);
    header.sequence = get_u64(wire + 12);
    return true;
}

bool decode_image_metadata(const std::uint8_t* wire, ImageMetadata& metadata)
{
    metadata.timestamp_ns = get_u64(wire);
    metadata.width = get_u32(wire + 8);
    metadata.height = get_u32(wire + 12);
    metadata.stride = get_u32(wire + 16);
    metadata.encoding = static_cast<ImageEncoding>(get_u16(wire + 20));
    metadata.data_size = get_u32(wire + 24);
    return true;
}

std::array<std::uint8_t, WIRE_RESULT_SIZE>
encode_result(const VisionResult& result)
{
    std::array<std::uint8_t, WIRE_RESULT_SIZE> wire{};
    put_u64(wire.data(), result.timestamp_ns);
    put_f32(wire.data() + 8, result.steering_rad);
    put_f32(wire.data() + 12, result.acceleration_ms2);
    put_f32(wire.data() + 16, result.cte_m);
    put_f32(wire.data() + 20, result.yaw_rad);
    put_f32(wire.data() + 24, result.curvature_1pm);
    put_f32(wire.data() + 28, result.cipo_distance_m);
    put_f32(wire.data() + 32, result.cipo_velocity_ms);
    put_f32(wire.data() + 36, result.inference_ms);
    std::uint32_t flags = 0;
    if (result.cipo_valid) flags |= 1U;
    if (result.path_valid) flags |= 2U;
    put_u32(wire.data() + 40, flags);
    return wire;
}

bool discard_bytes(int fd, std::uint32_t byte_count)
{
    std::array<std::uint8_t, 4096> scratch{};
    while (byte_count > 0) {
        const auto amount = std::min<std::size_t>(byte_count, scratch.size());
        if (!recv_all(fd, scratch.data(), amount))
            return false;
        byte_count -= static_cast<std::uint32_t>(amount);
    }
    return true;
}

} // namespace visionpilot::tcp::detail
