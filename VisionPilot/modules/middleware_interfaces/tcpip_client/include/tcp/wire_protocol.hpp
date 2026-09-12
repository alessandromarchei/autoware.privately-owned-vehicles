#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

#include <tcp/protocol.hpp>
#include <common/models.hpp>
#include <common/types.hpp>

#include <algorithm>
#include <cerrno>
#include <cstring>

#include <arpa/inet.h>
#include <poll.h>
#include <sys/socket.h>


#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>

inline void dump_wire(
    const std::string& label,
    const std::uint8_t* data,
    std::size_t size)
{
    std::cerr
        << label
        << " size=" << size
        << " bytes:";

    if (data == nullptr)
    {
        std::cerr << " <null>\n";
        return;
    }

    for (std::size_t i = 0; i < size; ++i)
    {
        std::cerr
            << ' '
            << std::hex
            << std::setw(2)
            << std::setfill('0')
            << static_cast<unsigned int>(data[i]);
    }

    std::cerr
        << std::dec
        << std::setfill(' ')
        << '\n';
}


inline void dump_visionpilot_result(const visionpilot::common::VisionPilotOutput& vpo) {
    const auto& inf = vpo.inference;

    std::cout << "====================================================\n";
    std::cout << "               VISIONPILOT OUTPUT DUMP              \n";
    std::cout << "====================================================\n";

    // ─── INFERENCE METADATA ──────────────────────────────────────────────
    std::cout << " [INFERENCE METRICS]\n"
              << "   Frame ID        : " << inf.frame_id << "\n"
              << "   Total Time       : " << std::fixed << std::setprecision(2) << inf.total_ms << " ms\n"
              << "   Pre-proc Time   : " << inf.pre_ms << " ms\n"
              << "   VisionPilot Time: " << inf.visionpilot_ms << " ms\n\n";

    // ─── AUTODRIVE OUTPUT ────────────────────────────────────────────────
    std::cout << " [AUTODRIVE]\n"
              << "   Valid           : " << (inf.auto_drive.valid ? "true" : "false") << "\n"
              << "   Dist Normalized : " << inf.auto_drive.dist_normalized << "\n"
              << "   Curvature Raw   : " << inf.auto_drive.curvature_raw << "\n"
              << "   Flag Prob (CIPO): " << inf.auto_drive.flag_prob << "\n\n";

    // ─── AUTOSTEER OUTPUT ────────────────────────────────────────────────
    std::cout << " [AUTOSTEER]\n"
              << "   Valid           : " << (inf.auto_steer.valid ? "true" : "false") << "\n"
              << "   Waypoints (xp)  : [";
    for (size_t i = 0; i < inf.auto_steer.xp.size(); ++i) {
        std::cout << inf.auto_steer.xp[i] << (i + 1 < inf.auto_steer.xp.size() ? ", " : "");
    }
    std::cout << "]\n   Confidence (h)  : [";
    for (size_t i = 0; i < inf.auto_steer.h_vector.size(); ++i) {
        std::cout << inf.auto_steer.h_vector[i] << (i + 1 < inf.auto_steer.h_vector.size() ? ", " : "");
    }
    std::cout << "]\n\n";

    // ─── AUTOSPEED OUTPUT ────────────────────────────────────────────────
    std::cout << " [AUTOSPEED]\n"
              << "   Valid           : " << (inf.auto_speed.valid ? "true" : "false") << "\n"
              << "   Detections Count: " << inf.auto_speed.detections.size() << "\n";
    for (size_t i = 0; i < inf.auto_speed.detections.size(); ++i) {
        const auto& d = inf.auto_speed.detections[i];
        std::cout << "     #" << i << " | Class: " << d.class_id
                  << " | Score: " << std::setprecision(3) << d.score
                  << " | BBox: [" << std::setprecision(1) 
                  << d.x1 << ", " << d.y1 << ", " << d.x2 << ", " << d.y2 << "]\n";
    }
    std::cout << "\n";

    // ─── CIPO FUSION ESTIMATE ────────────────────────────────────────────
    std::cout << " [CIPO FUSION]\n"
              << "   Valid           : " << (inf.cipo.valid ? "true" : "false") << "\n"
              << "   Distance        : " << std::setprecision(2) << inf.cipo.distance_m << " m (stddev: " << inf.cipo.distance_stddev_m << " m)\n"
              << "   Velocity        : " << inf.cipo.velocity_ms << " m/s\n"
              << "   CIPO Raw Found  : " << (inf.cipo.cipo_raw_found ? "true" : "false") << "\n"
              << "   CIPO Raw Dist   : " << inf.cipo.cipo_raw_dist_m << " m\n"
              << "   Cut-In Detected : " << (inf.cipo.cut_in_detected ? "true" : "false") << "\n\n";

    // ─── LATERAL FUSION ESTIMATE ─────────────────────────────────────────
    std::cout << " [LATERAL FUSION]\n"
              << "   Valid           : " << (inf.lateral.valid ? "true" : "false") << "\n"
              << "   CTE             : " << inf.lateral.cte_m << " m (stddev: " << inf.lateral.cte_stddev_m << " m, rate: " << inf.lateral.cte_rate_mps << " m/s)\n"
              << "   Yaw             : " << inf.lateral.yaw_rad << " rad (stddev: " << inf.lateral.yaw_stddev_rad << " rad, rate: " << inf.lateral.yaw_rate_rps << " rad/s)\n"
              << "   Curvature       : " << inf.lateral.curvature << " 1/m (stddev: " << inf.lateral.curv_stddev << ")\n"
              << "   RANSAC Path     : valid=" << (inf.lateral.path_valid ? "true" : "false")
              << ", inliers=" << inf.lateral.path_inliers << "/" << inf.lateral.path_points << "\n"
              << "   Polynomial      : y = (" << inf.lateral.path_a << ")x^2 + (" << inf.lateral.path_b << ")x + (" << inf.lateral.path_c << ")\n"
              << "   Path Extent X   : [" << inf.lateral.path_x_min_m << " m, " << inf.lateral.path_x_max_m << " m]\n\n";

    // ─── PLAN ────────────────────────────────────────────────────────────
    std::cout << " [PLAN]\n"
              << "   Acceleration    : " << std::setprecision(2) << vpo.plan.acceleration << " m/s^2\n"
              << "   Steering Points : " << vpo.plan.steering.size() << " [";
    for (size_t i = 0; i < vpo.plan.steering.size(); ++i) {
        std::cout << vpo.plan.steering[i] << (i + 1 < vpo.plan.steering.size() ? ", " : "");
    }
    std::cout << "]\n"
              << "   Warnings Count  : " << vpo.plan.warnings.size() << "\n";

    std::cout << "====================================================\n\n" << std::resetiosflags(std::ios_base::fixed);
}

namespace visionpilot::tcp::detail {


constexpr std::size_t WIRE_RESULT_FIXED_SIZE = 692;
constexpr std::size_t WIRE_DETECTION_SIZE = 24;
constexpr std::size_t WIRE_STEERING_SIZE = 8;
constexpr std::size_t WIRE_WARNING_SIZE = 1;

constexpr std::uint32_t MAX_WIRE_DETECTIONS = 4096;
constexpr std::uint32_t MAX_WIRE_STEERING   = 4096;
constexpr std::uint32_t MAX_WIRE_WARNINGS   = 4096;

constexpr std::size_t MAX_RESULT_BYTES =
    WIRE_RESULT_FIXED_SIZE +
    MAX_WIRE_DETECTIONS * WIRE_DETECTION_SIZE +
    MAX_WIRE_STEERING * WIRE_STEERING_SIZE +
    MAX_WIRE_WARNINGS * WIRE_WARNING_SIZE;
// ============================================================================
// Primitive writer
// ============================================================================

class WireWriter {
public:
    explicit WireWriter(std::size_t reserveSize = 0)
    {
        data_.reserve(reserveSize);
    }

    void writeU8(std::uint8_t value)
    {
        data_.push_back(value);
    }

    void writeBool(bool value)
    {
        writeU8(value ? 1U : 0U);
    }

    void writeU16(std::uint16_t value)
    {
        data_.push_back(static_cast<std::uint8_t>((value >> 8) & 0xFF));
        data_.push_back(static_cast<std::uint8_t>(value & 0xFF));
    }

    void writeU32(std::uint32_t value)
    {
        data_.push_back(static_cast<std::uint8_t>((value >> 24) & 0xFF));
        data_.push_back(static_cast<std::uint8_t>((value >> 16) & 0xFF));
        data_.push_back(static_cast<std::uint8_t>((value >> 8) & 0xFF));
        data_.push_back(static_cast<std::uint8_t>(value & 0xFF));
    }

    void writeI32(std::int32_t value)
    {
        writeU32(static_cast<std::uint32_t>(value));
    }

    void writeU64(std::uint64_t value)
    {
        for (int shift = 56; shift >= 0; shift -= 8) {
            data_.push_back(
                static_cast<std::uint8_t>((value >> shift) & 0xFF));
        }
    }

    void writeF32(float value)
    {
        static_assert(sizeof(float) == sizeof(std::uint32_t));

        std::uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        writeU32(bits);
    }

    void writeF64(double value)
    {
        static_assert(sizeof(double) == sizeof(std::uint64_t));

        std::uint64_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        writeU64(bits);
    }

    [[nodiscard]]
    std::vector<std::uint8_t> take()
    {
        return std::move(data_);
    }

private:
    std::vector<std::uint8_t> data_;
};

// ============================================================================
// Primitive reader
// ============================================================================

class WireReader {
public:

    WireReader(const std::uint8_t* data, std::size_t size)
        : data_(data),
        size_(size)
    {
    }

    template <std::size_t N>
    explicit WireReader(const std::array<std::uint8_t, N>& data)
        : WireReader(data.data(), data.size())
    {
    }

    explicit WireReader(const std::vector<std::uint8_t>& data)
        : WireReader(data.data(), data.size())
    {
    }

    [[nodiscard]]
    bool readU8(std::uint8_t& value)
    {
        if (!require(1))
            return false;

        value = data_[offset_++];
        return true;
    }

    [[nodiscard]]
    bool readBool(bool& value)
    {
        std::uint8_t wireValue = 0;

        if (!readU8(wireValue))
            return false;

        // Only 0 and 1 are accepted as valid wire values.
        if (wireValue > 1)
            return false;

        value = wireValue != 0;
        return true;
    }

    [[nodiscard]]
    bool readU32(std::uint32_t& value)
    {
        if (!require(4))
            return false;

        value =
            (static_cast<std::uint32_t>(data_[offset_]) << 24) |
            (static_cast<std::uint32_t>(data_[offset_ + 1]) << 16) |
            (static_cast<std::uint32_t>(data_[offset_ + 2]) << 8) |
            static_cast<std::uint32_t>(data_[offset_ + 3]);

        offset_ += 4;
        return true;
    }

    [[nodiscard]]
    bool readI32(std::int32_t& value)
    {
        std::uint32_t bits = 0;

        if (!readU32(bits))
            return false;

        static_assert(sizeof(value) == sizeof(bits));
        std::memcpy(&value, &bits, sizeof(value));
        return true;
    }

    [[nodiscard]]
    bool readU64(std::uint64_t& value)
    {
        if (!require(8))
            return false;

        value = 0;

        for (int i = 0; i < 8; ++i)
            value = (value << 8) | data_[offset_ + i];

        offset_ += 8;
        return true;
    }

    [[nodiscard]]
    bool readF32(float& value)
    {
        std::uint32_t bits = 0;

        if (!readU32(bits))
            return false;

        static_assert(sizeof(value) == sizeof(bits));
        std::memcpy(&value, &bits, sizeof(value));
        return true;
    }

    [[nodiscard]]
    bool readF64(double& value)
    {
        std::uint64_t bits = 0;

        if (!readU64(bits))
            return false;

        static_assert(sizeof(value) == sizeof(bits));
        std::memcpy(&value, &bits, sizeof(value));
        return true;
    }

    [[nodiscard]]
    bool finished() const
    {
        return offset_ == size_;
    }

private:
    [[nodiscard]]
    bool require(std::size_t count) const
    {
        return offset_ <= size_ && count <= size_ - offset_;
    }

    const std::uint8_t* data_;
    std::size_t size_;
    std::size_t offset_{0};
};

// ============================================================================
// Header
// ============================================================================

inline std::array<std::uint8_t, WIRE_HEADER_SIZE> encode_header(const MessageHeader& header)
{
    std::array<std::uint8_t, WIRE_HEADER_SIZE> wire{};

    auto writeU16At = [&](std::size_t offset, std::uint16_t value) {
        wire[offset] = static_cast<std::uint8_t>((value >> 8) & 0xFF);
        wire[offset + 1] = static_cast<std::uint8_t>(value & 0xFF);
    };

    auto writeU32At = [&](std::size_t offset, std::uint32_t value) {
        wire[offset] = static_cast<std::uint8_t>((value >> 24) & 0xFF);
        wire[offset + 1] =
            static_cast<std::uint8_t>((value >> 16) & 0xFF);
        wire[offset + 2] =
            static_cast<std::uint8_t>((value >> 8) & 0xFF);
        wire[offset + 3] = static_cast<std::uint8_t>(value & 0xFF);
    };

    auto writeU64At = [&](std::size_t offset, std::uint64_t value) {
        for (int i = 0; i < 8; ++i) {
            wire[offset + i] = static_cast<std::uint8_t>(
                (value >> (56 - i * 8)) & 0xFF);
        }
    };

    writeU32At(0, PROTOCOL_MAGIC);
    writeU16At(4, PROTOCOL_VERSION);
    writeU16At(6, static_cast<std::uint16_t>(header.type));
    writeU32At(8, header.payload_size);
    writeU64At(12, header.sequence);

    return wire;
}

inline bool decode_header(const std::array<std::uint8_t, WIRE_HEADER_SIZE>& wire,MessageHeader& header)
{
    WireReader reader{wire};

    std::uint32_t magic = 0;
    std::uint32_t payloadSize = 0;
    std::uint64_t sequence = 0;
    std::uint32_t typeAndVersion = 0;

    // Mantieni pure la tua decode_header precedente se preferisci.
    const auto readU16At = [&](std::size_t offset) {
        return static_cast<std::uint16_t>(
            (static_cast<std::uint16_t>(wire[offset]) << 8) |
            static_cast<std::uint16_t>(wire[offset + 1]));
    };

    const auto readU32At = [&](std::size_t offset) {
        return
            (static_cast<std::uint32_t>(wire[offset]) << 24) |
            (static_cast<std::uint32_t>(wire[offset + 1]) << 16) |
            (static_cast<std::uint32_t>(wire[offset + 2]) << 8) |
            static_cast<std::uint32_t>(wire[offset + 3]);
    };

    const auto readU64At = [&](std::size_t offset) {
        std::uint64_t value = 0;

        for (int i = 0; i < 8; ++i)
            value = (value << 8) | wire[offset + i];

        return value;
    };

    magic = readU32At(0);

    if (magic != PROTOCOL_MAGIC)
        return false;

    if (readU16At(4) != PROTOCOL_VERSION)
        return false;

    header.type = static_cast<MessageType>(readU16At(6));
    header.payload_size = readU32At(8);
    header.sequence = readU64At(12);

    return true;
}


inline bool decode_image_metadata(
    const std::array<std::uint8_t, WIRE_IMAGE_METADATA_SIZE>& wire,
    ImageMetadata& metadata)
{
    const auto readU16At = [&](std::size_t offset) {
        return static_cast<std::uint16_t>(
            (static_cast<std::uint16_t>(wire[offset]) << 8) |
            static_cast<std::uint16_t>(wire[offset + 1]));
    };

    const auto readU32At = [&](std::size_t offset) {
        return
            (static_cast<std::uint32_t>(wire[offset]) << 24) |
            (static_cast<std::uint32_t>(wire[offset + 1]) << 16) |
            (static_cast<std::uint32_t>(wire[offset + 2]) << 8) |
            static_cast<std::uint32_t>(wire[offset + 3]);
    };

    const auto readU64At = [&](std::size_t offset) {
        std::uint64_t value = 0;

        for (int i = 0; i < 8; ++i) {
            value =
                (value << 8) |
                static_cast<std::uint64_t>(
                    wire[offset + static_cast<std::size_t>(i)]);
        }

        return value;
    };

    metadata.timestamp_ns = readU64At(0);
    metadata.width = readU32At(8);
    metadata.height = readU32At(12);
    metadata.stride = readU32At(16);
    metadata.encoding =
        static_cast<ImageEncoding>(readU16At(20));
    metadata.data_size = readU32At(24);

    const std::uint32_t speedBits = readU32At(28);
    static_assert(
        sizeof(metadata.vehicle_speed_ms) == sizeof(speedBits));
    std::memcpy(
        &metadata.vehicle_speed_ms,
        &speedBits,
        sizeof(metadata.vehicle_speed_ms));

    return true;
}


inline std::array<std::uint8_t, WIRE_IMAGE_METADATA_SIZE> encode_image_metadata(const ImageMetadata& metadata)
{
    std::array<std::uint8_t, WIRE_IMAGE_METADATA_SIZE> wire{};

    auto writeU16At = [&](std::size_t offset, std::uint16_t value)
    {
        wire[offset] =
            static_cast<std::uint8_t>((value >> 8) & 0xFF);

        wire[offset + 1] =
            static_cast<std::uint8_t>(value & 0xFF);
    };

    auto writeU32At = [&](std::size_t offset, std::uint32_t value)
    {
        wire[offset] =
            static_cast<std::uint8_t>((value >> 24) & 0xFF);

        wire[offset + 1] =
            static_cast<std::uint8_t>((value >> 16) & 0xFF);

        wire[offset + 2] =
            static_cast<std::uint8_t>((value >> 8) & 0xFF);

        wire[offset + 3] =
            static_cast<std::uint8_t>(value & 0xFF);
    };

    auto writeU64At = [&](std::size_t offset, std::uint64_t value)
    {
        for (int i = 0; i < 8; ++i)
        {
            wire[offset + static_cast<std::size_t>(i)] =
                static_cast<std::uint8_t>(
                    (value >> (56 - i * 8)) & 0xFF);
        }
    };


    auto writeF32At = [&](std::size_t offset, float value)
    {
        std::uint32_t bits = 0;
        static_assert(sizeof(bits) == sizeof(value));
        std::memcpy(&bits, &value, sizeof(bits));

        writeU32At(offset, bits);
    };

    writeU64At(0, metadata.timestamp_ns);
    writeU32At(8, metadata.width);
    writeU32At(12, metadata.height);
    writeU32At(16, metadata.stride);

    writeU16At(
        20,
        static_cast<std::uint16_t>(metadata.encoding));

    // Bytes 22–23 reserved for future protocol extensions.
    writeU16At(22, 0);

    writeU32At(24, metadata.data_size);
    writeF32At(28, metadata.vehicle_speed_ms);

    return wire;
}


inline bool send_all(
    int socket,
    const void* data,
    std::size_t size)
{
    const auto* current =
        static_cast<const std::uint8_t*>(data);

    while (size > 0)
    {
        const ssize_t sent = ::send(
            socket,
            current,
            size,
            MSG_NOSIGNAL);

        if (sent <= 0)
            return false;

        current += sent;
        size -= static_cast<std::size_t>(sent);
    }

    return true;
}

inline bool recv_all(
    int socket,
    void* data,
    std::size_t size)
{
    auto* current = static_cast<std::uint8_t*>(data);

    while (size > 0)
    {
        const ssize_t received =
            ::recv(socket, current, size, 0);

        if (received <= 0)
            return false;

        current += received;
        size -= static_cast<std::size_t>(received);
    }

    return true;
}

inline bool wait_readable(int fd, int timeout_ms)
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

} // namespace visionpilot::tcp