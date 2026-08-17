#pragma once

#include <cstddef>
#include <cstdint>

namespace visionpilot::tcp {

inline constexpr std::uint32_t PROTOCOL_MAGIC = 0x56504E54U; // VPNT
inline constexpr std::uint16_t PROTOCOL_VERSION = 1;
inline constexpr std::size_t WIRE_HEADER_SIZE = 20;
inline constexpr std::size_t WIRE_IMAGE_METADATA_SIZE = 32;
inline constexpr std::size_t WIRE_RESULT_SIZE = 52;
inline constexpr std::uint32_t MAX_IMAGE_BYTES = 64U * 1024U * 1024U;

enum class MessageType : std::uint16_t {
    Image = 1,
    Result = 2,
    Ping = 3,
    Pong = 4,
};

enum class ImageEncoding : std::uint16_t {
    Bgr8 = 1,
    Rgb8 = 2,
    Gray8 = 3,
};

struct MessageHeader {
    MessageType type{MessageType::Image};
    std::uint32_t payload_size{0};
    std::uint64_t sequence{0};
};

struct ImageMetadata {
    std::uint64_t timestamp_ns{0};
    std::uint32_t width{0};
    std::uint32_t height{0};
    std::uint32_t stride{0};
    ImageEncoding encoding{ImageEncoding::Bgr8};
    std::uint32_t data_size{0};
    float vehicle_speed_ms{0.0F};
};

struct VisionResult {
    std::uint64_t frame_id{0};
    std::uint64_t timestamp_ns{0};
    float steering_rad{0.0F};
    float acceleration_ms2{0.0F};
    float cte_m{0.0F};
    float yaw_rad{0.0F};
    float curvature_1pm{0.0F};
    float cipo_distance_m{0.0F};
    float cipo_velocity_ms{0.0F};
    float inference_ms{0.0F};
    bool cipo_valid{false};
    bool path_valid{false};
};

struct ReceivedFrame {
    std::uint64_t frame_id{0};
    std::uint64_t timestamp_ns{0};
    float vehicle_speed_ms{0.0F};
};

} // namespace visionpilot::tcp
