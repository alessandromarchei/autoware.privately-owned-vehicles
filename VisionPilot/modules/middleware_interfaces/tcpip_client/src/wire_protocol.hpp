#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include <tcp/protocol.hpp>

namespace visionpilot::tcp::detail {

bool send_all(int fd, const void* data, std::size_t size);
bool recv_all(int fd, void* data, std::size_t size);
bool wait_readable(int fd, int timeout_ms);

std::array<std::uint8_t, WIRE_HEADER_SIZE>
encode_header(const MessageHeader& header);
bool decode_header(const std::uint8_t* wire, MessageHeader& header);

bool decode_image_metadata(const std::uint8_t* wire, ImageMetadata& metadata);

std::array<std::uint8_t, WIRE_RESULT_SIZE>
encode_result(const VisionResult& result);

bool discard_bytes(int fd, std::uint32_t byte_count);

} // namespace visionpilot::tcp::detail
