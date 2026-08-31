#ifndef VISIONPILOT_FLATBUFFER_CODEC_HPP
#define VISIONPILOT_FLATBUFFER_CODEC_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

#include <common/models.hpp>

namespace visionpilot::tcp {

std::vector<std::uint8_t> encode_result(
    const visionpilot::common::VisionPilotOutput& result
);

bool decode_result(
    const std::uint8_t* data,
    std::size_t size,
    visionpilot::common::VisionPilotOutput& result
);

inline bool decode_result(
    const std::vector<std::uint8_t>& data,
    visionpilot::common::VisionPilotOutput& result)
{
    return decode_result(
        data.data(),
        data.size(),
        result
    );
}

} // namespace visionpilot::tcp

#endif // VISIONPILOT_FLATBUFFER_CODEC_HPP