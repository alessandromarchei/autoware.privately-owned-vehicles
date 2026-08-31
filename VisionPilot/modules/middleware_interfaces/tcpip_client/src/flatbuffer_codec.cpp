#include <tcp/flatbuffer_codec.hpp>

#include <flatbuffers/flatbuffers.h>

#include "visionpilot_generated.h"

namespace visionpilot::tcp {

std::vector<std::uint8_t> encode_result(
    const visionpilot::common::VisionPilotOutput& result)
{
    flatbuffers::FlatBufferBuilder builder{1024};

    const auto& inference = result.inference;

    /*
     * AutoDrive
     */
    const auto autoDrive =
        wire::CreateAutoDriveOutput(
            builder,
            inference.auto_drive.dist_normalized,
            inference.auto_drive.curvature_raw,
            inference.auto_drive.flag_prob,
            inference.auto_drive.valid
        );

    /*
     * AutoSteer
     */
    const auto xp = builder.CreateVector(
        inference.auto_steer.xp.data(),
        inference.auto_steer.xp.size()
    );

    const auto hVector = builder.CreateVector(
        inference.auto_steer.h_vector.data(),
        inference.auto_steer.h_vector.size()
    );

    const auto autoSteer =
        wire::CreateAutoSteerOutput(
            builder,
            xp,
            hVector,
            inference.auto_steer.valid
        );

    /*
     * AutoSpeed detections
     */
    std::vector<wire::Detection> detections;

    detections.reserve(
        inference.auto_speed.detections.size()
    );

    for (const visionpilot::common::Detection& detection :
         inference.auto_speed.detections)
    {
        detections.emplace_back(
            detection.x1,
            detection.y1,
            detection.x2,
            detection.y2,
            detection.score,
            detection.class_id
        );
    }

    const auto detectionVector =
        builder.CreateVectorOfStructs(detections);

    const auto autoSpeed =
        wire::CreateAutoSpeedOutput(
            builder,
            detectionVector,
            inference.auto_speed.valid
        );

    /*
     * CIPO fusion
     */
    const auto& sourceCipo = inference.cipo;

    const auto cipo =
        wire::CreateCIPOFusionEstimate(
            builder,
            sourceCipo.valid,
            sourceCipo.distance_m,
            sourceCipo.velocity_ms,
            sourceCipo.distance_stddev_m,
            sourceCipo.cipo_raw_found,
            sourceCipo.cipo_raw_dist_m,
            sourceCipo.cut_in_detected
        );

    /*
     * Lateral fusion
     */
    const auto& sourceLateral = inference.lateral;

    const auto lateral =
        wire::CreateLateralFusionEstimate(
            builder,
            sourceLateral.valid,
            sourceLateral.cte_m,
            sourceLateral.cte_rate_mps,
            sourceLateral.yaw_rad,
            sourceLateral.yaw_rate_rps,
            sourceLateral.cte_stddev_m,
            sourceLateral.yaw_stddev_rad,
            sourceLateral.curvature,
            sourceLateral.curv_stddev,
            sourceLateral.path_valid,
            sourceLateral.raw_cte_m,
            sourceLateral.raw_yaw_rad,
            sourceLateral.raw_path_curvature,
            sourceLateral.raw_ad_curvature,
            sourceLateral.path_inliers,
            sourceLateral.path_points,
            sourceLateral.path_a,
            sourceLateral.path_b,
            sourceLateral.path_c,
            sourceLateral.path_x_min_m,
            sourceLateral.path_x_max_m
        );

    /*
     * Complete inference result
     */
    const auto wireInference =
        wire::CreateInferenceFrameResult(
            builder,
            inference.frame_id,
            inference.wall_ms,
            inference.pre_ms,
            inference.ad_ms,
            inference.as_ms,
            inference.asp_ms,
            autoDrive,
            autoSteer,
            autoSpeed,
            cipo,
            lateral
        );

    /*
     * Plan
     */
    const auto steering =
        builder.CreateVector(result.plan.steering);

    std::vector<wire::Warning> warnings;
    warnings.reserve(result.plan.warnings.size());

    for (const Warning warning :
         result.plan.warnings)
    {
        warnings.push_back(
            static_cast<wire::Warning>(warning)
        );
    }

    const auto warningVector =
        builder.CreateVector(warnings);

    const auto plan =
        wire::CreatePlan(
            builder,
            result.plan.acceleration,
            steering,
            warningVector
        );

    /*
     * Root object
     */
    const auto output =
        wire::CreateVisionPilotOutput(
            builder,
            wireInference,
            plan
        );

    wire::FinishVisionPilotOutputBuffer(
        builder,
        output
    );

    const std::uint8_t* begin =
        builder.GetBufferPointer();

    const std::uint8_t* end =
        begin + builder.GetSize();

    return std::vector<std::uint8_t>(begin, end);
}



bool decode_result(
    const std::uint8_t* data,
    std::size_t size,
    visionpilot::common::VisionPilotOutput& result)
{
    if (data == nullptr || size == 0) {
        return false;
    }

    flatbuffers::Verifier verifier(data, size);

    if (!wire::VerifyVisionPilotOutputBuffer(verifier)) {
        return false;
    }

    const wire::VisionPilotOutput* message =
        wire::GetVisionPilotOutput(data);

    if (message == nullptr ||
        message->inference() == nullptr ||
        message->plan() == nullptr)
    {
        return false;
    }

    const auto* sourceInference =
        message->inference();

    /*
     * Prima decodifichiamo in un oggetto temporaneo.
     * result viene modificato solo se tutto è valido.
     */
    visionpilot::common::VisionPilotOutput decoded{};

    decoded.inference.frame_id =
        sourceInference->frame_id();

    decoded.inference.wall_ms =
        sourceInference->wall_ms();

    decoded.inference.pre_ms =
        sourceInference->pre_ms();

    decoded.inference.ad_ms =
        sourceInference->ad_ms();

    decoded.inference.as_ms =
        sourceInference->as_ms();

    decoded.inference.asp_ms =
        sourceInference->asp_ms();

    /*
     * AutoDrive
     */
    if (const auto* source =
            sourceInference->auto_drive())
    {
        decoded.inference.auto_drive.dist_normalized =
            source->dist_normalized();

        decoded.inference.auto_drive.curvature_raw =
            source->curvature_raw();

        decoded.inference.auto_drive.flag_prob =
            source->flag_prob();

        decoded.inference.auto_drive.valid =
            source->valid();
    }

    /*
     * AutoSteer
     */
    if (const auto* source =
            sourceInference->auto_steer())
    {
        const auto* xp = source->xp();
        const auto* hVector = source->h_vector();

        if (xp == nullptr ||
            hVector == nullptr ||
            xp->size() != 64 ||
            hVector->size() != 64)
        {
            return false;
        }

        for (std::size_t i = 0; i < 64; ++i) {
            decoded.inference.auto_steer.xp[i] =
                xp->Get((unsigned int) i);

            decoded.inference.auto_steer.h_vector[i] =
                hVector->Get((unsigned int) i);
        }

        decoded.inference.auto_steer.valid =
            source->valid();
    }

    /*
     * AutoSpeed
     */
    if (const auto* source =
            sourceInference->auto_speed())
    {
        decoded.inference.auto_speed.valid =
            source->valid();

        if (const auto* sourceDetections =
                source->detections())
        {
            constexpr std::size_t MAX_DETECTIONS = 4096;

            if (sourceDetections->size() >
                MAX_DETECTIONS)
            {
                return false;
            }

            auto& destination =
                decoded.inference.auto_speed.detections;

            destination.reserve(
                sourceDetections->size()
            );

            for (const wire::Detection* detection :
                 *sourceDetections)
            {
                destination.push_back(
                    visionpilot::common::Detection{
                        detection->x1(),
                        detection->y1(),
                        detection->x2(),
                        detection->y2(),
                        detection->score(),
                        detection->class_id()
                    }
                );
            }
        }
    }

    /*
     * CIPO
     */
    if (const auto* source =
            sourceInference->cipo())
    {
        auto& destination = decoded.inference.cipo;

        destination.valid =
            source->valid();

        destination.distance_m =
            source->distance_m();

        destination.velocity_ms =
            source->velocity_ms();

        destination.distance_stddev_m =
            source->distance_stddev_m();

        destination.cipo_raw_found =
            source->cipo_raw_found();

        destination.cipo_raw_dist_m =
            source->cipo_raw_dist_m();

        destination.cut_in_detected =
            source->cut_in_detected();
    }

    /*
     * Copia analoga per LateralFusionEstimate.
     */

    /*
     * Plan
     */
    const auto* sourcePlan = message->plan();

    decoded.plan.acceleration =
        sourcePlan->acceleration();

    if (const auto* steering =
            sourcePlan->steering())
    {
        constexpr std::size_t MAX_STEERING = 4096;

        if (steering->size() > MAX_STEERING) {
            return false;
        }

        decoded.plan.steering.assign(
            steering->begin(),
            steering->end()
        );
    }

    if (const auto* warnings =
            sourcePlan->warnings())
    {
        constexpr std::size_t MAX_WARNINGS = 4096;

        if (warnings->size() > MAX_WARNINGS) {
            return false;
        }

        decoded.plan.warnings.reserve(
            warnings->size()
        );

        for (const wire::Warning warning : *warnings) {
            decoded.plan.warnings.push_back(
                static_cast<Warning>(warning)
            );
        }
    }

    result = std::move(decoded);
    return true;
}


} // namespace visionpilot::tcp