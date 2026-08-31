#include <common/utils.hpp>

std::string find_config(const std::string& filename)
{
    const std::string local = "config/" + filename;
    const std::string system = "/usr/share/visionpilot/config/" + filename;

    if (std::filesystem::exists(local)) return local;
    if (std::filesystem::exists(system)) return system;

    throw std::runtime_error("Config file not found: " + filename);
}

cv::Mat load_matrix(const std::string& filename, const std::string& matrix)
{
    // const std::string path = find_config(filename);
    const cv::FileStorage fs(filename, cv::FileStorage::READ);

    if (!fs.isOpened())
    {
        throw std::runtime_error("Failed to open calibration file: " + filename);
    }
    cv::Mat M;
    fs[matrix] >> M;

    VP_INFO("Loaded matrix '%s' from: %s", matrix.c_str(), filename.c_str());

    //print matrix values in a more readable format
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    for (int i = 0; i < M.rows; ++i)
    {
        for (int j = 0; j < M.cols; ++j)
        {
            oss << M.at<double>(i, j);
            if (j < M.cols - 1) oss << ", ";
        }
        oss << std::endl;
    }
    VP_INFO("Matrix values:\n%s", oss.str().c_str());

    return M;
}


#include <iomanip>
#include <ostream>

namespace visionpilot::common {

namespace {

template <typename Container>
void dump_container(
    std::ostream& stream,
    const char* name,
    const Container& values)
{
    stream << "    " << name
           << " [" << values.size() << "] = [";

    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            stream << ", ";
        }

        stream << values[i];
    }

    stream << "]\n";
}

}  // namespace

void dump_visionpilot_output(
    const VisionPilotOutput& output,
    std::ostream& stream)
{
    const auto previous_flags = stream.flags();
    const auto previous_precision = stream.precision();

    stream << std::boolalpha
           << std::fixed
           << std::setprecision(6);

    const auto& inference = output.inference;

    stream
        << "\n"
        << "============================================================\n"
        << "VISION PILOT OUTPUT DUMP\n"
        << "============================================================\n";

    // -------------------------------------------------------------------------
    // Frame and latency
    // -------------------------------------------------------------------------

    stream
        << "\n[Frame]\n"
        << "  frame_id: " << inference.frame_id << '\n'
        << "  latency:\n"
        << "    preprocess_ms: " << inference.pre_ms << '\n'
        << "    autodrive_ms:  " << inference.ad_ms << '\n'
        << "    autosteer_ms:  " << inference.as_ms << '\n'
        << "    autospeed_ms:  " << inference.asp_ms << '\n'
        << "    wall_ms:       " << inference.wall_ms << '\n';

    // -------------------------------------------------------------------------
    // AutoDrive
    // -------------------------------------------------------------------------

    const auto& auto_drive = inference.auto_drive;

    stream
        << "\n[AutoDrive]\n"
        << "  valid:           " << auto_drive.valid << '\n'
        << "  dist_normalized: " << auto_drive.dist_normalized << '\n'
        << "  curvature_raw:   " << auto_drive.curvature_raw << '\n'
        << "  flag_prob:       " << auto_drive.flag_prob << '\n';

    // -------------------------------------------------------------------------
    // AutoSteer
    // -------------------------------------------------------------------------

    const auto& auto_steer = inference.auto_steer;

    stream
        << "\n[AutoSteer]\n"
        << "  valid: " << auto_steer.valid << '\n';

    dump_container(stream, "xp", auto_steer.xp);
    dump_container(stream, "h_vector", auto_steer.h_vector);

    // -------------------------------------------------------------------------
    // AutoSpeed
    // -------------------------------------------------------------------------

    const auto& auto_speed = inference.auto_speed;

    stream
        << "\n[AutoSpeed]\n"
        << "  valid:      " << auto_speed.valid << '\n'
        << "  detections: " << auto_speed.detections.size() << '\n';

    for (std::size_t i = 0; i < auto_speed.detections.size(); ++i) {
        const auto& detection = auto_speed.detections[i];

        stream
            << "    detection[" << i << "]:\n"
            << "      class_id: " << detection.class_id << '\n'
            << "      score:    " << detection.score << '\n'
            << "      x1:       " << detection.x1 << '\n'
            << "      y1:       " << detection.y1 << '\n'
            << "      x2:       " << detection.x2 << '\n'
            << "      y2:       " << detection.y2 << '\n'
            << "      width:    "
            << detection.x2 - detection.x1 << '\n'
            << "      height:   "
            << detection.y2 - detection.y1 << '\n';
    }

    // -------------------------------------------------------------------------
    // CIPO fusion
    // -------------------------------------------------------------------------

    const auto& cipo = inference.cipo;

    stream
        << "\n[CIPO Fusion]\n"
        << "  valid:              " << cipo.valid << '\n'
        << "  distance_m:         " << cipo.distance_m << '\n'
        << "  velocity_ms:        " << cipo.velocity_ms << '\n'
        << "  distance_stddev_m:  " << cipo.distance_stddev_m << '\n'
        << "  cipo_raw_found:     " << cipo.cipo_raw_found << '\n'
        << "  cipo_raw_dist_m:    " << cipo.cipo_raw_dist_m << '\n'
        << "  cut_in_detected:    " << cipo.cut_in_detected << '\n';

    // -------------------------------------------------------------------------
    // Lateral fusion
    // -------------------------------------------------------------------------

    const auto& lateral = inference.lateral;

    stream
        << "\n[Lateral Fusion]\n"
        << "  valid:               " << lateral.valid << '\n'
        << "  cte_m:               " << lateral.cte_m << '\n'
        << "  cte_rate_mps:        " << lateral.cte_rate_mps << '\n'
        << "  yaw_rad:             " << lateral.yaw_rad << '\n'
        << "  yaw_rate_rps:        " << lateral.yaw_rate_rps << '\n'
        << "  cte_stddev_m:        " << lateral.cte_stddev_m << '\n'
        << "  yaw_stddev_rad:      " << lateral.yaw_stddev_rad << '\n'
        << "  curvature:           " << lateral.curvature << '\n'
        << "  curv_stddev:         " << lateral.curv_stddev << '\n'
        << "  path_valid:          " << lateral.path_valid << '\n'
        << "  raw_cte_m:           " << lateral.raw_cte_m << '\n'
        << "  raw_yaw_rad:         " << lateral.raw_yaw_rad << '\n'
        << "  raw_path_curvature:  "
        << lateral.raw_path_curvature << '\n'
        << "  raw_ad_curvature:    "
        << lateral.raw_ad_curvature << '\n'
        << "  path_inliers:        " << lateral.path_inliers << '\n'
        << "  path_points:         " << lateral.path_points << '\n'
        << "  path_a:              " << lateral.path_a << '\n'
        << "  path_b:              " << lateral.path_b << '\n'
        << "  path_c:              " << lateral.path_c << '\n'
        << "  path_x_min_m:        " << lateral.path_x_min_m << '\n'
        << "  path_x_max_m:        " << lateral.path_x_max_m << '\n';

    // -------------------------------------------------------------------------
    // Planner output
    // -------------------------------------------------------------------------

    const auto& plan = output.plan;

    stream
        << "\n[Plan]\n"
        << "  acceleration: " << plan.acceleration << '\n'
        << "  steering points: " << plan.steering.size() << '\n';

    dump_container(stream, "steering", plan.steering);

    // Warning cannot be dumped field-by-field without knowing its definition.
    stream
        << "  warnings: " << plan.warnings.size() << '\n';

    stream
        << "\n============================================================\n"
        << "END VISION PILOT OUTPUT DUMP\n"
        << "============================================================\n"
        << std::flush;

    stream.flags(previous_flags);
    stream.precision(previous_precision);
}

}  // namespace visionpilot::common