// VisionPilot — preprocess → inference → fusion → display
#include <chrono>
#include <memory>
#include <string>
#include <thread>

#include <config/vision_pilot_config.hpp>
#include <common/utils.hpp>
#include <engine/v4m_engine.hpp>
#include <vehicle_interface/vehicle_interface.hpp>
#include <vehicle_interface/can_interface.hpp>
#include <image_preprocessing/image_preprocessor.hpp>
#include <logging/logger.hpp>
#include <models/inference.hpp>
#include <planning/planning.hpp>
#include <debug/debug_draw.hpp>

#include <camera_interface/frames_interface.hpp>

#include "camera_interface/v4l2_camera_interface.hpp"
#include "camera_interface/file_interface.hpp"
#include "vehicle_interface/file_interface.hpp"


namespace ve = visionpilot::engine;
namespace vm = visionpilot::models;
namespace vd = visionpilot::debug;

int main(int argc, char** argv)
{
    Config cfg;

    // Default configuration file
    std::string config_path = "share/config/vision_pilot.conf";
    std::string homography_path = "share/config/H.yaml";
    std::string config_path_test = "share/config/vision_pilot_test.conf";

    std::string test_video_path;
    std::string test_vehicle_speed_path;
    std::string test_frames_path;

    SourceMode source_mode = SourceMode::Frames;

    // CLI flags
    bool debug_viz = false;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg(argv[i]);

        if (arg == "--debug-viz")
        {
            debug_viz = true;
        }
        else if (arg == "--config")
        {
            if (i + 1 >= argc)
            {
                VP_ERROR("Missing argument after --config");
                return 1;
            }

            config_path = argv[++i];
        }
        else if (arg == "--config-test")
        {
            if (i + 1 >= argc)
            {
                VP_ERROR("Missing argument after --config-test");
                return 1;
            }

            config_path_test = argv[++i];
        }
        else if (arg == "--H")
        {
            if (i + 1 >= argc)
            {
                VP_ERROR("Missing argument after --H. Specify path to homography YAML file.");
                return 1;
            }
            homography_path = argv[++i];
        }
        else if (arg == "--test-video")
        {
            if (i + 1 >= argc)
            {
                VP_ERROR("Missing argument after --test-video. Specify path to test video file.");
                return 1;
            }
            test_video_path = argv[++i];
        }
        else if (arg == "--test-vehicle-speed")
        {
            if (i + 1 >= argc)
            {
                VP_ERROR("Missing argument after --test-vehicle-speed. Specify path to test vehicle speed file.");
                return 1;
            }
            test_vehicle_speed_path = argv[++i];
        }
        else if (arg == "--test-frames")
        {
            if (i + 1 >= argc)
            {
                VP_ERROR(
                    "Missing argument after --test-frames. "
                    "Specify a directory containing extracted frames.");

                return 1;
            }

            test_frames_path = argv[++i];
        }
        else if (arg == "--source-mode")
        {
            if (i + 1 >= argc)
            {
                VP_ERROR("Missing argument after --source-mode. Specify source mode: video|ros2|v4l2|frames");
                return 1;
            }

            const std::string mode_str = argv[++i];
            try
            {
                source_mode = parse_source_mode(mode_str);
            }
            catch (const std::exception& e)
            {
                VP_ERROR("Invalid source mode: %s", e.what());
                return 1;
            }
        }
        else
        {
            VP_ERROR("Unknown argument: %s", arg.c_str());
            return 1;
        }
    }

    try
    {
        cfg = load_vision_pilot_config(config_path, config_path_test);
    }
    catch (const std::exception& e)
    {
        VP_ERROR("Config: %s", e.what());
        return 1;
    }   

    //apply changes from CLI flags to config
    if (!test_video_path.empty())
    {
        cfg.source.input_video = test_video_path;
        VP_INFO("Using test video: %s", test_video_path.c_str());
    }
    if (!test_vehicle_speed_path.empty())
    {
        cfg.source.input_vehicle_speed = test_vehicle_speed_path;
        VP_INFO("Using test vehicle speed file: %s", test_vehicle_speed_path.c_str());
    }
    if (!test_frames_path.empty())
    {
        cfg.source.test_frames_path = test_frames_path;
        VP_INFO("Using test raw frames: %s", test_frames_path.c_str());
    }

    //apply source mode to config
    cfg.source.mode = source_mode;

    //check mutually exclusive flags
    if (!test_video_path.empty() && !test_frames_path.empty())
    {
        VP_ERROR(
            "--test-video and --test-frames are mutually exclusive");

        return 1;
    }

    std::shared_ptr<CameraInterface> camera_interface;
    std::shared_ptr<VehicleInterface> vehicle_interface;

    ImagePreprocessor preprocessor;


    if (cfg.source.mode == SourceMode::Frames)
    {
        constexpr double TEST_VIDEO_FPS = 30.0;

        VP_INFO("Using extracted frames source mode");
        VP_INFO("Input frames directory: %s", test_frames_path.c_str());

        VP_INFO("Input vehicle speed: %s", cfg.source.input_vehicle_speed.c_str());

        camera_interface = std::make_shared<camera_interface::FramesInterface>(
                test_frames_path,
                cfg.source.video_loop,
                cfg.source.video_realtime,
                TEST_VIDEO_FPS);

        vehicle_interface = std::make_shared<FileInterface>(cfg.source.input_vehicle_speed);
    }
    else if (cfg.source.mode == SourceMode::Video)
    {
        VP_INFO("Using video source mode");
        VP_INFO(
            "Input video: %s",
            cfg.source.input_video.c_str());

        VP_INFO(
            "Input vehicle speed: %s",
            cfg.source.input_vehicle_speed.c_str());

        camera_interface =
            std::make_shared<camera_interface::FileInterface>(
                cfg.source.input_video,
                cfg.source.video_loop,
                cfg.source.video_realtime);

        vehicle_interface =
            std::make_shared<FileInterface>(
                cfg.source.input_vehicle_speed);
    }
    else
    {
        VP_INFO("Using live source mode: %s", source_label(cfg.source).c_str());

        VP_INFO("V4L2 device: %s", cfg.source.v4l2_device.c_str());

        VP_INFO("V4L2 FPS: %d", cfg.source.v4l2_fps);

        camera_interface = std::make_shared<camera_interface::V4L2CameraInterface>(
                    cfg.source.v4l2_device,
                    static_cast<uint32_t>(
                        cfg.source.v4l2_fps));

        vehicle_interface =
            std::make_shared<CanInterface>();
    }

    VP_INFO("Starting Inference Pipeline ...");
    //initialize inference pipeline (internally every hycoah for each model is initialized)
    vm::InferencePipeline pipeline(cfg.inference);

    VP_INFO("Starting Planner with speed limit: %.2f m/s and Lf: %.2f m", cfg.speed_limit, cfg.Lf);
    Planner planner(cfg.speed_limit, cfg.Lf);

    // ── Init visualization assets once based on mode ──────────────────────────
    if (debug_viz)
    {
        VP_INFO("[Viz] Debug mode — annotated telemetry overlay");
        vd::init_wheel_assets(cfg.wheel_dir);
        vd::init_homography();
    }
    else
    {
        VP_INFO("[Viz] Production mode — clean HUD");
    }

    // ── Initialize camera interface ───────────────────────────────────────────

    if (!camera_interface || !camera_interface->is_device_open())
    {
        VP_ERROR("Cannot open frame source");
        return 1;
    }


    const cv::Size net_size(vm::AutoDrive::NET_W, vm::AutoDrive::NET_H);
    cv::Mat frame, warped, resized;
    bool h_resized_set = false;
    cv::Mat H = load_matrix(homography_path, "H");

    VP_INFO("Starting main loop. Press Ctrl+C to exit.");

    while (true)
    {
        auto [ok, frame] = camera_interface->get_latest_frame();
        if (!ok || frame.empty())
        {
            if (cfg.source.mode == SourceMode::Video &&
                !cfg.source.video_loop) {
                break;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(5));

            continue;
        }

        preprocessor.preprocess(frame, warped, resized, net_size);
        cv::Size frame_size = frame.size();
        // One-time: tell the pipeline how to project AutoSteer/AutoSpeed outputs
        // back to world when those networks run on the plain-resized image.
        if (!h_resized_set)
        {
            pipeline.set_H_resized(H, frame_size);
            h_resized_set = true;
        }

        // ── Default frame no inference ────────────────────────────────────────────
        cv::Mat display_frame = resized;

        if (const auto r = pipeline.process(warped, resized))
        {
            pipeline.latency().print();

            const double ego_v = vehicle_interface->read();
            const double cte = r->lateral.cte_m;
            const double epsi = r->lateral.yaw_rad;
            const double kappa = r->lateral.curvature;

            // has_cipo: tracker-based — true only when filter tracks a target
            // closer than D_MAX. cipo_raw_found alone must not gate the planner.
            static constexpr double D_MAX = 150.0;
            const bool has_cipo = r->cipo.valid && r->cipo.distance_m < D_MAX;
            const double cipo_v = has_cipo ? r->cipo.velocity_ms : cfg.speed_limit;
            const double cipo_dist = r->cipo.distance_m;

            const double raw_cte = r->lateral.path_valid
                                       ? static_cast<double>(r->lateral.raw_cte_m)
                                       : cte;
            const Plan plan = planner.compute_plan(cte, epsi, kappa, ego_v, has_cipo, cipo_v, cipo_dist);

            VP_INFO(
                "plan: tyre=%.4f rad  accel=%.3f m/s²  |  cte=%.2fm(raw=%.2fm)  |  cipo=%s  dist=%.1f m  vel=%+.2f m/s",
                plan.steering.empty() ? 0.0 : plan.steering[0],
                plan.acceleration,
                cte,
                raw_cte,
                has_cipo ? "true" : "false",
                cipo_dist,
                r->cipo.velocity_ms);

            vehicle_interface->write(
                plan.steering.empty() ? 0.0 : plan.steering[0],
                plan.acceleration);

        }

    }

    return 0;
}