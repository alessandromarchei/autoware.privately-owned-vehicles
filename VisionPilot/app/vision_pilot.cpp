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
#include <tcp/tcp_frame_client.hpp>

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

    std::string tcp_server_address = "10.0.0.1";
    std::uint16_t tcp_server_port = 5000;

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
                VP_ERROR("Missing argument after --source-mode. Specify source mode: video|v4l2|frames|tcpip_frames");
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
        else if (arg == "--tcp-server")
        {
            if (i + 1 >= argc)
            {
                VP_ERROR("Missing argument after --tcp-server");
                return 1;
            }

            tcp_server_address = argv[++i];
        }
        else if (arg == "--tcp-port")
        {
            if (i + 1 >= argc)
            {
                VP_ERROR("Missing argument after --tcp-port");
                return 1;
            }

            const int port = std::stoi(argv[++i]);

            if (port <= 0 || port > 65535)
            {
                VP_ERROR("Invalid TCP port: %d", port);
                return 1;
            }

            tcp_server_port =
                static_cast<std::uint16_t>(port);
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
    std::unique_ptr<visionpilot::tcp::TCPClient> tcp_client;

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
    else if (cfg.source.mode == SourceMode::TCPIP_Frames)
    {
        VP_INFO("Using TCP/IP frame source mode");
        VP_INFO("TCP server: %s:%u", tcp_server_address.c_str(),static_cast<unsigned>(tcp_server_port));

        tcp_client = std::make_unique<visionpilot::tcp::TCPClient>();

        if (!tcp_client->connect_to(tcp_server_address, tcp_server_port, 5000))
        {
            VP_ERROR( "Cannot connect to TCP server %s:%u: %s",tcp_server_address.c_str(),
                static_cast<unsigned>(tcp_server_port),
                tcp_client->last_error().c_str());

            return 1;
        }

        VP_INFO("Connected to TCP image server");
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

    // ── Initialize camera interface or TCPIP Client interface ──────────────────────────────────

    if (cfg.source.mode == SourceMode::TCPIP_Frames)
    {
        if (!tcp_client ||
            !tcp_client->is_connected())
        {
            VP_ERROR("TCP client is not connected");
            return 1;
        }
    }
    else
    {
        if (!camera_interface ||
            !camera_interface->is_device_open())
        {
            VP_ERROR("Cannot open frame source");
            return 1;
        }

        if (!vehicle_interface)
        {
            VP_ERROR("Vehicle interface is not available");
            return 1;
        }
    }

    const cv::Size net_size(vm::AutoDrive::NET_W, vm::AutoDrive::NET_H);
    cv::Mat frame, warped, resized;
    bool h_resized_set = false;
    cv::Mat H = load_matrix(homography_path, "H");

    VP_INFO("Starting main loop. Press Ctrl+C to exit.");


    std::uint64_t current_frame_id = 0;
    std::uint64_t current_timestamp_ns = 0;
    double current_ego_speed_ms = 0.0;


    while (true)
    {
        bool ok = false;

        if (cfg.source.mode == SourceMode::TCPIP_Frames)
        {
            visionpilot::tcp::ReceivedFrame received{};

            ok = tcp_client->receive_frame(frame,received,10000);

            if (!ok)
            {
                VP_ERROR("TCP frame reception failed: %s", tcp_client->last_error().c_str());

                // La ricezione fallita chiude la socket.
                // Prova a riconnetterti al server Python.
                while (!tcp_client->reconnect(5000))
                {
                    VP_ERROR("TCP reconnect failed: %s", tcp_client->last_error().c_str());

                    std::this_thread::sleep_for(std::chrono::seconds(1));
                }

                VP_INFO("TCP connection restored");
                continue;
            }

            current_frame_id = received.frame_id;

            current_timestamp_ns = received.timestamp_ns;

            current_ego_speed_ms = static_cast<double>(received.vehicle_speed_ms);

            VP_INFO("TCP frame=%llu speed=%.3f m/s size=%dx%d", static_cast<unsigned long long>(current_frame_id),
                current_ego_speed_ms,
                frame.cols,
                frame.rows);
        }
        else
        {
            auto capture = camera_interface->get_latest_frame();

            //retrieve data from the tuple
            ok = std::get<0>(capture);
            frame = std::move(std::get<1>(capture));

            //retrieve speed from vehicle interface only if frame is valid
            if (ok && !frame.empty())
            {
                current_ego_speed_ms = vehicle_interface->read();
            }
        }

        if (!ok || frame.empty())
        {
            if (cfg.source.mode == SourceMode::Video || cfg.source.mode == SourceMode::Frames)
            {
                VP_INFO("End of video/frames reached. Exiting.");

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
            const Plan plan = planner.compute_plan(cte, epsi, kappa, current_ego_speed_ms, has_cipo, cipo_v, cipo_dist);

            VP_INFO(
                "plan: tyre=%.4f rad  accel=%.3f m/s²  |  cte=%.2fm(raw=%.2fm)  |  cipo=%s  dist=%.1f m  vel=%+.2f m/s",
                plan.steering.empty() ? 0.0 : plan.steering[0],
                plan.acceleration,
                cte,
                raw_cte,
                has_cipo ? "true" : "false",
                cipo_dist,
                r->cipo.velocity_ms);

            //send data to TCP server if in TCPIP_Frames mode, otherwise send to vehicle interface
            if (cfg.source.mode == SourceMode::TCPIP_Frames)
            {
                visionpilot::tcp::VisionResult tcp_result{};

                tcp_result.frame_id = current_frame_id;
                tcp_result.timestamp_ns = current_timestamp_ns;
                tcp_result.steering_rad = static_cast<float>(plan.steering.empty() ? 0.0 : plan.steering[0]);
                tcp_result.acceleration_ms2 = static_cast<float>(plan.acceleration);
                tcp_result.cte_m = static_cast<float>(cte);
                tcp_result.yaw_rad = static_cast<float>(epsi);
                tcp_result.curvature_1pm = static_cast<float>(kappa);
                tcp_result.cipo_distance_m = static_cast<float>(cipo_dist);
                tcp_result.cipo_velocity_ms = static_cast<float>(r->cipo.velocity_ms);
                tcp_result.cipo_valid = has_cipo;
                tcp_result.path_valid = r->lateral.path_valid;

                //send to TCP server
                if (!tcp_client->send_result(tcp_result))
                {
                    VP_ERROR("Cannot send result for frame %llu: %s",static_cast<unsigned long long>(
                            current_frame_id),
                        tcp_client->last_error().c_str());
                }
            }
            else
            {
                vehicle_interface->write( plan.steering.empty() ? 0.0 : plan.steering[0], plan.acceleration);
            }


        }

    }

    return 0;
}