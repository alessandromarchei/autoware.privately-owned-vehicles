#include <chrono>
#include <cstdint>
#include <iostream>

#include <opencv2/core.hpp>
#include <tcp/tcp_frame_client.hpp>

int main(int argc, char** argv)
{
    const std::string address = argc > 1 ? argv[1] : "10.0.0.1";
    const auto port = static_cast<std::uint16_t>(
        argc > 2 ? std::stoi(argv[2]) : 5000);

    visionpilot::tcp::TCPClient client;
    if (!client.connect_to(address, port, 5000)) {
        std::cerr << "Connection failed: " << client.last_error() << '\n';
        return 1;
    }

    std::cout << "Connected to " << address << ':' << port << '\n';
    while (true) {
        cv::Mat frame;
        visionpilot::tcp::ReceivedFrame received;
        const auto start = std::chrono::steady_clock::now();

        if (!client.receive_frame(frame, received, 10000)) {
            std::cerr << "Receive failed: " << client.last_error() << '\n';
            return 1;
        }

        const float receive_ms = static_cast<float>(
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count());

        std::cout << "frame=" << received.frame_id
                  << " size=" << frame.cols << 'x' << frame.rows
                  << " channels=" << frame.channels()
                  << " receive_ms=" << receive_ms << '\n';

        visionpilot::tcp::VisionResult result{};
        result.frame_id = received.frame_id;
        result.timestamp_ns = received.timestamp_ns;
        result.steering_rad = 0.1F;
        result.acceleration_ms2 = 0.2F;
        result.inference_ms = receive_ms;
        result.path_valid = true;

        if (!client.send_result(result)) {
            std::cerr << "Result send failed: " << client.last_error() << '\n';
            return 1;
        }
    }
}
