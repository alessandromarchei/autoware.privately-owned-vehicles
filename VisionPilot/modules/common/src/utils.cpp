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
