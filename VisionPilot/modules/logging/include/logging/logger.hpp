#pragma once

#include <cstdio>

#define VP_INFO(...)  do { std::printf("[INFO]  "); std::printf(__VA_ARGS__); std::printf("\n"); } while (false)
#define VP_WARN(...)  do { std::fprintf(stderr, "[WARN]  "); std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); } while (false)
#define VP_ERROR(...) do { std::fprintf(stderr, "[ERROR] "); std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); } while (false)