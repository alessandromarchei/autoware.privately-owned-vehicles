/***********************************************************************************************************************
 * DISCLAIMER
 *
 * The contents of this file (the "contents") are proprietary and confidential to Renesas Electronics Corporation
 * and/or its licensors ("Renesas") and subject to statutory and contractual protections.
 *
 * Unless otherwise expressly agreed in writing between Renesas and you: 1) you may not use, copy, modify, distribute,
 * display, or perform the contents; 2) you may not use any name or mark of Renesas for advertising or publicity
 * purposes or in connection with your use of the contents; 3) RENESAS MAKES NO WARRANTY OR REPRESENTATIONS ABOUT THE
 * SUITABILITY OF THE CONTENTS FOR ANY PURPOSE; THE CONTENTS ARE PROVIDED "AS IS" WITHOUT ANY EXPRESS OR IMPLIED
 * WARRANTY, INCLUDING THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
 * NON-INFRINGEMENT; AND 4) RENESAS SHALL NOT BE LIABLE FOR ANY DIRECT, INDIRECT, SPECIAL, OR CONSEQUENTIAL DAMAGES,
 * INCLUDING DAMAGES RESULTING FROM LOSS OF USE, DATA, OR PROJECTS, WHETHER IN AN ACTION OF CONTRACT OR TORT, ARISING
 * OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THE CONTENTS. Third-party contents included in this file may
 * be subject to different terms.
 *
 * Copyright [2025-2026] Renesas Electronics Corporation and/or its licensors. All Rights Reserved.
 ***********************************************************************************************************************/
/***********************************************************************************************************************
 * Version      : 2.0.0
 * XOS Version  : 3.47.0
 * Description  : Common functions used for application.
 ***********************************************************************************************************************/

#include "common.hpp"
#include <cstring>

int load_inputs_from_file(std::string filename, const std::vector<hycoah::InputMemory> &input_memories)
{
    // ------------------------------------------------------------
    // Validate AH input buffers and calculate expected total size
    // ------------------------------------------------------------
    size_t total_data_size = 0;

    for (size_t i = 0; i < input_memories.size(); ++i) {
        const auto& mem = input_memories[i];

        if (mem.cpuPtr() == nullptr || mem.size_bytes == 0) {
            std::cerr
                << "[Error] Invalid input memory at index " << i
                << " ptr=" << mem.cpuPtr()
                << " size=" << mem.size_bytes
                << std::endl;

            return -1;
        }

        total_data_size += mem.size_bytes;
    }

    std::cout
        << "[Input] Network expects "
        << input_memories.size()
        << " input buffer(s), total "
        << total_data_size
        << " bytes"
        << std::endl;

    // ------------------------------------------------------------
    // Try opening supplied input file
    // ------------------------------------------------------------
    std::ifstream file(filename, std::ios::binary);

    bool use_dummy = false;

    if (!file) {
        std::cerr
            << "[Warning] Could not open input file: "
            << filename
            << std::endl;

        use_dummy = true;
    } else {
        file.seekg(0, std::ios::end);
        const std::streamsize file_size = file.tellg();
        file.seekg(0, std::ios::beg);

        if (file_size < 0 ||
            static_cast<size_t>(file_size) != total_data_size) {

            std::cerr
                << "[Warning] Input file size mismatch!"
                << " Expected " << total_data_size
                << " bytes, got " << file_size
                << " bytes."
                << std::endl;

            use_dummy = true;
        }
    }

    // ------------------------------------------------------------
    // Generate dummy input directly inside AH input buffers
    // ------------------------------------------------------------
    if (use_dummy) {
        std::cout
            << "[Input] Generating zero-filled dummy input..."
            << std::endl;

        for (size_t i = 0; i < input_memories.size(); ++i) {
            const auto& mem = input_memories[i];

            void* dst =
                reinterpret_cast<uint8_t*>(mem.cpuPtr()) +
                mem.offset;

            std::memset(dst, 0, mem.size_bytes);

            std::cout
                << "  input[" << i << "]"
                << " size=" << mem.size_bytes
                << " bytes"
                << " offset=" << mem.offset
                << std::endl;
        }

        std::cout
            << "[Input] Dummy input ready ("
            << total_data_size
            << " bytes total)."
            << std::endl;

        return 0;
    }

    // ------------------------------------------------------------
    // Normal path: load binary input
    // ------------------------------------------------------------
    for (size_t i = 0; i < input_memories.size(); ++i) {
        const auto& mem = input_memories[i];

        char* dst =
            reinterpret_cast<char*>(mem.cpuPtr()) +
            mem.offset;

        file.read(dst, mem.size_bytes);

        if (!file) {
            std::cerr
                << "[Error] Failed reading input "
                << i
                << " (" << mem.size_bytes
                << " bytes)"
                << std::endl;

            return -1;
        }
    }

    std::cout
        << "[Input] Successfully loaded "
        << total_data_size
        << " bytes from "
        << filename
        << std::endl;

    return 0;
}

void save_outputs_to_file(const std::vector<hycoah::OutputMemory> &output_memories, const std::string &filename) {
  std::ofstream file(filename, std::ios::out | std::ios::binary);
  if (!file) {
    std::cerr << "[Error] Could not open file " << filename << " for writing." << std::endl;
  }

  for (size_t output_idx = 0; output_idx < output_memories.size(); ++output_idx) {
    const auto &output_memory = output_memories[output_idx];
    if ((output_memory.cpuPtr() == nullptr) || output_memory.size_bytes == 0) {
      std::cerr << "[Error] Invalid output memory at index " << output_idx << std::endl;
    }

    file.write(reinterpret_cast<const char *>(output_memory.cpuPtr()) + output_memory.offset, output_memory.size_bytes);
    if (!file) {
      std::cerr << "[Error] Failed to write output " << output_idx << " to file." << std::endl;
    }
  }

  file.close();
  std::cout << "Successfully wrote outputs to " << filename << std::endl;
}

void generate_latency_report(const std::string testcase_name, std::vector<LatencyStat> execute_time) {
  const std::string csv_filename = "latency_report.csv";
  bool csv_file_exists = std::ifstream(csv_filename).good();

  std::ofstream csv_file(csv_filename, std::ios::app);
  if (csv_file.is_open()) {
    if (!csv_file_exists) {
      csv_file << "case_name,"
                  "execute_total(avg) (ms),execute_cnnip(avg) (ms),execute_dsp_core0(avg) (ms),execute_dsp_core1(avg) "
                  "(ms),execute_dsp_core2(avg) (ms),execute_dsp_core3(avg) (ms),"
                  "execute_total(min) (ms),execute_cnnip(min) (ms),execute_dsp_core0(min) (ms),execute_dsp_core1(min) "
                  "(ms),execute_dsp_core2(min) (ms),execute_dsp_core3(min) (ms),"
                  "execute_total(max) (ms),execute_cnnip(max) (ms),execute_dsp_core0(max) (ms),execute_dsp_core1(max) "
                  "(ms),execute_dsp_core2(max) (ms),execute_dsp_core3(max) (ms)\n";
    }
    csv_file << std::fixed << std::setprecision(5) << testcase_name << "," << execute_time[0].avg << ","
             << execute_time[1].avg << "," << execute_time[2].avg << "," << execute_time[3].avg << ","
             << execute_time[4].avg << "," << execute_time[5].avg << "," << execute_time[0].min << ","
             << execute_time[1].min << "," << execute_time[2].min << "," << execute_time[3].min << ","
             << execute_time[4].min << "," << execute_time[5].min << "," << execute_time[0].max << ","
             << execute_time[1].max << "," << execute_time[2].max << "," << execute_time[3].max << ","
             << execute_time[4].max << "," << execute_time[5].max << "\n";
    csv_file.close();
  } else {
    std::cerr << "Failed to open latency_report.csv for writing." << std::endl;
  }
}

std::vector<std::vector<uint8_t>> core_list_from_name(const std::string &name) {
  std::vector<std::vector<uint8_t>> result;
  std::vector<uint8_t> current_cores;
  std::regex core_regex("core([0-3])");
  std::sregex_iterator next(name.begin(), name.end(), core_regex);
  std::sregex_iterator end;

  while (next != end) {
    std::smatch match = *next;
    if (match.size() > 1) {
      try {
        int core_id = std::stoi(match.str(1));
        current_cores.push_back(static_cast<uint8_t>(core_id));
      } catch (const std::exception &e) {
      }
    }
    next++;
  }

  result.push_back(current_cores);

  return result;
}

std::vector<std::vector<uint8_t>> core_list_from_name(const std::pair<std::string, std::string> &name_pair) {
  std::vector<std::vector<uint8_t>> result;

  result.push_back(core_list_from_name(name_pair.first)[0]);
  result.push_back(core_list_from_name(name_pair.second)[0]);

  return result;
}

std::string get_folder_name(const std::string &path) {
  if (path.empty() || path == "/") {
    return "";
  }

  size_t last_slash = path.rfind('/');
  if (last_slash == std::string::npos) {
    return "";
  }

  std::string before_last = path.substr(0, last_slash);
  size_t second_last_slash = before_last.rfind('/');

  if (second_last_slash == std::string::npos) {
    return before_last;
  }

  return before_last.substr(second_last_slash + 1);
}
