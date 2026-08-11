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
 * Description  : Header of common functions used for application.
 ***********************************************************************************************************************/

#ifndef COMMON_HPP
#define COMMON_HPP

#include <string.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <vector>

#include "rcar-xos/hycoah/r_hycoah.hpp"
#include "rcar-xos/hycoah/r_hycoah_io_desc.hpp"
#include "rcar-xos/hycoah/r_hycoah_types.hpp"

struct LatencyStat {
  float avg = 0;
  float min = 0;
  float max = 0;
};

int load_inputs_from_file(std::string filename, const std::vector<hycoah::InputMemory> &input_memories);
void save_outputs_to_file(const std::vector<hycoah::OutputMemory> &output_memories, const std::string &filename);
void generate_latency_report(const std::string testcase_name, std::vector<LatencyStat> execute_time);
std::vector<std::vector<uint8_t>> core_list_from_name(const std::string &name);
std::string get_folder_name(const std::string &path);

#endif
