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
 * Description  : Sample file of handling vsfwk component
 ***********************************************************************************************************************/

/***********************************************************************************************************************
 * \brief       Sample file of handling vsfwk component
 * \brief       These functions are expected to be provided and managed in the
 *application level by user
 ***********************************************************************************************************************/

#include <string.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>
#include <atomic>
#include <thread>

#include "rcar-xos/hycoah/r_hycoah.hpp"
#include "rcar-xos/hycoah/r_hycoah_types.hpp"
#include "common.hpp"
#include "rcar-xos/hycoah/r_hycoah_io_desc.hpp"
#include "rcar-xos/osal/r_osal.h"
#include "stdio.h"
#include "r_sample_init_config.hpp"

static volatile std::atomic<uint32_t> container_counter;
#define AH_SAMPLE_APP_TIMEOUT_COUNTER (100000)

using namespace hycoah;

size_t alignDataSize(const size_t align, size_t size) {
  auto new_size = size;
  if (align > size) {
    new_size = align;
  } else if (0 != (size % align)) {
    new_size = ((size / align) + 1) * align;
  }
  return new_size;
}

void createBuffer(BufMgr_BufferManager *buffer_manager, int &buffer_container_id, size_t size) {
  size_t buffer_size = alignDataSize(4096, size);
  int buffer_id = 0;

  auto buffer_config = BufMgr_ConfigParameter(1, buffer_size, 4096, CONTAINER_TYPE_PHYSIC, "hycoah_normal");
  buffer_config.alignSize();
  BufMgr_Buffer *buffer = nullptr;
  e_bufmngr_return_t ret = buffer_manager->R_BufMgr_CreateContainer(buffer_config, &buffer_container_id);
  if (RETURN_BUFMNGR_OK != ret) {
    printf("[%d] Failed to create buffer container.\n", ret);
  }
  if (RETURN_BUFMNGR_OK != buffer_manager->R_BufMgr_GetBufferHandle(buffer_container_id, buffer_id, buffer)) {
    printf("Failed to get buffer handle.\n");
  }
  if (nullptr == buffer) {
    printf("Buffer handle is null.\n");
  }
}

BufMgr_BufferManager *createBufferManager() {
  // Create Buffer Manager instance.
  e_bufmngr_return_t buffermanager_ret;
  e_bufmngr_multiprocess_type_t buffermgnr_type = e_bufmngr_multiprocess_type_t::R_HWA_BUFMNGR_SINGLE_PROCESS;
  auto *buffer_manager = new BufMgr_BufferManager(&buffermanager_ret, buffermgnr_type);
  if (buffer_manager == nullptr || buffermanager_ret != RETURN_BUFMNGR_OK) {
    return nullptr;
  }

  // Initialize Buffer Manager.
  buffermanager_ret = buffer_manager->R_BufMgr_Init(default_bufmgr_init_cfg);
  if (buffermanager_ret != RETURN_BUFMNGR_OK) {
    return nullptr;
  }
  return buffer_manager;
}

R_EXFWK *createExfwk(BufMgr_BufferManager *buffer_manager) {
  e_exfwk_multiprocess_type_t exfwk_type = e_exfwk_multiprocess_type_t::EXFWK_SINGLE_PROCESS;
  e_exfwk_return_t exfwk_ret;
  // Create EXFWK instance.
  auto *exfwk = new R_EXFWK(exfwk_type);

  exfwk_ret = exfwk->exfwk_init(default_exfwk_init_cfg, buffer_manager);
  if (exfwk_ret != RETURN_EXFWK_OK) {
    return nullptr;
  }
  return exfwk;
}

int32_t parserContainerCallback(e_exfwk_container_event_t event, void * const info)
{
    if (event == EXFWK_END_CONTAINER_EVENT)
    {
        container_counter--;
    }
    else if (event == EXFWK_TIMEOUT_EVENT)
    {
        s_exfwk_timeout_cbinfo_t * info_obj = (s_exfwk_timeout_cbinfo_t *)info;
        printf("    nb_job_aborted: %d", info_obj->nb_job_aborted);
        printf("    job_container_id: %d", info_obj->job_container_id);
    }
    else
    {
        printf("    (invalid)\n");
    }
    return 0;
}

// Execute function for single-job-container usecases
int32_t execute(R_EXFWK * exfwk, JobContainer * job_container)
{
    // Run jobs.
    e_exfwk_return_t           exfwk_ret;
    uint32_t                   parser_container_id;
    s_exfwk_container_cbinfo_t callback_info;
    container_counter    = 1;
    uint32_t timeout_u32 = 0;

    callback_info.timeout_msec  = 10000;
    callback_info.end_cbarg     = nullptr;
    callback_info.timeout_cbarg = nullptr;
    callback_info.cbfunc        = parserContainerCallback;

    exfwk_ret = exfwk->push(*job_container, 1, callback_info, parser_container_id);
    if (exfwk_ret != RETURN_EXFWK_OK)
    {
        printf("Failed to push jobs.\n");
        return e_hycoah_return_t::RETURN_HYCOAH_ERROR_JOB;
    }

    while ((0 < container_counter) && (AH_SAMPLE_APP_TIMEOUT_COUNTER > timeout_u32))
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        timeout_u32++;
    }

    printf("All Job Containers finished running.\n");
    return e_hycoah_return_t::RETURN_HYCOAH_OK;
}

// Time evaluation function for single-job-container usecases
void time_evaluate(R_EXFWK *exfwk, JobContainer *&job_container,
                   std::vector<Network *> &networks, int repeat_number,
                   std::vector<LatencyStat> &execute_time, std::vector<std::vector<uint8_t>> core_list,
                   bool warm_up = false) {
  st_osal_time_t start;
  st_osal_time_t end;
  osal_nano_sec_t timestamp;

  if (warm_up) {
    for (auto network : networks) {
      network->syncIO(e_sync_direction_t::H2D, 0);
    }

    auto exec_ret = execute(exfwk, job_container);
    if (exec_ret != e_hycoah_return_t::RETURN_HYCOAH_OK) {
      std::cout << " EXECUTION FAILED " << std::endl;
      return;
    }

    for (auto network : networks) {
      network->syncIO(e_sync_direction_t::D2H, 0);
    }
  }

  std::vector<std::vector<float>> benchmark_list;
  int number = 0;
  while (number++ < repeat_number) {
    (void)R_OSAL_ClockTimeGetTimeStamp(OSAL_CLOCK_TYPE_HIGH_RESOLUTION, &start);

    for (auto network : networks) {
      network->syncIO(e_sync_direction_t::H2D, 0);
    }

    auto exec_ret = execute(exfwk, job_container);
    if (exec_ret != e_hycoah_return_t::RETURN_HYCOAH_OK) {
      std::cout << " EXECUTION FAILED " << std::endl;
      return;
    }

    for (auto network : networks) {
      network->syncIO(e_sync_direction_t::D2H, 0);
    }

    (void)R_OSAL_ClockTimeGetTimeStamp(OSAL_CLOCK_TYPE_HIGH_RESOLUTION, &end);
    (void)R_OSAL_ClockTimeCalculateTimeDifference(&end, &start, &timestamp);
    std::chrono::nanoseconds time_ns(timestamp);
    float ms = std::chrono::duration<float, std::milli>(time_ns).count();

    float cnnip_time = 0;
    std::vector<float> dsp_time = {0, 0, 0, 0};
    uint8_t network_index = 0;
    for (auto network : networks) {
      auto latency = network->getLatency(0);
      cnnip_time += latency[0];
      if (!core_list.at(network_index).empty()) {
        dsp_time[core_list.at(network_index)[0]] += latency[1];
      }
      else {
        if (latency[1] > 0) {
          throw std::runtime_error("DSP core(s) are not correctly specified in msgpack artifact folder name.");
        }
      }
      network_index = static_cast<uint8_t>(network_index + 1);
    }

    benchmark_list.push_back({ms, cnnip_time, dsp_time[0], dsp_time[1], dsp_time[2], dsp_time[3]});
  }

  std::vector<float> sums(6, 0.0);
  std::vector<float> mins(6, std::numeric_limits<float>::max());
  std::vector<float> maxs(6, std::numeric_limits<float>::lowest());
  for (auto benchmark : benchmark_list) {
    for (size_t i = 0; i < 6; ++i) {
      sums[i] += benchmark[i];
      if (benchmark[i] < mins[i]) mins[i] = benchmark[i];
      if (benchmark[i] > maxs[i]) maxs[i] = benchmark[i];
    }
  }
  for (size_t i = 0; i < 6; ++i) {
    execute_time[i].avg = sums[i] / static_cast<float>(benchmark_list.size());
    execute_time[i].min = mins[i];
    execute_time[i].max = maxs[i];
  }
}
