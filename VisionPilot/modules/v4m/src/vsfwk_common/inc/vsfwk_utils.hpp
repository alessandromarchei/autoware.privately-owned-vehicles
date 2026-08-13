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
#pragma once

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

extern volatile std::atomic<uint32_t> container_counter;

#define AH_SAMPLE_APP_TIMEOUT_COUNTER (100000)

using namespace hycoah;

size_t alignDataSize(const size_t align, size_t size);

void createBuffer(BufMgr_BufferManager *buffer_manager, int &buffer_container_id, size_t size);

BufMgr_BufferManager *createBufferManager();

R_EXFWK *createExfwk(BufMgr_BufferManager *buffer_manager);

int32_t parserContainerCallback(e_exfwk_container_event_t event, void * const info);

// Execute function for single-job-container usecases
int32_t execute(R_EXFWK * exfwk, JobContainer * job_container);

// Time evaluation function for single-job-container usecases
void time_evaluate(R_EXFWK *exfwk, JobContainer *&job_container,
                   std::vector<Network *> &networks, int repeat_number,
                   std::vector<LatencyStat> &execute_time, std::vector<std::vector<uint8_t>> core_list,
                   bool warm_up = false);
