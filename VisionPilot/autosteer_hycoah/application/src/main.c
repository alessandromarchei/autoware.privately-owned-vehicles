
#include <string.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>

#include "common.hpp"
#include "rcar-xos/hycoah/r_hycoah.hpp"
#include "rcar-xos/hycoah/r_hycoah_io_desc.hpp"
#include "rcar-xos/hycoah/r_hycoah_types.hpp"
#include "rcar-xos/osal/r_osal.h"
#include "stdio.h"
#include "vsfwk_utils.hpp"

using namespace hycoah;

enum RunMode {
  ONE_CORE_ONE_CONTAINER = 0,
  ONE_CORE_TWO_CONTAINERS = 1,
  TWO_CORES_TWO_CONTAINERS = 2,
};

static std::vector<std::string> parseArguments(int argc, char *argv[]) {
  return std::vector<std::string>(argv + 1, argv + argc);
}

// Function to run one NN model on one DSP core in one job container
int run_one_model_one_core_one_container(std::string msgpack, std::string inputs, std::string testcase_name,
                                         std::vector<std::vector<uint8_t>> core_list) {
  // Initialize OSAL
  e_osal_return_t osal_ret = R_OSAL_Initialize();
  if (OSAL_RETURN_OK != osal_ret) {
    printf("OSAL Initialization failed with error %d\n", osal_ret);
    return -1;
  }

  // user created buffer_manager/exfwk/job_container
  BufMgr_BufferManager *buffer_manager = createBufferManager();
  R_EXFWK *exfwk = createExfwk(buffer_manager);
  JobContainer *job_container = new JobContainer;

  // Prepare input memories
  std::unordered_map<PipelineId, std::vector<InputMemory>> user_input_memories;
  std::unordered_map<PipelineId, std::vector<OutputMemory>> user_output_memories;
  user_input_memories[0] = {/* empty memories for PipelineId 0 */};
  user_output_memories[0] = {/* empty memories for PipelineId 0 */};

  // Initialize and get Network object
  ArtifactHelper helper;
  Network network(msgpack.c_str());
  auto config_ret = helper.config({network}, st_hycoah_config_t{});
  if (config_ret != e_hycoah_return_t::RETURN_HYCOAH_OK) {
    std::cout << " Config ARTIFACTHELPER FAILED " << std::endl;
  }
  e_hycoah_return_t ret = helper.init(network, buffer_manager, exfwk, user_input_memories, user_output_memories);

  if (ret != e_hycoah_return_t::RETURN_HYCOAH_OK) {
    std::cout << " INIT ARTIFACTHELPER FAILED " << std::endl;
  }

  //create input descriptors. for I/O METADATA
    const auto& input_descs  = network.getInputDesc();
    const auto& output_descs = network.getOutputDesc();

    std::cout << "INPUT DESCS  = " << input_descs.size() << std::endl;
    std::cout << "OUTPUT DESCS = " << output_descs.size() << std::endl;

    for (size_t i = 0; i < input_descs.size(); ++i) {
        std::cout
            << "Input[" << i << "]"
            << " name=" << input_descs[i].name
            << " size=" << input_descs[i].size_bytes
            << " layout=" << input_descs[i].layout
            << std::endl;
    }

    for (size_t i = 0; i < output_descs.size(); ++i) {
        std::cout
            << "Output[" << i << "]"
            << " name=" << output_descs[i].name
            << " size=" << output_descs[i].size_bytes
            << " layout=" << output_descs[i].layout
            << std::endl;
    }

    //I/O buffers physical memory allocation
    auto input_memories  = network.getInputMemory(0);
    auto output_memories = network.getOutputMemory(0);

    std::cout << "INPUT MEMORIES  = " << input_memories.size() << std::endl;
    std::cout << "OUTPUT MEMORIES = " << output_memories.size() << std::endl;

    for (size_t i = 0; i < input_memories.size(); ++i) {
        std::cout
            << "InputMemory[" << i << "]"
            << " size=" << input_memories[i].size_bytes
            << " container=" << input_memories[i].container_id
            << " buffer=" << input_memories[i].buffer_id
            << " offset=" << input_memories[i].offset
            << " cpu=" << input_memories[i].cpuPtr()
            << std::endl;
    }

    for (size_t i = 0; i < output_memories.size(); ++i) {
        std::cout
            << "OutputMemory[" << i << "]"
            << " size=" << output_memories[i].size_bytes
            << " container=" << output_memories[i].container_id
            << " buffer=" << output_memories[i].buffer_id
            << " offset=" << output_memories[i].offset
            << " cpu=" << output_memories[i].cpuPtr()
            << std::endl;
    }

  // prepare actual data
  load_inputs_from_file(inputs, input_memories);

  // prepare job
  // uint8_t pipeline_id = 0;
  std::vector<JobId> job_dependency;
  std::vector<JobId> job_ids;

  // add job.
  auto job_ret = network.addJobs(job_container, 0, job_dependency, job_ids);
  if (job_ret != e_hycoah_return_t::RETURN_HYCOAH_OK) {
    std::cout << " ADD JOB FAILED with error : " << job_ret << std::endl;
    return -1;
  }

  for (auto &&job_id : job_ids) {
    std::cout << "job_id: " << static_cast<int>(job_id) << " added" << std::endl;
  }

  // feed the input to HW
  network.syncIO(e_sync_direction_t::H2D, 0);

  auto exec_ret = execute(exfwk, job_container);
  if (exec_ret != e_hycoah_return_t::RETURN_HYCOAH_OK) {
    std::cout << " EXECUTION FAILED " << std::endl;
  }

  // fetch the output from HW
  network.syncIO(e_sync_direction_t::D2H, 0);

  save_outputs_to_file(output_memories, "outputs.bin");

  std::vector<LatencyStat> execute_time(6);  // "total", "cnnip", "dsp_core0", "dsp_core1", "dsp_core2", "dsp_core3"
  std::vector<Network *> networks{&network};
  time_evaluate(exfwk, job_container, networks, 80, execute_time, core_list, true);
  std::vector<std::string> scope_names = {"total", "cnnip", "dsp_core0", "dsp_core1", "dsp_core2", "dsp_core3"};
  for (size_t i = 0; i < 6; ++i) {
    std::cout << std::setw(9) << std::right << scope_names[i].c_str() << ": n = " << 80 << std::fixed
              << std::setprecision(5) << ", avg = " << std::setw(10) << std::left << execute_time[i].avg << " ms"
              << ", min = " << std::setw(10) << std::left << execute_time[i].min << " ms"
              << ", max = " << std::setw(10) << std::left << execute_time[i].max << " ms." << std::endl;
  }

  auto exfwk_ret = exfwk->exfwk_quit();
  if (exfwk_ret != RETURN_EXFWK_OK) {
    printf("exfwk quit failed with error %d\n", exfwk_ret);
    return -1;
  }
  delete exfwk;

  auto deinit_ret = helper.deinit();
  if (deinit_ret != e_hycoah_return_t::RETURN_HYCOAH_OK) {
    printf("Parser deinit failed with error %d\n", deinit_ret);
    return -1;
  }

  auto buffermanager_ret = buffer_manager->R_BufMgr_Close();
  if (buffermanager_ret != RETURN_BUFMNGR_OK) {
    printf("Buffer manager close failed with error %d\n", buffermanager_ret);
    return -1;
  }
  delete buffer_manager;

  delete job_container;

  // De-initialize OSAL
  osal_ret = R_OSAL_Deinitialize();
  if (OSAL_RETURN_OK != osal_ret) {
    printf("OSAL De-initialization failed with error %d\n", osal_ret);
    return -1;
  }

  generate_latency_report(testcase_name, execute_time);

  printf("Program ended successfully.\n");
  return 0;
}


int check_args(std::vector<std::string> args) {
  if (args.size() < 1) {
    printf("No arguments provided.\n");
    return -1;
  }

  return 0;
}

int main(int argc, char *argv[]) {
  auto args = parseArguments(argc, argv);
  if (check_args(args) != 0) {
    return -1;
  }

  std::string msgpack = args[0];
  std::string testcase_name = "unnamed";

  // The core information parsed from the folder name of the msgpack file is used to show latency information.
  // The folder name "sample_model1_cnnip4_dsp3_core0" stands for msgpack compiled with DSP core0.
  // If you want to specify the core information from the code directly, please modify the core_list variable below.
  // For example, please set core_list = {{0}} if the msgpack is compiled with DSP core0,
  // and set core_list = {{0, 1}} if the single NN multi cores msgpack is compiled with DSP core0 and core1.
  std::string folder_name = get_folder_name(msgpack);
  std::vector<std::vector<uint8_t>> core_list = core_list_from_name(folder_name);

  int ret = -1;

    std::string inputs = args[1];
    if (args.size() == 4) {
        testcase_name = args[2];
    }
    ret = run_one_model_one_core_one_container(msgpack, inputs, testcase_name, core_list);
    
  if (ret != 0) {
    return ret;
  }

  return 0;
}
