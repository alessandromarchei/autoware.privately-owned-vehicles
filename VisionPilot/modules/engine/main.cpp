
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

static std::vector<std::string> parseArguments(int argc, char *argv[]) {
  return std::vector<std::string>(argv + 1, argv + argc);
}


/*
assuming vision pilot will be giving the .msgpack file for the constructor
*/
int run_one_model_one_core_one_container_custom_io(std::string msgpack, std::string inputs, std::string testcase_name,
                                                   std::vector<std::vector<uint8_t>> core_list) {
  // Initialize OSAL
  e_osal_return_t osal_ret = R_OSAL_Initialize();
  if (OSAL_RETURN_OK != osal_ret) {
    printf("OSAL Initialization failed with error %d\n", osal_ret);
    return -1;
  }

  // user created buffer_manager/exfwk/job_container
  BufMgr_BufferManager *buffer_manager = createBufferManager();
  int input_container_id = 0;
  int output_container_id = 0;

  Network network(msgpack.c_str());
  // get inputs/outputs
  const std::vector<InputDesc> &input_descs = network.getInputDesc();
  const std::vector<OutputDesc> &output_descs = network.getOutputDesc();

  //print log info on the model . expected input/output size, etc.
  std::cout << "Model: " << msgpack << std::endl;
  std::cout << "Number of inputs: " << input_descs.size() << std::endl;
  for (size_t i = 0; i < input_descs.size(); ++i) {
    std::vector<int> shape = input_descs[i].shape;
    std::cout << "Input " << i << ": shape = [";
    for (size_t j = 0; j < shape.size(); ++j) {
      std::cout << shape[j];
      if (j < shape.size() - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]" << ", size = " << input_descs[i].size_bytes << " bytes" << std::endl;
  }


  std::cout << "Number of outputs: " << output_descs.size() << std::endl;
  for (size_t i = 0; i < output_descs.size(); ++i) {
    std::vector<int> shape = output_descs[i].shape;
    std::cout << "Output " << i << ": shape = [";
    for (size_t j = 0; j < shape.size(); ++j) {
      std::cout << shape[j];
      if (j < shape.size() - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]" << ", size = " << output_descs[i].size_bytes << " bytes" << std::endl;
  }


  // Create Input
  auto input_size = input_descs[0].size_bytes;
  createBuffer(buffer_manager, input_container_id, static_cast<size_t>(input_size));
  std::cout << "Created input buffer of size " << input_size << std::endl;

  // Create Output
  auto output_size = output_descs[0].size_bytes;
  createBuffer(buffer_manager, output_container_id, static_cast<size_t>(output_size));
  std::cout << "Created output buffer of size " << output_size << std::endl;

  R_EXFWK *exfwk = createExfwk(buffer_manager);
  JobContainer *job_container = new JobContainer;

  // Prepare input memories
  std::unordered_map<PipelineId, std::vector<InputMemory>> user_input_memories;
  std::unordered_map<PipelineId, std::vector<OutputMemory>> user_output_memories;
  user_input_memories[0] = {{buffer_manager, input_container_id, 0, static_cast<int64_t>(input_size), 0}};
  user_output_memories[0] = {{buffer_manager, output_container_id, 0, static_cast<int64_t>(output_size), 0}};

  // Initialize
  ArtifactHelper helper;
  auto config_ret = helper.config({network}, st_hycoah_config_t{});
  if (config_ret != e_hycoah_return_t::RETURN_HYCOAH_OK) {
    std::cout << " Config ARTIFACTHELPER FAILED " << std::endl;
    return -1;
  }

  e_hycoah_return_t ret = helper.init(network, buffer_manager, exfwk, {user_input_memories}, {user_output_memories});
  if (ret != e_hycoah_return_t::RETURN_HYCOAH_OK) {
    std::cout << " INIT ARTIFACTHELPER FAILED " << std::endl;
    return -1;
  }

  // lightable io memory
  auto input_memories = network.getInputMemory(0);
  auto output_memories = network.getOutputMemory(0);

  // prepare actual data
  load_inputs_from_file(inputs, input_memories);

  // prepare job
  // uint8_t pipeline_id = 0;
  std::vector<JobId> job_dependency;
  std::vector<JobId> job_ids;

  // add job.
  auto job_ret = network.addJobs(job_container, 0, job_dependency, job_ids);
  if (job_ret != e_hycoah_return_t::RETURN_HYCOAH_OK) {
    std::cout << " ADD JOB FAILED " << std::endl;
    return -1;
  }
  for (auto &&job_id : job_ids) {
    std::cout << "job_id: " << static_cast<int>(job_id) << std::endl;
  }

  network.syncIO(e_sync_direction_t::H2D, 0);

  auto exec_ret = execute(exfwk, job_container);
  if (exec_ret != e_hycoah_return_t::RETURN_HYCOAH_OK) {
    std::cout << " EXECUTION FAILED " << std::endl;
    return -1;
  }

  network.syncIO(e_sync_direction_t::D2H, 0);

  save_outputs_to_file(output_memories, "outputs.bin");

  std::vector<LatencyStat> execute_time(6);  // "total", "cnnip", "dsp_core0", "dsp_core1", "dsp_core2", "dsp_core3"
  std::vector<Network *> networks{&network};
  time_evaluate(exfwk, job_container, networks, 10, execute_time, core_list, true);
  std::vector<std::string> scope_names = {"total", "cnnip", "dsp_core0", "dsp_core1", "dsp_core2", "dsp_core3"};
  for (size_t i = 0; i < 6; ++i) {
    std::cout << std::setw(9) << std::right << scope_names[i].c_str() << ": n = " << 10 << std::fixed
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
  if (RETURN_BUFMNGR_OK != buffer_manager->R_BufMgr_DeleteContainer(input_container_id)) {
    printf("Failed to get deallocate buffer.\n");
    return -1;
  }
  if (RETURN_BUFMNGR_OK != buffer_manager->R_BufMgr_DeleteContainer(output_container_id)) {
    printf("Failed to get deallocate buffer.\n");
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

// Main function
int main(int argc, char *argv[]) {
  auto args = parseArguments(argc, argv);
  if (args.size() != 2 && args.size() != 3) {
    printf(
        "This example code is to run one model on one DSP core in one container with custom IO.\n"
        "Format: custom_io_app_{suffix} <msgpack> <inputs.bin> [testcase_name]\n"
        "Incorret number of arguments.\n");
    return -1;
  }

  std::string msgpack = args[0];
  std::string inputs = args[1];
  std::string testcase_name = "unnamed";
  if (args.size() == 3) {
    testcase_name = args[2];
  }

  // The core information parsed from the folder name of the msgpack file is used to show latency information.
  // The folder name "sample_model1_cnnip4_dsp3_core0" stands for msgpack compiled with DSP core0.
  // If you want to specify the core information from the code directly, please modify the core_list variable below.
  // For example, please set core_list = {{0}} if the msgpack is compiled with DSP core0.
  std::string folder_name = get_folder_name(msgpack);
  std::vector<std::vector<uint8_t>> core_list = core_list_from_name(folder_name);

  int ret = run_one_model_one_core_one_container_custom_io(msgpack, inputs, testcase_name, core_list);
  return ret;
}
