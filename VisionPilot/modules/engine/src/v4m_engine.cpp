#include <cstdio>
#include <stdexcept>
#include "v4m_engine.hpp"


using namespace hycoah;


namespace visionpilot::engine {

// OnnxEngine::OnnxEngine(const Config& cfg)
//     : env_(ORT_LOGGING_LEVEL_WARNING, "VisionPilot")
//     , cfg_(cfg)
// {
//     printf("[OnnxEngine] provider=%s", cfg_.provider.c_str());
//     if (cfg_.provider == "tensorrt" || cfg_.provider == "cuda") {
//         printf("  device=%d", cfg_.device_id);
//     }
//     if (cfg_.provider == "tensorrt") {
//         printf("  precision=%s  workspace=%.1fGB  cache=%s",
//                cfg_.precision.c_str(), cfg_.workspace_gb, cfg_.cache_dir.c_str());
//     }
//     printf("\n");
// }

// // ─── Public entry point ───────────────────────────────────────────────────────

// std::unique_ptr<Ort::Session> OnnxEngine::create_session(
//     const std::string& model_path,
//     const std::string& cache_prefix) const
// {
//     if (cfg_.provider == "cpu") {
//         return create_cpu_session(model_path);
//     }
//     throw std::runtime_error(
//         "[OnnxEngine] Unknown provider '" + cfg_.provider +
//         "'. Valid: cpu");
// }

// // ─── CPU ─────────────────────────────────────────────────────────────────────

// std::unique_ptr<Ort::Session> OnnxEngine::create_cpu_session(
//     const std::string& model_path) const
// {
//     Ort::SessionOptions opts;
//     opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);

//     printf("[OnnxEngine] Creating CPU session → %s\n", model_path.c_str());
//     return std::make_unique<Ort::Session>(env_, model_path.c_str(), opts);
// }

// }  // namespace visionpilot::engine


/*
v4m engine

class V4MEngine {
public:
    explicit V4MEngine(const Config& cfg);

    void create_session(const std::string& model_path) const;

    // Read-only access to config (models may inspect provider, etc.)
    const Config& config() const { return cfg_; }

private:
    void create_npu_session(const std::string& model_path) const;

    Config     cfg_;

};
*/


V4MEngine::V4MEngine(const Config & cfg)
{
    cfg_ = cfg;
    printf("[V4MEngine] provider=%s\n", cfg_.provider.c_str());
}


V4MEngine::~V4MEngine()
{
    //destructor
    cfg_ = Config();


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
}


// int run_one_model_one_core_one_container_custom_io(std::string msgpack, std::string inputs, std::string testcase_name,
//                                                    std::vector<std::vector<uint8_t>> core_list) 


void V4MEngine::create_session(const std::string & model_path) const
{
    //hyco Artifact Helper wrapper for loading a single model (.msgpack) bundle

    // Initialize OSAL
    osal_ret = R_OSAL_Initialize();
    if (OSAL_RETURN_OK != osal_ret) {
        printf("OSAL Initialization failed with error %d\n", osal_ret);
        return -1;
    }

    // user created buffer_manager/exfwk/job_container
    buffer_manager = createBufferManager();
    int input_container_id = 0;
    int output_container_id = 0;

    //create a network object from the .msgpack file
    network = std::make_unique<Network>(msgpack.c_str());


    // get inputs/outputs
    const std::vector<InputDesc> &input_descs = network->getInputDesc();
    const std::vector<OutputDesc> &output_descs = network->getOutputDesc();

    //print log info on the model . expected input/output size, etc.
    std::cout << "Model: " << msgpack << std::endl;
    std::cout << "Number of inputs: " << input_descs.size() << std::endl;
    for (size_t i = 0; i < input_descs.size(); ++i) {
        std::cout << "Input " << i << ": size = " << input_descs[i].size_bytes << " bytes" << std::endl;
    }

    std::cout << "Number of outputs: " << output_descs.size() << std::endl;
    
    for (size_t i = 0; i < output_descs.size(); ++i) {
        std::cout << "Output " << i << ": size = " << output_descs[i].size_bytes << " bytes " << std::endl;
    }


    // Create Input BUFFER
    auto input_size = input_descs[0].size_bytes;
    createBuffer(buffer_manager, input_container_id, static_cast<size_t>(input_size));
    std::cout << "Created input buffer of size " << input_size << std::endl;

    // Create Output BUFFER
    auto output_size = output_descs[0].size_bytes;
    createBuffer(buffer_manager, output_container_id, static_cast<size_t>(output_size));
    std::cout << "Created output buffer of size " << output_size << std::endl;

    //create execution framework and job container
    exfwk = createExfwk(buffer_manager);
    job_container = std::make_unique<JobContainer>();

    // Prepare input memories. set the input and output memories for the network
    user_input_memories[0] = {{buffer_manager, input_container_id, 0, static_cast<int64_t>(input_size), 0}};
    user_output_memories[0] = {{buffer_manager, output_container_id, 0, static_cast<int64_t>(output_size), 0}};

    // Initialize
    auto config_ret = helper.config({network}, st_hycoah_config_t{});
    if (config_ret != e_hycoah_return_t::RETURN_HYCOAH_OK) {
        std::cout << " Config ARTIFACTHELPER FAILED " << std::endl;
        return -1;
    }

    //initialize artifact helper
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


    // add job.
    auto job_ret = network.addJobs(job_container, 0, job_dependency, job_ids);
    if (job_ret != e_hycoah_return_t::RETURN_HYCOAH_OK) {
        std::cout << " ADD JOB FAILED " << std::endl;
        return -1;
    }
    
    for (auto &&job_id : job_ids) {
        std::cout << "job_id: " << static_cast<int>(job_id) << std::endl;
    }


    //move input and output data to/from device and execute the model
    //from host to device for inference
    network.syncIO(e_sync_direction_t::H2D, 0);


    //execute inference
    auto exec_ret = execute(exfwk, job_container);
    if (exec_ret != e_hycoah_return_t::RETURN_HYCOAH_OK) {
        std::cout << " EXECUTION FAILED " << std::endl;
        return -1;
    }


    //from device to host 
    network.syncIO(e_sync_direction_t::D2H, 0);

    //results are available in output_memories
    
}


}// namespace visionpilot::engine