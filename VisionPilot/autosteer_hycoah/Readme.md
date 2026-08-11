# Application of Single NN Model

# 1. License

Copyright [2025-2026] Renesas Electronics Corporation and/or its licensors. All Rights Reserved.

The contents of this repository (the "contents") are proprietary and confidential to Renesas Electronics Corporation and/or its licensors ("Renesas") and subject to statutory and contractual protections.
Unless otherwise expressly agreed in writing between Renesas and you: 1) you may not use, copy, modify, distribute, display, or perform the contents; 2) you may not use any name or mark of Renesas for advertising or publicity purposes or in connection with your use of the contents; 3) RENESAS MAKES NO WARRANTY OR REPRESENTATIONS ABOUT THE SUITABILITY OF THE CONTENTS FOR ANY PURPOSE; THE CONTENTS ARE PROVIDED "AS IS" WITHOUT ANY EXPRESS OR IMPLIED WARRANTY, INCLUDING THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT; AND 4) RENESAS SHALL NOT BE LIABLE FOR ANY DIRECT, INDIRECT, SPECIAL, OR CONSEQUENTIAL DAMAGES, INCLUDING DAMAGES RESULTING FROM LOSS OF USE, DATA, OR PROJECTS, WHETHER IN AN ACTION OF CONTRACT OR TORT, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THE CONTENTS. Third-party contents included in this file may be subject to different terms.

# 2. Description

This is a sample application to run single NN model
- on single DSP core in single container, or
- on single DSP core in multiple job containers, or
- on multiple DSP cores in multiple job containers.

# 3. How to use this sample application

## 3.1. Prepare data and scripts

1. Copy the compiled bin file `hycoah_sample_app_single_nn_{suffix}` (located under Renesas/rcar-xos/v3.xx.x/sw/<toolchain_env>/bin) of this sample application to the board, where `{suffix}` could be `v4h2` or `v4m`. If you need to build the bin file yourself, please refer to the README.md or README.html located at the top level of the rcar-xos release folder (Renesas/rcar-xos/v3.xx.x/ directory).
2. Copy shell scripts to run the sample application from `test_data` folder under this hycoah_sample_app_single_nn folder to the board.
3. Copy data from `test_data/{board}` to the board, where `{board}` could be `v4h` or `v4m`.
4. File architecture under board file system should look as below:

```
├── hycoah_sample_app_single_nn_{suffix}
├── run_single_nn_single_core_single_container.sh
├── run_single_nn_single_core_multi_container.sh
├── run_single_nn_multi_core_multi_container.sh (only for v4h)
├── sample_model1_cnnip4_dsp3_core0
│   ├── abundle.msgpack
|   ├── inputs.bin
│   └── inputs1.bin
└── sample_model1_cnnip4_dsp3_core0_core1 (only for v4h)
    ├── abundle.msgpack
    ├── inputs.bin
    └── inputs1.bin
```

Latency-related reporting parses the DSP core information from the msgpack artifact folder name (i.e., sample_model1_cnnip4_dsp3_core0, sample_model1_cnnip4_dsp3_core0_core1). The suffix _core0 indicates that the msgpack artifact was compiled for DSP core 0, whereas the suffix _core0_core1 indicates that the msgpack artifact was compiled for DSP core 0 and core 1. If the folder name format is invalid and the core information cannot be parsed, the following error will be shown in the execution log:
```
DSP core(s) are not correctly specified in msgpack artifact folder name.
```

## 3.2 Inference

1. Inference one frame on single DSP core in single job container:

```sh
sh run_single_nn_app_single_core_single_container.sh --{board}
```

2. Inference two frames on single DSP core in multiple job containers:

```sh
sh run_single_nn_app_single_core_multi_container.sh --{board}
```

3. Inference two frames on multiple DSP cores in multiple job containers:

```sh
sh run_single_nn_app_multi_core_multi_container.sh --v4h
```

where `{board}` could be either `v4h` or `v4m`.
