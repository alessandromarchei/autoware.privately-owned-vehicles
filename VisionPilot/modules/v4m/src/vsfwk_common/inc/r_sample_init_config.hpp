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
#ifndef __R_SAMPLE_INIT_CONFIG_HPP__
#define __R_SAMPLE_INIT_CONFIG_HPP__

/* HWA Execution Framework configuration for initial */
const s_exfwk_config_t default_exfwk_init_cfg =
{
    {   /* COMAL configuration */
        {   /* COMAL socket config settings for push communication */
            R_COMAL_CAF_SHM,            /* type */
            26,                         /* identifier */
            26,                         /* port */
            "127.0.0.1",                /* internal_addr */
            0                           /* channel number */
        },
        {   /* COMAL socket config settings for callback communication */
            R_COMAL_CAF_SHM,            /* type */
            27,                         /* identifier */
            27,                         /* port */
            "127.0.0.1",                /* internal_addr */
            0                           /* channel number */
        }
    },
    {   /* Execution Framework Scheduler Configuration */
        0xd000,                         /* OSAL Message Queue ID */
        100,                            /* Maximum number of messages */
        0xd000                          /* OSAL Thread ID */
    },
    {   /* Execution Framework Dispatcher Configuration */
        0xd000,                         /* OSAL mutex ID for main thread */
        0xd001,                         /* OSAL Thread ID for listen */
        {   /* Execution Framework Client Configuration (mq_id, thread_id, thread_id) */
            {0xd001, 0xd002, 0xd003},
            {0xd002, 0xd004, 0xd005},
            {0xd003, 0xd006, 0xd007},
            {0xd004, 0xd008, 0xd009},
            {0xd005, 0xd00a, 0xd00b},
            {0xd006, 0xd00c, 0xd00d},
            {0xd007, 0xd00e, 0xd00f},
            {0xd008, 0xd010, 0xd011},
            {0xd009, 0xd012, 0xd013},
            {0xd00a, 0xd014, 0xd015},
            {0xd00b, 0xd016, 0xd017},
            {0xd00c, 0xd018, 0xd019},
            {0xd00d, 0xd01a, 0xd01b},
            {0xd00e, 0xd01c, 0xd01d},
            {0xd00f, 0xd01e, 0xd01f},
            {0xd010, 0xd020, 0xd021}
        }
    },
    {   /* HWA configration */
        {   /* IMPX configuration */
            {   /* Common resource configuration */
                0xd011,                 /* OSAL message queue ID */
                5,                      /* OSAL message queue waiting timeout threshold */
                0xd001,                 /* OSAL mutex ID */
                5,                      /* OSAL mutex ID waiting timeout threshold */
                OSAL_INTERRUPT_PRIORITY_TYPE13  /* OSAL interrupt priority */
            },
            {   /* IMP channel configuration */
                {0, OSAL_PM_POLICY_HP},
                {1, OSAL_PM_POLICY_HP}
            },
            {   /* OCV channel configuration */
                {0, OSAL_PM_POLICY_HP},
                {1, OSAL_PM_POLICY_HP},
                {2, OSAL_PM_POLICY_HP},
                {3, OSAL_PM_POLICY_HP}
            },
            {   /* PSCEXE channel configuration */
                {0, OSAL_PM_POLICY_HP}
            },
            {   /* DMAC channel configuration */
                {0, OSAL_PM_POLICY_HP},
                {1, OSAL_PM_POLICY_HP},
                {2, OSAL_PM_POLICY_HP},
                {3, OSAL_PM_POLICY_HP}
            },
            {   /* DMAC_SLIM channel configuration */
                {0, OSAL_PM_POLICY_HP},
                {1, OSAL_PM_POLICY_HP},
                {2, OSAL_PM_POLICY_HP},
                {3, OSAL_PM_POLICY_HP}
            },
            {   /* CNN channel configuration */
                {0, OSAL_PM_POLICY_HP}
            },
            {   /* DSP channel configuration */
                #if defined(RCAR_V4H)
                {0, OSAL_PM_POLICY_HP},
                {1, OSAL_PM_POLICY_HP},
                {2, OSAL_PM_POLICY_HP},
                {3, OSAL_PM_POLICY_HP}
                #elif defined(RCAR_V4M)
                {0, OSAL_PM_POLICY_HP}
                #endif
            }
        },
        {   /* IMR configuration */
            {   /* IMR channel configuration */
                {0, 0xd002, 100, OSAL_INTERRUPT_PRIORITY_TYPE13, OSAL_PM_POLICY_HP},
                {1, 0xd003, 100, OSAL_INTERRUPT_PRIORITY_TYPE13, OSAL_PM_POLICY_HP},
                {2, 0xd004, 100, OSAL_INTERRUPT_PRIORITY_TYPE13, OSAL_PM_POLICY_HP},
                {3, 0xd005, 100, OSAL_INTERRUPT_PRIORITY_TYPE13, OSAL_PM_POLICY_HP},
            }
        },
        {   /* VIP configuration */
            {   /* Common resource configuration */
                0xd007, /* Mutex ID */
                100     /* Timeout Threshold for Mutex */
            },
            {   /* DOF channel configuration */
                {0, {0xd008, 0xd009}, 100, OSAL_PM_POLICY_HP, OSAL_INTERRUPT_PRIORITY_LOWEST, 10}
            },
            {   /* SPO channel configuration */
                {0, {0xd00a, 0xd00b}, 100, OSAL_PM_POLICY_HP, OSAL_INTERRUPT_PRIORITY_LOWEST, 10}
            },
            {   /* SPP channel configuration */
                {0, {0xd00c, 0xd00d}, 100, OSAL_PM_POLICY_HP, OSAL_INTERRUPT_PRIORITY_LOWEST, 10}
            }
        },
        {   /* ISP configuration */
            {   /* Common resource configuration */
                {0xd022, OSAL_THREAD_PRIORITY_TYPE10, 8192, 0xd012}
            },
            {   /* CISP channel configuration */
                {0, 0xd00e, 100, 0xd014, 5000, OSAL_INTERRUPT_PRIORITY_TYPE1}
            },
            {   /* TISP channel configuration */
                {0, 0xd010, 100, OSAL_INTERRUPT_PRIORITY_TYPE1, false}
            },
            {   /* VSPX channel configuration */
                {0, 0xd012, 100, 0xd016, 5000, OSAL_INTERRUPT_PRIORITY_TYPE1}
            }
        },
        {  /* UDF configuration */
            {
                {0xd024, 0xd024, OSAL_THREAD_PRIORITY_TYPE10, 8192},
                {0xd025, 0xd025, OSAL_THREAD_PRIORITY_TYPE10, 8192},
                {0xd026, 0xd026, OSAL_THREAD_PRIORITY_TYPE10, 8192},
            }
        }
    }
};

/* HWA Buffer Manager configuration for initial */
const s_bufmgr_config_t default_bufmgr_init_cfg =
{
    {                                       /* Memory Configuration */
        HWA_MMNGR_ALLOC_MODE_FREE_LIST,     /* allocation_mode */
        0xd030,                             /* mutex_page_table */
        0xd031,                             /* mutex_axi_bookkeeping */
        0xd032,                             /* mutex_buffer */
        0xd033,                             /* mutex_hwa_list */
        0xd034,                             /* mutex_va_manager */
        {                                   /* os_managed */
            {                               /* pools */
                {
                    "hwa_mmngr_pagetable",  /* name */
                    0,                      /* start */
                    0x100000,               /* size */
                    0,                      /* memory_region_idx */
                    0xd035,                 /* mutex_global */
                    0xd036                  /* mutex_monitor */
                },
                {
                    "hwa_mmngr_fcpr_atr",   /* name */
                    0x100000,               /* start */
                    0x1000000,              /* size */
                    0,                      /* memory_region_idx */
                    0xd037,                 /* mutex_global */
                    0xd038                  /* mutex_monitor */
                },
                {
                    "hycoah_normal",        /* name */
                    0x1100000,              /* start */
                    0x18800000,             /* size */
                    0,                      /* memory_region_idx */
                    0xd039,                 /* mutex_global */
                    0xd03a                  /* mutex_monitor */
                },
                #if defined(RCAR_V4H)
                {
                    "hycoah_dsp0",          /* name */
                    0,                      /* start */
                    0x00600000,             /* size */
                    1,                      /* memory_region_idx */
                    0xd03b,                 /* mutex_global */
                    0xd03c                  /* mutex_monitor */
                },
                {
                    "hycoah_dsp1",          /* name */
                    0,                      /* start */
                    0x00600000,             /* size */
                    2,                      /* memory_region_idx */
                    0xd03d,                 /* mutex_global */
                    0xd03e                  /* mutex_monitor */
                },
                {
                    "hycoah_dsp2",          /* name */
                    0,                      /* start */
                    0x00600000,             /* size */
                    3,                      /* memory_region_idx */
                    0xd03f,                 /* mutex_global */
                    0xd040                  /* mutex_monitor */
                },
                {
                    "hycoah_dsp3",          /* name */
                    0,                      /* start */
                    0x00600000,             /* size */
                    4,                      /* memory_region_idx */
                    0xd041,                 /* mutex_global */
                    0xd042                  /* mutex_monitor */
                }
                #elif defined(RCAR_V4M)
                {
                    "hycoah_dsp0",          /* name */
                    0,                      /* start */
                    0x00600000,             /* size */
                    1,                      /* memory_region_idx */
                    0xd03b,                 /* mutex_global */
                    0xd03c                  /* mutex_monitor */
                }
                #endif
            }
        },
        {                                   /* unmanaged */
            /* No data */
        }
    },
    {                                       /* Communication Configuration */
        {
            {
                R_COMAL_CAF_SHM,            /* type */
                25,                         /* identifier */
                25,                         /* port */
                "127.0.0.1",                /* internal_addr */
                0                           /* channel number */
            }
        }
    },
    {                                       /* Factory Configuration */
        0xd020                              /* mutex_id */
    },
    {                                       /* Dispatcher Configuration*/
        {
            {
                R_COMAL_CAF_SHM,            /* type */
                0xd040,                     /* server_thread_id */
                {                           /* dispatcher_thread_ids */
                    0xd041, 0xd042, 0xd043, 0xd044, 0xd045, 0xd046, 0xd047, 0xd048,
                    0xd049, 0xd04a, 0xd04b, 0xd04c, 0xd04d, 0xd04e, 0xd04f, 0xd050
                },
                OSAL_THREAD_PRIORITY_TYPE1, /* server_priority */
                OSAL_THREAD_PRIORITY_TYPE1, /* dispatcher_priority */
                0xd021,                     /* mutex id */
            }
        }
    }
};

#endif  /* __R_SAMPLE_INIT_CONFIG_HPP__ */
