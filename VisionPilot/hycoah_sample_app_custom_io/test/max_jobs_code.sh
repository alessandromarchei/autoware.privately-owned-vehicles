OBJDUMP=/home/sergey/Renesas/rcar-xos/v3.47.0/tools/toolchains/poky/sysroots/x86_64-pokysdk-linux/usr/bin/aarch64-poky-linux/aarch64-poky-linux-objdump

mkdir -p debug_jobs/

ar x /opt/rcar-xos/v3.47.0/sw/aarch64-gnu-linux/lib/libexfwk_v4m.a \
    r_exfwk_jobContainer.cpp.o

$OBJDUMP -d -C r_exfwk_jobContainer.cpp.o \
    > debug_jobs/jobcontainer_disassembly.txt

grep -n -A100 -B10 \
    'JobContainer::add_paired' \
    debug_jobs/jobcontainer_disassembly.txt