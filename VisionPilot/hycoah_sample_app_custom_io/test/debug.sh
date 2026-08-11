# ============================================================
# 0. SETUP
# ============================================================

mkdir -p debug

HYCOAH_LIB=/opt/rcar-xos/v3.47.0/sw/aarch64-gnu-linux/lib/libhycoah_v4m.a
EXFWK_LIB=/opt/rcar-xos/v3.47.0/sw/aarch64-gnu-linux/lib/libexfwk_v4m.a

OBJDUMP=/home/sergey/Renesas/rcar-xos/v3.47.0/tools/toolchains/poky/sysroots/x86_64-pokysdk-linux/usr/bin/aarch64-poky-linux/aarch64-poky-linux-objdump


# ============================================================
# 1. ELENCO DEGLI OBJECT FILE CONTENUTI IN libhycoah_v4m.a
# ============================================================

ar -t "$HYCOAH_LIB" \
    > debug/01_hycoah_archive_members.txt


# ============================================================
# 2. TUTTI I SIMBOLI addJobs DI HYCOAH
#
# Serve a ricostruire:
# Network::addJobs
#   -> ArtifactHelperImpl::addJobs
#   -> ModelHelper::addJobs
# ============================================================

nm -A -C "$HYCOAH_LIB" \
    | grep 'addJobs' \
    > debug/02_hycoah_addJobs_symbols.txt


# ============================================================
# 3. CERCA SPECIFICAMENTE ModelHelper::addJobs
#
# Qui vediamo anche quale .o della libreria la implementa.
# ============================================================

nm -A -C "$HYCOAH_LIB" \
    | grep 'ModelHelper::addJobs' \
    > debug/03_ModelHelper_addJobs_location.txt


# ============================================================
# 4. TUTTI I SIMBOLI JobContainer PRESENTI IN EXFWK
# ============================================================

nm -A -C "$EXFWK_LIB" \
    | grep 'JobContainer::' \
    > debug/04_exfwk_JobContainer_symbols.txt


# ============================================================
# 5. CERCA LE API DI JobContainer RELATIVE A JOB/PAIR
# ============================================================

nm -A -C "$EXFWK_LIB" \
    | grep 'JobContainer::' \
    | grep -Ei 'add|pair|insert|push|depend|job' \
    > debug/05_exfwk_JobContainer_job_functions.txt


# ============================================================
# 6. CERCA add_paired IN PARTICOLARE
# ============================================================

nm -A -C "$EXFWK_LIB" \
    | grep 'add_paired' \
    > debug/06_exfwk_add_paired_location.txt


# ============================================================
# 7. ESTRAI r_hycoah_model_helper.cpp.o
# ============================================================

cd debug

ar x "$HYCOAH_LIB" r_hycoah_model_helper.cpp.o

cd ..

ls -lh debug/r_hycoah_model_helper.cpp.o \
    > debug/07_ModelHelper_object_info.txt

file debug/r_hycoah_model_helper.cpp.o \
    >> debug/07_ModelHelper_object_info.txt


# ============================================================
# 8. TUTTI I SIMBOLI UNDEFINED RICHIESTI DA ModelHelper
#
# "U" = questa funzione viene chiamata da questo .o
#       ma è implementata altrove.
# ============================================================

nm -C debug/r_hycoah_model_helper.cpp.o \
    | grep ' U ' \
    > debug/08_ModelHelper_external_dependencies.txt


# ============================================================
# 9. SOLO DIPENDENZE RELATIVE A EXFWK / JOB CONTAINER
# ============================================================

nm -C debug/r_hycoah_model_helper.cpp.o \
    | grep ' U ' \
    | grep -Ei 'Job|Container|EXFWK|pair|depend|add|push' \
    > debug/09_ModelHelper_job_dependencies.txt


# ============================================================
# 10. DISASSEMBLY COMPLETO DI ModelHelper
#
# -d = disassemble
# -r = mostra relocations, FONDAMENTALE per vedere le chiamate
# -C = demangle dei simboli C++
# ============================================================

"$OBJDUMP" -drC \
    debug/r_hycoah_model_helper.cpp.o \
    > debug/10_ModelHelper_disassembly.txt


# ============================================================
# 11. TROVA ModelHelper::addJobs NEL DISASSEMBLY
# ============================================================

grep -n \
    'ModelHelper::addJobs' \
    debug/10_ModelHelper_disassembly.txt \
    > debug/11_ModelHelper_addJobs_locations_in_disassembly.txt


# ============================================================
# 12. TUTTE LE CHIAMATE A JobContainer NEL DISASSEMBLY
# ============================================================

grep -n \
    'JobContainer::' \
    debug/10_ModelHelper_disassembly.txt \
    > debug/12_ModelHelper_JobContainer_calls.txt


# ============================================================
# 13. CERCA SPECIFICAMENTE add_paired
# ============================================================

grep -n -B30 -A40 \
    'add_paired' \
    debug/10_ModelHelper_disassembly.txt \
    > debug/13_ModelHelper_add_paired_context.txt


# ============================================================
# 14. STRINGHE DI ERRORE PRESENTI NELL'OBJECT
# ============================================================

strings -t x debug/r_hycoah_model_helper.cpp.o \
    | grep -Ei 'paired|job container|EXFWK|Failed to add' \
    > debug/14_ModelHelper_error_strings.txt

# ============================================================
# 15. RELOCATION TABLE
#
# Molto utile perché permette di vedere esattamente
# quali funzioni vengono chiamate dalle istruzioni BL.
# ============================================================

"$OBJDUMP" -rC \
    debug/r_hycoah_model_helper.cpp.o \
    > debug/15_ModelHelper_relocations.txt

# ============================================================
# 16. SUMMARY DELLE COSE CHE CI INTERESSANO
# ============================================================

{
    echo "========================================"
    echo "HYCOAH addJobs"
    echo "========================================"
    cat debug/02_hycoah_addJobs_symbols.txt

    echo
    echo "========================================"
    echo "ModelHelper job dependencies"
    echo "========================================"
    cat debug/09_ModelHelper_job_dependencies.txt

    echo
    echo "========================================"
    echo "EXFWK add_paired"
    echo "========================================"
    cat debug/06_exfwk_add_paired_location.txt

    echo
    echo "========================================"
    echo "ModelHelper -> JobContainer calls"
    echo "========================================"
    cat debug/12_ModelHelper_JobContainer_calls.txt

} > debug/16_summary.txt