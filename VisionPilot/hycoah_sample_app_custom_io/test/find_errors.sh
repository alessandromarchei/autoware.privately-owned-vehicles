#!/bin/bash

# ============================================================
# Search HyCoAH / EXFWK debug information
# ============================================================

set -u

# Directory from which this script is launched
START_DIR="$(pwd)"

# R-Car xOS installation
XOS_ROOT="/opt/rcar-xos/v3.47.0"

# Results directory
DEBUG_DIR="${START_DIR}/debug_find"

mkdir -p "$DEBUG_DIR"

echo "============================================================"
echo " HyCoAH / EXFWK DEBUG SEARCH"
echo "============================================================"
echo "XOS root    : $XOS_ROOT"
echo "Output dir  : $DEBUG_DIR"
echo "Started at  : $(date)"
echo "============================================================"
echo


# ============================================================
# 0. ENVIRONMENT / TOOL INFORMATION
# ============================================================

echo "[0/9] Saving environment information..."

{
    echo "Date: $(date)"
    echo
    echo "XOS_ROOT=$XOS_ROOT"
    echo "START_DIR=$START_DIR"
    echo
    echo "grep:"
    command -v grep || true
    echo
    echo "strings:"
    command -v strings || true
    echo
    echo "nm:"
    command -v nm || true
    echo
    echo "pdftotext:"
    command -v pdftotext || true
} > "$DEBUG_DIR/00_environment.txt"

echo "      -> 00_environment.txt"
echo


# ============================================================
# 1. EXACT ERROR STRING IN TEXT FILES
# ============================================================

echo "[1/9] Searching exact error string in text/source files..."
echo "      Pattern: Failed to add paired jobs"

grep -RIna \
    --exclude='*.a' \
    --exclude='*.so' \
    --exclude='*.o' \
    --exclude='*.pdf' \
    "Failed to add paired jobs" \
    "$XOS_ROOT" \
    2>/dev/null \
    > "$DEBUG_DIR/01_exact_error_text.txt" || true

echo "      Matches: $(wc -l < "$DEBUG_DIR/01_exact_error_text.txt")"
echo "      -> 01_exact_error_text.txt"
echo


# ============================================================
# 2. GENERIC "paired" IN TEXT FILES
# ============================================================

echo "[2/9] Searching 'paired' in text/source files..."

grep -RIna \
    --exclude='*.a' \
    --exclude='*.so' \
    --exclude='*.o' \
    --exclude='*.pdf' \
    "paired" \
    "$XOS_ROOT" \
    2>/dev/null \
    > "$DEBUG_DIR/02_paired_text.txt" || true

echo "      Matches: $(wc -l < "$DEBUG_DIR/02_paired_text.txt")"
echo "      -> 02_paired_text.txt"
echo


# ============================================================
# 3. EXACT ERROR STRING IN BINARY FILES
# ============================================================

echo "[3/9] Searching exact error string inside .a/.so/.o..."
echo "      This may take a little while."

find "$XOS_ROOT" \
    -type f \( -name '*.a' -o -name '*.so' -o -name '*.o' \) \
    -exec sh -c '
        for f; do
            if strings "$f" 2>/dev/null |
               grep -Fq "Failed to add paired jobs"; then
                echo "FOUND: $f"
            fi
        done
    ' sh {} + \
    > "$DEBUG_DIR/03_exact_error_binaries.txt" 2>/dev/null || true

echo "      Matches: $(wc -l < "$DEBUG_DIR/03_exact_error_binaries.txt")"
echo "      -> 03_exact_error_binaries.txt"
echo


# ============================================================
# 4. ALL "paired" STRINGS IN BINARY FILES
# ============================================================

echo "[4/9] Searching all 'paired' strings inside .a/.so/.o..."

find "$XOS_ROOT" \
    -type f \( -name '*.a' -o -name '*.so' -o -name '*.o' \) \
    -exec sh -c '
        for f; do

            MATCHES=$(strings "$f" 2>/dev/null | grep -Fi "paired")

            if [ -n "$MATCHES" ]; then
                echo
                echo "============================================================"
                echo "FILE: $f"
                echo "============================================================"
                echo "$MATCHES"
            fi

        done
    ' sh {} + \
    > "$DEBUG_DIR/04_paired_binaries.txt" 2>/dev/null || true

echo "      -> 04_paired_binaries.txt"
echo


# ============================================================
# 5. "paired" IN PDFs WITH CONTEXT
# ============================================================

echo "[5/9] Searching 'paired' inside PDFs..."

if command -v pdftotext >/dev/null 2>&1; then

    find "$XOS_ROOT" \
        -type f \
        -iname '*.pdf' \
        -print0 |
    while IFS= read -r -d '' f; do

        if pdftotext "$f" - 2>/dev/null |
           grep -Fiq "paired"; then

            echo
            echo "============================================================"
            echo "PDF: $f"
            echo "============================================================"

            pdftotext "$f" - 2>/dev/null |
                grep -Fin -C 5 "paired"

        fi

    done > "$DEBUG_DIR/05_paired_pdfs.txt"

else
    echo "ERROR: pdftotext is not installed." \
        > "$DEBUG_DIR/05_paired_pdfs.txt"
fi

echo "      -> 05_paired_pdfs.txt"
echo


# ============================================================
# 6. SPECIFIC JOB PAIR TERMINOLOGY IN PDFs
# ============================================================

echo "[6/9] Searching job-pair terminology inside PDFs..."
echo "      Patterns:"
echo "        add_paired"
echo "        paired job"
echo "        paired jobs"
echo "        job pair"

if command -v pdftotext >/dev/null 2>&1; then

    find "$XOS_ROOT" \
        -type f \
        -iname '*.pdf' \
        -print0 |
    while IFS= read -r -d '' f; do

        MATCHES=$(
            pdftotext "$f" - 2>/dev/null |
            grep -Fin -m20 -E \
                'add_paired|paired job|paired jobs|job pair'
        )

        if [ -n "$MATCHES" ]; then
            echo
            echo "============================================================"
            echo "PDF: $f"
            echo "============================================================"
            echo "$MATCHES"
        fi

    done > "$DEBUG_DIR/06_job_pair_pdfs.txt"

else
    echo "ERROR: pdftotext is not installed." \
        > "$DEBUG_DIR/06_job_pair_pdfs.txt"
fi

echo "      -> 06_job_pair_pdfs.txt"
echo


# ============================================================
# 7. SEARCH add_paired IN TEXT/SOURCE FILES
# ============================================================

echo "[7/9] Searching literal 'add_paired' in source/text files..."

grep -RIna \
    --exclude='*.a' \
    --exclude='*.so' \
    --exclude='*.o' \
    --exclude='*.pdf' \
    "add_paired" \
    "$XOS_ROOT" \
    2>/dev/null \
    > "$DEBUG_DIR/07_add_paired_text.txt" || true

echo "      Matches: $(wc -l < "$DEBUG_DIR/07_add_paired_text.txt")"
echo "      -> 07_add_paired_text.txt"
echo


# ============================================================
# 8. SEARCH SYMBOL JobContainer::add_paired IN BINARIES
# ============================================================

echo "[8/9] Searching JobContainer::add_paired symbols..."
echo "      This should tell us which library/object implements it."

find "$XOS_ROOT" \
    -type f \( -name '*.a' -o -name '*.so' -o -name '*.o' \) \
    -exec nm -A -C {} + \
    2>/dev/null |
    grep -E 'JobContainer::add_paired|add_paired' \
    > "$DEBUG_DIR/08_add_paired_symbols.txt" || true

echo "      Matches: $(wc -l < "$DEBUG_DIR/08_add_paired_symbols.txt")"
echo "      -> 08_add_paired_symbols.txt"
echo


# ============================================================
# 9. SEARCH ALL JobContainer SYMBOLS
# ============================================================

echo "[9/9] Searching all JobContainer symbols..."

find "$XOS_ROOT" \
    -type f \( -name '*.a' -o -name '*.so' -o -name '*.o' \) \
    -exec nm -A -C {} + \
    2>/dev/null |
    grep 'JobContainer::' \
    > "$DEBUG_DIR/09_JobContainer_symbols.txt" || true

echo "      Matches: $(wc -l < "$DEBUG_DIR/09_JobContainer_symbols.txt")"
echo "      -> 09_JobContainer_symbols.txt"
echo


# ============================================================
# SUMMARY
# ============================================================

echo "Creating summary..."

{
    echo "============================================================"
    echo " HyCoAH / EXFWK DEBUG SEARCH SUMMARY"
    echo "============================================================"
    echo
    echo "Date: $(date)"
    echo "XOS root: $XOS_ROOT"
    echo

    echo "------------------------------------------------------------"
    echo "1. Exact error in text files"
    echo "------------------------------------------------------------"
    cat "$DEBUG_DIR/01_exact_error_text.txt"
    echo

    echo "------------------------------------------------------------"
    echo "2. Exact error found inside binaries"
    echo "------------------------------------------------------------"
    cat "$DEBUG_DIR/03_exact_error_binaries.txt"
    echo

    echo "------------------------------------------------------------"
    echo "3. add_paired in text/source"
    echo "------------------------------------------------------------"
    cat "$DEBUG_DIR/07_add_paired_text.txt"
    echo

    echo "------------------------------------------------------------"
    echo "4. add_paired binary symbols"
    echo "------------------------------------------------------------"
    cat "$DEBUG_DIR/08_add_paired_symbols.txt"
    echo

    echo "------------------------------------------------------------"
    echo "5. PDF job-pair references"
    echo "------------------------------------------------------------"
    cat "$DEBUG_DIR/06_job_pair_pdfs.txt"
    echo

} > "$DEBUG_DIR/10_summary.txt"


echo
echo "============================================================"
echo " SEARCH COMPLETED"
echo "============================================================"
echo "Results saved in:"
echo
echo "    $DEBUG_DIR"
echo
echo "Important files to inspect first:"
echo
echo "    10_summary.txt"
echo "    08_add_paired_symbols.txt"
echo "    06_job_pair_pdfs.txt"
echo "    04_paired_binaries.txt"
echo "    09_JobContainer_symbols.txt"
echo
echo "Finished at: $(date)"
echo "============================================================"