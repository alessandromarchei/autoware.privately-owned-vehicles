#!/usr/bin/env bash
set -Eeuo pipefail

# Recreate a relocatable AArch64/glibc Ipopt bundle for Renesas V4M Poky.
# Place this script in VisionPilot/thirdparty/Ipopt and run it from anywhere.


SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CACHE_DIR="${SCRIPT_DIR}/.downloads"

IPOPT_VERSION="300.1400.1902"
IPOPT_RELEASE="Ipopt-v300.1400.1902+0"
MUMPS_VERSION="500.900.0"
MUMPS_RELEASE="MUMPS_seq-v500.900.0+0"
SPRAL_VERSION="2025.9.18"
SPRAL_RELEASE="SPRAL-v2025.9.18+0"
LBT_VERSION="5.11.2"
LBT_RELEASE="libblastrampoline-v5.11.2+2"
METIS_VERSION="5.1.3"
METIS_RELEASE="METIS-v5.1.3+0"
OPENBLAS_VERSION="0.3.34"
OPENBLAS_RELEASE="OpenBLAS32-v0.3.34+0"
CSL_VERSION="1.5.7"
CSL_RELEASE="CompilerSupportLibraries-v1.5.7+0"

log() { printf '\n==> %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

command -v curl >/dev/null || die "curl is required"
command -v tar >/dev/null || die "tar is required"
command -v readelf >/dev/null || die "readelf is required"

mkdir -p "${CACHE_DIR}"

download() {
    local repository=$1
    local release=$2
    local filename=$3
    local destination="${CACHE_DIR}/${filename}"
    local url="https://github.com/JuliaBinaryWrappers/${repository}/releases/download/${release}/${filename}"

    if [[ -s "${destination}" ]] && tar -tzf "${destination}" >/dev/null 2>&1; then
        printf 'Using cached %s\n' "${filename}"
        return
    fi

    printf 'Downloading %s\n' "${filename}"
    curl --fail --location --retry 4 --retry-all-errors \
         --output "${destination}.part" "${url}"
    tar -tzf "${destination}.part" >/dev/null
    mv "${destination}.part" "${destination}"
}

IPOPT_ARCHIVE="Ipopt.v${IPOPT_VERSION}.aarch64-linux-gnu-libgfortran5-cxx11.tar.gz"
MUMPS_ARCHIVE="MUMPS_seq.v${MUMPS_VERSION}.aarch64-linux-gnu-libgfortran5.tar.gz"
SPRAL_ARCHIVE="SPRAL.v${SPRAL_VERSION}.aarch64-linux-gnu-libgfortran5.tar.gz"
LBT_ARCHIVE="libblastrampoline.v${LBT_VERSION}.aarch64-linux-gnu.tar.gz"
METIS_ARCHIVE="METIS.v${METIS_VERSION}.aarch64-linux-gnu.tar.gz"
OPENBLAS_ARCHIVE="OpenBLAS32.v${OPENBLAS_VERSION}.aarch64-linux-gnu-libgfortran5.tar.gz"
CSL_ARCHIVE="CompilerSupportLibraries.v${CSL_VERSION}.aarch64-linux-gnu-libgfortran5.tar.gz"

log "Downloading pinned AArch64 JLL artifacts"
download "Ipopt_jll.jl" "${IPOPT_RELEASE}" "${IPOPT_ARCHIVE}"
download "MUMPS_seq_jll.jl" "${MUMPS_RELEASE}" "${MUMPS_ARCHIVE}"
download "SPRAL_jll.jl" "${SPRAL_RELEASE}" "${SPRAL_ARCHIVE}"
download "libblastrampoline_jll.jl" "${LBT_RELEASE}" "${LBT_ARCHIVE}"
download "METIS_jll.jl" "${METIS_RELEASE}" "${METIS_ARCHIVE}"
download "OpenBLAS32_jll.jl" "${OPENBLAS_RELEASE}" "${OPENBLAS_ARCHIVE}"
download "CompilerSupportLibraries_jll.jl" "${CSL_RELEASE}" "${CSL_ARCHIVE}"

STAGING_DIR=$(mktemp -d "${TMPDIR:-/tmp}/ipopt-v4m.XXXXXXXX")
trap 'rm -rf -- "${STAGING_DIR}"' EXIT
PREFIX_DIR="${STAGING_DIR}/prefix"
RUNTIME_DIR="${STAGING_DIR}/runtime"
mkdir -p "${PREFIX_DIR}" "${RUNTIME_DIR}"

log "Assembling bundle in a staging directory"
for archive in \
    "${IPOPT_ARCHIVE}" \
    "${MUMPS_ARCHIVE}" \
    "${SPRAL_ARCHIVE}" \
    "${LBT_ARCHIVE}" \
    "${METIS_ARCHIVE}" \
    "${OPENBLAS_ARCHIVE}"
do
    tar -xzf "${CACHE_DIR}/${archive}" -C "${PREFIX_DIR}"
done

# Do not import the artifact's libgcc_s/libgomp: Poky provides native GCC 13
# runtimes. Only the missing Fortran ABI runtime and Quadmath belong here.
tar -xzf "${CACHE_DIR}/${CSL_ARCHIVE}" -C "${RUNTIME_DIR}"
mkdir -p "${PREFIX_DIR}/lib"
find "${RUNTIME_DIR}/lib" -maxdepth 1 \
    \( -name 'libgfortran.so*' -o -name 'libquadmath.so*' \) \
    -exec cp -a {} "${PREFIX_DIR}/lib/" \;

log "Validating the staged bundle"
required_files=(
    "include/coin-or/IpIpoptApplication.hpp"
    "lib/libipopt.so"
    "lib/libspral.so"
    "lib/libdmumps.so"
    "lib/libmumps_common.so"
    "lib/libpord.so"
    "lib/libmpiseq.so"
    "lib/libmetis.so"
    "lib/libblastrampoline.so.5"
    "lib/libopenblas.so"
    "lib/libgfortran.so.5"
)
for relative_path in "${required_files[@]}"; do
    [[ -e "${PREFIX_DIR}/${relative_path}" ]] || \
        die "artifact assembly did not produce ${relative_path}"
done

readelf -h "${PREFIX_DIR}/lib/libipopt.so" | grep -q 'AArch64' || \
    die "libipopt.so is not an AArch64 ELF library"
readelf -d "${PREFIX_DIR}/lib/libipopt.so" | grep -q 'libdmumps.so' || \
    die "unexpected Ipopt dependency set"

log "Installing directly into ${SCRIPT_DIR}"
# Only replace directories owned by these artifacts. The script itself, Git
# metadata and unrelated files in thirdparty/Ipopt remain untouched.
for directory in bin include lib modules share; do
    if [[ -e "${PREFIX_DIR}/${directory}" ]]; then
        rm -rf -- "${SCRIPT_DIR:?}/${directory}"
        mv "${PREFIX_DIR}/${directory}" "${SCRIPT_DIR}/${directory}"
    fi
done

cat > "${SCRIPT_DIR}/VERSIONS.txt" <<EOF
Target: aarch64-linux-gnu (glibc, libgfortran5, C++11 ABI)
Ipopt ${IPOPT_VERSION}
MUMPS sequential ${MUMPS_VERSION}
SPRAL ${SPRAL_VERSION}
libblastrampoline ${LBT_VERSION}
METIS ${METIS_VERSION}
OpenBLAS32 ${OPENBLAS_VERSION}
CompilerSupportLibraries ${CSL_VERSION} (libgfortran/libquadmath only)
EOF

log "Bundle ready"
printf 'IPOPT_ROOT=%s\n' "${SCRIPT_DIR}"
printf 'Configure with: -DIPOPT_ROOT=%q\n' "${SCRIPT_DIR}"
printf 'Runtime variables on V4M:\n'
printf '  export LD_LIBRARY_PATH=%q/lib\n' "${SCRIPT_DIR}"
printf '  export LBT_DEFAULT_LIBS=%q/lib/libopenblas.so\n' "${SCRIPT_DIR}"
printf '  export OPENBLAS_NUM_THREADS=1\n'
