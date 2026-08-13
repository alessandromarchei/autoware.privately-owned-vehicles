# Ipopt V4M cross-compiled smoke test

This program solves a small nonlinear lateral MPC-like problem using the Ipopt
C++ API. It exercises nonlinear constraints, objective gradients, a sparse
Jacobian, Ipopt, MUMPS, libblastrampoline, and OpenBLAS32.

## Cross-compile on the PC

```bash
cd ~/DEV/AI/vision_pilot/ipopt_v4m_smoke_test

unset LD_LIBRARY_PATH
source /home/sergey/Renesas/rcar-xos/v3.47.0/tools/toolchains/poky/environment-setup-cortexa76-poky-linux

cmake -S . -B build -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=/opt/rcar-xos/v3.47.0/cmake/toolchain_poky_5_0_adas.cmake \
  -DIPOPT_ROOT=$HOME/DEV/AI/vision_pilot/ipopt-jll-test/ipopt

cmake --build build --parallel
file build/ipopt_v4m_smoke_test
```

The final `file` command must report an AArch64 ELF executable.

If the actual toolchain file exists only below the Renesas installation rather
than `/opt`, pass its real absolute path instead.

## Copy and run on V4M

```bash
scp build/ipopt_v4m_smoke_test v4m:/home/root/ipopt-jll-test/
```

On V4M:

```bash
cd /home/root/ipopt-jll-test
IPOPT_PREFIX=$PWD/ipopt

LD_LIBRARY_PATH="$IPOPT_PREFIX/lib" \
LBT_DEFAULT_LIBS="$IPOPT_PREFIX/lib/libopenblas.so" \
OPENBLAS_NUM_THREADS=1 \
./ipopt_v4m_smoke_test
```

A successful test ends with:

```text
EXIT: Optimal Solution Found.
PASS: Ipopt + MUMPS + BLAS solved the nonlinear MPC test
```

This executable deliberately does not use `libipoptamplinterface`, `libasl`, or
the unused 64-bit-index MUMPS variants.
