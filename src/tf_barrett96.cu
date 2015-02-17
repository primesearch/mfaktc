/*
This file is part of mfaktc.
Copyright (C) 2009, 2010, 2011, 2012, 2014  Oliver Weihe (o.weihe@t-online.de)

mfaktc is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

mfaktc is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with mfaktc.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>  

#include "params.h"
#include "my_types.h"
#include "compatibility.h"
#include "my_intrinsics.h"

#define NVCC_EXTERN
#include "sieve.h"
#include "timer.h"
#include "output.h"
#undef NVCC_EXTERN

#include "tf_debug.h"
#include "tf_96bit_base_math.cu"
#include "tf_96bit_helper.cu"

#undef INV_160_96
#include "tf_barrett96_div.cu"
#define INV_160_96
#include "tf_barrett96_div.cu"
#undef INV_160_96

#define CPU_SIEVE
#include "tf_barrett96_core.cu"
#undef CPU_SIEVE


#if __CUDA_ARCH__ >= FERMI
  #define KERNEL_MIN_BLOCKS 2
#else
  #define KERNEL_MIN_BLOCKS 1
#endif


/*
The following functions receive a list of factor candidates (FCs) to test whether the divide M(exp) = (2^exp)-1 or not.
Each thread tests *exactly* one FC.

Input:
exp               the exponent (mersenne number currently working on)
k                 the common base of k
k_tab             offsets to k, exactly one value per thread (FC for thread X is: 2 * (k + k_tab[X]) * exp + 1)
shiftcount        how many trailing bits of exp are needed for exponentation
b                 precomputed value of exponentation without mod
bit_max64         the maximum size of the FCs (ln2(FC) - 64), only some kernels
modbasecase_debug array where debug data is written to

Output:
RES               integer array where the results (FCs which actually divide M(exp))
*/


__global__ void
#ifndef DEBUG_GPU_MATH
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS) mfaktc_barrett92(unsigned int exp, int96 k, unsigned int *k_tab, int shiftcount, int192 b, unsigned int *RES, int bit_max64)
#else
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS) mfaktc_barrett92(unsigned int exp, int96 k, unsigned int *k_tab, int shiftcount, int192 b, unsigned int *RES, int bit_max64, unsigned int *modbasecase_debug)
#endif
{
  int96 f;
  int index = blockDim.x * blockIdx.x + threadIdx.x;

  create_FC96(&f, exp, k, k_tab[index]);

  test_FC96_barrett92(f, b, exp, RES, bit_max64, shiftcount
#ifdef DEBUG_GPU_MATH
                      , modbasecase_debug
#endif
                      );
}


__global__ void
#ifndef DEBUG_GPU_MATH
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS) mfaktc_barrett88(unsigned int exp, int96 k, unsigned int *k_tab, int shiftcount, int192 b, unsigned int *RES, int bit_max64)
#else
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS) mfaktc_barrett88(unsigned int exp, int96 k, unsigned int *k_tab, int shiftcount, int192 b, unsigned int *RES, int bit_max64, unsigned int *modbasecase_debug)
#endif
{
  int96 f;
  int index = blockDim.x * blockIdx.x + threadIdx.x;

  create_FC96(&f, exp, k, k_tab[index]);
  
  test_FC96_barrett88(f, b, exp, RES, bit_max64, shiftcount
#ifdef DEBUG_GPU_MATH
                      , modbasecase_debug
#endif
                      );
}


__global__ void
#ifndef DEBUG_GPU_MATH
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS) mfaktc_barrett87(unsigned int exp, int96 k, unsigned int *k_tab, int shiftcount, int192 b, unsigned int *RES, int bit_max64)
#else
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS) mfaktc_barrett87(unsigned int exp, int96 k, unsigned int *k_tab, int shiftcount, int192 b, unsigned int *RES, int bit_max64, unsigned int *modbasecase_debug)
#endif
{
  int96 f;
  int index = blockDim.x * blockIdx.x + threadIdx.x;

  create_FC96(&f, exp, k, k_tab[index]);

  test_FC96_barrett87(f, b, exp, RES, bit_max64, shiftcount
#ifdef DEBUG_GPU_MATH
                      , modbasecase_debug
#endif
                      );
}


__global__ void
#ifndef DEBUG_GPU_MATH
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS) mfaktc_barrett79(unsigned int exp, int96 k, unsigned int *k_tab, int shiftcount, int192 b, unsigned int *RES)
#else
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS) mfaktc_barrett79(unsigned int exp, int96 k, unsigned int *k_tab, int shiftcount, int192 b, unsigned int *RES, int bit_max64, unsigned int *modbasecase_debug)
#endif
{
  int96 f;
  int index = blockDim.x * blockIdx.x + threadIdx.x;

  create_FC96_mad(&f, exp, k, k_tab[index]);

  test_FC96_barrett79(f, b, exp, RES, shiftcount
#ifdef DEBUG_GPU_MATH
                      , bit_max64, modbasecase_debug
#endif
                      );
}


__global__ void
#ifndef DEBUG_GPU_MATH
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS) mfaktc_barrett77(unsigned int exp, int96 k, unsigned int *k_tab, int shiftcount, int192 b, unsigned int *RES)
#else
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS) mfaktc_barrett77(unsigned int exp, int96 k, unsigned int *k_tab, int shiftcount, int192 b, unsigned int *RES, int bit_max64, unsigned int *modbasecase_debug)
#endif
{
  int96 f;
  int index = blockDim.x * blockIdx.x + threadIdx.x;

  create_FC96_mad(&f, exp, k, k_tab[index]);
  
  test_FC96_barrett77(f, b, exp, RES, shiftcount
#ifdef DEBUG_GPU_MATH
                      , bit_max64, modbasecase_debug
#endif
                      );
}


__global__ void
#ifndef DEBUG_GPU_MATH
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS) mfaktc_barrett76(unsigned int exp, int96 k, unsigned int *k_tab, int shiftcount, int192 b, unsigned int *RES)
#else
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS) mfaktc_barrett76(unsigned int exp, int96 k, unsigned int *k_tab, int shiftcount, int192 b, unsigned int *RES, int bit_max64, unsigned int *modbasecase_debug)
#endif
{
  int96 f;
  int index = blockDim.x * blockIdx.x + threadIdx.x;

  create_FC96_mad(&f, exp, k, k_tab[index]);
  
  test_FC96_barrett76(f, b, exp, RES, shiftcount
#ifdef DEBUG_GPU_MATH
                      , bit_max64, modbasecase_debug
#endif
                      );
}


#define TF_BARRETT

#define TF_BARRETT_92BIT
#include "tf_common.cu"
#undef TF_BARRETT_92BIT

#define TF_BARRETT_88BIT
#include "tf_common.cu"
#undef TF_BARRETT_88BIT

#define TF_BARRETT_87BIT
#include "tf_common.cu"
#undef TF_BARRETT_87BIT

#define TF_BARRETT_79BIT
#include "tf_common.cu"
#undef TF_BARRETT_79BIT

#define TF_BARRETT_77BIT
#include "tf_common.cu"
#undef TF_BARRETT_77BIT

#define TF_BARRETT_76BIT
#include "tf_common.cu"
#undef TF_BARRETT_76BIT

#undef TF_BARRETT
