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

#include "tf_barrett96_core.cu"

#include "gpusieve_helper.cu"

#if __CUDA_ARCH__ >= FERMI
#define KERNEL_MIN_BLOCKS 2
#else
#define KERNEL_MIN_BLOCKS 1
#endif

__global__ void
#ifndef DEBUG_GPU_MATH
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS)
    mfaktc_barrett92_gs(unsigned int exp, int96 k_base, unsigned int *bit_array, unsigned int bits_to_process, int shiftcount,
                        int192 b_preinit, unsigned int *RES, int bit_max64)
#else
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS)
    mfaktc_barrett92_gs(unsigned int exp, int96 k_base, unsigned int *bit_array, unsigned int bits_to_process, int shiftcount,
                        int192 b_preinit, unsigned int *RES, int bit_max64, unsigned int *modbasecase_debug)
#endif
/*
computes 2^exp mod f
shiftcount is used for precomputing without mod
a is precomputed on host ONCE.
bit_max64 is the number of bits in the factor (minus 64)
*/
{
    int96 f, f_base;
    int i, initial_shifter_value, total_bit_count, k_delta;
    extern __shared__ unsigned short k_deltas[]; // Write bits to test here.  Launching program must estimate
        // how much shared memory to allocate based on number of primes sieved.

    create_k_deltas(bit_array, bits_to_process, &total_bit_count, k_deltas);
    create_fbase96(&f_base, k_base, exp, bits_to_process);

    initial_shifter_value = exp << (32 - shiftcount); // Initial shifter value

    // Loop til the k values written to shared memory are exhausted
    for (i = threadIdx.x; i < total_bit_count; i += THREADS_PER_BLOCK) {
        // Get the (k - k_base) value to test
        k_delta = k_deltas[i];

        // Compute new f.  This is computed as f = f_base + 2 * (k - k_base) * exp.
        f.d0 = __add_cc(f_base.d0, __umul32(2 * k_delta * NUM_CLASSES, exp));
        f.d1 = __addc_cc(f_base.d1, __umul32hi(2 * k_delta * NUM_CLASSES, exp));
        f.d2 = __addc(f_base.d2, 0);

        test_FC96_barrett92(f, b_preinit, initial_shifter_value, RES, bit_max64
#ifdef DEBUG_GPU_MATH
                            ,
                            modbasecase_debug
#endif
        );
    }
}

__global__ void
#ifndef DEBUG_GPU_MATH
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS)
    mfaktc_barrett88_gs(unsigned int exp, int96 k_base, unsigned int *bit_array, unsigned int bits_to_process, int shiftcount,
                        int192 b_preinit, unsigned int *RES, int bit_max64)
#else
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS)
    mfaktc_barrett88_gs(unsigned int exp, int96 k_base, unsigned int *bit_array, unsigned int bits_to_process, int shiftcount,
                        int192 b_preinit, unsigned int *RES, int bit_max64, unsigned int *modbasecase_debug)
#endif
/*
computes 2^exp mod f
shiftcount is used for precomputing without mod
a is precomputed on host ONCE.
bit_max64 is the number of bits in the factor (minus 64)
*/
{
    int96 f, f_base;
    int i, initial_shifter_value, total_bit_count, k_delta;
    extern __shared__ unsigned short k_deltas[]; // Write bits to test here.  Launching program must estimate
        // how much shared memory to allocate based on number of primes sieved.

    create_k_deltas(bit_array, bits_to_process, &total_bit_count, k_deltas);
    create_fbase96(&f_base, k_base, exp, bits_to_process);

    initial_shifter_value = exp << (32 - shiftcount); // Initial shifter value

    // Loop til the k values written to shared memory are exhausted
    for (i = threadIdx.x; i < total_bit_count; i += THREADS_PER_BLOCK) {
        // Get the (k - k_base) value to test
        k_delta = k_deltas[i];

        // Compute new f.  This is computed as f = f_base + 2 * (k - k_base) * exp.
        f.d0 = __add_cc(f_base.d0, __umul32(2 * k_delta * NUM_CLASSES, exp));
        f.d1 = __addc_cc(f_base.d1, __umul32hi(2 * k_delta * NUM_CLASSES, exp));
        f.d2 = __addc(f_base.d2, 0);

        test_FC96_barrett88(f, b_preinit, initial_shifter_value, RES, bit_max64
#ifdef DEBUG_GPU_MATH
                            ,
                            modbasecase_debug
#endif
        );
    }
}

__global__ void
#ifndef DEBUG_GPU_MATH
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS)
    mfaktc_barrett87_gs(unsigned int exp, int96 k_base, unsigned int *bit_array, unsigned int bits_to_process, int shiftcount,
                        int192 b_preinit, unsigned int *RES, int bit_max64)
#else
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS)
    mfaktc_barrett87_gs(unsigned int exp, int96 k_base, unsigned int *bit_array, unsigned int bits_to_process, int shiftcount,
                        int192 b_preinit, unsigned int *RES, int bit_max64, unsigned int *modbasecase_debug)
#endif
/*
computes 2^exp mod f
shiftcount is used for precomputing without mod
a is precomputed on host ONCE.
bit_max64 is the number of bits in the factor (minus 64)
*/
{
    int96 f, f_base;
    int i, initial_shifter_value, total_bit_count, k_delta;
    extern __shared__ unsigned short k_deltas[]; // Write bits to test here.  Launching program must estimate
        // how much shared memory to allocate based on number of primes sieved.

    create_k_deltas(bit_array, bits_to_process, &total_bit_count, k_deltas);
    create_fbase96(&f_base, k_base, exp, bits_to_process);

    initial_shifter_value = exp << (32 - shiftcount); // Initial shifter value

    // Loop til the k values written to shared memory are exhausted
    for (i = threadIdx.x; i < total_bit_count; i += THREADS_PER_BLOCK) {
        // Get the (k - k_base) value to test
        k_delta = k_deltas[i];

        // Compute new f.  This is computed as f = f_base + 2 * (k - k_base) * exp.
        f.d0 = __add_cc(f_base.d0, __umul32(2 * k_delta * NUM_CLASSES, exp));
        f.d1 = __addc_cc(f_base.d1, __umul32hi(2 * k_delta * NUM_CLASSES, exp));
        f.d2 = __addc(f_base.d2, 0);

        test_FC96_barrett87(f, b_preinit, initial_shifter_value, RES, bit_max64
#ifdef DEBUG_GPU_MATH
                            ,
                            modbasecase_debug
#endif
        );
    }
}

__global__ void
#ifndef DEBUG_GPU_MATH
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS)
    mfaktc_barrett79_gs(unsigned int exp, int96 k_base, unsigned int *bit_array, unsigned int bits_to_process, int shiftcount,
                        int192 b_preinit, unsigned int *RES)
#else
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS)
    mfaktc_barrett79_gs(unsigned int exp, int96 k_base, unsigned int *bit_array, unsigned int bits_to_process, int shiftcount,
                        int192 b_preinit, unsigned int *RES, int bit_max64, unsigned int *modbasecase_debug)
#endif
/*
computes 2^exp mod f
shiftcount is used for precomputing without mod
a is precomputed on host ONCE.
*/
{
    int96 f, f_base;
    int i, initial_shifter_value, total_bit_count, k_delta;
    extern __shared__ unsigned short k_deltas[]; // Write bits to test here.  Launching program must estimate
        // how much shared memory to allocate based on number of primes sieved.

    create_k_deltas(bit_array, bits_to_process, &total_bit_count, k_deltas);
    create_fbase96(&f_base, k_base, exp, bits_to_process);

    initial_shifter_value = exp << (32 - shiftcount); // Initial shifter value

    // Loop til the k values written to shared memory are exhausted
    for (i = threadIdx.x; i < total_bit_count; i += THREADS_PER_BLOCK) {
        // Get the (k - k_base) value to test
        k_delta = k_deltas[i];

        // Compute new f.  This is computed as f = f_base + 2 * (k - k_base) * exp.
        f.d0 = __add_cc(f_base.d0, __umul32(2 * k_delta * NUM_CLASSES, exp));
        f.d1 = __addc_cc(f_base.d1, __umul32hi(2 * k_delta * NUM_CLASSES, exp));
        f.d2 = __addc(f_base.d2, 0);

        test_FC96_barrett79(f, b_preinit, initial_shifter_value, RES
#ifdef DEBUG_GPU_MATH
                            ,
                            bit_max64, modbasecase_debug
#endif
        );
    }
}

__global__ void
#ifndef DEBUG_GPU_MATH
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS)
    mfaktc_barrett77_gs(unsigned int exp, int96 k_base, unsigned int *bit_array, unsigned int bits_to_process, int shiftcount,
                        int192 b_preinit, unsigned int *RES)
#else
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS)
    mfaktc_barrett77_gs(unsigned int exp, int96 k_base, unsigned int *bit_array, unsigned int bits_to_process, int shiftcount,
                        int192 b_preinit, unsigned int *RES, int bit_max64, unsigned int *modbasecase_debug)
#endif
/*
computes 2^exp mod f
shiftcount is used for precomputing without mod
a is precomputed on host ONCE.
*/
{
    int96 f, f_base;
    int i, initial_shifter_value, total_bit_count, k_delta;
    extern __shared__ unsigned short k_deltas[]; // Write bits to test here.  Launching program must estimate
        // how much shared memory to allocate based on number of primes sieved.

    create_k_deltas(bit_array, bits_to_process, &total_bit_count, k_deltas);
    create_fbase96(&f_base, k_base, exp, bits_to_process);

    initial_shifter_value = exp << (32 - shiftcount); // Initial shifter value

    // Loop til the k values written to shared memory are exhausted
    for (i = threadIdx.x; i < total_bit_count; i += THREADS_PER_BLOCK) {
        // Get the (k - k_base) value to test
        k_delta = k_deltas[i];

        // Compute new f.  This is computed as f = f_base + 2 * (k - k_base) * exp.
        f.d0 = __add_cc(f_base.d0, __umul32(2 * k_delta * NUM_CLASSES, exp));
        f.d1 = __addc_cc(f_base.d1, __umul32hi(2 * k_delta * NUM_CLASSES, exp));
        f.d2 = __addc(f_base.d2, 0);

        test_FC96_barrett77(f, b_preinit, initial_shifter_value, RES
#ifdef DEBUG_GPU_MATH
                            ,
                            bit_max64, modbasecase_debug
#endif
        );
    }
}

__global__ void
#ifndef DEBUG_GPU_MATH
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS)
    mfaktc_barrett76_gs(unsigned int exp, int96 k_base, unsigned int *bit_array, unsigned int bits_to_process, int shiftcount,
                        int192 b_preinit, unsigned int *RES)
#else
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS)
    mfaktc_barrett76_gs(unsigned int exp, int96 k_base, unsigned int *bit_array, unsigned int bits_to_process, int shiftcount,
                        int192 b_preinit, unsigned int *RES, int bit_max64, unsigned int *modbasecase_debug)
#endif
/*
computes 2^exp mod f
shiftcount is used for precomputing without mod
a is precomputed on host ONCE.
*/
{
    int96 f, f_base;
    int i, initial_shifter_value, total_bit_count, k_delta;
    extern __shared__ unsigned short k_deltas[]; // Write bits to test here.  Launching program must estimate
        // how much shared memory to allocate based on number of primes sieved.

    create_k_deltas(bit_array, bits_to_process, &total_bit_count, k_deltas);
    create_fbase96(&f_base, k_base, exp, bits_to_process);

    initial_shifter_value = exp << (32 - shiftcount); // Initial shifter value

    // Loop til the k values written to shared memory are exhausted
    for (i = threadIdx.x; i < total_bit_count; i += THREADS_PER_BLOCK) {
        // Get the (k - k_base) value to test
        k_delta = k_deltas[i];

        // Compute new f.  This is computed as f = f_base + 2 * (k - k_base) * exp.
        f.d0 = __add_cc(f_base.d0, __umul32(2 * k_delta * NUM_CLASSES, exp));
        f.d1 = __addc_cc(f_base.d1, __umul32hi(2 * k_delta * NUM_CLASSES, exp));
        f.d2 = __addc(f_base.d2, 0);

        test_FC96_barrett76(f, b_preinit, initial_shifter_value, RES
#ifdef DEBUG_GPU_MATH
                            ,
                            bit_max64, modbasecase_debug
#endif
        );
    }
}

#define TF_BARRETT

#define TF_BARRETT_92BIT_GS
#include "tf_common_gs.cu"
#undef TF_BARRETT_92BIT_GS

#define TF_BARRETT_88BIT_GS
#include "tf_common_gs.cu"
#undef TF_BARRETT_88BIT_GS

#define TF_BARRETT_87BIT_GS
#include "tf_common_gs.cu"
#undef TF_BARRETT_87BIT_GS

#define TF_BARRETT_79BIT_GS
#include "tf_common_gs.cu"
#undef TF_BARRETT_79BIT_GS

#define TF_BARRETT_77BIT_GS
#include "tf_common_gs.cu"
#undef TF_BARRETT_77BIT_GS

#define TF_BARRETT_76BIT_GS
#include "tf_common_gs.cu"
#undef TF_BARRETT_76BIT_GS

#undef TF_BARRETT
