/*
This file is part of mfaktc.
Copyright (C) 2009, 2010, 2011, 2012  Oliver Weihe (o.weihe@t-online.de)

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

#undef DIV_160_96
#include "tf_barrett96_div.cu"
#define DIV_160_96
#include "tf_barrett96_div.cu"
#undef DIV_160_96

#include "tf_barrett96_core.cu"


// Inline to find the highest set bit in a word
// If no bit is set, CC 2.x returns 32, CC 1.x returns 31

__device__ static unsigned int ___clz (unsigned int a)
{
#if (__CUDA_ARCH__ >= FERMI) /* clz (count leading zeroes) is not available on CC 1.x devices */
	unsigned int r;
	asm("clz.b32 %0, %1;" : "=r" (r) : "r" (a));
	return r;
#else
	unsigned int r = 0;
	if ((a & 0xFFFF0000) == 0) r = 16, a <<= 16;
	if ((a & 0xFF000000) == 0) r += 8, a <<= 8;
	if ((a & 0xF0000000) == 0) r += 4, a <<= 4;
	if ((a & 0xC0000000) == 0) r += 2, a <<= 2;
	if ((a & 0x80000000) == 0) r += 1;
	return r;
#endif
}

// Inline to count the number of set bits in a word

__device__ static unsigned int ___popcnt (unsigned int a)
{
#if (__CUDA_ARCH__ >= FERMI) /* popc (population count) is not available on CC 1.x devices */
	unsigned int r;
	asm("popc.b32 %0, %1;" : "=r" (r) : "r" (a));
	return r;
#else
	a = (a&0x55555555) + ((a>> 1)&0x55555555);  // Generate sixteen 2-bit sums
	a = (a&0x33333333) + ((a>> 2)&0x33333333);  // Generate eight 3-bit sums
	a = (a&0x07070707) + ((a>> 4)&0x07070707);  // Generate four 4-bit sums
	a = (a&0x000F000F) + ((a>> 8)&0x000F000F);  // Generate two 5-bit sums
	a = (a&0x0000001F) + ((a>>16)&0x0000001F);  // Generate one 6-bit sum
	return a;
#endif
}


#if __CUDA_ARCH__ >= FERMI
  #define KERNEL_MIN_BLOCKS 2
#else
  #define KERNEL_MIN_BLOCKS 1
#endif

__global__ void
#ifndef CHECKS_MODBASECASE
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS) mfaktc_barrett92_gs(unsigned int exp, int96 k_base, unsigned int *bit_array, unsigned int bits_to_process, int shiftcount, int192 b_preinit, unsigned int *RES, int bit_max64)
#else
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS) mfaktc_barrett92_gs(unsigned int exp, int96 k_base, unsigned int *bit_array, unsigned int bits_to_process, int shiftcount, int192 b_preinit, unsigned int *RES, int bit_max64, unsigned int *modbasecase_debug)
#endif
/*
computes 2^exp mod f
shiftcount is used for precomputing without mod
a is precomputed on host ONCE.
bit_max64 is the number of bits in the factor (minus 64)
*/
{
  int96 f_base;
  int i, words_per_thread, initial_shifter_value, sieve_word, k_bit_base, total_bit_count;
  __shared__ volatile unsigned short bitcount[256];	// Each thread of our block puts bit-counts here
  extern __shared__ unsigned short smem[];		// Write bits to test here.  Launching program must estimate
							// how much shared memory to allocate based on number of primes sieved.

  // Get pointer to section of the bit_array this thread is processing.

  words_per_thread = bits_to_process / 8192;
  bit_array += blockIdx.x * bits_to_process / 32 + threadIdx.x * words_per_thread;

// Count number of bits set in this thread's word(s) from the bit_array

  bitcount[threadIdx.x] = 0;
  for (i = 0; i < words_per_thread; i++)
    bitcount[threadIdx.x] += ___popcnt(bit_array[i]);

// Create total count of bits set in block up to and including this threads popcnt.
// Kudos to Rocke Verser for the population counting code.
// CAUTION:  Following requires 256 threads per block

  // First five tallies remain within one warp.  Should be in lock-step.
  if (threadIdx.x & 1)        // If we are running on any thread 0bxxxxxxx1, tally neighbor's count.
    bitcount[threadIdx.x] += bitcount[threadIdx.x - 1];

  if (threadIdx.x & 2)        // If we are running on any thread 0bxxxxxx1x, tally neighbor's count.
    bitcount[threadIdx.x] += bitcount[(threadIdx.x - 2) | 1];

  if (threadIdx.x & 4)        // If we are running on any thread 0bxxxxx1xx, tally neighbor's count.
    bitcount[threadIdx.x] += bitcount[(threadIdx.x - 4) | 3];

  if (threadIdx.x & 8)        // If we are running on any thread 0bxxxx1xxx, tally neighbor's count.
    bitcount[threadIdx.x] += bitcount[(threadIdx.x - 8) | 7];

  if (threadIdx.x & 16)       // If we are running on any thread 0bxxx1xxxx, tally neighbor's count.
    bitcount[threadIdx.x] += bitcount[(threadIdx.x - 16) | 15];

  // Further tallies are across warps.  Must synchronize
  __syncthreads();
  if (threadIdx.x  & 32)      // If we are running on any thread 0bxx1xxxxx, tally neighbor's count.
    bitcount[threadIdx.x] += bitcount[(threadIdx.x - 32) | 31];

  __syncthreads();
  if (threadIdx.x & 64)       // If we are running on any thread 0bx1xxxxxx, tally neighbor's count.
    bitcount[threadIdx.x] += bitcount[(threadIdx.x - 64) | 63];

  __syncthreads();
  if (threadIdx.x & 128)       // If we are running on any thread 0b1xxxxxxx, tally neighbor's count.
    bitcount[threadIdx.x] += bitcount[127];

  // At this point, bitcount[...] contains the total number of bits for the indexed
  // thread plus all lower-numbered threads.  I.e., bitcount[255] is the total count.

  __syncthreads();
  total_bit_count = bitcount[255];

//POSSIBLE OPTIMIZATION - bitcounts and smem could use the same memory space if we'd read bitcount into a register
// and sync threads before doing any writes to smem.

//POSSIBLE SANITY CHECK -- is there any way to test if total_bit_count exceeds the amount of shared memory allocated?

// Loop til this thread's section of the bit array is finished.

  sieve_word = *bit_array;
  k_bit_base = threadIdx.x * words_per_thread * 32;
  for (i = total_bit_count - bitcount[threadIdx.x]; ; i++) {
    int bit_to_test;

// Make sure we have a non-zero sieve word

    while (sieve_word == 0) {
      if (--words_per_thread == 0) break;
      sieve_word = *++bit_array;
      k_bit_base += 32;
    }

// Check if this thread has processed all its set bits

    if (sieve_word == 0) break;

// Find a bit to test in the sieve word

    bit_to_test = 31 - ___clz (sieve_word);
    sieve_word &= ~(1 << bit_to_test);

// Copy the k value to the shared memory array

    smem[i] = k_bit_base + bit_to_test;
  }

  __syncthreads();

// Here, all warps in our block have placed their candidates in shared memory.
// Now we can start TFing candidates.

// Init some stuff that will be used for all k's tested

  initial_shifter_value = exp << (32 - shiftcount);	// Initial shifter value

// Compute factor corresponding to first sieve bit in this block.

  // Compute base k value
  k_base.d0 = __add_cc (k_base.d0, __umul32  (blockIdx.x * bits_to_process, NUM_CLASSES));
  k_base.d1 = __addc   (k_base.d1, __umul32hi(blockIdx.x * bits_to_process, NUM_CLASSES)); /* k values are limited to 64 bits */

  // Compute k * exp
  f_base.d0 =                                      __umul32(k_base.d0, exp);
  f_base.d1 = __add_cc(__umul32hi(k_base.d0, exp), __umul32(k_base.d1, exp));
  f_base.d2 = __addc  (__umul32hi(k_base.d1, exp),                       0);

  // Compute f_base = 2 * k * exp + 1
  shl_96(&f_base);
  f_base.d0 = f_base.d0 + 1;

// Loop til the k values written to shared memory are exhausted

  for (i = threadIdx.x; i < total_bit_count; i += THREADS_PER_BLOCK) {
    int96 f;
    int k_delta;

// Get the (k - k_base) value to test

    k_delta = smem[i];

// Compute new f.  This is computed as f = f_base + 2 * (k - k_base) * exp.

    f.d0 = __add_cc (f_base.d0, __umul32(2 * k_delta * NUM_CLASSES, exp));
    f.d1 = __addc_cc(f_base.d1, __umul32hi(2 * k_delta * NUM_CLASSES, exp));
    f.d2 = __addc   (f_base.d2, 0);

    test_FC96_barrett92(f, b_preinit, initial_shifter_value, RES, bit_max64
#ifdef CHECKS_MODBASECASE
                        , modbasecase_debug
#endif
                        );
  }
}


__global__ void
#ifndef CHECKS_MODBASECASE
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS) mfaktc_barrett88_gs(unsigned int exp, int96 k_base, unsigned int *bit_array, unsigned int bits_to_process, int shiftcount, int192 b_preinit, unsigned int *RES, int bit_max64)
#else
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS) mfaktc_barrett88_gs(unsigned int exp, int96 k_base, unsigned int *bit_array, unsigned int bits_to_process, int shiftcount, int192 b_preinit, unsigned int *RES, int bit_max64, unsigned int *modbasecase_debug)
#endif
/*
computes 2^exp mod f
shiftcount is used for precomputing without mod
a is precomputed on host ONCE.
bit_max64 is the number of bits in the factor (minus 64)
*/
{
  int96 f_base;
  int i, words_per_thread, initial_shifter_value, sieve_word, k_bit_base, total_bit_count;
  __shared__ volatile unsigned short bitcount[256];	// Each thread of our block puts bit-counts here
  extern __shared__ unsigned short smem[];		// Write bits to test here.  Launching program must estimate
							// how much shared memory to allocate based on number of primes sieved.

  // Get pointer to section of the bit_array this thread is processing.

  words_per_thread = bits_to_process / 8192;
  bit_array += blockIdx.x * bits_to_process / 32 + threadIdx.x * words_per_thread;

// Count number of bits set in this thread's word(s) from the bit_array

  bitcount[threadIdx.x] = 0;
  for (i = 0; i < words_per_thread; i++)
    bitcount[threadIdx.x] += ___popcnt(bit_array[i]);

// Create total count of bits set in block up to and including this threads popcnt.
// Kudos to Rocke Verser for the population counting code.
// CAUTION:  Following requires 256 threads per block

  // First five tallies remain within one warp.  Should be in lock-step.
  if (threadIdx.x & 1)        // If we are running on any thread 0bxxxxxxx1, tally neighbor's count.
    bitcount[threadIdx.x] += bitcount[threadIdx.x - 1];

  if (threadIdx.x & 2)        // If we are running on any thread 0bxxxxxx1x, tally neighbor's count.
    bitcount[threadIdx.x] += bitcount[(threadIdx.x - 2) | 1];

  if (threadIdx.x & 4)        // If we are running on any thread 0bxxxxx1xx, tally neighbor's count.
    bitcount[threadIdx.x] += bitcount[(threadIdx.x - 4) | 3];

  if (threadIdx.x & 8)        // If we are running on any thread 0bxxxx1xxx, tally neighbor's count.
    bitcount[threadIdx.x] += bitcount[(threadIdx.x - 8) | 7];

  if (threadIdx.x & 16)       // If we are running on any thread 0bxxx1xxxx, tally neighbor's count.
    bitcount[threadIdx.x] += bitcount[(threadIdx.x - 16) | 15];

  // Further tallies are across warps.  Must synchronize
  __syncthreads();
  if (threadIdx.x  & 32)      // If we are running on any thread 0bxx1xxxxx, tally neighbor's count.
    bitcount[threadIdx.x] += bitcount[(threadIdx.x - 32) | 31];

  __syncthreads();
  if (threadIdx.x & 64)       // If we are running on any thread 0bx1xxxxxx, tally neighbor's count.
    bitcount[threadIdx.x] += bitcount[(threadIdx.x - 64) | 63];

  __syncthreads();
  if (threadIdx.x & 128)       // If we are running on any thread 0b1xxxxxxx, tally neighbor's count.
    bitcount[threadIdx.x] += bitcount[127];

  // At this point, bitcount[...] contains the total number of bits for the indexed
  // thread plus all lower-numbered threads.  I.e., bitcount[255] is the total count.

  __syncthreads();
  total_bit_count = bitcount[255];

//POSSIBLE OPTIMIZATION - bitcounts and smem could use the same memory space if we'd read bitcount into a register
// and sync threads before doing any writes to smem.

//POSSIBLE SANITY CHECK -- is there any way to test if total_bit_count exceeds the amount of shared memory allocated?

// Loop til this thread's section of the bit array is finished.

  sieve_word = *bit_array;
  k_bit_base = threadIdx.x * words_per_thread * 32;
  for (i = total_bit_count - bitcount[threadIdx.x]; ; i++) {
    int bit_to_test;

// Make sure we have a non-zero sieve word

    while (sieve_word == 0) {
      if (--words_per_thread == 0) break;
      sieve_word = *++bit_array;
      k_bit_base += 32;
    }

// Check if this thread has processed all its set bits

    if (sieve_word == 0) break;

// Find a bit to test in the sieve word

    bit_to_test = 31 - ___clz (sieve_word);
    sieve_word &= ~(1 << bit_to_test);

// Copy the k value to the shared memory array

    smem[i] = k_bit_base + bit_to_test;
  }

  __syncthreads();

// Here, all warps in our block have placed their candidates in shared memory.
// Now we can start TFing candidates.

// Init some stuff that will be used for all k's tested

  initial_shifter_value = exp << (32 - shiftcount);	// Initial shifter value

// Compute factor corresponding to first sieve bit in this block.

  // Compute base k value
  k_base.d0 = __add_cc (k_base.d0, __umul32  (blockIdx.x * bits_to_process, NUM_CLASSES));
  k_base.d1 = __addc   (k_base.d1, __umul32hi(blockIdx.x * bits_to_process, NUM_CLASSES)); /* k values are limited to 64 bits */

  // Compute k * exp
  f_base.d0 =                                      __umul32(k_base.d0, exp);
  f_base.d1 = __add_cc(__umul32hi(k_base.d0, exp), __umul32(k_base.d1, exp));
  f_base.d2 = __addc  (__umul32hi(k_base.d1, exp),                       0);

  // Compute f_base = 2 * k * exp + 1
  shl_96(&f_base);
  f_base.d0 = f_base.d0 + 1;

// Loop til the k values written to shared memory are exhausted

  for (i = threadIdx.x; i < total_bit_count; i += THREADS_PER_BLOCK) {
    int96 f;
    int k_delta;

// Get the (k - k_base) value to test

    k_delta = smem[i];

// Compute new f.  This is computed as f = f_base + 2 * (k - k_base) * exp.

    f.d0 = __add_cc (f_base.d0, __umul32(2 * k_delta * NUM_CLASSES, exp));
    f.d1 = __addc_cc(f_base.d1, __umul32hi(2 * k_delta * NUM_CLASSES, exp));
    f.d2 = __addc   (f_base.d2, 0);

    test_FC96_barrett88(f, b_preinit, initial_shifter_value, RES, bit_max64
#ifdef CHECKS_MODBASECASE
                        , modbasecase_debug
#endif
                        );
  }
}


__global__ void
#ifndef CHECKS_MODBASECASE
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS) mfaktc_barrett87_gs(unsigned int exp, int96 k_base, unsigned int *bit_array, unsigned int bits_to_process, int shiftcount, int192 b_preinit, unsigned int *RES, int bit_max64)
#else
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS) mfaktc_barrett87_gs(unsigned int exp, int96 k_base, unsigned int *bit_array, unsigned int bits_to_process, int shiftcount, int192 b_preinit, unsigned int *RES, int bit_max64, unsigned int *modbasecase_debug)
#endif
/*
computes 2^exp mod f
shiftcount is used for precomputing without mod
a is precomputed on host ONCE.
bit_max64 is the number of bits in the factor (minus 64)
*/
{
  int96 f_base;
  int i, words_per_thread, initial_shifter_value, sieve_word, k_bit_base, total_bit_count;
  __shared__ volatile unsigned short bitcount[256];	// Each thread of our block puts bit-counts here
  extern __shared__ unsigned short smem[];		// Write bits to test here.  Launching program must estimate
							// how much shared memory to allocate based on number of primes sieved.

  // Get pointer to section of the bit_array this thread is processing.

  words_per_thread = bits_to_process / 8192;
  bit_array += blockIdx.x * bits_to_process / 32 + threadIdx.x * words_per_thread;

// Count number of bits set in this thread's word(s) from the bit_array

  bitcount[threadIdx.x] = 0;
  for (i = 0; i < words_per_thread; i++)
    bitcount[threadIdx.x] += ___popcnt(bit_array[i]);

// Create total count of bits set in block up to and including this threads popcnt.
// Kudos to Rocke Verser for the population counting code.
// CAUTION:  Following requires 256 threads per block

  // First five tallies remain within one warp.  Should be in lock-step.
  if (threadIdx.x & 1)        // If we are running on any thread 0bxxxxxxx1, tally neighbor's count.
    bitcount[threadIdx.x] += bitcount[threadIdx.x - 1];

  if (threadIdx.x & 2)        // If we are running on any thread 0bxxxxxx1x, tally neighbor's count.
    bitcount[threadIdx.x] += bitcount[(threadIdx.x - 2) | 1];

  if (threadIdx.x & 4)        // If we are running on any thread 0bxxxxx1xx, tally neighbor's count.
    bitcount[threadIdx.x] += bitcount[(threadIdx.x - 4) | 3];

  if (threadIdx.x & 8)        // If we are running on any thread 0bxxxx1xxx, tally neighbor's count.
    bitcount[threadIdx.x] += bitcount[(threadIdx.x - 8) | 7];

  if (threadIdx.x & 16)       // If we are running on any thread 0bxxx1xxxx, tally neighbor's count.
    bitcount[threadIdx.x] += bitcount[(threadIdx.x - 16) | 15];

  // Further tallies are across warps.  Must synchronize
  __syncthreads();
  if (threadIdx.x  & 32)      // If we are running on any thread 0bxx1xxxxx, tally neighbor's count.
    bitcount[threadIdx.x] += bitcount[(threadIdx.x - 32) | 31];

  __syncthreads();
  if (threadIdx.x & 64)       // If we are running on any thread 0bx1xxxxxx, tally neighbor's count.
    bitcount[threadIdx.x] += bitcount[(threadIdx.x - 64) | 63];

  __syncthreads();
  if (threadIdx.x & 128)       // If we are running on any thread 0b1xxxxxxx, tally neighbor's count.
    bitcount[threadIdx.x] += bitcount[127];

  // At this point, bitcount[...] contains the total number of bits for the indexed
  // thread plus all lower-numbered threads.  I.e., bitcount[255] is the total count.

  __syncthreads();
  total_bit_count = bitcount[255];

//POSSIBLE OPTIMIZATION - bitcounts and smem could use the same memory space if we'd read bitcount into a register
// and sync threads before doing any writes to smem.

//POSSIBLE SANITY CHECK -- is there any way to test if total_bit_count exceeds the amount of shared memory allocated?

// Loop til this thread's section of the bit array is finished.

  sieve_word = *bit_array;
  k_bit_base = threadIdx.x * words_per_thread * 32;
  for (i = total_bit_count - bitcount[threadIdx.x]; ; i++) {
    int bit_to_test;

// Make sure we have a non-zero sieve word

    while (sieve_word == 0) {
      if (--words_per_thread == 0) break;
      sieve_word = *++bit_array;
      k_bit_base += 32;
    }

// Check if this thread has processed all its set bits

    if (sieve_word == 0) break;

// Find a bit to test in the sieve word

    bit_to_test = 31 - ___clz (sieve_word);
    sieve_word &= ~(1 << bit_to_test);

// Copy the k value to the shared memory array

    smem[i] = k_bit_base + bit_to_test;
  }

  __syncthreads();

// Here, all warps in our block have placed their candidates in shared memory.
// Now we can start TFing candidates.

// Init some stuff that will be used for all k's tested

  initial_shifter_value = exp << (32 - shiftcount);	// Initial shifter value

// Compute factor corresponding to first sieve bit in this block.

  // Compute base k value
  k_base.d0 = __add_cc (k_base.d0, __umul32  (blockIdx.x * bits_to_process, NUM_CLASSES));
  k_base.d1 = __addc   (k_base.d1, __umul32hi(blockIdx.x * bits_to_process, NUM_CLASSES)); /* k values are limited to 64 bits */

  // Compute k * exp
  f_base.d0 =                                      __umul32(k_base.d0, exp);
  f_base.d1 = __add_cc(__umul32hi(k_base.d0, exp), __umul32(k_base.d1, exp));
  f_base.d2 = __addc  (__umul32hi(k_base.d1, exp),                       0);

  // Compute f_base = 2 * k * exp + 1
  shl_96(&f_base);
  f_base.d0 = f_base.d0 + 1;

// Loop til the k values written to shared memory are exhausted

  for (i = threadIdx.x; i < total_bit_count; i += THREADS_PER_BLOCK) {
    int96 f;
    int k_delta;

// Get the (k - k_base) value to test

    k_delta = smem[i];

// Compute new f.  This is computed as f = f_base + 2 * (k - k_base) * exp.

    f.d0 = __add_cc (f_base.d0, __umul32(2 * k_delta * NUM_CLASSES, exp));
    f.d1 = __addc_cc(f_base.d1, __umul32hi(2 * k_delta * NUM_CLASSES, exp));
    f.d2 = __addc   (f_base.d2, 0);

    test_FC96_barrett87(f, b_preinit, initial_shifter_value, RES, bit_max64
#ifdef CHECKS_MODBASECASE
                        , modbasecase_debug
#endif
                        );
  }
}


__global__ void
#ifndef CHECKS_MODBASECASE
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS) mfaktc_barrett79_gs(unsigned int exp, int96 k_base, unsigned int *bit_array, unsigned int bits_to_process, int shiftcount, int192 b_preinit, unsigned int *RES)
#else
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS) mfaktc_barrett79_gs(unsigned int exp, int96 k_base, unsigned int *bit_array, unsigned int bits_to_process, int shiftcount, int192 b_preinit, unsigned int *RES, int bit_max64, unsigned int *modbasecase_debug)
#endif
/*
computes 2^exp mod f
shiftcount is used for precomputing without mod
a is precomputed on host ONCE.
*/
{
  int96 f_base;
  int i, words_per_thread, initial_shifter_value, sieve_word, k_bit_base, total_bit_count;
  __shared__ volatile unsigned short bitcount[256];	// Each thread of our block puts bit-counts here
  extern __shared__ unsigned short smem[];		// Write bits to test here.  Launching program must estimate
							// how much shared memory to allocate based on number of primes sieved.

  // Get pointer to section of the bit_array this thread is processing.

  words_per_thread = bits_to_process / 8192;
  bit_array += blockIdx.x * bits_to_process / 32 + threadIdx.x * words_per_thread;

// Count number of bits set in this thread's word(s) from the bit_array

  bitcount[threadIdx.x] = 0;
  for (i = 0; i < words_per_thread; i++)
    bitcount[threadIdx.x] += ___popcnt(bit_array[i]);

// Create total count of bits set in block up to and including this threads popcnt.
// Kudos to Rocke Verser for the population counting code.
// CAUTION:  Following requires 256 threads per block

  // First five tallies remain within one warp.  Should be in lock-step.
  if (threadIdx.x & 1)        // If we are running on any thread 0bxxxxxxx1, tally neighbor's count.
    bitcount[threadIdx.x] += bitcount[threadIdx.x - 1];

  if (threadIdx.x & 2)        // If we are running on any thread 0bxxxxxx1x, tally neighbor's count.
    bitcount[threadIdx.x] += bitcount[(threadIdx.x - 2) | 1];

  if (threadIdx.x & 4)        // If we are running on any thread 0bxxxxx1xx, tally neighbor's count.
    bitcount[threadIdx.x] += bitcount[(threadIdx.x - 4) | 3];

  if (threadIdx.x & 8)        // If we are running on any thread 0bxxxx1xxx, tally neighbor's count.
    bitcount[threadIdx.x] += bitcount[(threadIdx.x - 8) | 7];

  if (threadIdx.x & 16)       // If we are running on any thread 0bxxx1xxxx, tally neighbor's count.
    bitcount[threadIdx.x] += bitcount[(threadIdx.x - 16) | 15];

  // Further tallies are across warps.  Must synchronize
  __syncthreads();
  if (threadIdx.x  & 32)      // If we are running on any thread 0bxx1xxxxx, tally neighbor's count.
    bitcount[threadIdx.x] += bitcount[(threadIdx.x - 32) | 31];

  __syncthreads();
  if (threadIdx.x & 64)       // If we are running on any thread 0bx1xxxxxx, tally neighbor's count.
    bitcount[threadIdx.x] += bitcount[(threadIdx.x - 64) | 63];

  __syncthreads();
  if (threadIdx.x & 128)       // If we are running on any thread 0b1xxxxxxx, tally neighbor's count.
    bitcount[threadIdx.x] += bitcount[127];

  // At this point, bitcount[...] contains the total number of bits for the indexed
  // thread plus all lower-numbered threads.  I.e., bitcount[255] is the total count.

  __syncthreads();
  total_bit_count = bitcount[255];

//POSSIBLE OPTIMIZATION - bitcounts and smem could use the same memory space if we'd read bitcount into a register
// and sync threads before doing any writes to smem.

//POSSIBLE SANITY CHECK -- is there any way to test if total_bit_count exceeds the amount of shared memory allocated?

// Loop til this thread's section of the bit array is finished.

  sieve_word = *bit_array;
  k_bit_base = threadIdx.x * words_per_thread * 32;
  for (i = total_bit_count - bitcount[threadIdx.x]; ; i++) {
    int bit_to_test;

// Make sure we have a non-zero sieve word

    while (sieve_word == 0) {
      if (--words_per_thread == 0) break;
      sieve_word = *++bit_array;
      k_bit_base += 32;
    }

// Check if this thread has processed all its set bits

    if (sieve_word == 0) break;

// Find a bit to test in the sieve word

    bit_to_test = 31 - ___clz (sieve_word);
    sieve_word &= ~(1 << bit_to_test);

// Copy the k value to the shared memory array

    smem[i] = k_bit_base + bit_to_test;
  }

  __syncthreads();

// Here, all warps in our block have placed their candidates in shared memory.
// Now we can start TFing candidates.

// Init some stuff that will be used for all k's tested

  initial_shifter_value = exp << (32 - shiftcount);	// Initial shifter value

// Compute factor corresponding to first sieve bit in this block.

  // Compute base k value
  k_base.d0 = __add_cc (k_base.d0, __umul32  (blockIdx.x * bits_to_process, NUM_CLASSES));
  k_base.d1 = __addc   (k_base.d1, __umul32hi(blockIdx.x * bits_to_process, NUM_CLASSES)); /* k values are limited to 64 bits */

  // Compute k * exp
  f_base.d0 =                                      __umul32(k_base.d0, exp);
  f_base.d1 = __add_cc(__umul32hi(k_base.d0, exp), __umul32(k_base.d1, exp));
  f_base.d2 = __addc  (__umul32hi(k_base.d1, exp),                       0);

  // Compute f_base = 2 * k * exp + 1
  shl_96(&f_base);
  f_base.d0 = f_base.d0 + 1;

// Loop til the k values written to shared memory are exhausted

  for (i = threadIdx.x; i < total_bit_count; i += THREADS_PER_BLOCK) {
    int96 f;
    int k_delta;

// Get the (k - k_base) value to test

    k_delta = smem[i];

// Compute new f.  This is computed as f = f_base + 2 * (k - k_base) * exp.

    f.d0 = __add_cc (f_base.d0, __umul32(2 * k_delta * NUM_CLASSES, exp));
    f.d1 = __addc_cc(f_base.d1, __umul32hi(2 * k_delta * NUM_CLASSES, exp));
    f.d2 = __addc   (f_base.d2, 0);

    test_FC96_barrett79(f, b_preinit, initial_shifter_value, RES
#ifdef CHECKS_MODBASECASE
                        , bit_max64, modbasecase_debug
#endif
                        );
  }
}


__global__ void
#ifndef CHECKS_MODBASECASE
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS) mfaktc_barrett77_gs(unsigned int exp, int96 k_base, unsigned int *bit_array, unsigned int bits_to_process, int shiftcount, int192 b_preinit, unsigned int *RES)
#else
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS) mfaktc_barrett77_gs(unsigned int exp, int96 k_base, unsigned int *bit_array, unsigned int bits_to_process, int shiftcount, int192 b_preinit, unsigned int *RES, int bit_max64, unsigned int *modbasecase_debug)
#endif
/*
computes 2^exp mod f
shiftcount is used for precomputing without mod
a is precomputed on host ONCE.
*/
{
  int96 f_base;
  int i, words_per_thread, initial_shifter_value, sieve_word, k_bit_base, total_bit_count;
  __shared__ volatile unsigned short bitcount[256];	// Each thread of our block puts bit-counts here
  extern __shared__ unsigned short smem[];		// Write bits to test here.  Launching program must estimate
							// how much shared memory to allocate based on number of primes sieved.

  // Get pointer to section of the bit_array this thread is processing.

  words_per_thread = bits_to_process / 8192;
  bit_array += blockIdx.x * bits_to_process / 32 + threadIdx.x * words_per_thread;

// Count number of bits set in this thread's word(s) from the bit_array

  bitcount[threadIdx.x] = 0;
  for (i = 0; i < words_per_thread; i++)
    bitcount[threadIdx.x] += ___popcnt(bit_array[i]);

// Create total count of bits set in block up to and including this threads popcnt.
// Kudos to Rocke Verser for the population counting code.
// CAUTION:  Following requires 256 threads per block

  // First five tallies remain within one warp.  Should be in lock-step.
  if (threadIdx.x & 1)        // If we are running on any thread 0bxxxxxxx1, tally neighbor's count.
    bitcount[threadIdx.x] += bitcount[threadIdx.x - 1];

  if (threadIdx.x & 2)        // If we are running on any thread 0bxxxxxx1x, tally neighbor's count.
    bitcount[threadIdx.x] += bitcount[(threadIdx.x - 2) | 1];

  if (threadIdx.x & 4)        // If we are running on any thread 0bxxxxx1xx, tally neighbor's count.
    bitcount[threadIdx.x] += bitcount[(threadIdx.x - 4) | 3];

  if (threadIdx.x & 8)        // If we are running on any thread 0bxxxx1xxx, tally neighbor's count.
    bitcount[threadIdx.x] += bitcount[(threadIdx.x - 8) | 7];

  if (threadIdx.x & 16)       // If we are running on any thread 0bxxx1xxxx, tally neighbor's count.
    bitcount[threadIdx.x] += bitcount[(threadIdx.x - 16) | 15];

  // Further tallies are across warps.  Must synchronize
  __syncthreads();
  if (threadIdx.x  & 32)      // If we are running on any thread 0bxx1xxxxx, tally neighbor's count.
    bitcount[threadIdx.x] += bitcount[(threadIdx.x - 32) | 31];

  __syncthreads();
  if (threadIdx.x & 64)       // If we are running on any thread 0bx1xxxxxx, tally neighbor's count.
    bitcount[threadIdx.x] += bitcount[(threadIdx.x - 64) | 63];

  __syncthreads();
  if (threadIdx.x & 128)       // If we are running on any thread 0b1xxxxxxx, tally neighbor's count.
    bitcount[threadIdx.x] += bitcount[127];

  // At this point, bitcount[...] contains the total number of bits for the indexed
  // thread plus all lower-numbered threads.  I.e., bitcount[255] is the total count.

  __syncthreads();
  total_bit_count = bitcount[255];

//POSSIBLE OPTIMIZATION - bitcounts and smem could use the same memory space if we'd read bitcount into a register
// and sync threads before doing any writes to smem.

//POSSIBLE SANITY CHECK -- is there any way to test if total_bit_count exceeds the amount of shared memory allocated?

// Loop til this thread's section of the bit array is finished.

  sieve_word = *bit_array;
  k_bit_base = threadIdx.x * words_per_thread * 32;
  for (i = total_bit_count - bitcount[threadIdx.x]; ; i++) {
    int bit_to_test;

// Make sure we have a non-zero sieve word

    while (sieve_word == 0) {
      if (--words_per_thread == 0) break;
      sieve_word = *++bit_array;
      k_bit_base += 32;
    }

// Check if this thread has processed all its set bits

    if (sieve_word == 0) break;

// Find a bit to test in the sieve word

    bit_to_test = 31 - ___clz (sieve_word);
    sieve_word &= ~(1 << bit_to_test);

// Copy the k value to the shared memory array

    smem[i] = k_bit_base + bit_to_test;
  }

  __syncthreads();

// Here, all warps in our block have placed their candidates in shared memory.
// Now we can start TFing candidates.

// Init some stuff that will be used for all k's tested

  initial_shifter_value = exp << (32 - shiftcount);	// Initial shifter value

// Compute factor corresponding to first sieve bit in this block.

  // Compute base k value
  k_base.d0 = __add_cc (k_base.d0, __umul32  (blockIdx.x * bits_to_process, NUM_CLASSES));
  k_base.d1 = __addc   (k_base.d1, __umul32hi(blockIdx.x * bits_to_process, NUM_CLASSES)); /* k values are limited to 64 bits */

  // Compute k * exp
  f_base.d0 =                                      __umul32(k_base.d0, exp);
  f_base.d1 = __add_cc(__umul32hi(k_base.d0, exp), __umul32(k_base.d1, exp));
  f_base.d2 = __addc  (__umul32hi(k_base.d1, exp),                       0);

  // Compute f_base = 2 * k * exp + 1
  shl_96(&f_base);
  f_base.d0 = f_base.d0 + 1;

// Loop til the k values written to shared memory are exhausted

  for (i = threadIdx.x; i < total_bit_count; i += THREADS_PER_BLOCK) {
    int96 f;
    int k_delta;

// Get the (k - k_base) value to test

    k_delta = smem[i];

// Compute new f.  This is computed as f = f_base + 2 * (k - k_base) * exp.

    f.d0 = __add_cc (f_base.d0, __umul32(2 * k_delta * NUM_CLASSES, exp));
    f.d1 = __addc_cc(f_base.d1, __umul32hi(2 * k_delta * NUM_CLASSES, exp));
    f.d2 = __addc   (f_base.d2, 0);

    test_FC96_barrett77(f, b_preinit, initial_shifter_value, RES
#ifdef CHECKS_MODBASECASE
                        , bit_max64, modbasecase_debug
#endif
                        );
  }
}


__global__ void
#ifndef CHECKS_MODBASECASE
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS) mfaktc_barrett76_gs(unsigned int exp, int96 k_base, unsigned int *bit_array, unsigned int bits_to_process, int shiftcount, int192 b_preinit, unsigned int *RES)
#else
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS) mfaktc_barrett76_gs(unsigned int exp, int96 k_base, unsigned int *bit_array, unsigned int bits_to_process, int shiftcount, int192 b_preinit, unsigned int *RES, int bit_max64, unsigned int *modbasecase_debug)
#endif
/*
computes 2^exp mod f
shiftcount is used for precomputing without mod
a is precomputed on host ONCE.
*/
{
  int96 f_base;
  int i, words_per_thread, initial_shifter_value, sieve_word, k_bit_base, total_bit_count;
  __shared__ volatile unsigned short bitcount[256];	// Each thread of our block puts bit-counts here
  extern __shared__ unsigned short smem[];		// Write bits to test here.  Launching program must estimate
							// how much shared memory to allocate based on number of primes sieved.

  // Get pointer to section of the bit_array this thread is processing.

  words_per_thread = bits_to_process / 8192;
  bit_array += blockIdx.x * bits_to_process / 32 + threadIdx.x * words_per_thread;

// Count number of bits set in this thread's word(s) from the bit_array

  bitcount[threadIdx.x] = 0;
  for (i = 0; i < words_per_thread; i++)
    bitcount[threadIdx.x] += ___popcnt(bit_array[i]);

// Create total count of bits set in block up to and including this threads popcnt.
// Kudos to Rocke Verser for the population counting code.
// CAUTION:  Following requires 256 threads per block

  // First five tallies remain within one warp.  Should be in lock-step.
  if (threadIdx.x & 1)        // If we are running on any thread 0bxxxxxxx1, tally neighbor's count.
    bitcount[threadIdx.x] += bitcount[threadIdx.x - 1];

  if (threadIdx.x & 2)        // If we are running on any thread 0bxxxxxx1x, tally neighbor's count.
    bitcount[threadIdx.x] += bitcount[(threadIdx.x - 2) | 1];

  if (threadIdx.x & 4)        // If we are running on any thread 0bxxxxx1xx, tally neighbor's count.
    bitcount[threadIdx.x] += bitcount[(threadIdx.x - 4) | 3];

  if (threadIdx.x & 8)        // If we are running on any thread 0bxxxx1xxx, tally neighbor's count.
    bitcount[threadIdx.x] += bitcount[(threadIdx.x - 8) | 7];

  if (threadIdx.x & 16)       // If we are running on any thread 0bxxx1xxxx, tally neighbor's count.
    bitcount[threadIdx.x] += bitcount[(threadIdx.x - 16) | 15];

  // Further tallies are across warps.  Must synchronize
  __syncthreads();
  if (threadIdx.x  & 32)      // If we are running on any thread 0bxx1xxxxx, tally neighbor's count.
    bitcount[threadIdx.x] += bitcount[(threadIdx.x - 32) | 31];

  __syncthreads();
  if (threadIdx.x & 64)       // If we are running on any thread 0bx1xxxxxx, tally neighbor's count.
    bitcount[threadIdx.x] += bitcount[(threadIdx.x - 64) | 63];

  __syncthreads();
  if (threadIdx.x & 128)       // If we are running on any thread 0b1xxxxxxx, tally neighbor's count.
    bitcount[threadIdx.x] += bitcount[127];

  // At this point, bitcount[...] contains the total number of bits for the indexed
  // thread plus all lower-numbered threads.  I.e., bitcount[255] is the total count.

  __syncthreads();
  total_bit_count = bitcount[255];

//POSSIBLE OPTIMIZATION - bitcounts and smem could use the same memory space if we'd read bitcount into a register
// and sync threads before doing any writes to smem.

//POSSIBLE SANITY CHECK -- is there any way to test if total_bit_count exceeds the amount of shared memory allocated?

// Loop til this thread's section of the bit array is finished.

  sieve_word = *bit_array;
  k_bit_base = threadIdx.x * words_per_thread * 32;
  for (i = total_bit_count - bitcount[threadIdx.x]; ; i++) {
    int bit_to_test;

// Make sure we have a non-zero sieve word

    while (sieve_word == 0) {
      if (--words_per_thread == 0) break;
      sieve_word = *++bit_array;
      k_bit_base += 32;
    }

// Check if this thread has processed all its set bits

    if (sieve_word == 0) break;

// Find a bit to test in the sieve word

    bit_to_test = 31 - ___clz (sieve_word);
    sieve_word &= ~(1 << bit_to_test);

// Copy the k value to the shared memory array

    smem[i] = k_bit_base + bit_to_test;
  }

  __syncthreads();

// Here, all warps in our block have placed their candidates in shared memory.
// Now we can start TFing candidates.

// Init some stuff that will be used for all k's tested

  initial_shifter_value = exp << (32 - shiftcount);	// Initial shifter value

// Compute factor corresponding to first sieve bit in this block.

  // Compute base k value
  k_base.d0 = __add_cc (k_base.d0, __umul32  (blockIdx.x * bits_to_process, NUM_CLASSES));
  k_base.d1 = __addc   (k_base.d1, __umul32hi(blockIdx.x * bits_to_process, NUM_CLASSES)); /* k values are limited to 64 bits */

  // Compute k * exp
  f_base.d0 =                                      __umul32(k_base.d0, exp);
  f_base.d1 = __add_cc(__umul32hi(k_base.d0, exp), __umul32(k_base.d1, exp));
  f_base.d2 = __addc  (__umul32hi(k_base.d1, exp),                       0);

  // Compute f_base = 2 * k * exp + 1
  shl_96(&f_base);
  f_base.d0 = f_base.d0 + 1;

// Loop til the k values written to shared memory are exhausted

  for (i = threadIdx.x; i < total_bit_count; i += THREADS_PER_BLOCK) {
    int96 f;
    int k_delta;

// Get the (k - k_base) value to test

    k_delta = smem[i];

// Compute new f.  This is computed as f = f_base + 2 * (k - k_base) * exp.

    f.d0 = __add_cc (f_base.d0, __umul32(2 * k_delta * NUM_CLASSES, exp));
    f.d1 = __addc_cc(f_base.d1, __umul32hi(2 * k_delta * NUM_CLASSES, exp));
    f.d2 = __addc   (f_base.d2, 0);

    test_FC96_barrett76(f, b_preinit, initial_shifter_value, RES
#ifdef CHECKS_MODBASECASE
                        , bit_max64, modbasecase_debug
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
