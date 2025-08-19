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

__device__ static void create_k_deltas(unsigned int *bit_array, unsigned int bits_to_process, int *total_bit_count,
                                       unsigned short *k_deltas)
{
    int i, words_per_thread, sieve_word, k_bit_base;
    __shared__ volatile unsigned short bitcount[256]; // Each thread of our block puts bit-counts here

    // Get pointer to section of the bit_array this thread is processing.

    words_per_thread = bits_to_process / 8192;
    bit_array += blockIdx.x * bits_to_process / 32 + threadIdx.x * words_per_thread;

    // Count number of bits set in this thread's word(s) from the bit_array

    bitcount[threadIdx.x] = 0;
    for (i = 0; i < words_per_thread; i++)
        bitcount[threadIdx.x] += __popc(bit_array[i]);

    // Create total count of bits set in block up to and including this threads popc.
    // Kudos to Rocke Verser for the population counting code.
    // CAUTION:  Following requires 256 threads per block

    // First five tallies remain within one warp.  Should be in lock-step.
    if (threadIdx.x & 1) // If we are running on any thread 0bxxxxxxx1, tally neighbor's count.
        bitcount[threadIdx.x] += bitcount[threadIdx.x - 1];

    if (threadIdx.x & 2) // If we are running on any thread 0bxxxxxx1x, tally neighbor's count.
        bitcount[threadIdx.x] += bitcount[(threadIdx.x - 2) | 1];

    if (threadIdx.x & 4) // If we are running on any thread 0bxxxxx1xx, tally neighbor's count.
        bitcount[threadIdx.x] += bitcount[(threadIdx.x - 4) | 3];

    if (threadIdx.x & 8) // If we are running on any thread 0bxxxx1xxx, tally neighbor's count.
        bitcount[threadIdx.x] += bitcount[(threadIdx.x - 8) | 7];

    if (threadIdx.x & 16) // If we are running on any thread 0bxxx1xxxx, tally neighbor's count.
        bitcount[threadIdx.x] += bitcount[(threadIdx.x - 16) | 15];

    // Further tallies are across warps.  Must synchronize
    __syncthreads();
    if (threadIdx.x & 32) // If we are running on any thread 0bxx1xxxxx, tally neighbor's count.
        bitcount[threadIdx.x] += bitcount[(threadIdx.x - 32) | 31];

    __syncthreads();
    if (threadIdx.x & 64) // If we are running on any thread 0bx1xxxxxx, tally neighbor's count.
        bitcount[threadIdx.x] += bitcount[(threadIdx.x - 64) | 63];

    __syncthreads();
    if (threadIdx.x & 128) // If we are running on any thread 0b1xxxxxxx, tally neighbor's count.
        bitcount[threadIdx.x] += bitcount[127];

    // At this point, bitcount[...] contains the total number of bits for the indexed
    // thread plus all lower-numbered threads.  I.e., bitcount[255] is the total count.

    __syncthreads();
    *total_bit_count = bitcount[255];

    //POSSIBLE OPTIMIZATION - bitcounts and k_deltas could use the same memory space if we'd read bitcount into a register
    // and sync threads before doing any writes to k_deltas.

    //POSSIBLE SANITY CHECK -- is there any way to test if total_bit_count exceeds the amount of shared memory allocated?

    // Loop til this thread's section of the bit array is finished.

    sieve_word = *bit_array;
    k_bit_base = threadIdx.x * words_per_thread * 32;
    for (i = *total_bit_count - bitcount[threadIdx.x];; i++) {
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

        bit_to_test = 31 - __clz(sieve_word);
        sieve_word &= ~(1 << bit_to_test);

        // Copy the k value to the shared memory array

        k_deltas[i] = k_bit_base + bit_to_test;
    }

    __syncthreads();
    // Here, all warps in our block have placed their candidates in shared memory.
    // Now we can start TFing candidates.
}

__device__ static void create_fbase96(int96 *f_base, int96 k_base, unsigned int exp, unsigned int bits_to_process)
{
    // Compute factor corresponding to first sieve bit in this block.

    // Compute base k value
    k_base.d0 = __add_cc(k_base.d0, __umul32(blockIdx.x * bits_to_process, NUM_CLASSES));
    k_base.d1 = __addc(k_base.d1, __umul32hi(blockIdx.x * bits_to_process, NUM_CLASSES)); /* k values are limited to 64 bits */

    // Compute k * exp
    f_base->d0 = __umul32(k_base.d0, exp);
    f_base->d1 = __add_cc(__umul32hi(k_base.d0, exp), __umul32(k_base.d1, exp));
    f_base->d2 = __addc(__umul32hi(k_base.d1, exp), 0);

    // Compute f_base = 2 * k * exp + 1
    shl_96(f_base);
    f_base->d0 = f_base->d0 + 1;
}
