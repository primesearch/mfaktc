/*
This file is part of mfaktc.
Copyright (C) 2009, 2010, 2011, 2012, 2014, 2015  Oliver Weihe (o.weihe@t-online.de)

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

#include "gpusieve.h"

#undef RAW_GPU_BENCH // FIXME

#ifdef TF_96BIT
#ifdef SHORTCUT_75BIT
extern "C" __host__ int tf_class_75_gs(unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff)
#define MFAKTC_FUNC mfaktc_75_gs
#else
extern "C" __host__ int tf_class_95_gs(unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff)
#define MFAKTC_FUNC mfaktc_95_gs
#endif
#endif
#ifdef TF_BARRETT
#ifdef TF_BARRETT_76BIT_GS
    extern "C" __host__ int tf_class_barrett76_gs(unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff)
#define MFAKTC_FUNC mfaktc_barrett76_gs
#elif defined TF_BARRETT_77BIT_GS
    extern "C" __host__ int tf_class_barrett77_gs(unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff)
#define MFAKTC_FUNC mfaktc_barrett77_gs
#elif defined TF_BARRETT_79BIT_GS
extern "C" __host__ int tf_class_barrett79_gs(unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff)
#define MFAKTC_FUNC mfaktc_barrett79_gs
#elif defined TF_BARRETT_87BIT_GS
extern "C" __host__ int tf_class_barrett87_gs(unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff)
#define MFAKTC_FUNC mfaktc_barrett87_gs
#elif defined TF_BARRETT_88BIT_GS
extern "C" __host__ int tf_class_barrett88_gs(unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff)
#define MFAKTC_FUNC mfaktc_barrett88_gs
#else
extern "C" __host__ int tf_class_barrett92_gs(unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff)
#define MFAKTC_FUNC mfaktc_barrett92_gs
#endif
#endif
{
    int i;
    timeval timer;
    int96 factor, k_base;
    int192 b_preinit;
    int shiftcount, ln2b, count = 0;
    int numblocks;
    unsigned long long k_remaining;
    char string[50];
    int shared_mem_required;
    int factorsfound = 0;

    // If we've never initialized the GPU sieving code, do so now.
    //  gpusieve_init (mystuff); // moved to main() function!

    // If we haven't initialized the GPU sieving code for this Mersenne exponent, do so now.
    gpusieve_init_exponent(mystuff);

    // Init the timer
    timer_init(&timer);

    // Pre-calculate some values

    shiftcount = 0;
    while ((1ULL << shiftcount) < (unsigned long long int)mystuff->exponent)
        shiftcount++;
    //  printf("\n\nshiftcount = %d\n",shiftcount);
    shiftcount -= 1;
    ln2b = 1;
    while (ln2b < 20 || ln2b < mystuff->bit_min) // how much preprocessing is possible
    {
        shiftcount--;
        ln2b <<= 1;
        if (mystuff->exponent & (1 << (shiftcount))) ln2b++;
    }
    //  printf("shiftcount = %d\n",shiftcount);
    //  printf("ln2b = %d\n",ln2b);
    b_preinit.d5 = 0;
    b_preinit.d4 = 0;
    b_preinit.d3 = 0;
    b_preinit.d2 = 0;
    b_preinit.d1 = 0;
    b_preinit.d0 = 0;
    if (ln2b < 32)
        b_preinit.d0 = 1 << ln2b;
    else if (ln2b < 64)
        b_preinit.d1 = 1 << (ln2b - 32);
    else if (ln2b < 96)
        b_preinit.d2 = 1 << (ln2b - 64);
    else if (ln2b < 128)
        b_preinit.d3 = 1 << (ln2b - 96);
    else if (ln2b < 160)
        b_preinit.d4 = 1 << (ln2b - 128);
    else
        b_preinit.d5 = 1 << (ln2b - 160); // b_preinit = 2^ln2b

    /* set result array to 0 */
    cudaMemset(mystuff->d_RES, 0, 1 * sizeof(int)); //first int of result array contains the number of factors found

#ifdef DEBUG_GPU_MATH
    cudaMemset(mystuff->d_modbasecase_debug, 0, 32 * sizeof(int));
#endif

    // Calculate the initial bit-to-clear values for this class
    gpusieve_init_class(mystuff, k_min);

    // Generously estimate the shared memory requirements for the TF kernel
#ifdef RAW_GPU_BENCH
    shared_mem_required = 100; // no sieving = 100%
#else
    if (mystuff->gpu_sieve_primes < 54)
        shared_mem_required = 100; // no sieving = 100%
    else if (mystuff->gpu_sieve_primes < 310)
        shared_mem_required = 50; // 54 primes expect 48.30%
    else if (mystuff->gpu_sieve_primes < 1846)
        shared_mem_required = 38; // 310 primes expect 35.50%
    else if (mystuff->gpu_sieve_primes < 21814)
        shared_mem_required = 30; // 1846 primes expect 28.10%
    else if (mystuff->gpu_sieve_primes < 67894)
        shared_mem_required = 24; // 21814 primes expect 21.93%
    else
        shared_mem_required = 22; // 67894 primes expect 19.94%
#endif
    shared_mem_required = mystuff->gpu_sieve_processing_size * sizeof(int) * shared_mem_required / 100;

    // FIXME: can't use all the shared memory for GPU sieve, lets keep 1kiB spare...
    if (mystuff->verbosity >= 3) printf("shared_mem_required = %d bytes\n", shared_mem_required + 1024);

    if ((shared_mem_required + 1024) > mystuff->max_shared_memory) {
        printf("ERROR: Not enough shared memory available!\n");
        printf("       Need %d bytes\n", shared_mem_required + 1024);
        printf("       You can lower GPUSieveProcessSize or increase GPUSievePrimes to lower\n");
        printf("       the amount of shared memory needed\n");
        exit(1);
    }

    // Loop until all the k's are processed
    for (;;) {
        // Calculate the number of k's remaining.  Round this up so that we sieve an array that is
        // a multiple of the bits processed by each TF kernel (my_stuff->gpu_sieve_processing_size).

        k_remaining = ((k_max - k_min + 1) + NUM_CLASSES - 1) / NUM_CLASSES;
        if (k_remaining < (unsigned long long)mystuff->gpu_sieve_size) {
            numblocks   = ((int)k_remaining + mystuff->gpu_sieve_processing_size - 1) / mystuff->gpu_sieve_processing_size;
            k_remaining = numblocks * mystuff->gpu_sieve_processing_size;
        } else
            numblocks = mystuff->gpu_sieve_size / mystuff->gpu_sieve_processing_size;

        // Do some sieving on the GPU.

        gpusieve(mystuff, k_remaining);

        // Set the k value corresponding to the first bit in the bit array

        k_base.d0 = (int)(k_min & 0xFFFFFFFF);
        k_base.d1 = (int)(k_min >> 32);
        k_base.d2 = 0;

        // Now let the GPU trial factor the candidates that survived the sieving

        MFAKTC_FUNC<<<numblocks, THREADS_PER_BLOCK, shared_mem_required> > >(
            mystuff->exponent, k_base, mystuff->d_bitarray, mystuff->gpu_sieve_processing_size, shiftcount, b_preinit, mystuff->d_RES
#if defined(TF_BARRETT) &&                                                                                                    \
    (defined(TF_BARRETT_87BIT_GS) || defined(TF_BARRETT_88BIT_GS) || defined(TF_BARRETT_92BIT_GS) || defined(DEBUG_GPU_MATH))
            ,
            mystuff->bit_min - 63
#endif
#ifdef DEBUG_GPU_MATH
            ,
            mystuff->d_modbasecase_debug
#endif
        );

        // Sync before doing more GPU sieving
        cudaDeviceSynchronize();

        // Count the number of blocks processed
        count += numblocks;

        // Move to next batch of k's
        k_min += (unsigned long long)mystuff->gpu_sieve_size * NUM_CLASSES;
        if (k_min > k_max) break;

        //BUG - we should call a different routine to advance the bit-to-clear values by gpusieve_size bits
        // This will be cheaper than recomputing the bit-to-clears from scratch
        // HOWEVER, the self-test code will ot check this new code unless we make the gpusieve_size much smaller
        gpusieve_init_class(mystuff, k_min);
    }

    /* download results from GPU */
    cudaMemcpy(mystuff->h_RES, mystuff->d_RES, 32 * sizeof(int), cudaMemcpyDeviceToHost);

#ifdef DEBUG_GPU_MATH
    cudaMemcpy(mystuff->h_modbasecase_debug, mystuff->d_modbasecase_debug, 32 * sizeof(int), cudaMemcpyDeviceToHost);
    for (i = 0; i < 32; i++)
        if (mystuff->h_modbasecase_debug[i] != 0) printf("h_modbasecase_debug[%2d] = %u\n", i, mystuff->h_modbasecase_debug[i]);
#endif

    // Set grid count to the number of blocks processed.  The print code will convert this to a
    // count of candidates processed (by multiplying by 8192 * THREADS_PER_BLOCK.
    // This count isn't an exact match to CPU sieving case as that counts candidates after sieving
    // and we are counting candidates before sieving.  We'd have to modify the TF kernels to count
    // the candidates processed to be completely compatible.
    mystuff->stats.grid_count = count;

    // Keep track of time spent TFing this class and this bitlevel
    /* prevent division by zero if timer resolution is too low */
    mystuff->stats.class_time = timer_diff(&timer) / 1000;
    mystuff->stats.bit_level_time += timer_diff(&timer) / 1000;
    if (mystuff->stats.class_time == 0) mystuff->stats.class_time = 1;

    // GPU sieving does not wait on the CPU (also used by print_status_line to indicate this is a GPU sieving kernel)
    mystuff->stats.cpu_wait = -2.0f;

    // Print out a useful status line
    print_status_line(mystuff);

    // Print out any found factors
    factorsfound = mystuff->h_RES[0];
    for (i = 0; (i < factorsfound) && (i < 10); i++) {
        factor.d2 = mystuff->h_RES[i * 3 + 1];
        factor.d1 = mystuff->h_RES[i * 3 + 2];
        factor.d0 = mystuff->h_RES[i * 3 + 3];
        print_dez96(factor, string);
        print_factor(mystuff, i, string);
    }
    if (factorsfound >= 10) {
        print_factor(mystuff, factorsfound, NULL);
    }

    return factorsfound;
}

#undef MFAKTC_FUNC
