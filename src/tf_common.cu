/*
This file is part of mfaktc.
Copyright (C) 2009, 2010, 2011, 2012, 2013, 2014  Oliver Weihe (o.weihe@t-online.de)

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

#ifdef TF_72BIT
extern "C" __host__ int tf_class_71(unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff)
#define MFAKTC_FUNC mfaktc_71
#endif
#ifdef TF_96BIT
#ifdef SHORTCUT_75BIT
    extern "C" __host__ int tf_class_75(unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff)
#define MFAKTC_FUNC mfaktc_75
#else
    extern "C" __host__ int tf_class_95(unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff)
#define MFAKTC_FUNC mfaktc_95
#endif
#endif
#ifdef TF_BARRETT
#ifdef TF_BARRETT_76BIT
        extern "C" __host__ int tf_class_barrett76(unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff)
#define MFAKTC_FUNC mfaktc_barrett76
#elif defined TF_BARRETT_77BIT
        extern "C" __host__ int tf_class_barrett77(unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff)
#define MFAKTC_FUNC mfaktc_barrett77
#elif defined TF_BARRETT_79BIT
    extern "C" __host__ int tf_class_barrett79(unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff)
#define MFAKTC_FUNC mfaktc_barrett79
#elif defined TF_BARRETT_87BIT
    extern "C" __host__ int tf_class_barrett87(unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff)
#define MFAKTC_FUNC mfaktc_barrett87
#elif defined TF_BARRETT_88BIT
    extern "C" __host__ int tf_class_barrett88(unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff)
#define MFAKTC_FUNC mfaktc_barrett88
#else
    extern "C" __host__ int tf_class_barrett92(unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff)
#define MFAKTC_FUNC mfaktc_barrett92
#endif
#endif
{
    size_t size = mystuff->threads_per_grid * sizeof(int);
    int i, index = 0, stream;
    cudaError_t cuda_ret;
    timeval timer;
    timeval timer2;
    unsigned long long int twait = 0;
#ifdef TF_72BIT
    int72 factor, k_base;
    int144 b_preinit;
#endif
#if defined(TF_96BIT) || defined(TF_BARRETT)
    int96 factor, k_base;
    int192 b_preinit;
#endif
    int shiftcount, ln2b, count = 0;
    unsigned long long int k_diff;
    char string[50];
    int factorsfound = 0;

    int h_ktab_index = 0;
    int h_ktab_cpu[CPU_STREAMS_MAX]; // the set of h_ktab[N]s currently ownt by CPU
        // 0 <= N < h_ktab_index: these h_ktab[]s are preprocessed
        // h_ktab_index <= N < mystuff.cpu_streams: these h_ktab[]s are NOT preprocessed
    int h_ktab_inuse[NUM_STREAMS_MAX]; // h_ktab_inuse[N] contains the number of h_ktab[] currently used by stream N
    unsigned long long int
        k_min_grid[CPU_STREAMS_MAX]; // k_min_grid[N] contains the k_min for h_ktab[h_ktab_cpu[N]], only valid for preprocessed h_ktab[]s

    timer_init(&timer);

    int threadsPerBlock = THREADS_PER_BLOCK;
    int blocksPerGrid   = (mystuff->threads_per_grid + threadsPerBlock - 1) / threadsPerBlock;

    unsigned int delay = 1000;

    for (i = 0; i < mystuff->num_streams; i++)
        h_ktab_inuse[i] = i;
    for (i = 0; i < mystuff->cpu_streams; i++)
        h_ktab_cpu[i] = i + mystuff->num_streams;
    for (i = 0; i < mystuff->cpu_streams; i++)
        k_min_grid[i] = 0;
    h_ktab_index = 0;

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
#ifdef TF_72BIT
    if (ln2b < 24)
        b_preinit.d0 = 1 << ln2b;
    else if (ln2b < 48)
        b_preinit.d1 = 1 << (ln2b - 24);
    else if (ln2b < 72)
        b_preinit.d2 = 1 << (ln2b - 48);
    else if (ln2b < 96)
        b_preinit.d3 = 1 << (ln2b - 72);
    else if (ln2b < 120)
        b_preinit.d4 = 1 << (ln2b - 96);
    else
        b_preinit.d5 = 1 << (ln2b - 120); // b_preinit = 2^ln2b
#endif
#if defined(TF_96BIT) || defined(TF_BARRETT)
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
#endif

    /* set result array to 0 */
    cudaMemset(mystuff->d_RES, 0, 1 * sizeof(int)); //first int of result array contains the number of factors found
    //  for(i=0;i<32;i++)mystuff->h_RES[i]=0;
    //  cudaMemcpy(mystuff->d_RES, mystuff->h_RES, 32*sizeof(int), cudaMemcpyHostToDevice);

#ifdef DEBUG_GPU_MATH
    //  cudaMemcpy(mystuff->d_modbasecase_debug, mystuff->h_RES, 32*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(mystuff->d_modbasecase_debug, 0, 32 * sizeof(int));
#endif

    timer_init(&timer2);
    while ((k_min <= k_max) || (h_ktab_index > 0)) {
        /* preprocessing: calculate a ktab (factor table) */
        if ((k_min <= k_max) && (h_ktab_index < mystuff->cpu_streams)) // if we have an empty h_ktab we can preprocess another one
        {
            delay = 1000;
            index = h_ktab_cpu[h_ktab_index];

            if (count > mystuff->num_streams) {
                twait += timer_diff(&timer2);
            }
#ifdef DEBUG_STREAM_SCHEDULE
            printf(" STREAM_SCHEDULE: preprocessing on h_ktab[%d] (count = %d)\n", index, count);
#endif

            sieve_candidates(mystuff->threads_per_grid, mystuff->h_ktab[index], mystuff->sieve_primes);
            k_diff = mystuff->h_ktab[index][mystuff->threads_per_grid - 1] + 1;
            k_diff *= NUM_CLASSES; /* NUM_CLASSES because classes are mod NUM_CLASSES */

            k_min_grid[h_ktab_index] = k_min;
            h_ktab_index++;

            count++;
            k_min += (unsigned long long int)k_diff;
            timer_init(&timer2);
        } else if (mystuff->allowsleep == 1) {
            /* no unused h_ktab for preprocessing. 
      This usually means that
      a) all GPU streams are busy 
      and
      b) we've preprocessed all available CPU streams
      so let's sleep for some time instead of running a busy loop on cudaStreamQuery() */
            my_usleep(delay);

            delay = delay * 3 / 2;
            if (delay > 500000) delay = 500000;
        }

        /* try upload ktab and start the calcualtion of a preprocessed dataset on the device */
        stream = 0;
        while ((stream < mystuff->num_streams) && (h_ktab_index > 0)) {
            if (cudaStreamQuery(mystuff->stream[stream]) == cudaSuccess) {
#ifdef DEBUG_STREAM_SCHEDULE
                printf(" STREAM_SCHEDULE: found empty stream: = %d (this releases h_ktab[%d])\n", stream, h_ktab_inuse[stream]);
#endif
                h_ktab_index--;
                i                        = h_ktab_inuse[stream];
                h_ktab_inuse[stream]     = h_ktab_cpu[h_ktab_index];
                h_ktab_cpu[h_ktab_index] = i;

                cudaMemcpyAsync(mystuff->d_ktab[stream], mystuff->h_ktab[h_ktab_inuse[stream]], size, cudaMemcpyHostToDevice,
                                mystuff->stream[stream]);

#ifdef TF_72BIT
                k_base.d0 = k_min_grid[h_ktab_index] & 0xFFFFFF;
                k_base.d1 = (k_min_grid[h_ktab_index] >> 24) & 0xFFFFFF;
                k_base.d2 = k_min_grid[h_ktab_index] >> 48;
#elif defined(TF_96BIT) || defined(TF_BARRETT)
                k_base.d0 = k_min_grid[h_ktab_index] & 0xFFFFFFFF;
                k_base.d1 = k_min_grid[h_ktab_index] >> 32;
                k_base.d2 = 0;
#endif

                MFAKTC_FUNC<<<blocksPerGrid, threadsPerBlock, 0, mystuff->stream[stream]> > >(
                    mystuff->exponent, k_base, mystuff->d_ktab[stream], shiftcount, b_preinit, mystuff->d_RES
#if defined(TF_BARRETT) && (defined(TF_BARRETT_87BIT) || defined(TF_BARRETT_88BIT) || defined(TF_BARRETT_92BIT) || defined(DEBUG_GPU_MATH))
                    ,
                    mystuff->bit_min - 63
#endif
#ifdef DEBUG_GPU_MATH
                    ,
                    mystuff->d_modbasecase_debug
#endif
                );

#ifdef DEBUG_STREAM_SCHEDULE
                printf(" STREAM_SCHEDULE: started GPU kernel on stream %d using h_ktab[%d]\n\n", stream, h_ktab_inuse[stream]);
#endif
#ifdef DEBUG_GPU_MATH
                cudaDeviceSynchronize(); /* needed to get the output from device printf() */
#endif
#ifdef DEBUG_STREAM_SCHEDULE_CHECK
                int j, index_count;
                for (i = 0; i < (mystuff->num_streams + mystuff->cpu_streams); i++) {
                    index_count = 0;
                    for (j = 0; j < mystuff->num_streams; j++)
                        if (h_ktab_inuse[j] == i) index_count++;
                    for (j = 0; j < mystuff->cpu_streams; j++)
                        if (h_ktab_cpu[j] == i) index_count++;
                    if (index_count != 1) {
                        printf("DEBUG_STREAM_SCHEDULE_CHECK: ERROR: index %d appeared %d times\n", i, index_count);
                        printf("  h_ktab_inuse[] =");
                        for (j = 0; j < mystuff->num_streams; j++)
                            printf(" %d", h_ktab_inuse[j]);
                        printf("\n  h_ktab_cpu[] =");
                        for (j = 0; j < mystuff->cpu_streams; j++)
                            printf(" %d", h_ktab_cpu[j]);
                        printf("\n");
                    }
                }
#endif
            }
            stream++;
        }
    }

    /* wait to finish the current calculations on the device */
    cuda_ret = cudaDeviceSynchronize();
    if (cuda_ret != cudaSuccess) printf("per class final cudaDeviceSynchronize failed!\n");

    /* download results from GPU */
    cudaMemcpy(mystuff->h_RES, mystuff->d_RES, 32 * sizeof(int), cudaMemcpyDeviceToHost);

#ifdef DEBUG_GPU_MATH
    cudaMemcpy(mystuff->h_modbasecase_debug, mystuff->d_modbasecase_debug, 32 * sizeof(int), cudaMemcpyDeviceToHost);
    for (i = 0; i < 32; i++)
        if (mystuff->h_modbasecase_debug[i] != 0) printf("h_modbasecase_debug[%2d] = %u\n", i, mystuff->h_modbasecase_debug[i]);
#endif

    mystuff->stats.grid_count = count;
    mystuff->stats.class_time = timer_diff(&timer) / 1000;
    mystuff->stats.bit_level_time += timer_diff(&timer) / 1000;
    /* prevent division by zero if timer resolution is too low */
    if (mystuff->stats.class_time == 0) mystuff->stats.class_time = 1;

    if (count > 2 * mystuff->num_streams)
        mystuff->stats.cpu_wait = (float)twait / ((float)mystuff->stats.class_time * 10);
    else
        mystuff->stats.cpu_wait = -1.0f;

    print_status_line(mystuff);

    if (mystuff->stats.cpu_wait >= 0.0f) {
        /* if SievePrimesAdjust is enable lets try to get 2 % < CPU wait < 6% */
        if (mystuff->sieve_primes_adjust == 1 && mystuff->stats.cpu_wait > 6.0f &&
            mystuff->sieve_primes < mystuff->sieve_primes_upper_limit && (mystuff->mode != MODE_SELFTEST_SHORT)) {
            mystuff->sieve_primes *= 9;
            mystuff->sieve_primes /= 8;
            if (mystuff->sieve_primes > mystuff->sieve_primes_upper_limit) mystuff->sieve_primes = mystuff->sieve_primes_upper_limit;
        }
        if (mystuff->sieve_primes_adjust == 1 && mystuff->stats.cpu_wait < 2.0f && mystuff->sieve_primes > mystuff->sieve_primes_min &&
            (mystuff->mode != MODE_SELFTEST_SHORT)) {
            mystuff->sieve_primes *= 7;
            mystuff->sieve_primes /= 8;
            if (mystuff->sieve_primes < mystuff->sieve_primes_min) mystuff->sieve_primes = mystuff->sieve_primes_min;
        }
    }

    factorsfound = mystuff->h_RES[0];
    for (i = 0; (i < factorsfound) && (i < 10); i++) {
        factor.d2 = mystuff->h_RES[i * 3 + 1];
        factor.d1 = mystuff->h_RES[i * 3 + 2];
        factor.d0 = mystuff->h_RES[i * 3 + 3];
#ifdef TF_72BIT
        print_dez72(factor, string);
#endif
#if defined(TF_96BIT) || defined(TF_BARRETT)
        print_dez96(factor, string);
#endif
        print_factor(mystuff, i, string);
    }
    if (factorsfound >= 10) {
        print_factor(mystuff, factorsfound, NULL);
    }

    return factorsfound;
}

#undef MFAKTC_FUNC
