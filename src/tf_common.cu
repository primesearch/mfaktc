/*
This file is part of mfaktc.
Copyright (C) 2009, 2010, 2011  Oliver Weihe (o.weihe@t-online.de)

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
extern "C" __host__ int tf_class_71(unsigned int exp, int bit_min, int bit_max, unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff)
#define MFAKTC_FUNC mfaktc_71
#define KERNEL_NAME "71bit_mul24"
#endif
#ifdef TF_96BIT
  #ifdef SHORTCUT_75BIT
extern "C" __host__ int tf_class_75(unsigned int exp, int bit_min, int bit_max, unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff)
#define MFAKTC_FUNC mfaktc_75
#define KERNEL_NAME "75bit_mul32"
  #else
extern "C" __host__ int tf_class_95(unsigned int exp, int bit_min, int bit_max, unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff)
#define MFAKTC_FUNC mfaktc_95
#define KERNEL_NAME "95bit_mul32"
  #endif
#endif
#ifdef TF_BARRETT
  #ifdef TF_BARRETT_79BIT
extern "C" __host__ int tf_class_barrett79(unsigned int exp, int bit_min, int bit_max, unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff)
#define MFAKTC_FUNC mfaktc_barrett79
#define KERNEL_NAME "barrett79_mul32"
  #else
extern "C" __host__ int tf_class_barrett92(unsigned int exp, int bit_min, int bit_max, unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff)
#define MFAKTC_FUNC mfaktc_barrett92
#define KERNEL_NAME "barrett92_mul32"
  #endif
#endif
{
  size_t size = mystuff->threads_per_grid * sizeof(int);
  int i, index = 0, stream;
  cudaError_t cuda_ret;
  timeval timer;
  timeval timer2;
  unsigned long long int twait = 0, eta;
  float cpuwait = 0.0;
#ifdef TF_72BIT  
  int72 factor,k_base;
  int144 b_preinit;
#endif
#if defined(TF_96BIT) || defined(TF_BARRETT)
  int96 factor,k_base;
  int192 b_preinit;
#endif
  int shiftcount, ln2b, count = 0;
  unsigned long long int k_diff;
  unsigned long long int t;
  char string[50];
  int factorsfound = 0;
  FILE *resultfile;
  
  int h_ktab_index = 0;
  int h_ktab_cpu[CPU_STREAMS_MAX];			// the set of h_ktab[N]s currently ownt by CPU
							// 0 <= N < h_ktab_index: these h_ktab[]s are preprocessed
                                                        // h_ktab_index <= N < mystuff.cpu_streams: these h_ktab[]s are NOT preprocessed
  int h_ktab_inuse[NUM_STREAMS_MAX];			// h_ktab_inuse[N] contains the number of h_ktab[] currently used by stream N
  unsigned long long int k_min_grid[CPU_STREAMS_MAX];	// k_min_grid[N] contains the k_min for h_ktab[h_ktab_cpu[N]], only valid for preprocessed h_ktab[]s
  
  timer_init(&timer);

  int threadsPerBlock = THREADS_PER_BLOCK;
  int blocksPerGrid = (mystuff->threads_per_grid + threadsPerBlock - 1) / threadsPerBlock;
  
  unsigned int delay = 1000;
  
  for(i=0; i<mystuff->num_streams; i++)h_ktab_inuse[i] = i;
  for(i=0; i<mystuff->cpu_streams; i++)h_ktab_cpu[i] = i + mystuff->num_streams;
  for(i=0; i<mystuff->cpu_streams; i++)k_min_grid[i] = 0;
  h_ktab_index = 0;
  
  shiftcount=0;
  while((1ULL<<shiftcount) < (unsigned long long int)exp)shiftcount++;
//  printf("\n\nshiftcount = %d\n",shiftcount);
  shiftcount-=1;ln2b=1;
  while(ln2b<20 || ln2b<bit_min)	// how much preprocessing is possible
  {
    shiftcount--;
    ln2b<<=1;
    if(exp&(1<<(shiftcount)))ln2b++;
  }
//  printf("shiftcount = %d\n",shiftcount);
//  printf("ln2b = %d\n",ln2b);
  b_preinit.d5=0;b_preinit.d4=0;b_preinit.d3=0;b_preinit.d2=0;b_preinit.d1=0;b_preinit.d0=0;
#ifdef TF_72BIT  
  if     (ln2b<24 )b_preinit.d0=1<< ln2b;
  else if(ln2b<48 )b_preinit.d1=1<<(ln2b-24);
  else if(ln2b<72 )b_preinit.d2=1<<(ln2b-48);
  else if(ln2b<96 )b_preinit.d3=1<<(ln2b-72);
  else if(ln2b<120)b_preinit.d4=1<<(ln2b-96);
  else             b_preinit.d5=1<<(ln2b-120);	// b_preinit = 2^ln2b
#endif
#if defined(TF_96BIT) || defined(TF_BARRETT)
  if     (ln2b<32 )b_preinit.d0=1<< ln2b;
  else if(ln2b<64 )b_preinit.d1=1<<(ln2b-32);
  else if(ln2b<96 )b_preinit.d2=1<<(ln2b-64);
  else if(ln2b<128)b_preinit.d3=1<<(ln2b-96);
  else if(ln2b<160)b_preinit.d4=1<<(ln2b-128);
  else             b_preinit.d5=1<<(ln2b-160);	// b_preinit = 2^ln2b
#endif  


#ifdef VERBOSE_TIMING
  printf("mfakt(%u,...) init:     %" PRIu64 "msec\n",exp,timer_diff(&timer)/1000);
#endif

/* set result array to 0 */  
  for(i=0;i<32;i++)mystuff->h_RES[i]=0;
  cudaMemcpy(mystuff->d_RES, mystuff->h_RES, 32*sizeof(int), cudaMemcpyHostToDevice);

#ifdef CHECKS_MODBASECASE  
  cudaMemcpy(mystuff->d_modbasecase_debug, mystuff->h_RES, 32*sizeof(int), cudaMemcpyHostToDevice);
#endif

  timer_init(&timer2);
  while((k_min <= k_max) || (h_ktab_index > 0))
  {
#ifdef VERBOSE_TIMING
    printf("##### k_start = %" PRIu64 " #####\n",k_min);
    printf("mfakt(%u,...) start:    %" PRIu64 "msec\n",exp,timer_diff(&timer)/1000);
#endif

    
/* preprocessing: calculate a ktab (factor table) */
    if((k_min <= k_max) && (h_ktab_index < mystuff->cpu_streams))	// if we have an empty h_ktab we can preprocess another one
    {
      delay = 1000;
      index = h_ktab_cpu[h_ktab_index];

      if(count > mystuff->num_streams)
      {
        twait+=timer_diff(&timer2);
      }
#ifdef DEBUG_STREAM_SCHEDULE
      printf(" STREAM_SCHEDULE: preprocessing on h_ktab[%d] (count = %d)\n", index, count);
#endif
    
      sieve_candidates(mystuff->threads_per_grid, mystuff->h_ktab[index], mystuff->sieve_primes);
      k_diff=mystuff->h_ktab[index][mystuff->threads_per_grid-1]+1;
      k_diff*=NUM_CLASSES;				/* NUM_CLASSES because classes are mod NUM_CLASSES */
      
      k_min_grid[h_ktab_index] = k_min;
      h_ktab_index++;
      
#ifdef VERBOSE_TIMING
      printf("mfakt(%u,...) sieving:  %" PRIu64 "msec\n",exp,timer_diff(&timer)/1000);
#endif
      count++;
      k_min += (unsigned long long int)k_diff;
      timer_init(&timer2);
    }
    else if(mystuff->allowsleep == 1)
    {
      /* no unused h_ktab for preprocessing. 
      This usually means that
      a) all GPU streams are busy 
      and
      b) we've preprocessed all available CPU streams
      so let's sleep for some time instead of running a busy loop on cudaStreamQuery() */
      my_usleep(delay);

      delay = delay * 3 / 2;
      if(delay > 500000) delay = 500000;
    }


/* try upload ktab and start the calcualtion of a preprocessed dataset on the device */
    stream = 0;
    while((stream < mystuff->num_streams) && (h_ktab_index > 0))
    {
      if(cudaStreamQuery(mystuff->stream[stream]) == cudaSuccess)
      {
#ifdef DEBUG_STREAM_SCHEDULE
        printf(" STREAM_SCHEDULE: found empty stream: = %d (this releases h_ktab[%d])\n", stream, h_ktab_inuse[stream]);
#endif
        h_ktab_index--;
        i                        = h_ktab_inuse[stream];
        h_ktab_inuse[stream]     = h_ktab_cpu[h_ktab_index];
        h_ktab_cpu[h_ktab_index] = i;

        cudaMemcpyAsync(mystuff->d_ktab[stream], mystuff->h_ktab[h_ktab_inuse[stream]], size, cudaMemcpyHostToDevice, mystuff->stream[stream]);

#ifdef TF_72BIT    
        k_base.d0 =  k_min_grid[h_ktab_index] & 0xFFFFFF;
        k_base.d1 = (k_min_grid[h_ktab_index] >> 24) & 0xFFFFFF;
        k_base.d2 =  k_min_grid[h_ktab_index] >> 48;
#elif defined(TF_96BIT) || defined(TF_BARRETT)
        k_base.d0 =  k_min_grid[h_ktab_index] & 0xFFFFFFFF;
        k_base.d1 =  k_min_grid[h_ktab_index] >> 32;
        k_base.d2 = 0;
#endif    

#if defined(TF_72BIT) || defined(TF_96BIT)
  #ifndef CHECKS_MODBASECASE
        MFAKTC_FUNC<<<blocksPerGrid, threadsPerBlock, 0, mystuff->stream[stream]>>>(exp, k_base, mystuff->d_ktab[stream], shiftcount, b_preinit, mystuff->d_RES);
  #else
        MFAKTC_FUNC<<<blocksPerGrid, threadsPerBlock, 0, mystuff->stream[stream]>>>(exp, k_base, mystuff->d_ktab[stream], shiftcount, b_preinit, mystuff->d_RES, mystuff->d_modbasecase_debug);
  #endif
#elif defined(TF_BARRETT)
  #ifndef CHECKS_MODBASECASE
    #ifndef TF_BARRETT_79BIT
        MFAKTC_FUNC<<<blocksPerGrid, threadsPerBlock, 0, mystuff->stream[stream]>>>(exp, k_base, mystuff->d_ktab[stream], shiftcount, b_preinit, mystuff->d_RES, bit_min-63);
    #else        
        MFAKTC_FUNC<<<blocksPerGrid, threadsPerBlock, 0, mystuff->stream[stream]>>>(exp, k_base, mystuff->d_ktab[stream], shiftcount, b_preinit, mystuff->d_RES);
    #endif
  #else
        MFAKTC_FUNC<<<blocksPerGrid, threadsPerBlock, 0, mystuff->stream[stream]>>>(exp, k_base, mystuff->d_ktab[stream], shiftcount, b_preinit, mystuff->d_RES, bit_min-63, mystuff->d_modbasecase_debug);
  #endif
#endif

#ifdef DEBUG_STREAM_SCHEDULE
        printf(" STREAM_SCHEDULE: started GPU kernel on stream %d using h_ktab[%d]\n\n", stream, h_ktab_inuse[stream]);
#endif
#ifdef CHECKS_MODBASECASE
        cudaThreadSynchronize(); /* needed to get the output from device printf() */
#endif
#ifdef DEBUG_STREAM_SCHEDULE_CHECK
        int j, index_count;
        for(i=0; i < (mystuff->num_streams + mystuff->cpu_streams); i++)
        {
          index_count = 0;
          for(j=0; j<mystuff->num_streams; j++)if(h_ktab_inuse[j] == i)index_count++;
          for(j=0; j<mystuff->cpu_streams; j++)if(h_ktab_cpu[j] == i)index_count++;
          if(index_count != 1)
          {
            printf("DEBUG_STREAM_SCHEDULE_CHECK: ERROR: index %d appeared %d times\n", i, index_count);
            printf("  h_ktab_inuse[] =");
            for(j=0; j<mystuff->num_streams; j++)printf(" %d", h_ktab_inuse[j]);
            printf("\n  h_ktab_cpu[] =");
            for(j=0; j<mystuff->cpu_streams; j++)printf(" %d", h_ktab_cpu[j]);
            printf("\n");
          }
        }
#endif
      }
      stream++;
    }
  }

/* wait to finish the current calculations on the device */
  cuda_ret = cudaThreadSynchronize();
  if(cuda_ret != cudaSuccess)printf("per class final cudaThreadSynchronize failed!\n");

#ifdef VERBOSE_TIMING
  printf("mfakt(%u,...) wait:     %" PRIu64 "msec\n",exp,timer_diff(&timer)/1000);
  printf("##### k_end = %" PRIu64 " #####\n",k_min);
#endif    

/* download results from GPU */
  cudaMemcpy(mystuff->h_RES, mystuff->d_RES, 32*sizeof(int), cudaMemcpyDeviceToHost);
#ifdef VERBOSE_TIMING
  printf("mfakt(%u,...) download: %" PRIu64 "msec\n",exp,timer_diff(&timer)/1000);
#endif


#ifdef CHECKS_MODBASECASE
  cudaMemcpy(mystuff->h_modbasecase_debug, mystuff->d_modbasecase_debug, 32*sizeof(int), cudaMemcpyDeviceToHost);
  for(i=0;i<32;i++)if(mystuff->h_modbasecase_debug[i] != 0)printf("h_modbasecase_debug[%2d] = %u\n", i, mystuff->h_modbasecase_debug[i]);
#endif  

#ifdef VERBOSE_TIMING
  printf("mfakt(%u,...) check:    %" PRIu64 "msec\n",exp,timer_diff(&timer)/1000);
#endif

  t=timer_diff(&timer)/1000;
  if(t==0)t=1;	/* prevent division by zero in the following printf(s) */

  if(mystuff->mode != MODE_SELFTEST_SHORT)
  {
    printf("%4" PRIu64 "/%4d", k_min%NUM_CLASSES, (int)NUM_CLASSES);

    if(((unsigned long long int)mystuff->threads_per_grid * (unsigned long long int)count) < 1000000000ULL)
      printf(" | %9.2fM", (double)mystuff->threads_per_grid * (double)count / 1000000.0);
    else
      printf(" | %9.2fG", (double)mystuff->threads_per_grid * (double)count / 1000000000.0);

         if(t < 100000ULL  )printf(" | %6.3fs", (double)t/1000.0);
    else if(t < 1000000ULL )printf(" | %6.2fs", (double)t/1000.0);
    else if(t < 10000000ULL)printf(" | %6.1fs", (double)t/1000.0);
    else                    printf(" | %6.0fs", (double)t/1000.0);

    if(mystuff->mode == MODE_NORMAL)
    {
      if(t > 250.0)
      {
        
#ifdef MORE_CLASSES      
        eta = (t * (960 - mystuff->class_counter) + 500)  / 1000;
#else
        eta = (t * (96 - mystuff->class_counter) + 500)  / 1000;
#endif
             if(eta < 3600) printf(" | %2" PRIu64 "m%02" PRIu64 "s", eta / 60, eta % 60);
        else if(eta < 86400)printf(" | %2" PRIu64 "h%02" PRIu64 "m", eta / 3600, (eta / 60) % 60);
        else                printf(" | %2" PRIu64 "d%02" PRIu64 "h", eta / 86400, (eta / 3600) % 24);
      }
      else printf(" |   n.a.");
    }
    else if(mystuff->mode == MODE_SELFTEST_FULL)printf(" |   n.a.");

    printf(" | %6.2fM/s", (double)mystuff->threads_per_grid * (double)count / ((double)t * 1000.0));
    
    printf(" | %11d", mystuff->sieve_primes);
  }

  if(count > 2 * mystuff->num_streams)
  {
    cpuwait = (float)twait / ((float)t * 10);
    if(mystuff->mode != MODE_SELFTEST_SHORT)printf(" |   %5.2f%%", cpuwait);
/* if SievePrimesAdjust is enable lets try to get 2 % < CPU wait < 6% */
    if(mystuff->sieve_primes_adjust == 1 && cpuwait > 6.0 && mystuff->sieve_primes < mystuff->sieve_primes_max && (mystuff->mode != MODE_SELFTEST_SHORT))
    {
      mystuff->sieve_primes *= 9;
      mystuff->sieve_primes /= 8;
      if(mystuff->sieve_primes > mystuff->sieve_primes_max) mystuff->sieve_primes = mystuff->sieve_primes_max;
    }
    if(mystuff->sieve_primes_adjust == 1 && cpuwait < 2.0  && mystuff->sieve_primes > SIEVE_PRIMES_MIN && (mystuff->mode != MODE_SELFTEST_SHORT))
    {
      mystuff->sieve_primes *= 7;
      mystuff->sieve_primes /= 8;
      if(mystuff->sieve_primes < SIEVE_PRIMES_MIN) mystuff->sieve_primes = SIEVE_PRIMES_MIN;
    }
  }
  else if(mystuff->mode != MODE_SELFTEST_SHORT)printf(" |     n.a.");


  if(mystuff->mode == MODE_NORMAL)
  {
    if(mystuff->printmode == 1)printf("\r");
    else printf("\n");
  }
  if(mystuff->mode == MODE_SELFTEST_FULL && mystuff->printmode == 0)
  {
    printf("\n");
  }

  factorsfound=mystuff->h_RES[0];
  for(i=0; (i<factorsfound) && (i<10); i++)
  {
    factor.d2=mystuff->h_RES[i*3 + 1];
    factor.d1=mystuff->h_RES[i*3 + 2];
    factor.d0=mystuff->h_RES[i*3 + 3];
#ifdef TF_72BIT    
    print_dez72(factor,string);
#endif    
#if defined(TF_96BIT) || defined(TF_BARRETT)
    print_dez96(factor,string);
#endif
    if(mystuff->mode != MODE_SELFTEST_SHORT)
    {
      if(mystuff->printmode == 1 && i == 0)printf("\n");
      printf("M%u has a factor: %s\n", exp, string);
    }
    if(mystuff->mode == MODE_NORMAL)
    {
      resultfile=fopen("results.txt", "a");
#ifndef MORE_CLASSES      
      fprintf(resultfile,"M%u has a factor: %s [TF:%d:%d%s:mfaktc %s %s]\n", exp, string, bit_min, bit_max, ((mystuff->stopafterfactor == 2) && (mystuff->class_counter <  96)) ? "*" : "" , MFAKTC_VERSION, KERNEL_NAME);
#else      
      fprintf(resultfile,"M%u has a factor: %s [TF:%d:%d%s:mfaktc %s %s]\n", exp, string, bit_min, bit_max, ((mystuff->stopafterfactor == 2) && (mystuff->class_counter < 960)) ? "*" : "" , MFAKTC_VERSION, KERNEL_NAME);
#endif
      fclose(resultfile);
    }
  }
  if(factorsfound>=10)
  {
    if(mystuff->mode != MODE_SELFTEST_SHORT)printf("M%u: %d additional factors not shown\n",exp,factorsfound-10);
    if(mystuff->mode == MODE_NORMAL)
    {
      resultfile=fopen("results.txt", "a");
      fprintf(resultfile,"M%u: %d additional factors not shown\n",exp,factorsfound-10);
      fclose(resultfile);
    }
  }

  return factorsfound;
}

#undef MFAKTC_FUNC
#undef KERNEL_NAME
