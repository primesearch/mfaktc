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

#include <stdio.h>
#include <stdlib.h>
#ifndef _MSC_VER
#include <unistd.h>
#endif
#include <string.h>
#include <errno.h> 

#include <cuda.h>
#include <cuda_runtime.h>  

#include "params.h"
#include "my_types.h"
#include "compatibility.h"

#include "sieve.h"
#include "read_config.h"
#include "parse.h"
#include "timer.h"
#include "tf_72bit.h"
#include "tf_96bit.h"
#include "tf_barrett96.h"
#include "checkpoint.h"

unsigned long long int calculate_k(unsigned int exp, int bits)
/* calculates biggest possible k in "2 * exp * k + 1 < 2^bits" */
{
  unsigned long long int k = 0, tmp_low, tmp_hi;
  
  if((bits > 65) && exp < (1 << (bits - 65))) k = 0; // k would be >= 2^64...
  else if(bits <= 64)
  {
    tmp_low = 1ULL << (bits - 1);
    tmp_low--;
    k = tmp_low / exp;
  }
  else if(bits <= 96)
  {
    tmp_hi = 1ULL << (bits - 33);
    tmp_hi--;
    tmp_low = 0xFFFFFFFFULL;
    
    k = tmp_hi / exp;
    tmp_low += (tmp_hi % exp) << 32;
    k <<= 32;
    k += tmp_low / exp;
  }
  
  if(k == 0)k = 1;
  return k;
}




int tf(unsigned int exp, int bit_min, int bit_max, mystuff_t *mystuff, int class_hint, unsigned long long int k_hint, int kernel)
/*
tf M<exp> from 2^bit_min to 2^bit_max

kernel: see my_types.h -> enum GPUKernels

return value (mystuff->mode = MODE_NORMAL):
number of factors found
123456 cudaGetLastError() returned an error

return value (mystuff->mode = MODE_SELFTEST_SHORT or MODE_SELFTEST_FULL):
0 for a successfull selftest (known factor was found)
1 no factor found
2 wrong factor returned
123456 cudaGetLastError() returned an error

other return value 
-1 unknown mode
*/
{
  int cur_class, max_class = NUM_CLASSES-1, i, count = 0;
  unsigned long long int k_min, k_max, k_range, tmp;
  unsigned int f_hi, f_med, f_low;
  struct timeval timer;
#ifdef VERBOSE_TIMING  
  struct timeval timer2;
#endif  
  int factorsfound = 0, restart = 0;
  FILE *resultfile=NULL;

  char kernelname[20];
  int retval = 0;
  
  cudaError_t cudaError;
  
  unsigned long long int time_run, time_est;


  if(mystuff->mode != MODE_SELFTEST_SHORT)printf("tf(%u, %d, %d, ...);\n",exp,bit_min,bit_max);
  if((mystuff->mode != MODE_NORMAL) && (mystuff->mode != MODE_SELFTEST_SHORT) && (mystuff->mode != MODE_SELFTEST_FULL))
  {
    printf("ERROR, invalid mode for tf(): %d\n", mystuff->mode);
    return -1;
  }
  timer_init(&timer);
  
  mystuff->class_counter = 0;
  
  k_min=calculate_k(exp,bit_min);
  k_max=calculate_k(exp,bit_max);
  
  if((mystuff->mode == MODE_SELFTEST_FULL) || (mystuff->mode == MODE_SELFTEST_SHORT))
  {
/* a shortcut for the selftest, bring k_min a k_max "close" to the known factor */
    if(NUM_CLASSES == 420)k_range = 10000000000ULL;
    else                  k_range = 100000000000ULL;
    if(mystuff->mode == MODE_SELFTEST_SHORT)k_range /= 5; /* even smaller ranges for the "small" selftest */
    if((k_max - k_min) > (3ULL * k_range))
    {
      tmp = k_hint - (k_hint % k_range) - k_range;
      if(tmp > k_min) k_min = tmp;

      tmp += 3ULL * k_range;
      if((tmp < k_max) || (k_max < k_min)) k_max = tmp; /* check for k_max < k_min enables some selftests where k_max >= 2^64 but the known factor itself has a k < 2^64 */
    }
  }

  k_min -= k_min % NUM_CLASSES;	/* k_min is now 0 mod NUM_CLASSES */

  if(mystuff->mode != MODE_SELFTEST_SHORT)
  {
    printf(" k_min = %" PRIu64 "\n",k_min);
    printf(" k_max = %" PRIu64 "\n",k_max);
  }

  if(kernel == AUTOSELECT_KERNEL)
  {
/* select the GPU kernel (fastest GPU kernel has highest priority) */
    if(mystuff->compcapa_major == 1)
    {
      if                         (bit_max <= 71)                              kernel = _71BIT_MUL24;
      else if((bit_min >= 64) && (bit_max <= 79))                             kernel = BARRETT79_MUL32;
      else if                    (bit_max <= 75)                              kernel = _75BIT_MUL32;
      else if((bit_min >= 64) && (bit_max <= 92) && (bit_max - bit_min == 1)) kernel = BARRETT92_MUL32;
      else                                                                    kernel = _95BIT_MUL32;
    }
    else // mystuff->compcapa_major != 1
    {
           if((bit_min >= 64) && (bit_max <= 79))                             kernel = BARRETT79_MUL32;
      else if((bit_min >= 64) && (bit_max <= 92) && (bit_max - bit_min == 1)) kernel = BARRETT92_MUL32;
      else if                    (bit_max <= 75)                              kernel = _75BIT_MUL32;
      else                                                                    kernel = _95BIT_MUL32;
    }
  }

       if(kernel == _71BIT_MUL24)    sprintf(kernelname, "71bit_mul24");
  else if(kernel == _75BIT_MUL32)    sprintf(kernelname, "75bit_mul32");
  else if(kernel == _95BIT_MUL32)    sprintf(kernelname, "95bit_mul32");
  else if(kernel == BARRETT79_MUL32) sprintf(kernelname, "barrett79_mul32");
  else if(kernel == BARRETT92_MUL32) sprintf(kernelname, "barrett92_mul32");
  else                               sprintf(kernelname, "UNKNOWN kernel");

  if(mystuff->mode != MODE_SELFTEST_SHORT)printf("Using GPU kernel \"%s\"\n",kernelname);

  if(mystuff->mode == MODE_NORMAL)
  {
    if((mystuff->checkpoints == 1) && (checkpoint_read(exp, bit_min, bit_max, &cur_class, &factorsfound) == 1))
    {
      printf("\nfound a valid checkpoint file!\n");
      printf("  last finished class was: %d\n", cur_class);
      printf("  found %d factor(s) already\n\n", factorsfound);
      cur_class++; // the checkpoint contains the last complete processed class!

/* calculate the number of classes which are allready processed. This value is needed to estimate ETA */
      for(i = 0; i < cur_class; i++)
      {
/* check if class is NOT "3 or 5 mod 8", "0 mod 3", "0 mod 5", "0 mod 7" (or "0 mod 11") */
        if( ((2 * (exp% 8) * ((k_min+i)% 8)) % 8 !=  2) && \
            ((2 * (exp% 8) * ((k_min+i)% 8)) % 8 !=  4) && \
            ((2 * (exp% 3) * ((k_min+i)% 3)) % 3 !=  2) && \
            ((2 * (exp% 5) * ((k_min+i)% 5)) % 5 !=  4) && \
            ((2 * (exp% 7) * ((k_min+i)% 7)) % 7 !=  6))
#ifdef MORE_CLASSES        
        if(  (2 * (exp%11) * ((k_min+i)%11)) %11 != 10 )
#endif    
        {
          mystuff->class_counter++;
        }
      }
      restart = mystuff->class_counter;
    }
    else
    {
      cur_class=0;
    }
  }
  else // mystuff->mode != MODE_NORMAL
  {
    cur_class = class_hint % NUM_CLASSES;
    max_class = cur_class;
  }

  for(; cur_class <= max_class; cur_class++)
  {
/* check if class is NOT "3 or 5 mod 8", "0 mod 3", "0 mod 5", "0 mod 7" (or "0 mod 11") */
    if( ((2 * (exp% 8) * ((k_min+cur_class)% 8)) % 8 !=  2) && \
        ((2 * (exp% 8) * ((k_min+cur_class)% 8)) % 8 !=  4) && \
        ((2 * (exp% 3) * ((k_min+cur_class)% 3)) % 3 !=  2) && \
        ((2 * (exp% 5) * ((k_min+cur_class)% 5)) % 5 !=  4) && \
        ((2 * (exp% 7) * ((k_min+cur_class)% 7)) % 7 !=  6))
#ifdef MORE_CLASSES        
    if(  (2 * (exp%11) * ((k_min+cur_class)%11)) %11 != 10 )
#endif    
    {
#ifdef VERBOSE_TIMING
      timer_init(&timer2);
#endif    
      sieve_init_class(exp, k_min+cur_class, mystuff->sieve_primes);
#ifdef VERBOSE_TIMING      
      printf("tf(): time spent for sieve_init_class(exp, k_min+cur_class, mystuff->sieve_primes): %" PRIu64 "ms\n",timer_diff(&timer2)/1000);
#endif
      if(mystuff->mode != MODE_SELFTEST_SHORT && (count == 0 || (count%20 == 0 && mystuff->printmode == 0)))
      {
        printf("    class | candidates |    time | avg. rate | SievePrimes |    ETA | avg. wait\n");
      }
      count++;
      mystuff->class_counter++;
      
           if(kernel == _71BIT_MUL24)    factorsfound+=tf_class_71       (exp, bit_min, k_min+cur_class, k_max, mystuff);
      else if(kernel == _75BIT_MUL32)    factorsfound+=tf_class_75       (exp, bit_min, k_min+cur_class, k_max, mystuff);
      else if(kernel == _95BIT_MUL32)    factorsfound+=tf_class_95       (exp, bit_min, k_min+cur_class, k_max, mystuff);
      else if(kernel == BARRETT79_MUL32) factorsfound+=tf_class_barrett79(exp, bit_min, k_min+cur_class, k_max, mystuff);
      else if(kernel == BARRETT92_MUL32) factorsfound+=tf_class_barrett92(exp, bit_min, k_min+cur_class, k_max, mystuff);
      else
      {
        printf("ERROR: Unknown kernel selected (%d)!\n", kernel);
        return 123456;
      }
      cudaError = cudaGetLastError();
      if(cudaError != cudaSuccess)
      {
        printf("ERROR: cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
        return 123456; /* bail out, we might have a serios problem (detected by cudaGetLastError())... */
      }
      if(mystuff->mode == MODE_NORMAL)
      {
        if(mystuff->checkpoints == 1)checkpoint_write(exp, bit_min, bit_max, cur_class, factorsfound);
        if((mystuff->stopafterfactor >= 2) && (factorsfound > 0) && (cur_class != max_class))cur_class = max_class + 1;
      }
    }
fflush(NULL);    
  }
  if(mystuff->mode != MODE_SELFTEST_SHORT && mystuff->printmode == 1)printf("\n");
  if(mystuff->mode == MODE_NORMAL)resultfile=fopen("results.txt", "a");
  if(factorsfound)
  {
    if((mystuff->mode == MODE_NORMAL) && (mystuff->stopafterfactor >= 2))
    {
      fprintf(resultfile, "found %d factor(s) for M%u from 2^%2d to 2^%2d (partially tested) [mfaktc %s %s]\n", factorsfound, exp, bit_min, bit_max, MFAKTC_VERSION, kernelname);
      printf(             "found %d factor(s) for M%u from 2^%2d to 2^%2d (partially tested) [mfaktc %s %s]\n", factorsfound, exp, bit_min, bit_max, MFAKTC_VERSION, kernelname);
    }
    else
    {
      if(mystuff->mode == MODE_NORMAL)        fprintf(resultfile, "found %d factor(s) for M%u from 2^%2d to 2^%2d [mfaktc %s %s]\n", factorsfound, exp, bit_min, bit_max, MFAKTC_VERSION, kernelname);
      if(mystuff->mode != MODE_SELFTEST_SHORT)printf(             "found %d factor(s) for M%u from 2^%2d to 2^%2d [mfaktc %s %s]\n", factorsfound, exp, bit_min, bit_max, MFAKTC_VERSION, kernelname);
    }
  }
  else
  {
    if(mystuff->mode == MODE_NORMAL)        fprintf(resultfile, "no factor for M%u from 2^%d to 2^%d [mfaktc %s %s]\n", exp, bit_min, bit_max, MFAKTC_VERSION, kernelname);
    if(mystuff->mode != MODE_SELFTEST_SHORT)printf(             "no factor for M%u from 2^%d to 2^%d [mfaktc %s %s]\n", exp, bit_min, bit_max, MFAKTC_VERSION, kernelname);
  }
  if(mystuff->mode == MODE_NORMAL)
  {
    retval = factorsfound;
    fclose(resultfile);
    if(mystuff->checkpoints == 1)checkpoint_delete(exp);
  }
  else // mystuff->mode != MODE_NORMAL
  {
    if(mystuff->h_RES[0] == 0)
    {
      printf("ERROR: selftest failed for M%u\n", exp);
      printf("  no factor found\n");
      retval = 1;
    }
    else // mystuff->h_RES[0] > 0
    {
/*
calculate the value of the known factor in f_{hi|med|low} and compare with the
results from the selftest.
k_max and k_min are used as 64bit temporary integers here...
*/    
      f_hi    = (k_hint >> 63);
      f_med   = (k_hint >> 31) & 0xFFFFFFFFULL;
      f_low   = (k_hint <<  1) & 0xFFFFFFFFULL; /* f_{hi|med|low} = 2 * k_hint */
      
      k_max   = (unsigned long long int)exp * f_low;
      f_low   = (k_max & 0xFFFFFFFFULL) + 1;
      k_min   = (k_max >> 32);

      k_max   = (unsigned long long int)exp * f_med;
      k_min  += k_max & 0xFFFFFFFFULL;
      f_med   = k_min & 0xFFFFFFFFULL;
      k_min >>= 32;
      k_min  += (k_max >> 32);

      f_hi  = k_min + (exp * f_hi); /* f_{hi|med|low} = 2 * k_hint * exp +1 */
      
      if(kernel == _71BIT_MUL24) /* 71bit kernel uses only 24bit per int */
      {
        f_hi  <<= 16;
        f_hi   += f_med >> 16;

        f_med <<= 8;
        f_med  += f_low >> 24;
        f_med  &= 0x00FFFFFF;
        
        f_low  &= 0x00FFFFFF;
      }
      k_min=0;
      for(i=0; (i<mystuff->h_RES[0]) && (i<10); i++)
      {
        if(mystuff->h_RES[i*3 + 1] == f_hi  && \
           mystuff->h_RES[i*3 + 2] == f_med && \
           mystuff->h_RES[i*3 + 3] == f_low) k_min++;
      }
      if(k_min != 1) /* the factor should appear ONCE */
      {
        printf("ERROR: selftest failed for M%u!\n", exp);
        printf("  expected result: %08X %08X %08X\n", f_hi, f_med, f_low);
        for(i=0; (i<mystuff->h_RES[0]) && (i<10); i++)
        {
          printf("  reported result: %08X %08X %08X\n", mystuff->h_RES[i*3 + 1], mystuff->h_RES[i*3 + 2], mystuff->h_RES[i*3 + 3]);
        }
        retval = 2;
      }
      else
      {
        if(mystuff->mode != MODE_SELFTEST_SHORT)printf("selftest for M%u passed!\n", exp);
      }
    }
  }
  if(mystuff->mode != MODE_SELFTEST_SHORT)
  {
    time_run = timer_diff(&timer)/1000;
    
    if(restart == 0)printf("tf(): total time spent: ");
    else            printf("tf(): time spent since restart:   ");

/*  restart == 0 ==> time_est = time_run */
#ifndef MORE_CLASSES
    time_est = (time_run * 96ULL  ) / (unsigned long long int)(96 -restart);
#else
    time_est = (time_run * 960ULL ) / (unsigned long long int)(960-restart);
#endif

    if(time_est > 86400000ULL)printf("%" PRIu64 "d ",   time_run / 86400000ULL);
    if(time_est > 3600000ULL) printf("%2" PRIu64 "h ", (time_run /  3600000ULL) % 24ULL);
    if(time_est > 60000ULL)   printf("%2" PRIu64 "m ", (time_run /    60000ULL) % 60ULL);
                              printf("%2" PRIu64 ".%03" PRIu64 "s\n", (time_run / 1000ULL) % 60ULL, time_run % 1000ULL);
    if(restart != 0)
    {
      printf("      estimated total time spent: ");
      if(time_est > 86400000ULL)printf("%" PRIu64 "d ",   time_est / 86400000ULL);
      if(time_est > 3600000ULL) printf("%2" PRIu64 "h ", (time_est /  3600000ULL) % 24ULL);
      if(time_est > 60000ULL)   printf("%2" PRIu64 "m ", (time_est /    60000ULL) % 60ULL);
                                printf("%2" PRIu64 ".%03" PRIu64 "s\n", (time_est / 1000ULL) % 60ULL, time_est % 1000ULL);
    }
    printf("\n");
  }
  return retval;
}


void print_help(char *string)
{
  printf("mfaktc v%s Copyright (C) 2009, 2010, 2011  Oliver Weihe (o.weihe@t-online.de)\n", MFAKTC_VERSION);
  printf("This program comes with ABSOLUTELY NO WARRANTY; for details see COPYING.\n");
  printf("This is free software, and you are welcome to redistribute it\n");
  printf("under certain conditions; see COPYING for details.\n\n\n");

  printf("Usage: %s [options]\n", string);
  printf("  -h                     display this help and exit\n");
  printf("  -d <device number>     specify the device number used by this program\n");
  printf("  -tf <exp> <min> <max>  trial factor M<exp> from 2^<min> to 2^<max> and exit\n");
  printf("                         instead of parsing the worktodo file\n");
  printf("  -st                    run builtin selftest and exit\n");
  printf("\n");
  printf("options for debuging purposes\n");
  printf("  --timertest            run test of timer functions and exit\n");
  printf("  --sleeptest            run test of sleep functions and exit\n");
}


int selftest(mystuff_t *mystuff, int type)
/*
type = 0: full selftest
type = 1: small selftest (this is executed EACH time mfaktc is started)

return value
0 selftest passed
1 selftest failed
123456 we might have a serios problem (detected by cudaGetLastError())
*/
{
  int i, j, tf_res, st_success=0, st_nofactor=0, st_wrongfactor=0, st_unknown=0;

#define NUM_SELFTESTS 1557
  unsigned int exp[NUM_SELFTESTS], index[9], num_selftests=0;
  int bit_min[NUM_SELFTESTS], f_class;
  unsigned long long int k[NUM_SELFTESTS];
  int retval=1;
  int kernels[5];
  
#include "selftest-data.c"  

  if(type == 0)
  {
    for(i=0; i<NUM_SELFTESTS; i++)
    {
      f_class = (int)(k[i] % NUM_CLASSES);

/* create a list which kernels can handle this testcase */
      j = 0;
      if((bit_min[i] >= 64) && (bit_min[i]+1) <= 92)kernels[j++] = BARRETT92_MUL32; /* no need to check bit_max - bit_min == 1 ;) */
      if((bit_min[i] >= 64) && (bit_min[i]+1) <= 79)kernels[j++] = BARRETT79_MUL32; /* no need to check bit_max - bit_min == 1 ;) */
                                                    kernels[j++] = _95BIT_MUL32;
      if((bit_min[i]+1) <= 75)                      kernels[j++] = _75BIT_MUL32;
      if((bit_min[i]+1) <= 71)                      kernels[j++] = _71BIT_MUL24;

      do
      {
        num_selftests++;
        tf_res=tf(exp[i], bit_min[i], bit_min[i]+1, mystuff, f_class, k[i], kernels[--j]);
             if(tf_res == 0)st_success++;
        else if(tf_res == 1)st_nofactor++;
        else if(tf_res == 2)st_wrongfactor++;
        else if(tf_res == 123456)return 123456; /* bail out, we might have a serios problem (detected by cudaGetLastError())... */
        else           st_unknown++;
      }
      while(j>0);
    }
  }
  else if(type == 1)
  {
    index[0]=   2; index[1]=  25; index[2]=  57; /* some factors below 2^71 (test the 71/75 bit kernel depending on compute capability) */
    index[3]=  70; index[4]=  88; index[5]= 106; /* some factors below 2^75 (test 75 bit kernel) */
    index[6]=1547; index[7]=1552; index[8]=1556; /* some factors below 2^95 (test 95 bit kernel) */
    for(i=0; i<9; i++)
    {
      f_class = (int)(k[index[i]] % NUM_CLASSES);

      j = 0;
      if((bit_min[index[i]] >= 64) && (bit_min[index[i]]+1) <= 92)kernels[j++] = BARRETT92_MUL32; /* no need to check bit_max - bit_min == 1 ;) */
      if((bit_min[index[i]] >= 64) && (bit_min[index[i]]+1) <= 79)kernels[j++] = BARRETT79_MUL32; /* no need to check bit_max - bit_min == 1 ;) */
                                                                  kernels[j++] = _95BIT_MUL32;
      if((bit_min[index[i]]+1) <= 75)                             kernels[j++] = _75BIT_MUL32;
      if((bit_min[index[i]]+1) <= 71)                             kernels[j++] = _71BIT_MUL24;

      do
      {
        num_selftests++;
        tf_res=tf(exp[index[i]], bit_min[index[i]], bit_min[index[i]]+1, mystuff, f_class, k[index[i]], kernels[--j]);
             if(tf_res == 0)st_success++;
        else if(tf_res == 1)st_nofactor++;
        else if(tf_res == 2)st_wrongfactor++;
        else if(tf_res == 123456)return 123456; /* bail out, we might have a serios problem (detected by cudaGetLastError())... */
        else           st_unknown++;
      }
      while(j>0);
    }
  }
//  if((type == 0) || (st_success != num_selftests))
  {
    printf("Selftest statistics\n");
    printf("  number of tests           %d\n", num_selftests);
    printf("  successfull tests         %d\n", st_success);
    if(st_nofactor > 0)   printf("  no factor found           %d\n", st_nofactor);
    if(st_wrongfactor > 0)printf("  wrong factor reported     %d\n", st_wrongfactor);
    if(st_unknown > 0)    printf("  unknown return value      %d\n", st_unknown);
    printf("\n");
  }

  if(st_success == num_selftests)
  {
    printf("selftest PASSED!\n\n");
    retval=0;
  }
  else
  {
    printf("selftest FAILED!\n\n");
  }
  return retval;
}

void print_last_CUDA_error()
/* just run cudaGetLastError() and print the error message if its return value is not cudaSuccess */
{
  cudaError_t cudaError;
  
  cudaError = cudaGetLastError();
  if(cudaError != cudaSuccess)
  {
    printf("  cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
  }
}

int main(int argc, char **argv)
{
  unsigned int exp = 0;
  int bit_min = -1, bit_max = -1, bit_min_stage, bit_max_stage;
  int parse_ret = -1;
  int devicenumber = 0;
  mystuff_t mystuff;
#ifdef VERBOSE_TIMING  
  struct timeval timer;
#endif
  struct cudaDeviceProp deviceinfo;
  int i, tmp = 0;
  char *ptr;
  int use_worktodo = 1;
  
  i = 0;
  mystuff.mode=MODE_NORMAL;
  
  while(i<argc)
  {
    if(!strcmp((char*)"-h", argv[i]))
    {
      print_help(argv[0]);
      return 0;
    }
    else if(!strcmp((char*)"-d", argv[i]))
    {
      if(i+1 >= argc)
      {
        printf("ERROR: no device number specified for option \"-d\"\n");
        return 1;
      }
      devicenumber=(int)strtol(argv[i+1],&ptr,10);
      if(*ptr || errno || devicenumber != strtol(argv[i+1],&ptr,10) )
      {
        printf("ERROR: can't parse <device number> for option \"-d\"\n");
        return 1;
      }
      i++;
    }
    else if(!strcmp((char*)"-tf", argv[i]))
    {
      if(i+3 >= argc)
      {
        printf("ERROR: missing parameters for option \"-tf\"\n");
        return 1;
      }
      exp=(unsigned int)strtoul(argv[i+1],&ptr,10);
      if(*ptr || errno || (unsigned long)exp != strtoul(argv[i+1],&ptr,10) )
      {
        printf("ERROR: can't parse parameter <exp> for option \"-tf\"\n");
        return 1;
      }
      bit_min=(int)strtol(argv[i+2],&ptr,10);
      if(*ptr || errno || (long)bit_min != strtol(argv[i+2],&ptr,10) )
      {
        printf("ERROR: can't parse parameter <min> for option \"-tf\"\n");
        return 1;
      }
      bit_max=(int)strtol(argv[i+3],&ptr,10);
      if(*ptr || errno || (long)bit_max != strtol(argv[i+3],&ptr,10) )
      {
        printf("ERROR: can't parse parameter <max> for option \"-tf\"\n");
        return 1;
      }
      if(!valid_assignment(exp, bit_min, bit_max))
      {
        return 1;
      }
      use_worktodo = 0;
      parse_ret = 0;
      i += 3;
    }
    else if(!strcmp((char*)"-st", argv[i]))
    {
      mystuff.mode = MODE_SELFTEST_FULL;
    }
    else if(!strcmp((char*)"--timertest", argv[i]))
    {
      timertest();
      return 0;
    }
    else if(!strcmp((char*)"--sleeptest", argv[i]))
    {
      sleeptest();
      return 0;
    }
    i++;
  }

  printf("mfaktc v%s (%dbit built)\n\n", MFAKTC_VERSION, (int)(sizeof(void*)*8));

  if(cudaSetDevice(devicenumber)!=cudaSuccess)
  {
    printf("cudaSetDevice(%d) failed\n",devicenumber);
    print_last_CUDA_error();
    return 1;
  }
  
/* print current configuration */
  printf("Compiletime options\n");
  printf("  THREADS_PER_BLOCK         %d\n", THREADS_PER_BLOCK);
  printf("  SIEVE_SIZE_LIMIT          %dkiB\n", SIEVE_SIZE_LIMIT);
  printf("  SIEVE_SIZE                %dbits\n", SIEVE_SIZE);
  if(SIEVE_SIZE <= 0)
  {
    printf("ERROR: SIEVE_SIZE is <= 0, consider to increase SIEVE_SIZE_LIMIT in params.h\n");
    return 1;
  }
  printf("  SIEVE_SPLIT               %d\n", SIEVE_SPLIT);
  if(SIEVE_SPLIT > SIEVE_PRIMES_MIN)
  {
    printf("ERROR: SIEVE_SPLIT must be <= SIEVE_PRIMES_MIN\n");
    return 1;
  }
#ifdef MORE_CLASSES
  printf("  MORE_CLASSES              enabled\n");
#else
  printf("  MORE_CLASSES              disabled\n");
#endif

#ifdef VERBOSE_TIMING
  printf("  VERBOSE_TIMING            enabled (DEBUG option)\n");
#endif
#ifdef USE_DEVICE_PRINTF
  printf("  USE_DEVICE_PRINTF         enabled (DEBUG option)\n");
#endif
#ifdef CHECKS_MODBASECASE
  printf("  CHECKS_MODBASECASE        enabled (DEBUG option)\n");
#endif
#ifdef DEBUG_STREAM_SCHEDULE
  printf("  DEBUG_STREAM_SCHEDULE     enabled (DEBUG option)\n");
#endif
#ifdef DEBUG_STREAM_SCHEDULE_CHECK
  printf("  DEBUG_STREAM_SCHEDULE_CHECK\n                            enabled (DEBUG option)\n");
#endif
#ifdef RAW_GPU_BENCH
  printf("  RAW_GPU_BENCH             enabled (DEBUG option)\n");
#endif

  read_config(&mystuff);

  cudaGetDeviceProperties(&deviceinfo, devicenumber);
  printf("\nCUDA device info\n");
  printf("  name                      %s\n",deviceinfo.name);
  mystuff.compcapa_major = deviceinfo.major;
  mystuff.compcapa_minor = deviceinfo.minor;
  printf("  compute capability        %d.%d\n",deviceinfo.major,deviceinfo.minor);
  if((mystuff.compcapa_major == 1) && (mystuff.compcapa_minor == 0))
  {
    printf("Sorry, devices with compute capability 1.0 are not supported!\n");
    return 1;
  }
  printf("  maximum threads per block %d\n",deviceinfo.maxThreadsPerBlock);
#if CUDART_VERSION >= 2000
  i=0;
       if(deviceinfo.major == 1)i=8;                            /* devices with compute capability 1.x have 8 shader cores per multiprocessor */
  else if(deviceinfo.major == 2 && deviceinfo.minor == 0)i=32;	/* devices with compute capability 2.0 have 32 shader cores per multiprocessor */
  else if(deviceinfo.major == 2 && deviceinfo.minor == 1)i=48;	/* devices with compute capability 2.1 have 48 shader cores per multiprocessor */
  if(i != 0)printf("  number of multiprocessors %d (%d shader cores)\n", deviceinfo.multiProcessorCount, deviceinfo.multiProcessorCount * i);
  else      printf("  number of mutliprocessors %d (unknown number of shader cores)\n", deviceinfo.multiProcessorCount);
#endif
  printf("  clock rate                %dMHz\n", deviceinfo.clockRate / 1000);
  if(THREADS_PER_BLOCK > deviceinfo.maxThreadsPerBlock)
  {
    printf("\nERROR: THREADS_PER_BLOCK > deviceinfo.maxThreadsPerBlock\n");
    return 1;
  }

#if CUDART_VERSION >= 2020
  int drv_ver, rt_ver;
  printf("\nCUDA version info\n");
  printf("  binary compiled for CUDA  %d.%d\n", CUDART_VERSION/1000, CUDART_VERSION%100);
  cudaDriverGetVersion(&drv_ver);  
  cudaRuntimeGetVersion(&rt_ver);
  printf("  CUDA driver version       %d.%d\n", drv_ver/1000, drv_ver%100);
  printf("  CUDA runtime version      %d.%d\n", rt_ver/1000, rt_ver%100);
  
  if(drv_ver < CUDART_VERSION)
  {
    printf("WARNING: current CUDA driver version is lower than the CUDA toolkit version used during compile\n");
  }
  if(rt_ver < CUDART_VERSION)
  {
    printf("WARNING: current CUDA runtime version is lower than the CUDA toolkit version used during compile\n");
  }
#endif  

  printf("\nAutomatic parameters\n");
#if CUDART_VERSION >= 2000
  i = THREADS_PER_BLOCK * deviceinfo.multiProcessorCount;
  while( (i * 2) <= mystuff.threads_per_grid_max) i = i * 2;
  mystuff.threads_per_grid = i;
#else
  mystuff.threads_per_grid = mystuff.threads_per_grid_max;
#endif
  printf("  threads per grid          %d\n", mystuff.threads_per_grid);
  

  if(mystuff.threads_per_grid % THREADS_PER_BLOCK)
  {
    printf("ERROR: mystuff.threads_per_grid is _NOT_ a multiple of THREADS_PER_BLOCK\n");
    return 1;
  }
  printf("\n");
  
  for(i=0;i<mystuff.num_streams;i++)
  {
    if( cudaStreamCreate(&(mystuff.stream[i])) != cudaSuccess)
    {
      printf("cudaStreamCreate() failed for stream %d\n", i);
      print_last_CUDA_error();
      return 1;
    }
  }
/* Allocate some memory arrays */  
  for(i=0;i<(mystuff.num_streams + mystuff.cpu_streams);i++)
  {
    if( cudaHostAlloc((void**)&(mystuff.h_ktab[i]), mystuff.threads_per_grid * sizeof(int), 0) != cudaSuccess )
    {
      printf("ERROR: cudaHostAlloc(h_ktab[%d]) failed\n", i);
      print_last_CUDA_error();
      return 1;
    }
  }
  for(i=0;i<mystuff.num_streams;i++)
  {
    if( cudaMalloc((void**)&(mystuff.d_ktab[i]), mystuff.threads_per_grid * sizeof(int)) != cudaSuccess )
    {
      printf("ERROR: cudaMalloc(d_ktab1[%d]) failed\n", i);
      print_last_CUDA_error();
      return 1;
    }
  }
  if( cudaHostAlloc((void**)&(mystuff.h_RES),32 * sizeof(int), 0) != cudaSuccess )
  {
    printf("ERROR: cudaHostAlloc(h_RES) failed\n");
    print_last_CUDA_error();
    return 1;
  }
  if( cudaMalloc((void**)&(mystuff.d_RES), 32 * sizeof(int)) != cudaSuccess )
  {
    printf("ERROR: cudaMalloc(d_RES) failed\n");
    print_last_CUDA_error();
    return 1;
  }
#ifdef CHECKS_MODBASECASE
  if( cudaHostAlloc((void**)&(mystuff.h_modbasecase_debug), 32 * sizeof(int), 0) != cudaSuccess )
  {
    printf("ERROR: cudaHostAlloc(h_modbasecase_debug) failed\n");
    print_last_CUDA_error();
    return 1;
  }
  if( cudaMalloc((void**)&(mystuff.d_modbasecase_debug), 32 * sizeof(int)) != cudaSuccess )
  {
    printf("ERROR: cudaMalloc(d_modbasecase_debug) failed\n");
    print_last_CUDA_error();
    return 1;
  }
#endif  
  
#ifdef VERBOSE_TIMING
  timer_init(&timer);
#endif
  sieve_init();
#ifdef VERBOSE_TIMING
  printf("tf(): time spent for sieve_init(): %" PRIu64 "ms\n",timer_diff(&timer)/1000);
#endif

  mystuff.sieve_primes_max = SIEVE_PRIMES_MAX;
  if(mystuff.mode == MODE_NORMAL)
  {

/* before we start real work run a small selftest */  
    mystuff.mode = MODE_SELFTEST_SHORT;
    printf("running a simple selftest...\n");
    if(selftest(&mystuff, 1) != 0)return 1; /* selftest failed :( */
    mystuff.mode = MODE_NORMAL;

    do
    {
      if(use_worktodo)parse_ret = get_next_assignment(mystuff.workfile, &exp, &bit_min, &bit_max);
      if(parse_ret == 0)
      {
        printf("got assignment: exp=%u bit_min=%d bit_max=%d\n",exp,bit_min,bit_max);

        bit_min_stage = bit_min;
        bit_max_stage = bit_max;

        mystuff.sieve_primes_max = sieve_sieve_primes_max(exp);
        if(mystuff.sieve_primes > mystuff.sieve_primes_max)
        {
          mystuff.sieve_primes = mystuff.sieve_primes_max;
          printf("WARNING: SievePrimes is too big for the current assignment, lowering to %u\n", mystuff.sieve_primes_max);
          printf("         It is not allowed to sieve primes which are equal or bigger than the \n");
          printf("         exponent itself!\n");
        }
        if(mystuff.stages == 1)
        {
          while( ((calculate_k(exp,bit_max_stage) - calculate_k(exp,bit_min_stage)) > (250000000ULL * NUM_CLASSES)) && ((bit_max_stage - bit_min_stage) > 1) )bit_max_stage--;
        }
        tmp = 0;
        while(bit_max_stage <= bit_max)
        {
          tmp = tf(exp, bit_min_stage, bit_max_stage, &mystuff, 0, 0, AUTOSELECT_KERNEL);
          if(tmp == 123456) return 1; /* bail out, we might have a serios problem (detected by cudaGetLastError())... */

          if( (mystuff.stopafterfactor > 0) && (tmp > 0) )
          {
            bit_max_stage = bit_max;
          }

          if(use_worktodo)
          {
            if(bit_max_stage == bit_max)parse_ret = clear_assignment(mystuff.workfile, exp, bit_min_stage, bit_max, 0);
            else                        parse_ret = clear_assignment(mystuff.workfile, exp, bit_min_stage, bit_max, bit_max_stage);

                 if(parse_ret == 3) printf("ERROR: clear_assignment() / modify_assignment(): can't open \"%s\"\n", mystuff.workfile);
            else if(parse_ret == 4) printf("ERROR: clear_assignment() / modify_assignment(): can't open \"__worktodo__.tmp\"\n");
            else if(parse_ret == 5) printf("ERROR: clear_assignment() / modify_assignment(): assignment not found in \"%s\"\n", mystuff.workfile);
            else if(parse_ret == 6) printf("ERROR: clear_assignment() / modify_assignment(): can't rename workfiles\n");
            else if(parse_ret != 0) printf("ERROR: clear_assignment() / modify_assignment(): Unknown error (%d)\n", parse_ret);
          }

          bit_min_stage = bit_max_stage;
          bit_max_stage++;
        }
      }
      else if(parse_ret == 1) printf("ERROR: get_next_assignment(): can't open \"%s\"\n", mystuff.workfile);
      else if(parse_ret == 2) printf("ERROR: get_next_assignment(): no valid assignment found in \"%s\"\n", mystuff.workfile);
      else if(parse_ret != 0) printf("ERROR: get_next_assignment(): Unknown error (%d)\n", parse_ret);
    }
    while(parse_ret == 0 && use_worktodo);
  }
  else // mystuff.mode != MODE_NORMAL
  {
    selftest(&mystuff, 0);
  }

  for(i=0;i<mystuff.num_streams;i++)
  {
    cudaStreamDestroy(mystuff.stream[i]);
  }
#ifdef CHECKS_MODBASECASE
  cudaFree(mystuff.d_modbasecase_debug);
  cudaFree(mystuff.h_modbasecase_debug);
#endif  
  cudaFree(mystuff.d_RES);
  cudaFree(mystuff.h_RES);
  for(i=0;i<(mystuff.num_streams + mystuff.cpu_streams);i++)cudaFreeHost(mystuff.h_ktab[i]);
  for(i=0;i<mystuff.num_streams;i++)cudaFree(mystuff.d_ktab[i]);
  sieve_free();

  return 0;
}
