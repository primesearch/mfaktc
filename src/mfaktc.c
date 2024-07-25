/*
This file is part of mfaktc.
Copyright (C) 2009, 2010, 2011, 2012, 2013, 2014, 2015  Oliver Weihe (o.weihe@t-online.de)

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
#include <time.h>

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
#include "signal_handler.h"
#include "output.h"
#include "gpusieve.h"

unsigned long long int calculate_k(unsigned int exp, int bits)
/* calculates biggest possible k in "2 * exp * k + 1 < 2^bits" */
{
  unsigned long long int k = 0, tmp_low, tmp_hi;
  
  if((bits > 65) && exp < (1U << (bits - 65))) k = 0; // k would be >= 2^64...
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


int kernel_possible(int kernel, mystuff_t *mystuff)
/* returns 1 if the selected kernel can handle the assignment, 0 otherwise
The variables mystuff->exponent, mystuff->bit_min and mystuff->bit_max_stage
must be set to a valid assignment prior call of this function!
Because all currently available kernels can handle the full supported range
of exponents this isn't used here for now. */
{
  int ret = 0;

  if( kernel == _71BIT_MUL24                                                                                                    && mystuff->bit_max_stage <= 71) ret = 1;

  if((kernel == _75BIT_MUL32    || (mystuff->gpu_sieving && mystuff->exponent >= mystuff->gpu_sieve_min_exp && kernel == _75BIT_MUL32_GS))    && mystuff->bit_max_stage <= 75) ret = 1;
  if((kernel == _95BIT_MUL32    || (mystuff->gpu_sieving && mystuff->exponent >= mystuff->gpu_sieve_min_exp && kernel == _95BIT_MUL32_GS))    && mystuff->bit_max_stage <= 95) ret = 1;

  if((kernel == BARRETT76_MUL32 || (mystuff->gpu_sieving && mystuff->exponent >= mystuff->gpu_sieve_min_exp && kernel == BARRETT76_MUL32_GS)) && mystuff->bit_min >= 64 && mystuff->bit_max_stage <= 76) ret = 1;
  if((kernel == BARRETT77_MUL32 || (mystuff->gpu_sieving && mystuff->exponent >= mystuff->gpu_sieve_min_exp && kernel == BARRETT77_MUL32_GS)) && mystuff->bit_min >= 64 && mystuff->bit_max_stage <= 77) ret = 1;
  if((kernel == BARRETT79_MUL32 || (mystuff->gpu_sieving && mystuff->exponent >= mystuff->gpu_sieve_min_exp && kernel == BARRETT79_MUL32_GS)) && mystuff->bit_min >= 64 && mystuff->bit_max_stage <= 79) ret = 1;
  if((kernel == BARRETT87_MUL32 || (mystuff->gpu_sieving && mystuff->exponent >= mystuff->gpu_sieve_min_exp && kernel == BARRETT87_MUL32_GS)) && mystuff->bit_min >= 65 && mystuff->bit_max_stage <= 87 && (mystuff->bit_max_stage - mystuff->bit_min) == 1) ret = 1;
  if((kernel == BARRETT88_MUL32 || (mystuff->gpu_sieving && mystuff->exponent >= mystuff->gpu_sieve_min_exp && kernel == BARRETT88_MUL32_GS)) && mystuff->bit_min >= 65 && mystuff->bit_max_stage <= 88 && (mystuff->bit_max_stage - mystuff->bit_min) == 1) ret = 1;
  if((kernel == BARRETT92_MUL32 || (mystuff->gpu_sieving && mystuff->exponent >= mystuff->gpu_sieve_min_exp && kernel == BARRETT92_MUL32_GS)) && mystuff->bit_min >= 65 && mystuff->bit_max_stage <= 92 && (mystuff->bit_max_stage - mystuff->bit_min) == 1) ret = 1;

  return ret;
}


int class_needed(unsigned int exp, unsigned long long int k_min, int c)
{
/*
checks whether the class c must be processed or can be ignored at all because
all factor candidates within the class c are a multiple of 3, 5, 7 or 11 (11
only if MORE_CLASSES is definied) or are 3 or 5 mod 8 (Mersenne) or are 5 or 7 mod 8 (Wagstaff)

k_min *MUST* be aligned in that way that k_min is in class 0!
*/
#ifdef WAGSTAFF
  if  ((2 * (exp %  8) * ((k_min + c) %  8)) %  8 !=  6)
#else /* Mersennes */
  if  ((2 * (exp %  8) * ((k_min + c) %  8)) %  8 !=  2)
#endif
  if( ((2 * (exp %  8) * ((k_min + c) %  8)) %  8 !=  4) && \
      ((2 * (exp %  3) * ((k_min + c) %  3)) %  3 !=  2) && \
      ((2 * (exp %  5) * ((k_min + c) %  5)) %  5 !=  4) && \
      ((2 * (exp %  7) * ((k_min + c) %  7)) %  7 !=  6))
#ifdef MORE_CLASSES        
  if  ((2 * (exp % 11) * ((k_min + c) % 11)) % 11 != 10 )
#endif
  {
    return 1;
  }

  return 0;
}


void close_log(mystuff_t *mystuff)
{
    if (mystuff->logfileptr != NULL)
    {
        fclose(mystuff->logfileptr);
        mystuff->logfileptr = NULL;
    }
}


int tf(mystuff_t *mystuff, int class_hint, unsigned long long int k_hint, int kernel)
/*
tf M<mystuff->exponent> from 2^<mystuff->bit_min> to 2^<mystuff->mystuff->bit_max_stage>

kernel: see my_types.h -> enum GPUKernels

return value (mystuff->mode = MODE_NORMAL):
number of factors found
RET_CUDA_ERROR cudaGetLastError() returned an error
RET_QUIT if early exit was requested by SIGINT

return value (mystuff->mode = MODE_SELFTEST_SHORT or MODE_SELFTEST_FULL):
0 for a successfull selftest (known factor was found)
1 no factor found
2 wrong factor returned
RET_CUDA_ERROR cudaGetLastError() returned an error

other return value 
-1 unknown mode
*/
{
  int cur_class, max_class = NUM_CLASSES-1, i;
  unsigned long long int k_min, k_max, k_range, tmp;
  unsigned int f_hi, f_med, f_low;
  struct timeval timer, timer_last_checkpoint;
  static struct timeval timer_last_addfilecheck;
  int factorsfound = 0, numfactors = 0, restart = 0;

  int retval = 0;
  
  cudaError_t cudaError;
  
  unsigned long long int time_run, time_est;
  
  mystuff->stats.output_counter = 0; /* reset output counter, needed for status headline */
  mystuff->stats.ghzdays = primenet_ghzdays(mystuff->exponent, mystuff->bit_min, mystuff->bit_max_stage);

  if(mystuff->mode != MODE_SELFTEST_SHORT)logprintf(mystuff, "Starting trial factoring %s%u from 2^%d to 2^%d (%.2f GHz-days)\n", NAME_NUMBERS, mystuff->exponent, mystuff->bit_min, mystuff->bit_max_stage, mystuff->stats.ghzdays);
  if((mystuff->mode != MODE_NORMAL) && (mystuff->mode != MODE_SELFTEST_SHORT) && (mystuff->mode != MODE_SELFTEST_FULL))
  {
    logprintf(mystuff, "ERROR, invalid mode for tf(): %d\n", mystuff->mode);
    return -1;
  }
  timer_init(&timer);
  timer_init(&timer_last_checkpoint);
  if(mystuff->addfilestatus == -1)
  {
    mystuff->addfilestatus = 0;
    timer_init(&timer_last_addfilecheck);
  }
  
  mystuff->stats.class_counter = 0;
  mystuff->stats.bit_level_time = 0;
  
  k_min=calculate_k(mystuff->exponent, mystuff->bit_min);
  k_max=calculate_k(mystuff->exponent, mystuff->bit_max_stage);

  if((mystuff->mode == MODE_SELFTEST_FULL) || (mystuff->mode == MODE_SELFTEST_SHORT))
  {
/* a shortcut for the selftest, bring k_min a k_max "close" to the known factor
   0 <= mystuff->selftestrandomoffset < 25000000, thus k_range must be greater than 25000000 */
    if(NUM_CLASSES == 420)k_range = 50000000ULL; 
    else                  k_range = 500000000ULL;

/* greatly increased k_range for the -st2 selftest */    
    if(mystuff->selftestsize == 2) k_range*=100;

    tmp = k_hint - (k_hint % k_range) - (2ULL * k_range) - mystuff->selftestrandomoffset;
    if((tmp <= k_hint) && (tmp > k_min)) k_min = tmp; /* check for tmp <= k_hint prevents integer underflow (k_hint < ( k_range + mystuff->selftestrandomoffset) */
    
    tmp += 4ULL * k_range;
    if((tmp >= k_hint) && ((tmp < k_max) || (k_max < k_min))) k_max = tmp; /* check for k_max < k_min enables some selftests where k_max >= 2^64 but the known factor itself has a k < 2^64 */
  }

  k_min -= k_min % NUM_CLASSES;	/* k_min is now 0 mod NUM_CLASSES */

  if(mystuff->mode != MODE_SELFTEST_SHORT && (mystuff->verbosity >= 2 || (mystuff->mode == MODE_NORMAL && mystuff->verbosity >= 1)))
  {
    logprintf(mystuff, " k_min =  %" PRIu64 "\n", k_min);
    if(k_hint > 0)logprintf(mystuff, " k_hint = %" PRIu64 "\n", k_hint);
    logprintf(mystuff, " k_max =  %" PRIu64 "\n", k_max);
  }

  if(kernel == AUTOSELECT_KERNEL)
  {
/* select the GPU kernel (fastest GPU kernel has highest priority)
see benchmarks in src/kernel_benchmarks.txt */
    if(mystuff->compcapa_major == 1)
    {
           if(kernel_possible(BARRETT76_MUL32_GS, mystuff)) kernel = BARRETT76_MUL32_GS;
      else if(kernel_possible(BARRETT77_MUL32_GS, mystuff)) kernel = BARRETT77_MUL32_GS;
      else if(kernel_possible(BARRETT87_MUL32_GS, mystuff)) kernel = BARRETT87_MUL32_GS;
      else if(kernel_possible(BARRETT88_MUL32_GS, mystuff)) kernel = BARRETT88_MUL32_GS;
      else if(kernel_possible(BARRETT79_MUL32_GS, mystuff)) kernel = BARRETT79_MUL32_GS;
      else if(kernel_possible(BARRETT92_MUL32_GS, mystuff)) kernel = BARRETT92_MUL32_GS;
      else if(kernel_possible(_75BIT_MUL32_GS,    mystuff)) kernel = _75BIT_MUL32_GS;
      else if(kernel_possible(_95BIT_MUL32_GS,    mystuff)) kernel = _95BIT_MUL32_GS;

      else if(kernel_possible(BARRETT76_MUL32,    mystuff)) kernel = BARRETT76_MUL32;
      else if(kernel_possible(BARRETT77_MUL32,    mystuff)) kernel = BARRETT77_MUL32;
      else if(kernel_possible(_71BIT_MUL24,       mystuff)) kernel = _71BIT_MUL24;
      else if(kernel_possible(BARRETT87_MUL32,    mystuff)) kernel = BARRETT87_MUL32;
      else if(kernel_possible(BARRETT88_MUL32,    mystuff)) kernel = BARRETT88_MUL32;
      else if(kernel_possible(BARRETT79_MUL32,    mystuff)) kernel = BARRETT79_MUL32;
      else if(kernel_possible(BARRETT92_MUL32,    mystuff)) kernel = BARRETT92_MUL32;
      else if(kernel_possible(_75BIT_MUL32,       mystuff)) kernel = _75BIT_MUL32;
      else if(kernel_possible(_95BIT_MUL32,       mystuff)) kernel = _95BIT_MUL32;
    }
    else // mystuff->compcapa_major != 1
    {
           if(kernel_possible(BARRETT76_MUL32_GS, mystuff)) kernel = BARRETT76_MUL32_GS;
      else if(kernel_possible(BARRETT87_MUL32_GS, mystuff)) kernel = BARRETT87_MUL32_GS;
      else if(kernel_possible(BARRETT88_MUL32_GS, mystuff)) kernel = BARRETT88_MUL32_GS;
      else if(kernel_possible(BARRETT77_MUL32_GS, mystuff)) kernel = BARRETT77_MUL32_GS;
      else if(kernel_possible(BARRETT79_MUL32_GS, mystuff)) kernel = BARRETT79_MUL32_GS;
      else if(kernel_possible(BARRETT92_MUL32_GS, mystuff)) kernel = BARRETT92_MUL32_GS;
      else if(kernel_possible(_75BIT_MUL32_GS,    mystuff)) kernel = _75BIT_MUL32_GS;
      else if(kernel_possible(_95BIT_MUL32_GS,    mystuff)) kernel = _95BIT_MUL32_GS;

      else if(kernel_possible(BARRETT76_MUL32,    mystuff)) kernel = BARRETT76_MUL32;
      else if(kernel_possible(BARRETT87_MUL32,    mystuff)) kernel = BARRETT87_MUL32;
      else if(kernel_possible(BARRETT88_MUL32,    mystuff)) kernel = BARRETT88_MUL32;
      else if(kernel_possible(BARRETT77_MUL32,    mystuff)) kernel = BARRETT77_MUL32;
      else if(kernel_possible(BARRETT79_MUL32,    mystuff)) kernel = BARRETT79_MUL32;
      else if(kernel_possible(BARRETT92_MUL32,    mystuff)) kernel = BARRETT92_MUL32;
      else if(kernel_possible(_75BIT_MUL32,       mystuff)) kernel = _75BIT_MUL32;
      else if(kernel_possible(_95BIT_MUL32,       mystuff)) kernel = _95BIT_MUL32;
    }
  }

       if(kernel == _71BIT_MUL24)       sprintf(mystuff->stats.kernelname, "71bit_mul24");
  else if(kernel == _75BIT_MUL32)       sprintf(mystuff->stats.kernelname, "75bit_mul32");
  else if(kernel == _95BIT_MUL32)       sprintf(mystuff->stats.kernelname, "95bit_mul32");

  else if(kernel == _75BIT_MUL32_GS)    sprintf(mystuff->stats.kernelname, "75bit_mul32_gs");
  else if(kernel == _95BIT_MUL32_GS)    sprintf(mystuff->stats.kernelname, "95bit_mul32_gs");

  else if(kernel == BARRETT76_MUL32)    sprintf(mystuff->stats.kernelname, "barrett76_mul32");
  else if(kernel == BARRETT77_MUL32)    sprintf(mystuff->stats.kernelname, "barrett77_mul32");
  else if(kernel == BARRETT79_MUL32)    sprintf(mystuff->stats.kernelname, "barrett79_mul32");
  else if(kernel == BARRETT87_MUL32)    sprintf(mystuff->stats.kernelname, "barrett87_mul32");
  else if(kernel == BARRETT88_MUL32)    sprintf(mystuff->stats.kernelname, "barrett88_mul32");
  else if(kernel == BARRETT92_MUL32)    sprintf(mystuff->stats.kernelname, "barrett92_mul32");

  else if(kernel == BARRETT76_MUL32_GS) sprintf(mystuff->stats.kernelname, "barrett76_mul32_gs");
  else if(kernel == BARRETT77_MUL32_GS) sprintf(mystuff->stats.kernelname, "barrett77_mul32_gs");
  else if(kernel == BARRETT79_MUL32_GS) sprintf(mystuff->stats.kernelname, "barrett79_mul32_gs");
  else if(kernel == BARRETT87_MUL32_GS) sprintf(mystuff->stats.kernelname, "barrett87_mul32_gs");
  else if(kernel == BARRETT88_MUL32_GS) sprintf(mystuff->stats.kernelname, "barrett88_mul32_gs");
  else if(kernel == BARRETT92_MUL32_GS) sprintf(mystuff->stats.kernelname, "barrett92_mul32_gs");
  
  else                                  sprintf(mystuff->stats.kernelname, "UNKNOWN kernel");

  if(mystuff->mode != MODE_SELFTEST_SHORT && mystuff->verbosity >= 1)logprintf(mystuff, "Using GPU kernel \"%s\"\n", mystuff->stats.kernelname);

  if(mystuff->mode == MODE_NORMAL)
  {
    if((mystuff->checkpoints == 1) && (checkpoint_read(mystuff->exponent, mystuff->bit_min, mystuff->bit_max_stage, &cur_class, &factorsfound, mystuff->factors_string, &(mystuff->stats.bit_level_time)) == 1))
    {
      logprintf(mystuff, "\nfound a valid checkpoint file!\n");
      if(mystuff->verbosity >= 1)logprintf(mystuff, "  last finished class was: %d\n", cur_class);
      if(mystuff->verbosity >= 1)logprintf(mystuff, "  found %d factor(s) already%s%s\n", factorsfound, factorsfound > 0 ? ": " : "", mystuff->factors_string);
      if(mystuff->verbosity >= 1)logprintf(mystuff, "  previous work took %llu ms\n\n", mystuff->stats.bit_level_time);
      else                       logprintf(mystuff, "\n");
      cur_class++; // the checkpoint contains the last complete processed class!

/* calculate the number of classes which are allready processed. This value is needed to estimate ETA */
      for(i = 0; i < cur_class; i++)
      {
        if(class_needed(mystuff->exponent, k_min, i))mystuff->stats.class_counter++;
      }
      restart = mystuff->stats.class_counter;
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
    if(class_needed(mystuff->exponent, k_min, cur_class))
    {
      mystuff->stats.class_number = cur_class;
      if(mystuff->quit)
      {
/* check if quit is requested. Because this is at the begining of the class
   we can be sure that if RET_QUIT is returned the last class hasn't
   finished. The signal handler which sets mystuff->quit not active during
   selftests so we need to check for RET_QUIT only when doing real work. */
        if(mystuff->printmode == 1)logprintf(mystuff, "\n");
        return RET_QUIT;
      }
      else
      {
	if(kernel != _75BIT_MUL32_GS &&
	   kernel != _95BIT_MUL32_GS &&
	   kernel != BARRETT76_MUL32_GS &&
	   kernel != BARRETT77_MUL32_GS &&
	   kernel != BARRETT79_MUL32_GS &&
	   kernel != BARRETT87_MUL32_GS &&
	   kernel != BARRETT88_MUL32_GS &&
	   kernel != BARRETT92_MUL32_GS)
	{
	  sieve_init_class(mystuff->exponent, k_min+cur_class, mystuff->sieve_primes);
	}
        mystuff->stats.class_counter++;
      
             if(kernel == _71BIT_MUL24)       numfactors = tf_class_71          (k_min+cur_class, k_max, mystuff);
        else if(kernel == _75BIT_MUL32)       numfactors = tf_class_75          (k_min+cur_class, k_max, mystuff);
        else if(kernel == _95BIT_MUL32)       numfactors = tf_class_95          (k_min+cur_class, k_max, mystuff);

        else if(kernel == _75BIT_MUL32_GS)    numfactors = tf_class_75_gs       (k_min+cur_class, k_max, mystuff);
        else if(kernel == _95BIT_MUL32_GS)    numfactors = tf_class_95_gs       (k_min+cur_class, k_max, mystuff);

        else if(kernel == BARRETT76_MUL32)    numfactors = tf_class_barrett76   (k_min+cur_class, k_max, mystuff);
        else if(kernel == BARRETT77_MUL32)    numfactors = tf_class_barrett77   (k_min+cur_class, k_max, mystuff);
        else if(kernel == BARRETT79_MUL32)    numfactors = tf_class_barrett79   (k_min+cur_class, k_max, mystuff);
        else if(kernel == BARRETT87_MUL32)    numfactors = tf_class_barrett87   (k_min+cur_class, k_max, mystuff);
        else if(kernel == BARRETT88_MUL32)    numfactors = tf_class_barrett88   (k_min+cur_class, k_max, mystuff);
        else if(kernel == BARRETT92_MUL32)    numfactors = tf_class_barrett92   (k_min+cur_class, k_max, mystuff);

	else if(kernel == BARRETT76_MUL32_GS) numfactors = tf_class_barrett76_gs(k_min+cur_class, k_max, mystuff);
	else if(kernel == BARRETT77_MUL32_GS) numfactors = tf_class_barrett77_gs(k_min+cur_class, k_max, mystuff);
	else if(kernel == BARRETT79_MUL32_GS) numfactors = tf_class_barrett79_gs(k_min+cur_class, k_max, mystuff);
	else if(kernel == BARRETT87_MUL32_GS) numfactors = tf_class_barrett87_gs(k_min+cur_class, k_max, mystuff);
	else if(kernel == BARRETT88_MUL32_GS) numfactors = tf_class_barrett88_gs(k_min+cur_class, k_max, mystuff);
	else if(kernel == BARRETT92_MUL32_GS) numfactors = tf_class_barrett92_gs(k_min+cur_class, k_max, mystuff);

        else
        {
          logprintf(mystuff, "ERROR: Unknown kernel selected (%d)!\n", kernel);
          return RET_CUDA_ERROR;
        }
        cudaError = cudaGetLastError();
        if(cudaError != cudaSuccess)
        {
          logprintf(mystuff, "ERROR: cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
          return RET_CUDA_ERROR; /* bail out, we might have a serios problem (detected by cudaGetLastError())... */
        }
        factorsfound += numfactors;
        if(mystuff->mode == MODE_NORMAL)
        {
          if(mystuff->checkpoints == 1)
          {
            if (numfactors > 0)
            {
              char factorstring[50];
              int96 factor;
              for (i = 0; (i < numfactors) && (i < 10); i++)
              {
                factor.d2 = mystuff->h_RES[i * 3 + 1];
                factor.d1 = mystuff->h_RES[i * 3 + 2];
                factor.d0 = mystuff->h_RES[i * 3 + 3];
                print_dez96(factor, factorstring);
                sprintf(mystuff->factors_string, mystuff->factors_string[0] ? "%s,\"%s\"" : "%s\"%s\"", mystuff->factors_string, factorstring);
              }
            }
            if (numfactors > 0 || timer_diff(&timer_last_checkpoint) / 1000000 >= (unsigned long long int)mystuff->checkpointdelay || mystuff->quit)
            {
                timer_init(&timer_last_checkpoint);
                checkpoint_write(mystuff->exponent, mystuff->bit_min, mystuff->bit_max_stage, cur_class, factorsfound, mystuff->factors_string, mystuff->stats.bit_level_time);
            }
          }
          if((mystuff->addfiledelay > 0) && timer_diff(&timer_last_addfilecheck) / 1000000 >= (unsigned long long int)mystuff->addfiledelay)
          {
            timer_init(&timer_last_addfilecheck);
            if(process_add_file(mystuff->workfile, mystuff->addfile, &(mystuff->addfilestatus), mystuff->verbosity) != OK)
            {
              mystuff->addfiledelay = 0; /* disable for until exit at least... */
            }
          }
          if((mystuff->stopafterfactor >= 2) && (factorsfound > 0) && (cur_class != max_class))cur_class = max_class + 1;
        }
      }
      fflush(NULL);
    }
  }
  if(mystuff->mode != MODE_SELFTEST_SHORT && mystuff->printmode == 1)logprintf(mystuff, "\n");
  print_result_line(mystuff, factorsfound);

  if(mystuff->mode == MODE_NORMAL)
  {
    retval = factorsfound;
    if(mystuff->checkpoints == 1)checkpoint_delete(mystuff->exponent);
  }
  else // mystuff->mode != MODE_NORMAL
  {
    if(mystuff->h_RES[0] == 0)
    {
      logprintf(mystuff, "ERROR: selftest failed for %s%u\n", NAME_NUMBERS, mystuff->exponent);
      logprintf(mystuff, "  no factor found\n");
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
      
      k_max   = (unsigned long long int)mystuff->exponent * f_low;
      f_low   = (k_max & 0xFFFFFFFFULL) + 1;
      k_min   = (k_max >> 32);

      k_max   = (unsigned long long int)mystuff->exponent * f_med;
      k_min  += k_max & 0xFFFFFFFFULL;
      f_med   = k_min & 0xFFFFFFFFULL;
      k_min >>= 32;
      k_min  += (k_max >> 32);

      f_hi  = k_min + (mystuff->exponent * f_hi); /* f_{hi|med|low} = 2 * k_hint * mystuff->exponent +1 */
      
      if(kernel == _71BIT_MUL24) /* 71bit kernel uses only 24bit per int */
      {
        f_hi  <<= 16;
        f_hi   += f_med >> 16;

        f_med <<= 8;
        f_med  += f_low >> 24;
        f_med  &= 0x00FFFFFF;
        
        f_low  &= 0x00FFFFFF;
      }
      k_min=0; /* using k_min for counting number of matches here */
      for(i=0; ((unsigned int)i < mystuff->h_RES[0]) && (i < 10); i++)
      {
        if(mystuff->h_RES[i*3 + 1] == f_hi  && \
           mystuff->h_RES[i*3 + 2] == f_med && \
           mystuff->h_RES[i*3 + 3] == f_low) k_min++;
      }
      if(k_min != 1) /* the factor should appear ONCE */
      {
        logprintf(mystuff, "ERROR: selftest failed for %s%u!\n", NAME_NUMBERS, mystuff->exponent);
        logprintf(mystuff, "  expected result: %08X %08X %08X\n", f_hi, f_med, f_low);
        for(i=0; ((unsigned int)i < mystuff->h_RES[0]) && (i < 10); i++)
        {
          logprintf(mystuff, "  reported result: %08X %08X %08X\n", mystuff->h_RES[i*3 + 1], mystuff->h_RES[i*3 + 2], mystuff->h_RES[i*3 + 3]);
        }
        retval = 2;
      }
      else
      {
        if(mystuff->mode != MODE_SELFTEST_SHORT)logprintf(mystuff, "selftest for %s%u passed!\n", NAME_NUMBERS, mystuff->exponent);
      }
    }
  }
  if(mystuff->mode != MODE_SELFTEST_SHORT)
  {
    time_run = timer_diff(&timer)/1000;
    
    if(restart == 0)logprintf(mystuff, "tf(): total time spent: ");
    else            logprintf(mystuff, "tf(): time spent since restart:   ");

/*  restart == 0 ==> time_est = time_run */
#ifndef MORE_CLASSES
    time_est = (time_run * 96ULL  ) / (unsigned long long int)(96 -restart);
#else
    time_est = (time_run * 960ULL ) / (unsigned long long int)(960-restart);
#endif

    if(time_est > 86400000ULL)logprintf(mystuff, "%" PRIu64 "d ",   time_run / 86400000ULL);
    if(time_est > 3600000ULL) logprintf(mystuff, "%2" PRIu64 "h ", (time_run /  3600000ULL) % 24ULL);
    if(time_est > 60000ULL)   logprintf(mystuff, "%2" PRIu64 "m ", (time_run /    60000ULL) % 60ULL);
                              logprintf(mystuff, "%2" PRIu64 ".%03" PRIu64 "s\n", (time_run / 1000ULL) % 60ULL, time_run % 1000ULL);
    if(restart != 0)
    {
      logprintf(mystuff, "      estimated total time spent: ");
      if(time_est > 86400000ULL)logprintf(mystuff, "%" PRIu64 "d ",   time_est / 86400000ULL);
      if(time_est > 3600000ULL) logprintf(mystuff, "%2" PRIu64 "h ", (time_est /  3600000ULL) % 24ULL);
      if(time_est > 60000ULL)   logprintf(mystuff, "%2" PRIu64 "m ", (time_est /    60000ULL) % 60ULL);
                                logprintf(mystuff, "%2" PRIu64 ".%03" PRIu64 "s\n", (time_est / 1000ULL) % 60ULL, time_est % 1000ULL);
    }
    logprintf(mystuff, "\n");
  }
  return retval;
}


int selftest(mystuff_t *mystuff, int type)
/*
type = 0: full selftest (1557 testcases)
type = 1: full selftest (all testcases)
type = 1: small selftest (this is executed EACH time mfaktc is started)

return value
0 selftest passed
1 selftest failed
RET_CUDA_ERROR we might have a serios problem (detected by cudaGetLastError())
*/
{
  int i, j, tf_res, st_success=0, st_nofactor=0, st_wrongfactor=0, st_unknown=0;

#ifdef WAGSTAFF
  #define NUM_SELFTESTS 1591
#else /* Mersennes */
  #define NUM_SELFTESTS 2867
#endif
  unsigned int exp[NUM_SELFTESTS], index[9];
  int num_selftests=0;
  int bit_min[NUM_SELFTESTS], f_class;
  unsigned long long int k[NUM_SELFTESTS];
  int retval=1;

  #define NUM_KERNEL 17
  int kernels[NUM_KERNEL+1]; // currently there are <NUM_KERNEL> different kernels, kernel numbers start at 1!
  int kernel_success[NUM_KERNEL+1], kernel_fail[NUM_KERNEL+1];

#ifdef WAGSTAFF
  #include "selftest-data-wagstaff.c"  
#else /* Mersennes */
  #include "selftest-data-mersenne.c"  
#endif

  for(i = 0; i <= NUM_KERNEL; i++)
  {
    kernel_success[i] = 0;
    kernel_fail[i] = 0;
  }

  if(type == 0)
  {
    for(i = 0; i < NUM_SELFTESTS; i++)
    {
      logprintf(mystuff, "########## testcase %d/%d ##########\n", i+1, NUM_SELFTESTS);
      f_class = (int)(k[i] % NUM_CLASSES);

      mystuff->exponent           = exp[i];
      mystuff->bit_min            = bit_min[i];
      mystuff->bit_max_assignment = bit_min[i] + 1;
      mystuff->bit_max_stage      = mystuff->bit_max_assignment;

/* create a list which kernels can handle this testcase */
      j = 0;
      if(kernel_possible(BARRETT92_MUL32,    mystuff)) kernels[j++] = BARRETT92_MUL32;
      if(kernel_possible(BARRETT88_MUL32,    mystuff)) kernels[j++] = BARRETT88_MUL32;
      if(kernel_possible(BARRETT87_MUL32,    mystuff)) kernels[j++] = BARRETT87_MUL32;
      if(kernel_possible(BARRETT79_MUL32,    mystuff)) kernels[j++] = BARRETT79_MUL32;
      if(kernel_possible(BARRETT77_MUL32,    mystuff)) kernels[j++] = BARRETT77_MUL32;
      if(kernel_possible(BARRETT76_MUL32,    mystuff)) kernels[j++] = BARRETT76_MUL32;
      if(kernel_possible(BARRETT92_MUL32_GS, mystuff)) kernels[j++] = BARRETT92_MUL32_GS;
      if(kernel_possible(BARRETT88_MUL32_GS, mystuff)) kernels[j++] = BARRETT88_MUL32_GS;
      if(kernel_possible(BARRETT87_MUL32_GS, mystuff)) kernels[j++] = BARRETT87_MUL32_GS;
      if(kernel_possible(BARRETT79_MUL32_GS, mystuff)) kernels[j++] = BARRETT79_MUL32_GS;
      if(kernel_possible(BARRETT77_MUL32_GS, mystuff)) kernels[j++] = BARRETT77_MUL32_GS;
      if(kernel_possible(BARRETT76_MUL32_GS, mystuff)) kernels[j++] = BARRETT76_MUL32_GS;
      if(kernel_possible(_95BIT_MUL32,       mystuff)) kernels[j++] = _95BIT_MUL32;
      if(kernel_possible(_75BIT_MUL32,       mystuff)) kernels[j++] = _75BIT_MUL32;
      if(kernel_possible(_71BIT_MUL24,       mystuff)) kernels[j++] = _71BIT_MUL24;
      if(kernel_possible(_95BIT_MUL32_GS,    mystuff)) kernels[j++] = _95BIT_MUL32_GS;
      if(kernel_possible(_75BIT_MUL32_GS,    mystuff)) kernels[j++] = _75BIT_MUL32_GS;

      do
      {
        num_selftests++;
        tf_res=tf(mystuff, f_class, k[i], kernels[--j]);
             if(tf_res == 0)st_success++;
        else if(tf_res == 1)st_nofactor++;
        else if(tf_res == 2)st_wrongfactor++;
        else if(tf_res == RET_CUDA_ERROR)return RET_CUDA_ERROR; /* bail out, we might have a serios problem (detected by cudaGetLastError())... */
        else           st_unknown++;
        
        if(tf_res == 0)kernel_success[kernels[j]]++;
        else           kernel_fail[kernels[j]]++;
      }
      while(j>0);
    }
  }
  else if(type == 1)
  {
#ifdef WAGSTAFF
    index[0]=  26; index[1]=1000; index[2]=1078; /* some factors below 2^71 (test the 71/75 bit kernel depending on compute capability) */
    index[3]=1290; index[4]=1291; index[5]=1292; /* some factors below 2^75 (test 75 bit kernel) */
    index[6]=1566; index[7]=1577; index[8]=1588; /* some factors below 2^95 (test 95 bit kernel) */
#else /* Mersennes */
    index[0]=   2; index[1]=  25; index[2]=  57; /* some factors below 2^71 (test the 71/75 bit kernel depending on compute capability) */
    index[3]=  70; index[4]=  88; index[5]= 106; /* some factors below 2^75 (test 75 bit kernel) */
    index[6]=1547; index[7]=1552; index[8]=1556; /* some factors below 2^95 (test 95 bit kernel) */
#endif
        
    for(i = 0; i < 9; i++)
    {
      f_class = (int)(k[index[i]] % NUM_CLASSES);
      
      mystuff->exponent           = exp[index[i]];
      mystuff->bit_min            = bit_min[index[i]];
      mystuff->bit_max_assignment = bit_min[index[i]] + 1;
      mystuff->bit_max_stage      = mystuff->bit_max_assignment;

      j = 0;
      if(kernel_possible(BARRETT92_MUL32,    mystuff)) kernels[j++] = BARRETT92_MUL32;
      if(kernel_possible(BARRETT88_MUL32,    mystuff)) kernels[j++] = BARRETT88_MUL32;
      if(kernel_possible(BARRETT87_MUL32,    mystuff)) kernels[j++] = BARRETT87_MUL32;
      if(kernel_possible(BARRETT79_MUL32,    mystuff)) kernels[j++] = BARRETT79_MUL32;
      if(kernel_possible(BARRETT77_MUL32,    mystuff)) kernels[j++] = BARRETT77_MUL32;
      if(kernel_possible(BARRETT76_MUL32,    mystuff)) kernels[j++] = BARRETT76_MUL32;
      if(kernel_possible(BARRETT92_MUL32_GS, mystuff)) kernels[j++] = BARRETT92_MUL32_GS;
      if(kernel_possible(BARRETT88_MUL32_GS, mystuff)) kernels[j++] = BARRETT88_MUL32_GS;
      if(kernel_possible(BARRETT87_MUL32_GS, mystuff)) kernels[j++] = BARRETT87_MUL32_GS;
      if(kernel_possible(BARRETT79_MUL32_GS, mystuff)) kernels[j++] = BARRETT79_MUL32_GS;
      if(kernel_possible(BARRETT77_MUL32_GS, mystuff)) kernels[j++] = BARRETT77_MUL32_GS;
      if(kernel_possible(BARRETT76_MUL32_GS, mystuff)) kernels[j++] = BARRETT76_MUL32_GS;
      if(kernel_possible(_95BIT_MUL32,       mystuff)) kernels[j++] = _95BIT_MUL32;
      if(kernel_possible(_75BIT_MUL32,       mystuff)) kernels[j++] = _75BIT_MUL32;
      if(kernel_possible(_71BIT_MUL24,       mystuff)) kernels[j++] = _71BIT_MUL24;
      if(kernel_possible(_95BIT_MUL32_GS,    mystuff)) kernels[j++] = _95BIT_MUL32_GS;
      if(kernel_possible(_75BIT_MUL32_GS,    mystuff)) kernels[j++] = _75BIT_MUL32_GS;

      do
      {
        num_selftests++;
        tf_res=tf(mystuff, f_class, k[index[i]], kernels[--j]);
             if(tf_res == 0)st_success++;
        else if(tf_res == 1)st_nofactor++;
        else if(tf_res == 2)st_wrongfactor++;
        else if(tf_res == RET_CUDA_ERROR)return RET_CUDA_ERROR; /* bail out, we might have a serios problem (detected by cudaGetLastError())... */
        else           st_unknown++;
      }
      while(j>0);
    }
  }

  logprintf(mystuff, "Selftest statistics\n");
  logprintf(mystuff, "  number of tests           %d\n", num_selftests);
  logprintf(mystuff, "  successfull tests         %d\n", st_success);
  if(st_nofactor > 0)   logprintf(mystuff, "  no factor found           %d\n", st_nofactor);
  if(st_wrongfactor > 0)logprintf(mystuff, "  wrong factor reported     %d\n", st_wrongfactor);
  if(st_unknown > 0)    logprintf(mystuff, "  unknown return value      %d\n", st_unknown);
  if(type == 0)
  {
    logprintf(mystuff, "\n");
    logprintf(mystuff, "  kernel             | success |   fail\n");
    logprintf(mystuff, "  -------------------+---------+-------\n");
    for(i = 0; i <= NUM_KERNEL; i++)
    {
           if(i == _71BIT_MUL24)       logprintf(mystuff, "  71bit_mul24        | %6d  | %6d\n", kernel_success[i], kernel_fail[i]);
      else if(i == _75BIT_MUL32)       logprintf(mystuff, "  75bit_mul32        | %6d  | %6d\n", kernel_success[i], kernel_fail[i]);
      else if(i == _95BIT_MUL32)       logprintf(mystuff, "  95bit_mul32        | %6d  | %6d\n", kernel_success[i], kernel_fail[i]);

      else if(i == _75BIT_MUL32_GS)    logprintf(mystuff, "  75bit_mul32_gs     | %6d  | %6d\n", kernel_success[i], kernel_fail[i]);
      else if(i == _95BIT_MUL32_GS)    logprintf(mystuff, "  95bit_mul32_gs     | %6d  | %6d\n", kernel_success[i], kernel_fail[i]);

      else if(i == BARRETT76_MUL32)    logprintf(mystuff, "  barrett76_mul32    | %6d  | %6d\n", kernel_success[i], kernel_fail[i]);
      else if(i == BARRETT77_MUL32)    logprintf(mystuff, "  barrett77_mul32    | %6d  | %6d\n", kernel_success[i], kernel_fail[i]);
      else if(i == BARRETT79_MUL32)    logprintf(mystuff, "  barrett79_mul32    | %6d  | %6d\n", kernel_success[i], kernel_fail[i]);
      else if(i == BARRETT87_MUL32)    logprintf(mystuff, "  barrett87_mul32    | %6d  | %6d\n", kernel_success[i], kernel_fail[i]);
      else if(i == BARRETT88_MUL32)    logprintf(mystuff, "  barrett88_mul32    | %6d  | %6d\n", kernel_success[i], kernel_fail[i]);
      else if(i == BARRETT92_MUL32)    logprintf(mystuff, "  barrett92_mul32    | %6d  | %6d\n", kernel_success[i], kernel_fail[i]);

      else if(i == BARRETT76_MUL32_GS) logprintf(mystuff, "  barrett76_mul32_gs | %6d  | %6d\n", kernel_success[i], kernel_fail[i]);
      else if(i == BARRETT77_MUL32_GS) logprintf(mystuff, "  barrett77_mul32_gs | %6d  | %6d\n", kernel_success[i], kernel_fail[i]);
      else if(i == BARRETT79_MUL32_GS) logprintf(mystuff, "  barrett79_mul32_gs | %6d  | %6d\n", kernel_success[i], kernel_fail[i]);
      else if(i == BARRETT87_MUL32_GS) logprintf(mystuff, "  barrett87_mul32_gs | %6d  | %6d\n", kernel_success[i], kernel_fail[i]);
      else if(i == BARRETT88_MUL32_GS) logprintf(mystuff, "  barrett88_mul32_gs | %6d  | %6d\n", kernel_success[i], kernel_fail[i]);
      else if(i == BARRETT92_MUL32_GS) logprintf(mystuff, "  barrett92_mul32_gs | %6d  | %6d\n", kernel_success[i], kernel_fail[i]);

      else                             logprintf(mystuff, "  UNKNOWN kernel     | %6d  | %6d\n", kernel_success[i], kernel_fail[i]);
    }
  }
  logprintf(mystuff, "\n");

  if(st_success == num_selftests)
  {
    logprintf(mystuff, "selftest PASSED!\n\n");
    retval=0;
  }
  else
  {
    logprintf(mystuff, "selftest FAILED!\n");
    logprintf(mystuff, "  random selftest offset was: %d\n\n", mystuff->selftestrandomoffset);
  }
  return retval;
}


void print_last_CUDA_error(mystuff_t *mystuff)
/* just run cudaGetLastError() and print the error message if its return value is not cudaSuccess */
{
  cudaError_t cudaError;
  
  cudaError = cudaGetLastError();
  if(cudaError != cudaSuccess)
  {
    logprintf(mystuff, "  cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
  }
}


int main(int argc, char **argv)
{
  unsigned int exponent = 1;
  int bit_min = -1, bit_max = -1;
  int parse_ret = -1;
  int devicenumber = 0;
  mystuff_t mystuff;
  struct cudaDeviceProp deviceinfo;
  int i, tmp = 0;
  char *ptr;
  int use_worktodo = 1;
    
  i = 1;
  mystuff.mode = MODE_NORMAL;
  mystuff.quit = 0;
  mystuff.verbosity = 1;
  mystuff.bit_min = -1;
  mystuff.bit_max_assignment = -1;
  mystuff.bit_max_stage = -1;
  mystuff.logging = -1;
  mystuff.gpu_sieving = 0;
  mystuff.gpu_sieve_size = GPU_SIEVE_SIZE_DEFAULT * 1024 * 1024;		/* Size (in bits) of the GPU sieve.  Default is 128M bits. */
  mystuff.gpu_sieve_primes = GPU_SIEVE_PRIMES_DEFAULT;				/* Default to sieving primes below about 1.05M */
  mystuff.gpu_sieve_processing_size = GPU_SIEVE_PROCESS_SIZE_DEFAULT * 1024;	/* Default to 8K bits processed by each block in a Barrett kernel. */
  sprintf(mystuff.resultfile, "results.txt");
  sprintf(mystuff.jsonresultfile, "results.json.txt");
  sprintf(mystuff.logfile, "mfaktc.log");
  sprintf(mystuff.workfile, "worktodo.txt");
  sprintf(mystuff.addfile, "worktodo.add");
  mystuff.addfilestatus = -1;                                                   /* -1 -> timer not initialized! */
  
  while(i < argc)
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
        logprintf(&mystuff, "ERROR: no device number specified for option \"-d\"\n");
        return 1;
      }
      devicenumber = (int)strtol(argv[i+1], &ptr, 10);
      if(*ptr || errno || devicenumber != strtol(argv[i+1], &ptr, 10) )
      {
        logprintf(&mystuff, "ERROR: can't parse <device number> for option \"-d\"\n");
        return 1;
      }
      i++;
    }
    else if(!strcmp((char*)"-tf", argv[i]))
    {
      if(i+3 >= argc)
      {
        logprintf(&mystuff, "ERROR: missing parameters for option \"-tf\"\n");
        return 1;
      }
      exponent = (unsigned int)strtoul(argv[i+1], &ptr, 10);
      if(*ptr || errno || (unsigned long)exponent != strtoul(argv[i+1],&ptr,10) )
      {
        logprintf(&mystuff, "ERROR: can't parse parameter <exp> for option \"-tf\"\n");
        return 1;
      }
      bit_min = (int)strtol(argv[i+2], &ptr, 10);
      if(*ptr || errno || (long)bit_min != strtol(argv[i+2],&ptr,10) )
      {
        logprintf(&mystuff, "ERROR: can't parse parameter <min> for option \"-tf\"\n");
        return 1;
      }
      bit_max = (int)strtol(argv[i+3], &ptr, 10);
      if(*ptr || errno || (long)bit_max != strtol(argv[i+3],&ptr,10) )
      {
        logprintf(&mystuff, "ERROR: can't parse parameter <max> for option \"-tf\"\n");
        return 1;
      }
      if(!valid_assignment(exponent, bit_min, bit_max, mystuff.verbosity))
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
      mystuff.selftestsize = 1;
    }
    else if(!strcmp((char*)"-st2", argv[i]))
    {
      mystuff.mode = MODE_SELFTEST_FULL;
      mystuff.selftestsize = 2;
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
    else if(!strcmp((char*)"-v", argv[i]))
    {
      if(i+1 >= argc)
      {
        logprintf(&mystuff, "ERROR: no verbosity level specified for option \"-v\"\n");
        return 1;
      }
      tmp = (int)strtol(argv[i+1], &ptr, 10);
      if(*ptr || errno || tmp != strtol(argv[i+1], &ptr, 10) )
      {
        logprintf(&mystuff, "ERROR: can't parse verbosity level for option \"-v\"\n");
        return 1;
      }
      i++;
      
      if(tmp > 3)
      {
        logprintf(&mystuff, "WARNING: maximum verbosity level is 3\n");
        tmp = 3;
      }
      
      if(tmp < 0)
      {
        logprintf(&mystuff, "WARNING: minumum verbosity level is 0\n");
        tmp = 0;
      }

      mystuff.verbosity = tmp;
    }
    i++;
  }

  read_config(&mystuff);

  logprintf(&mystuff, "mfaktc v%s (%dbit built)\n\n", MFAKTC_VERSION, (int)(sizeof(void*)*8));

/* print current configuration */
  
  if(mystuff.verbosity >= 1)logprintf(&mystuff, "Compiletime options\n");
  if(mystuff.verbosity >= 1)logprintf(&mystuff, "  THREADS_PER_BLOCK         %d\n", THREADS_PER_BLOCK);
  if(mystuff.verbosity >= 1)logprintf(&mystuff, "  SIEVE_SIZE_LIMIT          %dkiB\n", SIEVE_SIZE_LIMIT);
  if(mystuff.verbosity >= 1)logprintf(&mystuff, "  SIEVE_SIZE                %dbits\n", SIEVE_SIZE);
  if(SIEVE_SIZE <= 0)
  {
    logprintf(&mystuff, "ERROR: SIEVE_SIZE is <= 0, consider to increase SIEVE_SIZE_LIMIT in params.h\n");
    close_log(&mystuff);
    return 1;
  }
  if(mystuff.verbosity >= 1)logprintf(&mystuff, "  SIEVE_SPLIT               %d\n", SIEVE_SPLIT);
  if(SIEVE_SPLIT > SIEVE_PRIMES_MIN)
  {
    logprintf(&mystuff, "ERROR: SIEVE_SPLIT must be <= SIEVE_PRIMES_MIN\n");
    close_log(&mystuff);
    return 1;
  }
#ifdef MORE_CLASSES
  if(mystuff.verbosity >= 1)logprintf(&mystuff, "  MORE_CLASSES              enabled\n");
#else
  if(mystuff.verbosity >= 1)logprintf(&mystuff, "  MORE_CLASSES              disabled\n");
#endif

#ifdef WAGSTAFF
  if(mystuff.verbosity >= 1)logprintf(&mystuff, "  Wagstaff mode             enabled\n");  
#endif

#ifdef USE_DEVICE_PRINTF
  if(mystuff.verbosity >= 1)logprintf(&mystuff, "  USE_DEVICE_PRINTF         enabled (DEBUG option)\n");
#endif
#ifdef DEBUG_GPU_MATH
  if(mystuff.verbosity >= 1)logprintf(&mystuff, "  DEBUG_GPU_MATH            enabled (DEBUG option)\n");
#endif
#ifdef DEBUG_STREAM_SCHEDULE
  if(mystuff.verbosity >= 1)logprintf(&mystuff, "  DEBUG_STREAM_SCHEDULE     enabled (DEBUG option)\n");
#endif
#ifdef DEBUG_STREAM_SCHEDULE_CHECK
  if(mystuff.verbosity >= 1)logprintf(&mystuff, "  DEBUG_STREAM_SCHEDULE_CHECK\n                            enabled (DEBUG option)\n");
#endif
#ifdef RAW_GPU_BENCH
  if(mystuff.verbosity >= 1)logprintf(&mystuff, "  RAW_GPU_BENCH             enabled (DEBUG option)\n");
#endif

  int drv_ver, rt_ver;
  if(mystuff.verbosity >= 1)logprintf(&mystuff, "\nCUDA version info\n");
  if(mystuff.verbosity >= 1)logprintf(&mystuff, "  binary compiled for CUDA  %d.%d\n", CUDART_VERSION/1000, CUDART_VERSION%100);
#if CUDART_VERSION >= 2020
  cudaRuntimeGetVersion(&rt_ver);
  if(mystuff.verbosity >= 1)logprintf(&mystuff, "  CUDA runtime version      %d.%d\n", rt_ver/1000, rt_ver%100);
  cudaDriverGetVersion(&drv_ver);  
  if(mystuff.verbosity >= 1)logprintf(&mystuff, "  CUDA driver version       %d.%d\n", drv_ver/1000, drv_ver%100);
  
  if(drv_ver < CUDART_VERSION)
  {
    logprintf(&mystuff, "ERROR: current CUDA driver version is lower than the CUDA toolkit version used during compile!\n");
    logprintf(&mystuff, "       Please update your graphics driver.\n");
    close_log(&mystuff);
    return 1;
  }
  if(rt_ver != CUDART_VERSION)
  {
    logprintf(&mystuff, "ERROR: CUDA runtime version must match the CUDA toolkit version used during compile!\n");
    close_log(&mystuff);
    return 1;
  }
#endif  

  if(cudaSetDevice(devicenumber)!=cudaSuccess)
  {
    logprintf(&mystuff, "cudaSetDevice(%d) failed\n",devicenumber);
    print_last_CUDA_error(&mystuff);
    close_log(&mystuff);
    return 1;
  }

  cudaGetDeviceProperties(&deviceinfo, devicenumber);
  mystuff.compcapa_major = deviceinfo.major;
  mystuff.compcapa_minor = deviceinfo.minor;
#if CUDART_VERSION >= 6050
  mystuff.max_shared_memory = (int)deviceinfo.sharedMemPerMultiprocessor;
#else
  if(mystuff.compcapa_major == 1)mystuff.max_shared_memory = 16384; /* assume 16kiB for CC 1.x */
  else                           mystuff.max_shared_memory = 49152; /* assume 48kiB for all other */
#endif
  if(mystuff.verbosity >= 1)
  {
    logprintf(&mystuff, "\nCUDA device info\n");
    logprintf(&mystuff, "  name                      %s\n",deviceinfo.name);
    logprintf(&mystuff, "  compute capability        %d.%d\n",deviceinfo.major,deviceinfo.minor);
    logprintf(&mystuff, "  max threads per block     %d\n",deviceinfo.maxThreadsPerBlock);
    logprintf(&mystuff, "  max shared memory per MP  %d byte\n", mystuff.max_shared_memory);
    logprintf(&mystuff, "  number of multiprocessors %d\n", deviceinfo.multiProcessorCount);
   
/* map deviceinfo.major + deviceinfo.minor to number of CUDA cores per MP. 
   This is just information, I doesn't matter whether it is correct or not */
    i=0;
         if(deviceinfo.major == 1)                          i =   8;
    else if(deviceinfo.major == 2 && deviceinfo.minor == 0) i =  32;
    else if(deviceinfo.major == 2 && deviceinfo.minor == 1) i =  48;
    else if(deviceinfo.major == 3)                          i = 192;
    else if(deviceinfo.major == 5)                          i = 128;
    
    if(i > 0)
    {             
      logprintf(&mystuff, "  CUDA cores per MP         %d\n", i);
      logprintf(&mystuff, "  CUDA cores - total        %d\n", i * deviceinfo.multiProcessorCount);
    }
    
    logprintf(&mystuff, "  clock rate (CUDA cores)   %dMHz\n", deviceinfo.clockRate / 1000);
#if CUDART_VERSION >= 5000
    logprintf(&mystuff, "  memory clock rate:        %dMHz\n", deviceinfo.memoryClockRate / 1000);
    logprintf(&mystuff, "  memory bus width:         %d bit\n", deviceinfo.memoryBusWidth);
#endif
  }

  if((mystuff.compcapa_major == 1) && (mystuff.compcapa_minor == 0))
  {
    logprintf(&mystuff, "Sorry, devices with compute capability 1.0 are not supported!\n");
    close_log(&mystuff);
    return 1;
  }

  if(THREADS_PER_BLOCK > deviceinfo.maxThreadsPerBlock)
  {
    logprintf(&mystuff, "\nERROR: THREADS_PER_BLOCK > deviceinfo.maxThreadsPerBlock\n");
    close_log(&mystuff);
    return 1;
  }

  // Don't do a CPU spin loop waiting for the GPU
  cudaSetDeviceFlags(cudaDeviceBlockingSync);

  if(mystuff.verbosity >= 1)logprintf(&mystuff, "\nAutomatic parameters\n");
#if CUDART_VERSION >= 2000
  i = THREADS_PER_BLOCK * deviceinfo.multiProcessorCount;
  while( (i * 2) <= mystuff.threads_per_grid_max) i = i * 2;
  mystuff.threads_per_grid = i;
#else
  mystuff.threads_per_grid = mystuff.threads_per_grid_max;
#endif
  if(mystuff.verbosity >= 1)logprintf(&mystuff, "  threads per grid          %d\n", mystuff.threads_per_grid);
  
  if(mystuff.threads_per_grid % THREADS_PER_BLOCK)
  {
    logprintf(&mystuff, "ERROR: mystuff.threads_per_grid is _NOT_ a multiple of THREADS_PER_BLOCK\n");
    close_log(&mystuff);
    return 1;
  }

  srandom(time(NULL));
  mystuff.selftestrandomoffset = random() % 25000000;
  if(mystuff.verbosity >= 2)logprintf(&mystuff, "  random selftest offset    %d\n", mystuff.selftestrandomoffset);
  
  for(i=0;i<mystuff.num_streams;i++)
  {
    if( cudaStreamCreate(&(mystuff.stream[i])) != cudaSuccess)
    {
      logprintf(&mystuff, "ERROR: cudaStreamCreate() failed for stream %d\n", i);
      print_last_CUDA_error(&mystuff);
      close_log(&mystuff);
      return 1;
    }
  }
/* Allocate some memory arrays */  
  for(i=0;i<(mystuff.num_streams + mystuff.cpu_streams);i++)
  {
    if( cudaHostAlloc((void**)&(mystuff.h_ktab[i]), mystuff.threads_per_grid * sizeof(int), 0) != cudaSuccess )
    {
      logprintf(&mystuff, "ERROR: cudaHostAlloc(h_ktab[%d]) failed\n", i);
      print_last_CUDA_error(&mystuff);
      close_log(&mystuff);
      return 1;
    }
  }
  for(i=0;i<mystuff.num_streams;i++)
  {
    if( cudaMalloc((void**)&(mystuff.d_ktab[i]), mystuff.threads_per_grid * sizeof(int)) != cudaSuccess )
    {
      logprintf(&mystuff, "ERROR: cudaMalloc(d_ktab1[%d]) failed\n", i);
      print_last_CUDA_error(&mystuff);
      close_log(&mystuff);
      return 1;
    }
  }
  if( cudaHostAlloc((void**)&(mystuff.h_RES),32 * sizeof(int), 0) != cudaSuccess )
  {
    logprintf(&mystuff, "ERROR: cudaHostAlloc(h_RES) failed\n");
    print_last_CUDA_error(&mystuff);
    close_log(&mystuff);
    return 1;
  }
  if( cudaMalloc((void**)&(mystuff.d_RES), 32 * sizeof(int)) != cudaSuccess )
  {
    logprintf(&mystuff, "ERROR: cudaMalloc(d_RES) failed\n");
    print_last_CUDA_error(&mystuff);
    close_log(&mystuff);
    return 1;
  }
#ifdef DEBUG_GPU_MATH
  if( cudaHostAlloc((void**)&(mystuff.h_modbasecase_debug), 32 * sizeof(int), 0) != cudaSuccess )
  {
    logprintf(&mystuff, "ERROR: cudaHostAlloc(h_modbasecase_debug) failed\n");
    print_last_CUDA_error(&mystuff);
    close_log(&mystuff);
    return 1;
  }
  if( cudaMalloc((void**)&(mystuff.d_modbasecase_debug), 32 * sizeof(int)) != cudaSuccess )
  {
    logprintf(&mystuff, "ERROR: cudaMalloc(d_modbasecase_debug) failed\n");
    print_last_CUDA_error(&mystuff);
    close_log(&mystuff);
    return 1;
  }
#endif  
  
  sieve_init();
  if(mystuff.gpu_sieving)gpusieve_init(&mystuff);

  if(mystuff.verbosity >= 1)logprintf(&mystuff, "\n");

  mystuff.sieve_primes_upper_limit = mystuff.sieve_primes_max;
  if(mystuff.mode == MODE_NORMAL)
  {

/* before we start real work run a small selftest */  
    mystuff.mode = MODE_SELFTEST_SHORT;
    logprintf(&mystuff, "running a simple selftest...\n");
    if(selftest(&mystuff, 1) != 0)return 1; /* selftest failed :( */
    mystuff.mode = MODE_NORMAL;
    
/* signal handler blablabla */
    register_signal_handler(&mystuff);
    
    if(use_worktodo && mystuff.addfiledelay != 0)
    {
      if(process_add_file(mystuff.workfile, mystuff.addfile, &(mystuff.addfilestatus), mystuff.verbosity) != OK)
      {
        mystuff.addfiledelay = 0; /* disable for until exit at least... */
      }
    }
    if(!use_worktodo)mystuff.addfiledelay = 0; /* disable addfile if not using worktodo at all (-tf on command line) */
    do
    {
      if (use_worktodo)parse_ret = get_next_assignment(mystuff.workfile, &((mystuff.exponent)), &((mystuff.bit_min)), &((mystuff.bit_max_assignment)), &((mystuff.assignment_key)), mystuff.verbosity);
      else /* got work from command */
      {
        mystuff.exponent           = exponent;
        mystuff.bit_min            = bit_min;
        mystuff.bit_max_assignment = bit_max;
        mystuff.assignment_key[0] = 0;
      }
      mystuff.factors_string[0] = 0;
      if(parse_ret == OK)
      {
        if(mystuff.verbosity >= 1)logprintf(&mystuff, "got assignment: exp=%u bit_min=%d bit_max=%d (%.2f GHz-days)\n", mystuff.exponent, mystuff.bit_min, mystuff.bit_max_assignment, primenet_ghzdays(mystuff.exponent, mystuff.bit_min, mystuff.bit_max_assignment));
        if(mystuff.gpu_sieving && mystuff.exponent < mystuff.gpu_sieve_min_exp)
        {
          logprintf(&mystuff, "ERROR: GPU sieve requested but current settings don't allow exponents below\n");
          logprintf(&mystuff, "       %u. You can decrease the value of GPUSievePrimes in mfaktc.ini \n", mystuff.gpu_sieve_min_exp);
          logprintf(&mystuff, "       lower this limit.\n");
          return 1;
        }

        mystuff.bit_max_stage = mystuff.bit_max_assignment;

        if(mystuff.stages == 1)
        {
          while( ((calculate_k(mystuff.exponent, mystuff.bit_max_stage) - calculate_k(mystuff.exponent, mystuff.bit_min)) > (250000000ULL * NUM_CLASSES)) && ((mystuff.bit_max_stage - mystuff.bit_min) > 1) )mystuff.bit_max_stage--;
        }
        tmp = 0;
        while(mystuff.bit_max_stage <= mystuff.bit_max_assignment && !mystuff.quit)
        {
          tmp = tf(&mystuff, 0, 0, AUTOSELECT_KERNEL);
//          tmp = tf(&mystuff, 0, 0, _71BIT_MUL24);
//          tmp = tf(&mystuff, 0, 0, _75BIT_MUL32);
//          tmp = tf(&mystuff, 0, 0, _75BIT_MUL32_GS);
//          tmp = tf(&mystuff, 0, 0, _95BIT_MUL32);
//          tmp = tf(&mystuff, 0, 0, _95BIT_MUL32_GS);
//          tmp = tf(&mystuff, 0, 0, BARRETT76_MUL32);
//          tmp = tf(&mystuff, 0, 0, BARRETT76_MUL32_GS);
//          tmp = tf(&mystuff, 0, 0, BARRETT77_MUL32);
//          tmp = tf(&mystuff, 0, 0, BARRETT77_MUL32_GS);
//          tmp = tf(&mystuff, 0, 0, BARRETT79_MUL32);
//          tmp = tf(&mystuff, 0, 0, BARRETT79_MUL32_GS);
//          tmp = tf(&mystuff, 0, 0, BARRETT87_MUL32);
//          tmp = tf(&mystuff, 0, 0, BARRETT87_MUL32_GS);
//          tmp = tf(&mystuff, 0, 0, BARRETT88_MUL32);
//          tmp = tf(&mystuff, 0, 0, BARRETT88_MUL32_GS);
//          tmp = tf(&mystuff, 0, 0, BARRETT92_MUL32);
//          tmp = tf(&mystuff, 0, 0, BARRETT92_MUL32_GS);
          
          
          if(tmp == RET_CUDA_ERROR) return 1; /* bail out, we might have a serios problem (detected by cudaGetLastError())... */

          if(tmp != RET_QUIT)
          {
            if( (mystuff.stopafterfactor > 0) && (tmp > 0) )
            {
              mystuff.bit_max_stage = mystuff.bit_max_assignment;
            }

            if(use_worktodo)
            {
              if(mystuff.bit_max_stage == mystuff.bit_max_assignment)parse_ret = clear_assignment(mystuff.workfile, mystuff.exponent, mystuff.bit_min, mystuff.bit_max_assignment, 0);
              else                                                   parse_ret = clear_assignment(mystuff.workfile, mystuff.exponent, mystuff.bit_min, mystuff.bit_max_assignment, mystuff.bit_max_stage);

                   if(parse_ret == CANT_OPEN_WORKFILE)   logprintf(&mystuff, "ERROR: clear_assignment() / modify_assignment(): can't open \"%s\"\n", mystuff.workfile);
              else if(parse_ret == CANT_OPEN_TEMPFILE)   logprintf(&mystuff, "ERROR: clear_assignment() / modify_assignment(): can't open \"__worktodo__.tmp\"\n");
              else if(parse_ret == ASSIGNMENT_NOT_FOUND) logprintf(&mystuff, "ERROR: clear_assignment() / modify_assignment(): assignment not found in \"%s\"\n", mystuff.workfile);
              else if(parse_ret == CANT_RENAME)          logprintf(&mystuff, "ERROR: clear_assignment() / modify_assignment(): can't rename workfiles\n");
              else if(parse_ret != OK)                   logprintf(&mystuff, "ERROR: clear_assignment() / modify_assignment(): Unknown error (%d)\n", parse_ret);
            }

            mystuff.bit_min = mystuff.bit_max_stage;
            mystuff.bit_max_stage++;
          }
        }
      }
      else if(parse_ret == CANT_OPEN_FILE)             logprintf(&mystuff, "ERROR: get_next_assignment(): can't open \"%s\"\n", mystuff.workfile);
      else if(parse_ret == VALID_ASSIGNMENT_NOT_FOUND) logprintf(&mystuff, "ERROR: get_next_assignment(): no valid assignment found in \"%s\"\n", mystuff.workfile);
      else if(parse_ret != OK)                         logprintf(&mystuff, "ERROR: get_next_assignment(): Unknown error (%d)\n", parse_ret);
    }
    while(parse_ret == OK && use_worktodo && !mystuff.quit);
  }
  else // mystuff.mode != MODE_NORMAL
  {
    selftest(&mystuff, 0);
  }

  for(i=0;i<mystuff.num_streams;i++)
  {
    cudaStreamDestroy(mystuff.stream[i]);
  }
#ifdef DEBUG_GPU_MATH
  cudaFree(mystuff.d_modbasecase_debug);
  cudaFree(mystuff.h_modbasecase_debug);
#endif  
  cudaFree(mystuff.d_RES);
  cudaFree(mystuff.h_RES);
  for(i=0;i<(mystuff.num_streams + mystuff.cpu_streams);i++)cudaFreeHost(mystuff.h_ktab[i]);
  for(i=0;i<mystuff.num_streams;i++)cudaFree(mystuff.d_ktab[i]);
  sieve_free();

  // Free GPU sieve data structures
  cudaFree(mystuff.d_bitarray);
  cudaFree(mystuff.d_sieve_info);
  cudaFree(mystuff.d_calc_bit_to_clear_info);
  close_log(&mystuff);
  return 0;
}
