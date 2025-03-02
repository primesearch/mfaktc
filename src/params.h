/*
This file is part of mfaktc.
Copyright (C) 2009, 2010, 2011, 2012, 2013, 2015, 2019, 2024  Oliver Weihe (o.weihe@t-online.de)

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

/*
SIEVE_SIZE_LIMIT is the maximum segment size of the sieve.
too small => to much overhead
too big => doesn't fit into (fast) CPU-caches
The size given here is in kiB (1024 bytes). A good starting point is the size
of your CPUs L1-Data cache.
This is just the upper LIMIT of the SIEVE_SIZE, the actual sieve size depends
on some other factors asweel, but you don't have to worry about.
*/

#define SIEVE_SIZE_LIMIT 32



/*
If MORE_CLASSES is defined than the while TF process is split into 4620
(4 * 3*5*7*11) classes. Otherwise it will be split into 420 (4 * 3*5*7)
classes. With 4620 the siever runs a bit more efficent at the cost of 10 times
more sieve initializations. This will allow to increase SIEVE_PRIMES a little
bit further.
This starts to become usefull on my system for e.g. TF M66xxxxxx from 2^66 to
2^67.
*/

#define MORE_CLASSES


/* use WAGSTAFF to build mfaktc doing TF on Wagstaff numbers instead of
Mersenne numbers */

//#define WAGSTAFF



/******************
** DEBUG options **
******************/

/* do some checks on math done on GPU (mainly division stuff) */
//#define DEBUG_GPU_MATH

/* define TRACE_FC to enable tracing of a specific Factor Candidate
DEBUG_GPU_MATH has to be enabled, too. */

//#define TRACE_FC

/* M49635893 has a factor: 280164061095680036711, this is part of the
"simple selftest" AND the full selftest */
//#define TRACE_D2 0x0000000F
//#define TRACE_D1 0x300EB131
//#define TRACE_D0 0x96D84F67

/* print stream and h_ktab usage */
//#define DEBUG_STREAM_SCHEDULE

/* perform a sanity check on the h_ktab usage */
//#define DEBUG_STREAM_SCHEDULE_CHECK


/* disable sieve code to measure raw GPU performance */
//#define RAW_GPU_BENCH






/******************************************************************************
*******************************************************************************
*** DO NOT EDIT DEFINES BELOW THIS LINE UNLESS YOU REALLY KNOW WHAT YOU DO! ***
*** DO NOT EDIT DEFINES BELOW THIS LINE UNLESS YOU REALLY KNOW WHAT YOU DO! ***
*** DO NOT EDIT DEFINES BELOW THIS LINE UNLESS YOU REALLY KNOW WHAT YOU DO! ***
*******************************************************************************
******************************************************************************/

#define MFAKTC_VERSION "0.24.0-alpha.3" /* DO NOT CHANGE! */
#define MFAKTC_CHECKPOINT_VERSION "0.24" /* DO NOT CHANGE! */
#define MFAKTC_CHECKSUM_VERSION "1" /* DO NOT CHANGE! */



/*
THREADS_PER_BLOCK has a hardware limit, 512 on GPUs with compute capability
1.x and 1024 on GPUs with compute capability 2.0. 
256 should be OK for most cases. Anyway there is usually no need to increase
THREADS_PER_BLOCK above 256 because if enough resources are available
(e.g. registers, shared memory) multiple blocks are launched at the same
time. When it is increased too much you might run out of register space
(especially on GPUs with compute capability 1.0 and 1.1)
*/

#define THREADS_PER_BLOCK 256 /* DO NOT CHANGE! */



/*
SIEVE_PRIMES defines how far we sieve the factor candidates.
The first <SIEVE_PRIMES> odd primes are sieved.
The optimal value depends greatly on the speed of the CPU (one core) and the
speed of the CPU.
The actual configuration is done in mfaktc.ini.
The following lines define the min, default and max value.
*/

#define SIEVE_PRIMES_MIN      2000 /* DO NOT CHANGE! */
#define SIEVE_PRIMES_DEFAULT 25000 /* DO NOT CHANGE! */
#define SIEVE_PRIMES_MAX    200000 /* DO NOT CHANGE! */



/* the first SIEVE_SPLIT primes have a special code in sieve.c. This defines
when the siever switches between those two code variants. */

#define SIEVE_SPLIT 250 /* DO NOT CHANGE! */



/*
The number of CUDA streams used by mfaktc.
The actual configuration is done in mfaktc.ini. This ini-file contains
a small describtion, too
The following lines define the min, default and max value.
*/

#define NUM_STREAMS_MIN     1 /* DO NOT CHANGE! */
#define NUM_STREAMS_DEFAULT 3 /* DO NOT CHANGE! */
#define NUM_STREAMS_MAX     10 /* DO NOT CHANGE! */

#define CPU_STREAMS_MIN     1 /* DO NOT CHANGE! */
#define CPU_STREAMS_DEFAULT 3 /* DO NOT CHANGE! */
#define CPU_STREAMS_MAX     5 /* DO NOT CHANGE! */



/* set NUM_CLASSES and SIEVE_SIZE depending on MORE_CLASSES and SIEVE_SIZE_LIMIT */
#ifndef MORE_CLASSES
  #define NUM_CLASSES 420 /* 2 * 2 * 3 * 5 * 7 */  /* DO NOT CHANGE! */
  #define SIEVE_SIZE ((SIEVE_SIZE_LIMIT<<13) - (SIEVE_SIZE_LIMIT<<13) % (11*13*17*19)) /* DO NOT CHANGE! */
#else
  #define NUM_CLASSES 4620 /* 2 * 2 * 3 * 5 * 7 * 11 */ /* DO NOT CHANGE! */
  #define SIEVE_SIZE ((SIEVE_SIZE_LIMIT<<13) - (SIEVE_SIZE_LIMIT<<13) % (13*17*19*23)) /* DO NOT CHANGE! */
#endif



/*
GPU_SIEVE_PRIMES defines how far we sieve the factor candidates on the GPU.
The first <GPU_SIEVE_PRIMES> primes are sieved.

GPU_SIEVE_SIZE defines how big of a GPU sieve we use (in M bits).

GPU_SIEVE_PROCESS_SIZE defines how far many bits of the sieve each TF block processes (in K bits).
Larger values may lead to less wasted cycles by reducing the number of times all threads in a warp
are not TFing a candidate.  However, more shared memory is used which may reduce occupancy.
Smaller values should lead to a more responsive system (each kernel takes less time to execute).

The actual configuration is done in mfaktc.ini.
The following lines define the min, default and max value.
*/

#define GPU_SIEVE_PRIMES_MIN                 0 /* GPU sieving code can work (inefficiently) with very small numbers */
#define GPU_SIEVE_PRIMES_DEFAULT         82486 /* Default is to sieve primes up to about 1.05M */
#define GPU_SIEVE_PRIMES_MAX           1075000 /* Primes to 16,729,793. GPU sieve should be able to handle up to 16M. */

#define GPU_SIEVE_SIZE_MIN                   4 /* A 4M bit sieve seems like a reasonable minimum */
#define GPU_SIEVE_SIZE_DEFAULT            2047 /* Default is a 128M bit sieve */
#define GPU_SIEVE_SIZE_MAX                2047 /* We've only tested up to 128M bits.  The GPU sieve code may be able to go higher. */

#define GPU_SIEVE_PROCESS_SIZE_MIN           8 /* Processing 8Ki bits in each block is minimum (256 threads * 1 word of 32 bits) */
#define GPU_SIEVE_PROCESS_SIZE_DEFAULT      16 /* Default is processing 16Ki bits */
#define GPU_SIEVE_PROCESS_SIZE_MAX          32 /* Upper limit is 64K, since we store k values as "short".
                                                  Not validated and shared memory might be an issue! */



#ifdef WAGSTAFF
  #define NAME_NUMBERS "W"
#else /* Mersennes */
  #define NAME_NUMBERS "M"
#endif

/* For worktodo.txt files */
#define MAX_LINE_LENGTH 100
