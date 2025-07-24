/*
This file is part of mfaktc.
Copyright (C) 2009, 2010, 2011, 2012, 2014, 2015, 2018, 2019  Oliver Weihe (o.weihe@t-online.de)

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

/* 72bit (3x 24bit) integer
D=d0 + d1*(2^24) + d2*(2^48) */
typedef struct {
    unsigned int d0, d1, d2;
} int72;

/* 144bit (6x 24bit) integer
D=d0 + d1*(2^24) + d2*(2^48) + ... */
typedef struct {
    unsigned int d0, d1, d2, d3, d4, d5;
} int144;

/* int72 and int96 are the same but this way the compiler warns
when an int96 is passed to a function designed to handle 72 bit int.
The applies to int144 and int192, too. */

/* 96bit (3x 32bit) integer
D= d0 + d1*(2^32) + d2*(2^64) */
typedef struct {
    unsigned int d0, d1, d2;
} int96;

/* 192bit (6x 32bit) integer
D=d0 + d1*(2^32) + d2*(2^64) + ... */
typedef struct {
    unsigned int d0, d1, d2, d3, d4, d5;
} int192;

// clang-format off
typedef struct {
    char progressheader[256];              /* userconfigureable progress header */
    char progressformat[256];              /* userconfigureable progress line */
    int class_number;                      /* the number of the last processed class */
    int grid_count;                        /* number of grids processed in the last processed class */
    unsigned long long int class_time;     /* time (in ms) needed to process the last processed class */
    unsigned long long int bit_level_time; /* time (in ms) since the bitlevel was started */
    float cpu_wait;                        /* percentage CPU was waiting for the GPU */
    int output_counter;                    /* count how often the status line was written since last headline */
    int class_counter;                     /* number of finished classes of the current job */
    double ghzdays;                        /* primenet GHZdays for the current assignment (current stage) */
    char kernelname[30];
} stats_t;

typedef struct {
    cudaStream_t stream[NUM_STREAMS_MAX];
    unsigned int *h_ktab[NUM_STREAMS_MAX + CPU_STREAMS_MAX];
    unsigned int *d_ktab[NUM_STREAMS_MAX];
    unsigned int *h_RES;
    unsigned int *d_RES;

    unsigned int exponent;                  /* the exponent we're currently working on */
    int bit_min;                            /* where do we start TFing */
    int bit_max_assignment;                 /* the upper size of factors we're searching for */
    int bit_max_stage;                      /* as above, but only for the current stage */
  
    int sieve_primes;                       /* the actual number of odd primes using for sieving */
    int sieve_primes_adjust;                /* allow automated adjustment of sieve_primes? */
    int sieve_primes_upper_limit;           /* the upper limit of sieve_primes for the current exponent */
    int sieve_primes_min, sieve_primes_max; /* user configurable sieve_primes min/max */
  
    char workfile[51];                      /* allow filenames up to 50 chars... */
    char addfile[51];                       /* allow filenames up to 50 chars... */
    char resultfile[51];                    /* allow filenames up to 50 chars... */
    char jsonresultfile[51];                /* allow filenames up to 50 chars... */
    char logfile[51];                       /* allow filenames up to 50 chars... */
    FILE* logfileptr;
    int num_streams, cpu_streams;
  
    int compcapa_major;                      /* compute capability major - actual GPU */
    int compcapa_minor;                      /* compute capability minor - actual GPU */
    int cuda_toolkit;                        /* CUDA toolkit used for compiling) */
    int cuda_arch;                           /* CUDA arch (--generate-code during compiletime) selected for GPU */
    int max_shared_memory;                   /* maximum size of shared memory per multiprocessor (in byte) */
  
    int checkpoints, checkpointdelay, mode, stages, stopafterfactor;
    int threads_per_grid_max, threads_per_grid;
  
    int addfiledelay, addfilestatus; /* status: -1: timer not initialized
                                                 0: last check didn't found an addfile
                                                 1: last check found an addfile */

// clang-format on
#ifdef DEBUG_GPU_MATH
    unsigned int *d_modbasecase_debug;
    unsigned int *h_modbasecase_debug;
#endif

    int gpu_sieving; /* TRUE if we're letting the GPU do the sieving */
    int gpu_sieve_size; /* Size (in bits) of the GPU sieve.  Default is 128M bits. */
    int gpu_sieve_primes; /* the actual number of primes using for sieving */
    int gpu_sieve_processing_size; /* The number of GPU sieve bits each thread in a Barrett kernel will process.  Default is 2K bits. */
    unsigned int gpu_sieve_min_exp; /* the minimum exponent allowed for GPU sieving */
    unsigned int *d_bitarray; /* 128M bit array for GPU sieve */
    unsigned int *d_sieve_info; /* Device array containing compressed info needed for prime number GPU sieves */
    unsigned int *d_calc_bit_to_clear_info; /* Device array containing uncompressed info needed to calculate initial bit-to-clear */

    int printmode;
    int allowsleep;

    int print_timestamp;

    int quit;
    int verbosity; /* 0 = reduced number of screen printfs, 1 = default, >= 2 = some additional printfs */
    int logging;   /* 0 = logging disabled (default), 1 = logging enabled */
    int legacy_results_txt; /* 0 = output to results.txt disabled (default), 1 = output to results.txt enabled */

    int selftestsize;
    int selftestrandomoffset;

    stats_t stats; /* stuff for statistics, etc. */

    // clang-format off
    char V5UserID[51];                        /* primenet V5UserID and ComputerID */
    char ComputerID[51];                      /* currently only used for screen/result output */
    char assignment_key[MAX_LINE_LENGTH + 1]; /* the assignment ID */
    int96 factors[MAX_FACTORS_PER_JOB];       /* store factors in global state */
    // clang-format on

} mystuff_t; /* FIXME: proper name needed */

enum GPUKernels {
    AUTOSELECT_KERNEL,
    _75BIT_MUL32,
    _95BIT_MUL32,
    BARRETT76_MUL32,
    BARRETT77_MUL32,
    BARRETT79_MUL32,
    BARRETT87_MUL32,
    BARRETT88_MUL32,
    BARRETT92_MUL32,
    _75BIT_MUL32_GS,
    _95BIT_MUL32_GS,
    BARRETT76_MUL32_GS,
    BARRETT77_MUL32_GS,
    BARRETT79_MUL32_GS,
    BARRETT87_MUL32_GS,
    BARRETT88_MUL32_GS,
    BARRETT92_MUL32_GS
};

enum MODES { MODE_NORMAL, MODE_SELFTEST_SHORT, MODE_SELFTEST_FULL };

#define RET_CUDA_ERROR 1000000001
#define RET_QUIT       1000000002

// Capabilities
#define TESLA                     100
#define FERMI                     200
#define KEPLER                    300
#define KEPLER_WITH_FUNNELSHIFT   320
#define MAXWELL                   500
#define COMPUTE_CAPABILITY_PASCAL 600
#define VOLTA                     700
#define TURING                    750
