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

#include "gpusieve_helper.cu"


#ifndef DEBUG_GPU_MATH
__device__ static void mod_192_96(int96 *res, int192 q, int96 n, float nf)
#else
__device__ static void mod_192_96(int96 *res, int192 q, int96 n, float nf, unsigned int *modbasecase_debug)
#endif
/* res = q mod n */
{
  float qf;
  unsigned int qi;
  int192 nn;

/********** Step 1, Offset 2^75 (2*32 + 11) **********/
/*
the 75 bit kernel has only one difference: the first iteration of the
division will be skipped
*/
#ifndef SHORTCUT_75BIT
  qf= __uint2float_rn(q.d5);
  qf= qf * 4294967296.0f + __uint2float_rn(q.d4);
  qf= qf * 4294967296.0f + __uint2float_rn(q.d3);
  qf*= 2097152.0f;

  qi=__float2uint_rz(qf*nf);

  MODBASECASE_QI_ERROR(1<<22, 1, qi, 0);


// nn = n * qi
  nn.d2 =                                 __umul32(n.d0, qi);
  nn.d3 = __add_cc (__umul32hi(n.d0, qi), __umul32(n.d1, qi));
  nn.d4 = __addc_cc(__umul32hi(n.d1, qi), __umul32(n.d2, qi));
  nn.d5 = __addc   (__umul32hi(n.d2, qi),                  0);

// shiftleft nn 11 bits
  nn.d5 = (nn.d5 << 11) + (nn.d4 >> 21);
  nn.d4 = (nn.d4 << 11) + (nn.d3 >> 21);
  nn.d3 = (nn.d3 << 11) + (nn.d2 >> 21);
  nn.d2 =  nn.d2 << 11;

//  q = q - nn
  q.d2 = __sub_cc (q.d2, nn.d2);
  q.d3 = __subc_cc(q.d3, nn.d3);
  q.d4 = __subc_cc(q.d4, nn.d4);
  q.d5 = __subc   (q.d5, nn.d5);
#endif // SHORTCUT_75BIT
/********** Step 2, Offset 2^55 (1*32 + 23) **********/
#ifndef SHORTCUT_75BIT  
  qf= __uint2float_rn(q.d5);
  qf= qf * 4294967296.0f + __uint2float_rn(q.d4);
#else
  qf= __uint2float_rn(q.d4);
#endif  
  qf= qf * 4294967296.0f + __uint2float_rn(q.d3);
  qf= qf * 4294967296.0f + __uint2float_rn(q.d2);
  qf*= 512.0f;

  qi=__float2uint_rz(qf*nf);

  MODBASECASE_QI_ERROR(1<<22, 2, qi, 1);


// nn = n * qi
  nn.d1 =                                 __umul32(n.d0, qi);
  nn.d2 = __add_cc (__umul32hi(n.d0, qi), __umul32(n.d1, qi));
  nn.d3 = __addc_cc(__umul32hi(n.d1, qi), __umul32(n.d2, qi));
  nn.d4 = __addc   (__umul32hi(n.d2, qi),                  0);

// shiftleft nn 23 bits
#ifdef DEBUG_GPU_MATH
  nn.d5 =                  nn.d4 >> 9;
#endif  
  nn.d4 = (nn.d4 << 23) + (nn.d3 >> 9);
  nn.d3 = (nn.d3 << 23) + (nn.d2 >> 9);
  nn.d2 = (nn.d2 << 23) + (nn.d1 >> 9);
  nn.d1 =  nn.d1 << 23;

// q = q - nn
  q.d1 = __sub_cc (q.d1, nn.d1);
  q.d2 = __subc_cc(q.d2, nn.d2);
  q.d3 = __subc_cc(q.d3, nn.d3);
#ifndef DEBUG_GPU_MATH  
  q.d4 = __subc   (q.d4, nn.d4);
#else
  q.d4 = __subc_cc(q.d4, nn.d4);
  q.d5 = __subc   (q.d5, nn.d5);
#endif

/********** Step 3, Offset 2^35 (1*32 + 3) **********/
  MODBASECASE_NONZERO_ERROR(q.d5, 3, 5, 2);

  qf= __uint2float_rn(q.d4);
  qf= qf * 4294967296.0f + __uint2float_rn(q.d3);
  qf= qf * 4294967296.0f + __uint2float_rn(q.d2);
  qf*= 536870912.0f; // add (q.d1 >> 3) ???
//  qf*= 4294967296.0f; /* this includes the shiftleft of qi by 3 bits! */

  qi=__float2uint_rz(qf*nf);

  MODBASECASE_QI_ERROR(1<<22, 3, qi, 3);

// shiftleft qi 3 bits to avoid "long shiftleft" after multiplication
  qi <<= 3;

// nn = n * qi
  nn.d1 =                                 __umul32(n.d0, qi);
  nn.d2 = __add_cc (__umul32hi(n.d0, qi), __umul32(n.d1, qi));
  nn.d3 = __addc_cc(__umul32hi(n.d1, qi), __umul32(n.d2, qi));
  nn.d4 = __addc   (__umul32hi(n.d2, qi),                  0);

//  q = q - nn
  q.d1 = __sub_cc (q.d1, nn.d1);
  q.d2 = __subc_cc(q.d2, nn.d2);
  q.d3 = __subc_cc(q.d3, nn.d3);
  q.d4 = __subc   (q.d4, nn.d4);

/********** Step 4, Offset 2^15 (0*32 + 15) **********/
  MODBASECASE_NONZERO_ERROR(q.d5, 4, 5, 4);

  qf= __uint2float_rn(q.d4);
  qf= qf * 4294967296.0f + __uint2float_rn(q.d3);
  qf= qf * 4294967296.0f + __uint2float_rn(q.d2);
  qf= qf * 4294967296.0f + __uint2float_rn(q.d1);
  qf*= 131072.0f;
  
  qi=__float2uint_rz(qf*nf);

  MODBASECASE_QI_ERROR(1<<22, 4, qi, 5);

// nn = n * qi
  nn.d0 =                                 __umul32(n.d0, qi);
  nn.d1 = __add_cc (__umul32hi(n.d0, qi), __umul32(n.d1, qi));
  nn.d2 = __addc_cc(__umul32hi(n.d1, qi), __umul32(n.d2, qi));
  nn.d3 = __addc   (__umul32hi(n.d2, qi),                  0);

// shiftleft nn 15 bits
#ifdef DEBUG_GPU_MATH
  nn.d4 =                  nn.d3 >> 17;
#endif
  nn.d3 = (nn.d3 << 15) + (nn.d2 >> 17);
  nn.d2 = (nn.d2 << 15) + (nn.d1 >> 17);
  nn.d1 = (nn.d1 << 15) + (nn.d0 >> 17);
  nn.d0 =  nn.d0 << 15;

//  q = q - nn
  q.d0 = __sub_cc (q.d0, nn.d0);
  q.d1 = __subc_cc(q.d1, nn.d1);
  q.d2 = __subc_cc(q.d2, nn.d2);
#ifndef DEBUG_GPU_MATH
  q.d3 = __subc   (q.d3, nn.d3);
#else
  q.d3 = __subc_cc(q.d3, nn.d3);
  q.d4 = __subc   (q.d4, nn.d4);
#endif

/********** Step 5, Offset 2^0 (0*32 + 0) **********/
  MODBASECASE_NONZERO_ERROR(q.d5, 5, 5, 6);
  MODBASECASE_NONZERO_ERROR(q.d4, 5, 4, 7);

  qf= __uint2float_rn(q.d3);
  qf= qf * 4294967296.0f + __uint2float_rn(q.d2);
  qf= qf * 4294967296.0f + __uint2float_rn(q.d1);
  qf= qf * 4294967296.0f + __uint2float_rn(q.d0);
  
  qi=__float2uint_rz(qf*nf);

  MODBASECASE_QI_ERROR(1<<20, 5, qi, 8);

// nn = n * qi
  nn.d0 =                                 __umul32(n.d0, qi);
  nn.d1 = __add_cc (__umul32hi(n.d0, qi), __umul32(n.d1, qi));
#ifndef DEBUG_GPU_MATH  
  nn.d2 = __addc   (__umul32hi(n.d1, qi), __umul32(n.d2, qi));
#else
  nn.d2 = __addc_cc(__umul32hi(n.d1, qi), __umul32(n.d2, qi));
  nn.d3 = __addc   (__umul32hi(n.d2, qi),                  0);
#endif  

//  q = q - nn
  q.d0 = __sub_cc (q.d0, nn.d0);
  q.d1 = __subc_cc(q.d1, nn.d1);
#ifndef DEBUG_GPU_MATH
  q.d2 = __subc   (q.d2, nn.d2);
#else
  q.d2 = __subc_cc(q.d2, nn.d2);
  q.d3 = __subc   (q.d3, nn.d3);
#endif

  res->d0=q.d0;
  res->d1=q.d1;
  res->d2=q.d2;
  
  MODBASECASE_NONZERO_ERROR(q.d5, 6, 5, 9);
  MODBASECASE_NONZERO_ERROR(q.d4, 6, 4, 10);
  MODBASECASE_NONZERO_ERROR(q.d3, 6, 3, 11);

/*
qi is allways a little bit too small, this is OK for all steps except the last
one. Sometimes the result is a little bit bigger than n
*/
/*  if(cmp_ge_96(*res,n))
  {
    sub_96(&tmp96,*res,n);
    copy_96(res,tmp96);
  }*/
}


__device__ static void test_FC96_mfaktc_95(int96 f, int192 b, unsigned int exp, unsigned int *RES, int shiftcount
#ifdef DEBUG_GPU_MATH
                                           , unsigned int *modbasecase_debug
#endif
                                           )
{
  int96 a;
  float ff;

/*
ff = f as float, needed in mod_192_96().
Precalculated here since it is the same for all steps in the following loop */
  ff= __uint2float_rn(f.d2);
  ff= ff * 4294967296.0f + __uint2float_rn(f.d1);
  ff= ff * 4294967296.0f + __uint2float_rn(f.d0);

  ff=__int_as_float(0x3f7ffffb) / ff;	// just a little bit below 1.0f so we allways underestimate the quotient
        
#ifndef DEBUG_GPU_MATH
  mod_192_96(&a,b,f,ff);			// a = b mod f
#else
  mod_192_96(&a,b,f,ff,modbasecase_debug);	// a = b mod f
#endif
  exp<<= 32 - shiftcount;
  while(exp)
  {
#ifdef SHORTCUT_75BIT
    square_96_160(&b,a);			// b = a^2
#else
    square_96_192(&b,a);			// b = a^2
#endif
    if(exp&0x80000000)shl_192(&b);              // "optional multiply by 2" in Prime 95 documentation
#ifndef DEBUG_GPU_MATH
      mod_192_96(&a,b,f,ff);			// a = b mod f
#else
      mod_192_96(&a,b,f,ff,modbasecase_debug);	// a = b mod f
#endif
    exp<<=1;
  }

  if(cmp_ge_96(a,f))				// final adjustment in case a >= f
  {
    sub_96(&a,a,f);
  }

#if defined DEBUG_GPU_MATH
  if(cmp_ge_96(a,f))
  {
    printf("EEEEEK, final a is >= f\n");
  }
#endif

/* finally check if we found a factor and write the factor to RES[]
this kernel has a lower FC limit of 2^64 so we can use check_big_factor96() */
  check_factor96(f, a, RES);
}


__global__ void
#ifdef SHORTCUT_75BIT
__launch_bounds__(THREADS_PER_BLOCK,2) mfaktc_75(unsigned int exp, int96 k, unsigned int *k_tab, int shiftcount, int192 b, unsigned int *RES
#else
__launch_bounds__(THREADS_PER_BLOCK,2) mfaktc_95(unsigned int exp, int96 k, unsigned int *k_tab, int shiftcount, int192 b, unsigned int *RES
#endif
#ifdef DEBUG_GPU_MATH
                                                 , unsigned int *modbasecase_debug
#endif
                                                 )                                                
/*
computes 2^exp mod f
shiftcount is used for precomputing without mod
a is precomputed on host ONCE. */
{
  int96 f;
  int index = blockDim.x * blockIdx.x + threadIdx.x;

  create_FC96_mad(&f, exp, k, k_tab[index]);    // f = 2 * (k + k_tab[index]) * exp + 1
  
  test_FC96_mfaktc_95(f, b, exp, RES, shiftcount
#ifdef DEBUG_GPU_MATH
                      , modbasecase_debug
#endif
                      );

}


__global__ void
#ifdef SHORTCUT_75BIT
__launch_bounds__(THREADS_PER_BLOCK,2) mfaktc_75_gs(unsigned int exp, int96 k_base, unsigned int *bit_array, unsigned int bits_to_process, int shiftcount, int192 b, unsigned int *RES
#else
__launch_bounds__(THREADS_PER_BLOCK,2) mfaktc_95_gs(unsigned int exp, int96 k_base, unsigned int *bit_array, unsigned int bits_to_process, int shiftcount, int192 b, unsigned int *RES
#endif
#ifdef DEBUG_GPU_MATH
                                                 , unsigned int *modbasecase_debug
#endif
                                                 )                                                
/*
computes 2^exp mod f
shiftcount is used for precomputing without mod
a is precomputed on host ONCE. */
{
  int96 f, f_base;
  int i, total_bit_count, k_delta;
  extern __shared__ unsigned short k_deltas[];		// Write bits to test here.  Launching program must estimate
							// how much shared memory to allocate based on number of primes sieved.

  create_k_deltas(bit_array, bits_to_process, &total_bit_count, k_deltas);
  create_fbase96(&f_base, k_base, exp, bits_to_process);

// Loop til the k values written to shared memory are exhausted
  for (i = threadIdx.x; i < total_bit_count; i += THREADS_PER_BLOCK) {

// Get the (k - k_base) value to test
    k_delta = k_deltas[i];

// Compute new f.  This is computed as f = f_base + 2 * (k - k_base) * exp.
    f.d0 = __add_cc (f_base.d0, __umul32(2 * k_delta * NUM_CLASSES, exp));
    f.d1 = __addc_cc(f_base.d1, __umul32hi(2 * k_delta * NUM_CLASSES, exp));
    f.d2 = __addc   (f_base.d2, 0);

    test_FC96_mfaktc_95(f, b, exp, RES, shiftcount
#ifdef DEBUG_GPU_MATH
                        , modbasecase_debug
#endif
                        );
  }
}

#define TF_96BIT
#include "tf_common.cu"
#include "tf_common_gs.cu"
#undef TF_96BIT
