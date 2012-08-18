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

#include "tf_96bit_base_math.cu"
#include "tf_debug.h"


#undef DIV_160_96
#include "tf_barrett96_div.cu"
#define DIV_160_96
#include "tf_barrett96_div.cu"
#undef DIV_160_96


#ifndef CHECKS_MODBASECASE
__device__ static void mod_simple_96(int96 *res, int96 q, int96 n, float nf)
#else
__device__ static void mod_simple_96(int96 *res, int96 q, int96 n, float nf, int bit_max64, unsigned int limit, unsigned int *modbasecase_debug)
#endif
/*
res = q mod n
used for refinement in barrett modular multiplication
assumes q < Xn where X is a small integer
*/
{
  float qf;
  unsigned int qi;
  int96 nn;

  qf = __uint2float_rn(q.d2);
  qf = qf * 4294967296.0f + __uint2float_rn(q.d1);
  
  qi=__float2uint_rz(qf*nf);

#ifdef CHECKS_MODBASECASE
/* both barrett based kernels are made for factor candidates above 2^64,
atleast the 79bit variant fails on factor candidates less than 2^64!
Lets ignore those errors...
Factor candidates below 2^64 can occur when TFing from 2^64 to 2^65, the
first candidate in each class can be smaller than 2^64.
This is NOT an issue because those exponents should be TFed to 2^64 with a
kernel which can handle those "small" candidates before starting TF from
2^64 to 2^65. So in worst case we have a false positive which is catched
easily from the primenetserver.
The same applies to factor candidates which are bigger than 2^bit_max for the
barrett92 kernel. If the factor candidate is bigger than 2^bit_max than
usually just the correction factor is bigger than expected. There are tons
of messages that qi is to high (better: higher than expected) e.g. when trial
factoring huge exponents from 2^64 to 2^65 with the barrett92 kernel (during
selftest). The factor candidates might be as high a 2^68 in some of these
cases! This is related to the _HUGE_ blocks that mfaktc processes at once.
To make it short: let's ignore warnings/errors from factor candidates which
are "out of range".
*/
  if(n.d2 != 0 && n.d2 < (1 << bit_max64))
  {
    MODBASECASE_QI_ERROR(limit, 100, qi, 12);
  }
#endif

#if (__CUDA_ARCH__ >= FERMI) && (CUDART_VERSION >= 4010) /* multiply-add with carry is not available on CC 1.x devices and before CUDA 4.1 */
  nn.d0 =                          __umul32(n.d0, qi);
  nn.d1 = __umad32hi_cc (n.d0, qi, __umul32(n.d1, qi));
  nn.d2 = __umad32hic   (n.d1, qi, __umul32(n.d2, qi));
#else
  nn.d0 =                                 __umul32(n.d0, qi);
  nn.d1 = __add_cc (__umul32hi(n.d0, qi), __umul32(n.d1, qi));
  nn.d2 = __addc   (__umul32hi(n.d1, qi), __umul32(n.d2, qi));
#endif
  
  res->d0 = __sub_cc (q.d0, nn.d0);
  res->d1 = __subc_cc(q.d1, nn.d1);
  res->d2 = __subc   (q.d2, nn.d2);

// perfect refinement not needed, barrett's modular reduction can handle numbers which are a little bit "too big".
/*  if(cmp_ge_96(*res,n))
  {
    sub_96(res, *res, n);
  }*/
}


#if __CUDA_ARCH__ >= FERMI
  #define KERNEL_MIN_BLOCKS 2
#else
  #define KERNEL_MIN_BLOCKS 1
#endif

__global__ void
#ifndef CHECKS_MODBASECASE
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS) mfaktc_barrett92(unsigned int exp, int96 k, unsigned int *k_tab, int shiftcount, int192 b, unsigned int *RES, int bit_max64)
#else
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS) mfaktc_barrett92(unsigned int exp, int96 k, unsigned int *k_tab, int shiftcount, int192 b, unsigned int *RES, int bit_max64, unsigned int *modbasecase_debug)
#endif
/*
computes 2^exp mod f
shiftcount is used for precomputing without mod
a is precomputed on host ONCE.

bit_max64 is bit_max - 64!
*/
{
  int96 exp96, f;
  int96 a, u;
  int192 tmp192;
  int96 tmp96;
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  float ff;
  int bit_max64_32 = 32 - bit_max64; /* used for bit shifting... */

  exp96.d2 = 0; exp96.d1 = exp >> 31; exp96.d0 = exp + exp;	// exp96 = 2 * exp

/* umad32 is slower here?! */
  k.d0 = __add_cc (k.d0, __umul32  (k_tab[index], NUM_CLASSES));
  k.d1 = __addc   (k.d1, __umul32hi(k_tab[index], NUM_CLASSES));	/* k is limited to 2^64 -1 so there is no need for k.d2 */

/* umad32 is slower here?! */
  f.d0 = 1 +                                  __umul32(k.d0, exp96.d0); /* exp96.d0 is even so there is no carry when adding 1 */
  f.d1 = __add_cc(__umul32hi(k.d0, exp96.d0), __umul32(k.d1, exp96.d0));
  f.d2 = __addc  (__umul32hi(k.d1, exp96.d0),                        0);

  if(exp96.d1) /* exp96.d1 is 0 or 1 */
  {
    f.d1 = __add_cc(f.d1, k.d0);
    f.d2 = __addc  (f.d2, k.d1);  
  }						// f = 2 * k * exp + 1

/*
ff = f as float, needed in mod_192_96().
Precalculated here since it is the same for all steps in the following loop */
  ff= __uint2float_rn(f.d2);
  ff= ff * 4294967296.0f + __uint2float_rn(f.d1);		// f.d0 ingored because lower limit for this kernel are 64 bit which yields at least 32 significant digits without f.d0!

  ff=__int_as_float(0x3f7ffffb) / ff;		// just a little bit below 1.0f so we allways underestimate the quotient
        
  tmp192.d5 = 0x40000000 >> ((bit_max64_32-1) << 1);			// tmp192 = 2^(2*bit_max)
  tmp192.d4 = 0; tmp192.d3 = 0; tmp192.d2 = 0; tmp192.d1 = 0; tmp192.d0 = 0;

#ifndef CHECKS_MODBASECASE
  div_192_96(&u,tmp192,f,ff);						// u = floor(tmp192 / f)
#else
  div_192_96(&u,tmp192,f,ff,modbasecase_debug);				// u = floor(tmp192 / f)
#endif

  a.d0 = (b.d2 >> bit_max64) + (b.d3 << bit_max64_32);			// a = b / (2^bit_max)
  a.d1 = (b.d3 >> bit_max64) + (b.d4 << bit_max64_32);
  a.d2 = (b.d4 >> bit_max64) + (b.d5 << bit_max64_32);

  mul_96_192_no_low2(&tmp192, a, u);					// tmp192 = (b / (2^bit_max)) * u # at least close to ;)

  a.d0 = (tmp192.d2 >> bit_max64) + (tmp192.d3 << bit_max64_32);	// a = ((b / (2^bit_max)) * u) / (2^bit_max)
  a.d1 = (tmp192.d3 >> bit_max64) + (tmp192.d4 << bit_max64_32);
  a.d2 = (tmp192.d4 >> bit_max64) + (tmp192.d5 << bit_max64_32);

  mul_96(&tmp96, a, f);							// tmp96 = (((b / (2^bit_max)) * u) / (2^bit_max)) * f

  tmp96.d0 = __sub_cc (b.d0, tmp96.d0);					// we do not need the upper digits of b and tmp96 because they are 0 after this subtraction!
  tmp96.d1 = __subc_cc(b.d1, tmp96.d1);
  tmp96.d2 = __subc   (b.d2, tmp96.d2);

#ifndef CHECKS_MODBASECASE
  mod_simple_96(&a, tmp96, f, ff);					// adjustment, plain barrett returns N = AB mod M where N < 3M!
#else
  int limit = 6;
  mod_simple_96(&a, tmp96, f, ff, bit_max64, limit, modbasecase_debug);
#endif
  
  exp<<= 32 - shiftcount;
  while(exp)
  {
    square_96_192(&b, a);						// b = a^2

    a.d0 = (b.d2 >> bit_max64) + (b.d3 << bit_max64_32);		// a = b / (2^bit_max)
    a.d1 = (b.d3 >> bit_max64) + (b.d4 << bit_max64_32);
    a.d2 = (b.d4 >> bit_max64) + (b.d5 << bit_max64_32);

    mul_96_192_no_low2(&tmp192, a, u);					// tmp192 = (b / (2^bit_max)) * u # at least close to ;)

    a.d0 = (tmp192.d2 >> bit_max64) + (tmp192.d3 << bit_max64_32);	// a = ((b / (2^bit_max)) * u) / (2^bit_max)
    a.d1 = (tmp192.d3 >> bit_max64) + (tmp192.d4 << bit_max64_32);
    a.d2 = (tmp192.d4 >> bit_max64) + (tmp192.d5 << bit_max64_32);

    mul_96(&tmp96, a, f);						// tmp96 = (((b / (2^bit_max)) * u) / (2^bit_max)) * f

    tmp96.d0 = __sub_cc (b.d0, tmp96.d0);				// we do not need the upper digits of b and tmp96 because they are 0 after this subtraction!
    tmp96.d1 = __subc_cc(b.d1, tmp96.d1);
    tmp96.d2 = __subc   (b.d2, tmp96.d2);
    
    if(exp&0x80000000)shl_96(&tmp96);					// "optional multiply by 2" in Prime 95 documentation

#ifndef CHECKS_MODBASECASE
    mod_simple_96(&a, tmp96, f, ff);					// adjustment, plain barrett returns N = AB mod M where N < 3M!
#else
    int limit = 6;
    mod_simple_96(&a, tmp96, f, ff, bit_max64, limit, modbasecase_debug);
#endif

    exp<<=1;
  }
  
  if(cmp_ge_96(a,f))				// final adjustment in case a >= f
  {
    sub_96(&a, a, f);
  }

#if defined CHECKS_MODBASECASE && defined USE_DEVICE_PRINTF && __CUDA_ARCH__ >= FERMI
  if(cmp_ge_96(a,f) && f.d2)
  {
    printf("EEEEEK, final a is >= f\n");
  }
#endif
  
/* finally check if we found a factor and write the factor to RES[] */
  if( ((a.d2|a.d1)==0 && a.d0==1) )
  {
/* in contrast to the other kernels the two barrett based kernels are only allowed for factors above 2^64 so there is no need to check for f = 1! */  
    index=atomicInc(&RES[0],10000);
    if(index<10)				/* limit to 10 factors per class */
    {
      RES[index*3 + 1]=f.d2;
      RES[index*3 + 2]=f.d1;
      RES[index*3 + 3]=f.d0;
    }
  }
}

__global__ void
#ifndef CHECKS_MODBASECASE
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS) mfaktc_barrett79(unsigned int exp, int96 k, unsigned int *k_tab, int shiftcount, int192 b, unsigned int *RES)
#else
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS) mfaktc_barrett79(unsigned int exp, int96 k, unsigned int *k_tab, int shiftcount, int192 b, unsigned int *RES, int bit_max64, unsigned int *modbasecase_debug)
#endif
/*
computes 2^exp mod f
shiftcount is used for precomputing without mod
a is precomputed on host ONCE.
*/
{
  int96 exp96, f;
  int96 a, u;
  int192 tmp192;
  int96 tmp96;
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  float ff;

  exp96.d2 = 0; exp96.d1 = exp >> 31; exp96.d0 = exp + exp;	// exp96 = 2 * exp

/* umad32 is slower here?! */
  k.d0 = __add_cc (k.d0, __umul32  (k_tab[index], NUM_CLASSES));
  k.d1 = __addc   (k.d1, __umul32hi(k_tab[index], NUM_CLASSES));	/* k is limited to 2^64 -1 so there is no need for k.d2 */

/* umad32 is slower here?! */
  f.d0 = 1 +                                  __umul32(k.d0, exp96.d0); /* exp96.d0 is even so there is no carry when adding 1 */
  f.d1 = __add_cc(__umul32hi(k.d0, exp96.d0), __umul32(k.d1, exp96.d0));
  f.d2 = __addc  (__umul32hi(k.d1, exp96.d0),                        0);

  if(exp96.d1) /* exp96.d1 is 0 or 1 */
  {
    f.d1 = __add_cc(f.d1, k.d0);
    f.d2 = __addc  (f.d2, k.d1);  
  }						// f = 2 * k * exp + 1

/*
ff = f as float, needed in mod_160_96().
Precalculated here since it is the same for all steps in the following loop */
  ff= __uint2float_rn(f.d2);
  ff= ff * 4294967296.0f + __uint2float_rn(f.d1);		// f.d0 ingored because lower limit for this kernel are 64 bit which yields at least 32 significant digits without f.d0!

  ff=__int_as_float(0x3f7ffffb) / ff;		// just a little bit below 1.0f so we allways underestimate the quotient

  tmp192.d4 = 0xFFFFFFFF;						// tmp is nearly 2^(80*2)
  tmp192.d3 = 0xFFFFFFFF;
  tmp192.d2 = 0xFFFFFFFF;
  tmp192.d1 = 0xFFFFFFFF;
  tmp192.d0 = 0xFFFFFFFF;

#ifndef CHECKS_MODBASECASE
  div_160_96(&u,tmp192,f,ff);						// u = floor(2^(80*2) / f)
#else
  div_160_96(&u,tmp192,f,ff,modbasecase_debug);				// u = floor(2^(80*2) / f)
#endif

  a.d0 = b.d2;// & 0xFFFF8000;						// a = b / (2^80) (the result is leftshifted by 15 bits, this is corrected later)
  a.d1 = b.d3;
  a.d2 = b.d4;

  mul_96_192_no_low3(&tmp192, a, u);					// tmp192 = (b / (2^80)) * u

  a.d0 = tmp192.d3;							// a = ((b / (2^80)) * u) / (2^80)
  a.d1 = tmp192.d4;							// this includes the shiftleft by 32 bits, read above...
  a.d2 = tmp192.d5;

  mul_96(&tmp96, a, f);							// tmp96 = (((b / (2^80)) * u) / (2^80)) * f

  tmp96.d0 = __sub_cc (b.d0, tmp96.d0);					// we do not need the upper digits of b and tmp96 because they are 0 after this subtraction!
  tmp96.d1 = __subc_cc(b.d1, tmp96.d1);
  tmp96.d2 = __subc   (b.d2, tmp96.d2);

#ifndef CHECKS_MODBASECASE
  mod_simple_96(&a, tmp96, f, ff);					// adjustment, plain barrett returns N = AB mod M where N < 3M!
#else                                                                   // because of loss of accuracy in mul_96_192_no_low3() the error will increase
  int limit = 6;
  if(bit_max64 == 15) limit = 9;					// bit_max == 79, due to decreased accuracy of mul_96_192_no_low3() above we need a higher threshold
  mod_simple_96(&a, tmp96, f, ff, 79 - 64, limit << (15 - bit_max64), modbasecase_debug);	// limit is 6 * 2^(79 - bit_max)
#endif

  
  exp<<= 32 - shiftcount;
  while(exp)
  {
    square_96_160(&b, a);						// b = a^2

    a.d0 = b.d2;// & 0xFFFF8000;					// a = b / (2^80) (the result is leftshifted by 15 bits, this is corrected later)
    a.d1 = b.d3;
    a.d2 = b.d4;
    
    mul_96_192_no_low3(&tmp192, a, u);					// tmp192 = (b / (2^80)) * u

    a.d0 = tmp192.d3;							// a = ((b / (2^80)) * u) / (2^80)
    a.d1 = tmp192.d4;							// this includes the shiftleft by 32 bits, read above...
    a.d2 = tmp192.d5;

    mul_96(&tmp96, a, f);						// tmp96 = (((b / (2^80)) * u) / (2^80)) * f

    tmp96.d0 = __sub_cc (b.d0, tmp96.d0);				// we do not need the upper digits of b and tmp96 because they are 0 after this subtraction!
    tmp96.d1 = __subc_cc(b.d1, tmp96.d1);
    tmp96.d2 = __subc   (b.d2, tmp96.d2);
    
    if(exp&0x80000000)shl_96(&tmp96);					// "optional multiply by 2" in Prime 95 documentation

#ifndef CHECKS_MODBASECASE
    mod_simple_96(&a, tmp96, f, ff);					// adjustment, plain barrett returns N = AB mod M where N < 3M!
#else                                                                   // because of loss of accuracy in mul_96_192_no_low3() the error will increase
    int limit = 6;
    if(bit_max64 == 15) limit = 9;					// bit_max == 79, due to decreased accuracy of mul_96_192_no_low3() above we need a higher threshold
    mod_simple_96(&a, tmp96, f, ff, 79 - 64, limit << (15 - bit_max64), modbasecase_debug);	// limit is 6 * 2^(79 - bit_max)
#endif


//    exp<<=1;
    exp += exp;
  }
  if(cmp_ge_96(a,f))							// final adjustment in case a >= f
  {
    sub_96(&a, a, f);
  }
  
#if defined CHECKS_MODBASECASE && defined USE_DEVICE_PRINTF && __CUDA_ARCH__ >= FERMI
  if(cmp_ge_96(a,f) && f.d2)						// factors < 2^64 are not supported by this kernel
  {
    printf("EEEEEK, final a is >= f\n");
  }
#endif
  
/* finally check if we found a factor and write the factor to RES[] */
  if( ((a.d2|a.d1)==0 && a.d0==1) )
  {
/* in contrast to the other kernels the two barrett based kernels are only allowed for factors above 2^64 so there is no need to check for f = 1! */  
    index=atomicInc(&RES[0],10000);
    if(index<10)				/* limit to 10 factors per class */
    {
      RES[index*3 + 1]=f.d2;
      RES[index*3 + 2]=f.d1;
      RES[index*3 + 3]=f.d0;
    }
  }
}


__global__ void
#ifndef CHECKS_MODBASECASE
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS) mfaktc_barrett76(unsigned int exp, int96 k, unsigned int *k_tab, int shiftcount, int192 b, unsigned int *RES)
#else
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS) mfaktc_barrett76(unsigned int exp, int96 k, unsigned int *k_tab, int shiftcount, int192 b, unsigned int *RES, int bit_max64, unsigned int *modbasecase_debug)
#endif
/*
computes 2^exp mod f
shiftcount is used for precomputing without mod
a is precomputed on host ONCE.

This kernel is a copy of mfaktc_barrett79(), the difference is the simplified correction step.
*/
{
  int96 exp96, f;
  int96 a, u;
  int192 tmp192;
  int96 tmp96;
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  float ff;
  
  exp96.d2 = 0; exp96.d1 = exp >> 31; exp96.d0 = exp + exp;	// exp96 = 2 * exp

#if (__CUDA_ARCH__ >= FERMI) && (CUDART_VERSION >= 4010) /* multiply-add with carry is not available on CC 1.x devices and before CUDA 4.1 */
  k.d0 = __umad32_cc(k_tab[index], NUM_CLASSES, k.d0);
  k.d1 = __umad32hic(k_tab[index], NUM_CLASSES, k.d1);
#else
  k.d0 = __add_cc (k.d0, __umul32  (k_tab[index], NUM_CLASSES));
  k.d1 = __addc   (k.d1, __umul32hi(k_tab[index], NUM_CLASSES));	/* k is limited to 2^64 -1 so there is no need for k.d2 */
#endif  

/* umad32 is slower here?! */
  f.d0 = 1 +                                  __umul32(k.d0, exp96.d0); /* exp96.d0 is even so there is no carry when adding 1 */
  f.d1 = __add_cc(__umul32hi(k.d0, exp96.d0), __umul32(k.d1, exp96.d0));
  f.d2 = __addc  (__umul32hi(k.d1, exp96.d0),                        0);

  if(exp96.d1) /* exp96.d1 is 0 or 1 */
  {
    f.d1 = __add_cc(f.d1, k.d0);
    f.d2 = __addc  (f.d2, k.d1);  
  }						// f = 2 * k * exp + 1
  
/*
ff = f as float, needed in mod_160_96().
Precalculated here since it is the same for all steps in the following loop */
  ff= __uint2float_rn(f.d2);
  ff= ff * 4294967296.0f + __uint2float_rn(f.d1);		// f.d0 ingored because lower limit for this kernel are 64 bit which yields at least 32 significant digits without f.d0!

  ff=__int_as_float(0x3f7ffffb) / ff;		// just a little bit below 1.0f so we allways underestimate the quotient

  tmp192.d4 = 0xFFFFFFFF;						// tmp is nearly 2^(80*2)
  tmp192.d3 = 0xFFFFFFFF;
  tmp192.d2 = 0xFFFFFFFF;
  tmp192.d1 = 0xFFFFFFFF;
  tmp192.d0 = 0xFFFFFFFF;

#ifndef CHECKS_MODBASECASE
  div_160_96(&u,tmp192,f,ff);						// u = floor(2^(80*2) / f)
#else
  div_160_96(&u,tmp192,f,ff,modbasecase_debug);				// u = floor(2^(80*2) / f)
#endif

  a.d0 = b.d2;// & 0xFFFF8000;						// a = b / (2^80) (the result is leftshifted by 15 bits, this is corrected later)
  a.d1 = b.d3;
  a.d2 = b.d4;

  mul_96_192_no_low3(&tmp192, a, u);					// tmp192 = (b / (2^80)) * u

  a.d0 = tmp192.d3;							// a = ((b / (2^80)) * u) / (2^80)
  a.d1 = tmp192.d4;							// this includes the shiftleft by 32 bits, read above...
  a.d2 = tmp192.d5;

  mul_96(&tmp96, a, f);							// tmp96 = (((b / (2^80)) * u) / (2^80)) * f
  
  a.d0 = __sub_cc (b.d0, tmp96.d0);					// we do not need the upper digits of b and tmp96 because they are 0 after this subtraction!
  a.d1 = __subc_cc(b.d1, tmp96.d1);
  a.d2 = __subc   (b.d2, tmp96.d2);
  
#ifdef CHECKS_MODBASECASE
  if(f.d2)								// check only when f is >= 2^64 (f <= 2^64 is not supported by this kernel
  {
    MODBASECASE_QI_ERROR(0xC000, 99, a.d2, 13)				// a should never have a value >= 2^80, if so square_96_160() will overflow!
  }									// this will warn whenever a becomes close to 2^80
#endif

  exp<<= 32 - shiftcount;
  while(exp)
  {
    square_96_160(&b, a);						// b = a^2

    a.d0 = b.d2;// & 0xFFFF8000;					// a = b / (2^80) (the result is leftshifted by 15 bits, this is corrected later)
    a.d1 = b.d3;
    a.d2 = b.d4;
    
    mul_96_192_no_low3(&tmp192, a, u);					// tmp192 = (b / (2^80)) * u

    a.d0 = tmp192.d3;							// a = ((b / (2^80)) * u) / (2^80)
    a.d1 = tmp192.d4;							// this includes the shiftleft by 32 bits, read above...
    a.d2 = tmp192.d5;

    mul_96(&tmp96, a, f);						// tmp96 = (((b / (2^80)) * u) / (2^80)) * f

    a.d0 = __sub_cc (b.d0, tmp96.d0);					// we do not need the upper digits of b and tmp96 because they are 0 after this subtraction!
    a.d1 = __subc_cc(b.d1, tmp96.d1);
    a.d2 = __subc   (b.d2, tmp96.d2);

    if(exp&0x80000000)shl_96(&a);					// "optional multiply by 2" in Prime 95 documentation

#ifdef CHECKS_MODBASECASE
    if(f.d2)								// check only when f is >= 2^64 (f <= 2^64 is not supported by this kernel
    {
      MODBASECASE_QI_ERROR(0xC000, 99, a.d2, 13)			// a should never have a value >= 2^80, if so square_96_160() will overflow!
    }									// this will warn whenever a becomes close to 2^80
#endif

//    exp<<=1;
    exp += exp;
  }
  
  tmp96.d0 = a.d0;
  tmp96.d1 = a.d1;
  tmp96.d2 = a.d2;

#ifndef CHECKS_MODBASECASE
  mod_simple_96(&a, tmp96, f, ff);					// adjustment, plain barrett returns N = AB mod M where N < 3M!
#else                                                                   // because of loss of accuracy in mul_96_192_no_low3() the error will increase
  int limit = 6;
  if(bit_max64 == 15) limit = 9;					// bit_max == 79, due to decreased accuracy of mul_96_192_no_low3() above we need a higher threshold
  mod_simple_96(&a, tmp96, f, ff, 79 - 64, limit << (15 - bit_max64), modbasecase_debug);	// limit is 6 * 2^(79 - bit_max)
#endif

  if(cmp_ge_96(a,f))							// final adjustment in case a >= f
  {
    sub_96(&a, a, f);
  }
  
#if defined CHECKS_MODBASECASE && defined USE_DEVICE_PRINTF && __CUDA_ARCH__ >= FERMI
  if(cmp_ge_96(a,f) && f.d2)						// factors < 2^64 are not supported by this kernel
  {
    printf("EEEEEK, final a is >= f\n");
  }
#endif
  
/* finally check if we found a factor and write the factor to RES[] */
  if( ((a.d2|a.d1)==0 && a.d0==1) )
  {
/* in contrast to the other kernels the two barrett based kernels are only allowed for factors above 2^64 so there is no need to check for f = 1! */  
    index=atomicInc(&RES[0],10000);
    if(index<10)				/* limit to 10 factors per class */
    {
      RES[index*3 + 1]=f.d2;
      RES[index*3 + 2]=f.d1;
      RES[index*3 + 3]=f.d0;
    }
  }
}

#define TF_BARRETT

#define TF_BARRETT_92BIT
#include "tf_common.cu"
#undef TF_BARRETT_92BIT

#define TF_BARRETT_79BIT
#include "tf_common.cu"
#undef TF_BARRETT_79BIT

#define TF_BARRETT_76BIT
#include "tf_common.cu"
#undef TF_BARRETT_76BIT

#undef TF_BARRETT
