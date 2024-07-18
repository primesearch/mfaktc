/*
This file is part of mfaktc.
Copyright (C) 2009, 2010, 2011, 2012, 2013  Oliver Weihe (o.weihe@t-online.de)

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


__device__ static void check_factor96(int96 f, int96 a, unsigned int *RES)
/* Check whether f is a factor or not. If f != 1 and a == 1 then f is a factor,
in this case f is written into the RES array. */
{
  int index;
#ifdef WAGSTAFF
  if(a.d2 == f.d2 && a.d1 == f.d1 && a.d0 == (f.d0 - 1))
#else /* Mersennes */
  if((a.d2|a.d1) == 0 && a.d0 == 1)
#endif
  {
    if(f.d2 != 0 || f.d1 != 0 || f.d0 != 1)	/* 1 isn't really a factor ;) */
    {
      index=atomicInc(&RES[0], 10000);
      if(index < 10)				/* limit to 10 factors per class */
      {
        RES[index * 3 + 1] = f.d2;
        RES[index * 3 + 2] = f.d1;
        RES[index * 3 + 3] = f.d0;
      }
    }
  }
}


__device__ static void check_big_factor96(int96 f, int96 a, unsigned int *RES)
/* Similar to check_factor96() but without checking f != 1. This is a little
bit faster but only safe for kernel which have a lower limit well above 1. The
barrett based kernels have a lower limit of 2^64 so this function is used
there. */
{
  int index;
#ifdef WAGSTAFF
  if(a.d2 == f.d2 && a.d1 == f.d1 && a.d0 == (f.d0 - 1))
#else /* Mersennes */
  if((a.d2|a.d1) == 0 && a.d0 == 1)
#endif
  {
    index = atomicInc(&RES[0], 10000);
    if(index < 10)				/* limit to 10 factors per class */
    {
      RES[index * 3 + 1] = f.d2;
      RES[index * 3 + 2] = f.d1;
      RES[index * 3 + 3] = f.d0;
    }
  }
}


__device__ static void create_FC96(int96 *f, unsigned int exp, int96 k, unsigned int k_offset)
/* calculates f = 2 * (k+k_offset) * exp + 1 */
{
  int96 exp96;

  exp96.d1 = exp >> 31;
  exp96.d0 = exp << 1;			// exp96 = 2 * exp

  k.d0 = __add_cc (k.d0, __umul32  (k_offset, NUM_CLASSES));
  k.d1 = __addc   (k.d1, __umul32hi(k_offset, NUM_CLASSES));	/* k is limited to 2^64 -1 so there is no need for k.d2 */

//  mul_96(&f,k,exp96);					// f = 2 * k * exp
//  f.d0 += 1;						// f = 2 * k * exp + 1

  f->d0 = 1 +                                  __umul32(k.d0, exp96.d0);
  f->d1 = __add_cc(__umul32hi(k.d0, exp96.d0), __umul32(k.d1, exp96.d0));
  f->d2 = __addc  (__umul32hi(k.d1, exp96.d0),                        0);

  if(exp96.d1) /* exp96.d1 is 0 or 1 */
  {
    f->d1 = __add_cc(f->d1, k.d0);
    f->d2 = __addc  (f->d2, k.d1);  
  }							// f = 2 * k * exp + 1
}


__device__ static void create_FC96_mad(int96 *f, unsigned int exp, int96 k, unsigned int k_offset)
/* similar to create_FC96(), this versions uses multiply-add with carry which
is faster for _SOME_ kernels. */
{
#if (__CUDA_ARCH__ < FERMI) || (CUDART_VERSION < 4010) /* multiply-add with carry is not available on CC 1.x devices and before CUDA 4.1 */
  create_FC96(f, exp, k, k_offset);
#else
  int96 exp96;

  exp96.d1 = exp >> 31;
  exp96.d0 = exp << 1;			// exp96 = 2 * exp

  k.d0 = __umad32_cc(k_offset, NUM_CLASSES, k.d0);
  k.d1 = __umad32hic(k_offset, NUM_CLASSES, k.d1);
        
  /* umad32 is slower here?! */
  f->d0 = 1 +                                  __umul32(k.d0, exp96.d0);
  f->d1 = __add_cc(__umul32hi(k.d0, exp96.d0), __umul32(k.d1, exp96.d0));
  f->d2 = __addc  (__umul32hi(k.d1, exp96.d0),                        0);

  if(exp96.d1) /* exp96.d1 is 0 or 1 */
  {
    f->d1 = __add_cc(f->d1, k.d0);
    f->d2 = __addc  (f->d2, k.d1);  
  }							// f = 2 * k * exp + 1
#endif
}


#ifndef DEBUG_GPU_MATH
__device__ static void mod_simple_96(int96 *res, int96 q, int96 n, float nf)
#else
__device__ static void mod_simple_96(int96 *res, int96 q, int96 n, float nf, int bit_min64, int bit_max64, unsigned int limit, unsigned int *modbasecase_debug)
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

#ifdef DEBUG_GPU_MATH
/* Barrett based kernels are made for factor candidates above 2^64,
at least the 79bit variant fails on factor candidates less than 2^64!
Lets ignore those errors...
Factor candidates below 2^64 can occur when TFing from 2^64 to 2^65, the
first candidate in each class can be smaller than 2^64.
This is NOT an issue because those exponents should be TFed to 2^64 with a
kernel which can handle those "small" candidates before starting TF from
2^64 to 2^65. So in worst case we have a false positive which is caught
easily by the primenet server.
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
//  if(n.d2 != 0 && n.d2 < (1 << bit_max64))
  if(n.d2 >= (1 << bit_min64) && n.d2 < (1 << bit_max64))
  {
    MODBASECASE_QI_ERROR(limit, 100, qi, 12);
  #if defined USE_DEVICE_PRINTF && __CUDA_ARCH__ >= FERMI
    if(qi > limit)
    {
      printf("n = 0x %08X %08X %08X\n", n.d2, n.d1, n.d0);
      printf("bit_min = %d (%08X)\n", bit_min64, 1<<bit_min64);
    }
  #endif
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


__device__ static void mod_simple_96_and_check_big_factor96(int96 q, int96 n, float nf, unsigned int *RES)
/*
This function is a combination of mod_simple_96(), check_big_factor96() and an additional correction step.
If q mod n == 1 then n is a factor and written into the RES array.
q must be less than 100n!
*/
{
  float qf;
  unsigned int qi, res;
  int96 nn;

  qf = __uint2float_rn(q.d2);
  qf = qf * 4294967296.0f + __uint2float_rn(q.d1);

  qi=__float2uint_rz(qf*nf);
  
#ifdef WAGSTAFF
  qi++; /* cause in underflow in subtraction so we can check for (-1) instead of (q - 1) */
#endif
/* at this point the quotient still is sometimes to small (the error is 1 in this case)
--> final res odd and qi correct: n might be a factor
    final res odd and qi too small: n can't be a factor (because the correct res is even)
    final res even and qi correct: n can't be a factor (because the res is even)
    final res even and qi too small: n might be a factor
so we compare the LSB of qi and q.d0, if they are the same (both even or both odd) the res (without correction) would be even. In this case increment qi by one.*/

  qi += ((~qi) ^ q.d0) & 1;
 
  nn.d0 = __umul32(n.d0, qi);

#ifdef WAGSTAFF
  if((q.d0 - nn.d0) == 0xFFFFFFFF) /* is the lowest word of the result -1 (only in this case n might be a factor) */
#else
  if((q.d0 - nn.d0) == 1) /* is the lowest word of the result 1 (only in this case n might be a factor) */
#endif
  {
#if (__CUDA_ARCH__ >= FERMI) && (CUDART_VERSION >= 4010) /* multiply-add with carry is not available on CC 1.x devices and before CUDA 4.1 */
    nn.d1 = __umad32hi_cc (n.d0, qi, __umul32(n.d1, qi));
    nn.d2 = __umad32hic   (n.d1, qi, __umul32(n.d2, qi));
#else
    nn.d1 = __add_cc (__umul32hi(n.d0, qi), __umul32(n.d1, qi));
    nn.d2 = __addc   (__umul32hi(n.d1, qi), __umul32(n.d2, qi));
#endif

#ifdef WAGSTAFF
    res  = __sub_cc (q.d0, nn.d0);
    res &= __subc_cc(q.d1, nn.d1);
    res &= __subc   (q.d2, nn.d2);
    
    if(res == 0xFFFFFFFF)
#else /* Mersennes */
    nn.d0++;
    res  = __sub_cc (q.d0, nn.d0);
//           __sub_cc (q.d0, nn.d0); /* the compiler (release 5.0, V0.2.1221) doesn't want to execute this so we need the TWO lines above... */
    res |= __subc_cc(q.d1, nn.d1);
    res |= __subc   (q.d2, nn.d2);

    if(res == 0)
#endif
    {
      int index;
      index = atomicInc(&RES[0], 10000);
      if(index < 10)                              /* limit to 10 factors per class */
      {
        RES[index * 3 + 1] = n.d2;
        RES[index * 3 + 2] = n.d1;
        RES[index * 3 + 3] = n.d0;
      }
    }
  }
}
