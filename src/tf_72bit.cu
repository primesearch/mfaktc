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


__device__ static void mul_24_48(unsigned int *res_hi, unsigned int *res_lo, unsigned int a, unsigned int b)
/* res_hi*(2^24) + res_lo = a * b */
{
  *res_lo = __umul24(a,b) & 0xFFFFFF;
  *res_hi = __umul24hi(a,b) >> 8;
}


__device__ static int cmp_ge_72(int72 a, int72 b)
/* checks if a is greater or equal than b */
{
  if(a.d2 == b.d2)
  {
    if(a.d1 == b.d1)return(a.d0 >= b.d0);
    else            return(a.d1 >  b.d1);
  }
  else              return(a.d2 >  b.d2);
}


__device__ static void sub_72(int72 *res, int72 a, int72 b)
/* a must be greater or equal b!
res = a - b */
{
  res->d0 = __sub_cc (a.d0, b.d0) & 0xFFFFFF;
  res->d1 = __subc_cc(a.d1, b.d1) & 0xFFFFFF;
  res->d2 = __subc   (a.d2, b.d2) & 0xFFFFFF;
}


__device__ static void mul_72(int72 *res, int72 a, int72 b)
/* res = (a * b) mod (2^72) */
{
  unsigned int hi,lo;

  mul_24_48(&hi, &lo, a.d0, b.d0);
  res->d0 = lo;
  res->d1 = hi;

  mul_24_48(&hi, &lo, a.d1, b.d0);
  res->d1 += lo;
  res->d2 = hi;

  mul_24_48(&hi, &lo, a.d0, b.d1);
  res->d1 += lo;
  res->d2 += hi;

  res->d2 += __umul24(a.d2,b.d0);

  res->d2 += __umul24(a.d1,b.d1);

  res->d2 += __umul24(a.d0,b.d2);

//  no need to carry res->d0

  res->d2 += res->d1 >> 24;
  res->d1 &= 0xFFFFFF;

  res->d2 &= 0xFFFFFF;
}


__device__ static void square_72_144(int144 *res, int72 a)
/* res = a^2 */
{
  res->d0  =  __umul24(a.d0, a.d0)       & 0xFFFFFF;
  res->d1  =  __umul24hi(a.d0, a.d0) >> 8;
  
  res->d1 += (__umul24(a.d1, a.d0) << 1) & 0xFFFFFF;
  res->d2  =  __umul24hi(a.d1, a.d0) >> 7;

  res->d2 += (__umul24(a.d2, a.d0) << 1) & 0xFFFFFF;
  res->d3  =  __umul24hi(a.d2, a.d0) >> 7;
  
  res->d2 +=  __umul24(a.d1, a.d1)       & 0xFFFFFF;
  res->d3 +=  __umul24hi(a.d1, a.d1) >> 8;
  
  res->d3 += (__umul24(a.d2, a.d1) << 1) & 0xFFFFFF;
  res->d4  =  __umul24hi(a.d2, a.d1) >> 7;

  res->d4 +=  __umul24(a.d2, a.d2)       & 0xFFFFFF;
  res->d5  =  __umul24hi(a.d2, a.d2) >> 8;

/*  res->d0 doesn't need carry */
  res->d2 += res->d1 >> 24;
  res->d1 &= 0xFFFFFF;

  res->d3 += res->d2 >> 24;
  res->d2 &= 0xFFFFFF;

  res->d4 += res->d3 >> 24;
  res->d3 &= 0xFFFFFF;

  res->d5 += res->d4 >> 24;
  res->d4 &= 0xFFFFFF;
/*  res->d5 doesn't need carry */
}


__device__ static void square_72_144_shl(int144 *res, int72 a)
/* res = 2* a^2 */
{
  res->d0  = (__umul24(a.d0, a.d0) << 1) & 0xFFFFFF;
  res->d1  =  __umul24hi(a.d0, a.d0) >> 7;
  
  res->d1 += (__umul24(a.d1, a.d0) << 2) & 0xFFFFFF;
  res->d2  =  __umul24hi(a.d1, a.d0) >> 6;

  res->d2 += (__umul24(a.d2, a.d0) << 2) & 0xFFFFFF;
  res->d3  =  __umul24hi(a.d2, a.d0) >> 6;
  
  res->d2 += (__umul24(a.d1, a.d1) << 1) & 0xFFFFFF;
  res->d3 +=  __umul24hi(a.d1, a.d1) >> 7;
  
  res->d3 += (__umul24(a.d2, a.d1) << 2) & 0xFFFFFF;
  res->d4  =  __umul24hi(a.d2, a.d1) >> 6;

  res->d4 += (__umul24(a.d2, a.d2) << 1) & 0xFFFFFF;
  res->d5  =  __umul24hi(a.d2, a.d2) >> 7;

/*  res->d0 doesn't need carry */
  res->d2 += res->d1 >> 24;
  res->d1 &= 0xFFFFFF;

  res->d3 += res->d2 >> 24;
  res->d2 &= 0xFFFFFF;

  res->d4 += res->d3 >> 24;
  res->d3 &= 0xFFFFFF;

  res->d5 += res->d4 >> 24;
  res->d4 &= 0xFFFFFF;
/*  res->d5 doesn't need carry */
}


#ifndef DEBUG_GPU_MATH
__device__ static void mod_144_72(int72 *res, int144 q, int72 n, float nf)
#else
__device__ static void mod_144_72(int72 *res, int144 q, int72 n, float nf, unsigned int *modbasecase_debug)
#endif
/* res = q mod n */
{
  float qf;
  unsigned int qi;
  int144 nn;

/********** Step 1, Offset 2^51 (2*24 + 3) **********/
  qf= __uint2float_rn(q.d5);
  qf= qf * 16777216.0f + __uint2float_rn(q.d4);
  qf= qf * 16777216.0f + __uint2float_rn(q.d3);
  qf*= 2097152.0f;

  qi=__float2uint_rz(qf*nf);

  MODBASECASE_QI_ERROR(1<<22, 1, qi, 0);

//  nn.d0=0;
//  nn.d1=0;
// nn = n * qi AND shiftleft 3 bits at once, carry is done later
  nn.d2  = (__umul24(n.d0, qi) << 3) & 0xFFFFFF;
  nn.d3  =  __umul24hi(n.d0, qi) >> 5;

  nn.d3 += (__umul24(n.d1, qi) << 3) & 0xFFFFFF;
  nn.d4  =  __umul24hi(n.d1, qi) >> 5;

  nn.d4 += (__umul24(n.d2, qi) << 3) & 0xFFFFFF;
  nn.d5  =  __umul24hi(n.d2, qi) >> 5;


/* do carry */
  nn.d4 += nn.d3 >> 24; nn.d3 &= 0xFFFFFF;
  nn.d5 += nn.d4 >> 24; nn.d4 &= 0xFFFFFF;
  
  MODBASECASE_VALUE_BIG_ERROR(0xFFFFFF, "nn.d5", 1, nn.d5, 1);

/*  q = q - nn */
/* subtraction using sub.cc.u32, subc.cc.u32 and subc.u32 instructions */
  q.d2 = __sub_cc (q.d2, nn.d2) & 0xFFFFFF;
  q.d3 = __subc_cc(q.d3, nn.d3) & 0xFFFFFF;
  q.d4 = __subc_cc(q.d4, nn.d4) & 0xFFFFFF;
  q.d5 = __subc   (q.d5, nn.d5);

/********** Step 2, Offset 2^31 (1*24 + 7) **********/
  qf= __uint2float_rn(q.d5);
  qf= qf * 16777216.0f + __uint2float_rn(q.d4);
  qf= qf * 16777216.0f + __uint2float_rn(q.d3);
  qf= qf * 16777216.0f + __uint2float_rn(q.d2);
  qf*= 131072.0f;

  qi=__float2uint_rz(qf*nf);

  MODBASECASE_QI_ERROR(1<<22, 2, qi, 2);

//  nn.d0=0;
// nn = n * qi AND shiftleft 7 bits at once, carry is done later
  nn.d1  = (__umul24(n.d0, qi) << 7) & 0xFFFFFF;
  nn.d2  =  __umul24hi(n.d0, qi) >> 1;

  nn.d2 += (__umul24(n.d1, qi) << 7) & 0xFFFFFF;
  nn.d3  =  __umul24hi(n.d1, qi) >> 1;

  nn.d3 += (__umul24(n.d2, qi) << 7) & 0xFFFFFF;
  nn.d4  =  __umul24hi(n.d2, qi) >> 1;
  #ifdef DEBUG_GPU_MATH
  nn.d5=0;
  #endif
  
/* do carry */
  nn.d3 += nn.d2 >> 24; nn.d2 &= 0xFFFFFF;
  nn.d4 += nn.d3 >> 24; nn.d3 &= 0xFFFFFF;
#ifdef DEBUG_GPU_MATH
  nn.d5 += nn.d4 >> 24; nn.d4 &= 0xFFFFFF;
#endif  

/* q = q - nn */
/* subtraction using sub.cc.u32, subc.cc.u32 and subc.u32 instructions */
  q.d1 = __sub_cc (q.d1, nn.d1) & 0xFFFFFF;
  q.d2 = __subc_cc(q.d2, nn.d2) & 0xFFFFFF;
  q.d3 = __subc_cc(q.d3, nn.d3) & 0xFFFFFF;
#ifndef DEBUG_GPU_MATH  
  q.d4 = __subc   (q.d4, nn.d4) & 0xFFFFFF;
#else
  q.d4 = __subc_cc(q.d4, nn.d4) & 0xFFFFFF;
  q.d5 = __subc   (q.d5, nn.d5);
#endif

/********** Step 3, Offset 2^11 (0*24 + 11) **********/
  MODBASECASE_NONZERO_ERROR(q.d5, 3, 5, 3);

  qf= __uint2float_rn(q.d4);
  qf= qf * 16777216.0f + __uint2float_rn(q.d3);
  qf= qf * 16777216.0f + __uint2float_rn(q.d2);
  qf= qf * 16777216.0f + __uint2float_rn(q.d1);
  qf*= 8192.0f;

  qi=__float2uint_rz(qf*nf);

  MODBASECASE_QI_ERROR(1<<22, 3, qi, 4);

//nn = n * qi, shiftleft is done later
  nn.d0 =                                      __umul24(n.d0, qi)               & 0xFFFFFF;
  nn.d1 = __add_cc (__umul24hi(n.d0, qi) >> 8, __umul24(n.d1, qi) | 0xFF000000) & 0xFFFFFF;
  nn.d2 = __addc_cc(__umul24hi(n.d1, qi) >> 8, __umul24(n.d2, qi) | 0xFF000000) & 0xFFFFFF;
  nn.d3 = __addc   (__umul24hi(n.d2, qi) >> 8, 0);

// shiftleft 11 bits
#ifdef DEBUG_GPU_MATH
  nn.d4 =                           nn.d3>>13;
  nn.d3 = ((nn.d3 & 0x1FFF)<<11) + (nn.d2>>13);
#else  
  nn.d3 = ( nn.d3          <<11) + (nn.d2>>13);	// we don't need to clear top bits here, this is done during q = q - nn
#endif  
  nn.d2 = ((nn.d2 & 0x1FFF)<<11) + (nn.d1>>13);
  nn.d1 = ((nn.d1 & 0x1FFF)<<11) + (nn.d0>>13);
  nn.d0 = ((nn.d0 & 0x1FFF)<<11);
  
/*  q = q - nn */
/* subtraction using sub.cc.u32, subc.cc.u32 and subc.u32 instructions */
  q.d0 = __sub_cc (q.d0, nn.d0) & 0xFFFFFF;
  q.d1 = __subc_cc(q.d1, nn.d1) & 0xFFFFFF;
  q.d2 = __subc_cc(q.d2, nn.d2) & 0xFFFFFF;
#ifndef DEBUG_GPU_MATH
  q.d3 = __subc   (q.d3, nn.d3) & 0xFFFFFF;
#else
  q.d3 = __subc_cc(q.d3, nn.d3) & 0xFFFFFF;
  q.d4 = __subc   (q.d4, nn.d4);
#endif

/********** Step 4, Offset 2^0 (0*24 + 0) **********/
  MODBASECASE_NONZERO_ERROR(q.d5, 4, 5, 5);
  MODBASECASE_NONZERO_ERROR(q.d4, 4, 4, 6);

  qf= __uint2float_rn(q.d3);
  qf= qf * 16777216.0f + __uint2float_rn(q.d2);
  qf= qf * 16777216.0f + __uint2float_rn(q.d1);
  qf= qf * 16777216.0f + __uint2float_rn(q.d0);
  
  qi=__float2uint_rz(qf*nf);

  MODBASECASE_QI_ERROR(1<<22, 4, qi, 7);

  nn.d0 =                                      __umul24(n.d0, qi)               & 0xFFFFFF;
  nn.d1 = __add_cc (__umul24hi(n.d0, qi) >> 8, __umul24(n.d1, qi) | 0xFF000000) & 0xFFFFFF;
#ifndef DEBUG_GPU_MATH  
  nn.d2 = __addc   (__umul24hi(n.d1, qi) >> 8, __umul24(n.d2, qi));
#else
  nn.d2 = __addc_cc(__umul24hi(n.d1, qi) >> 8, __umul24(n.d2, qi) | 0xFF000000) & 0xFFFFFF;
  nn.d3 = __addc   (__umul24hi(n.d2, qi) >> 8, 0);
#endif

/* q = q - nn */
/* subtraction using sub.cc.u32, subc.cc.u32 and subc.u32 instructions */
  q.d0 = __sub_cc (q.d0, nn.d0) & 0xFFFFFF;
  q.d1 = __subc_cc(q.d1, nn.d1) & 0xFFFFFF;
#ifndef DEBUG_GPU_MATH  
  q.d2 = __subc   (q.d2, nn.d2) & 0xFFFFFF;
#else
  q.d2 = __subc_cc(q.d2, nn.d2) & 0xFFFFFF;
  q.d3 = __subc   (q.d3, nn.d3);
#endif


  res->d0=q.d0;
  res->d1=q.d1;
  res->d2=q.d2;

  MODBASECASE_NONZERO_ERROR(q.d5, 5, 5, 8);
  MODBASECASE_NONZERO_ERROR(q.d4, 5, 4, 9);
  MODBASECASE_NONZERO_ERROR(q.d3, 5, 3, 10);

/*
qi is allways a little bit too small, this is OK for all steps except the last
one. Sometimes the result is a little bit bigger than n
*/
/*  if(cmp_ge_72(*res,n))
  {
    sub_72(&tmp72,*res,n);
    copy_72(res,tmp72);
  }*/
}

__global__ void
#ifndef DEBUG_GPU_MATH
__launch_bounds__(THREADS_PER_BLOCK,2) mfaktc_71(unsigned int exp, int72 k, unsigned int *k_tab, int shiftcount, int144 b, unsigned int *RES)
#else
__launch_bounds__(THREADS_PER_BLOCK,2) mfaktc_71(unsigned int exp, int72 k, unsigned int *k_tab, int shiftcount, int144 b, unsigned int *RES, unsigned int *modbasecase_debug)
#endif
/*
computes 2^exp mod f
shiftcount is used for precomputing without mod
a is precomputed on host ONCE. */
{
  int72 exp72,f;
  int72 a;
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  float ff;

  exp72.d2=0;exp72.d1=exp>>23;exp72.d0=(exp&0x7FFFFF)<<1;	// exp72 = 2 * exp

  mul_24_48(&(a.d1),&(a.d0),k_tab[index],NUM_CLASSES);
  k.d0 += a.d0;
  k.d1 += a.d1;
  k.d1 += k.d0 >> 24; k.d0 &= 0xFFFFFF;
  k.d2 += k.d1 >> 24; k.d1 &= 0xFFFFFF;		// k = k + k_tab[index] * NUM_CLASSES
        
  mul_72(&f,k,exp72);				// f = 2 * k * exp
  f.d0 += 1;					// f = 2 * k * exp + 1

/*
ff = f as float, needed in mod_144_72().
Precalculated here since it is the same for all steps in the following loop */
  ff= __uint2float_rn(f.d2);
  ff= ff * 16777216.0f + __uint2float_rn(f.d1);
  ff= ff * 16777216.0f + __uint2float_rn(f.d0);

  ff=__int_as_float(0x3f7ffffb) / ff;	// just a little bit below 1.0f so we allways underestimate the quotient
        
#ifndef DEBUG_GPU_MATH
  mod_144_72(&a,b,f,ff);			// a = b mod f
#else
  mod_144_72(&a,b,f,ff,modbasecase_debug);	// a = b mod f
#endif    
  exp<<= 32 - shiftcount;
  while(exp)
  {
    if(exp&0x80000000)square_72_144_shl(&b,a);	// b = 2 * a^2 ("optional multiply by 2" in Prime 95 documentation)
    else              square_72_144(&b,a);	// b = a^2
#ifndef DEBUG_GPU_MATH
    mod_144_72(&a,b,f,ff);			// a = b mod f
#else
    mod_144_72(&a,b,f,ff,modbasecase_debug);	// a = b mod f
#endif    
    exp<<=1;
  }

  if(cmp_ge_72(a,f))				// final adjustment in case a >= f
  {
    sub_72(&a,a,f);
  }

#if defined DEBUG_GPU_MATH && defined USE_DEVICE_PRINTF && __CUDA_ARCH__ >= FERMI
  if(cmp_ge_72(a,f))
  {
    printf("EEEEEK, final a is >= f\n");
  }
#endif

/* finally check if we found a factor and write the factor to RES[] */
#ifdef WAGSTAFF
  if(a.d2 == f.d2 && a.d1 == f.d1 && a.d0 == (f.d0 - 1))
#else /* Mersennes */
  if((a.d2|a.d1)==0 && a.d0==1)
#endif
  {
    if(f.d2!=0 || f.d1!=0 || f.d0!=1)		/* 1 isn't really a factor ;) */
    {
      index=atomicInc(&RES[0],10000);
      if(index<10)				/* limit to 10 factors per class */
      {
        RES[index*3 + 1]=f.d2;
        RES[index*3 + 2]=f.d1;
        RES[index*3 + 3]=f.d0;
      }
    }
  }
}


#define TF_72BIT
#include "tf_common.cu"
#undef TF_72BIT
