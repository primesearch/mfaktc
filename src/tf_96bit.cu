/*
This file is part of mfaktc.
Copyright (C) 2009, 2010  Oliver Weihe (o.weihe@t-online.de)

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
#undef NVCC_EXTERN

#include "tf_debug.h"

#ifdef SHORTCUT_75BIT
__host__ void print_dez96(int96 a, char *buf);
__host__ void print_dez192(int192 a, char *buf);
__host__ void setui_96(int96 *res, unsigned long long int a);
#else
__host__ void print_dez96(int96 a, char *buf)
/*
writes "a" into "buf" in decimal
"buf" must be at least 30 bytes
*/
{
  char digit[29];
  int digits=0,carry,i=0;
  long long int tmp;
  
  while((a.d0!=0 || a.d1!=0 || a.d2!=0) && digits<29)
  {
                                                   carry=a.d2%10; a.d2/=10;
    tmp = a.d1; tmp += (long long int)carry << 32; carry=tmp%10;  a.d1=tmp/10;
    tmp = a.d0; tmp += (long long int)carry << 32; carry=tmp%10;  a.d0=tmp/10;
    digit[digits++]=carry;
  }
  if(digits==0)sprintf(buf,"0");
  else
  {
    digits--;
    while(digits >= 0)
    {
      sprintf(&(buf[i++]),"%1d",digit[digits--]);
    }
  }
}


__host__ void print_dez192(int192 a, char *buf)
{
/*
writes "a" into "buf" in decimal
"buf" must be at least 59 bytes
*/
  char digit[58];
  int digits=0,carry,i=0;
  long long int tmp;
  
  while((a.d0!=0 || a.d1!=0 || a.d2!=0 || a.d3!=0 || a.d4!=0 || a.d5!=0) && digits<58)
  {
                                                   carry=a.d5%10; a.d5/=10;
    tmp = a.d4; tmp += (long long int)carry << 32; carry=tmp%10;  a.d4=tmp/10;
    tmp = a.d3; tmp += (long long int)carry << 32; carry=tmp%10;  a.d3=tmp/10;
    tmp = a.d2; tmp += (long long int)carry << 32; carry=tmp%10;  a.d2=tmp/10;
    tmp = a.d1; tmp += (long long int)carry << 32; carry=tmp%10;  a.d1=tmp/10;
    tmp = a.d0; tmp += (long long int)carry << 32; carry=tmp%10;  a.d0=tmp/10;
    digit[digits++]=carry;
  }
  if(digits==0)sprintf(buf,"0 bla");
  else
  {
    digits--;
    while(digits >= 0)
    {
      sprintf(&(buf[i++]),"%1d",digit[digits--]);
    }
  }
}


__host__ void setui_96(int96 *res, unsigned long long int a)
/* sets res to a */
{
  res->d0 = (unsigned int)(a & 0xFFFFFFFF);
  a>>=32;
  res->d1 = (unsigned int)(a);
  res->d2 = 0;
}
#endif


__device__ static void copy_96(int96 *a, int96 b)
/* a = b */
{
  a->d0 = b.d0;
  a->d1 = b.d1;
  a->d2 = b.d2;
}


__device__ static int cmp_96(int96 a, int96 b)
/* returns
-1 if a < b
0  if a = b
1  if a > b */
{
  if(a.d2 < b.d2)return -1;
  if(a.d2 > b.d2)return 1;
  if(a.d1 < b.d1)return -1;
  if(a.d1 > b.d1)return 1;
  if(a.d0 < b.d0)return -1;
  if(a.d0 > b.d0)return 1;
  return 0;
}


__device__ static void sub_96(int96 *res, int96 a, int96 b)
/* a must be greater or equal b!
res = a - b */
{
  res->d0 = __sub_cc (a.d0, b.d0);
  res->d1 = __subc_cc(a.d1, b.d1);
  res->d2 = __subc   (a.d2, b.d2);
}


__device__ static void square_96_192(int192 *res, int96 a)
/* res = a^2 */
{
  unsigned int A01_lo, A01_hi;
  unsigned int TWO_A02_lo, TWO_A02_hi, TWO_A02_c;
  unsigned int A12_lo, A12_hi;

/*  A01 = a.d0 * a.d1 */
  A01_lo = __umul32  (a.d0, a.d1);
  A01_hi = __umul32hi(a.d0, a.d1);

/*  TWO_A02 = 2 * a.d0 * a.d2 */
  TWO_A02_lo = __umul32  (a.d0, a.d2); TWO_A02_lo = __add_cc (TWO_A02_lo, TWO_A02_lo);
  TWO_A02_hi = __umul32hi(a.d0, a.d2); TWO_A02_hi = __addc_cc(TWO_A02_hi, TWO_A02_hi);
                                       TWO_A02_c  = __addc   (         0,          0);

/*  A12 = a.d1 * a.d2 */
  A12_lo = __umul32  (a.d1, a.d2);
  A12_hi = __umul32hi(a.d1, a.d2);

  res->d0 =           __umul32  (a.d0, a.d0);
  res->d1 = __add_cc (__umul32hi(a.d0, a.d0),                 A01_lo);
  res->d2 = __addc_cc(                A01_hi,             TWO_A02_lo);
  res->d3 = __addc_cc(            TWO_A02_hi, __umul32hi(a.d1, a.d1));
  res->d4 = __addc   (             TWO_A02_c, __umul32  (a.d2, a.d2));
/*
highest possible value for __umul32  (a.d2, a.d2) is 0xFFFFFFF9
this occurs for a.d2 = {479772853, 1667710795, 2627256501, 3815194443}
TWO_A02_c is 0 or 1 and we have at most 1 from carry of res->d3
So the result is <= 0xFFFFFFFB and we don't need to carry the res->d5!
*/
  
  res->d1 = __add_cc (               res->d1,                 A01_lo);
  res->d2 = __addc_cc(               res->d2, __umul32  (a.d1, a.d1));
  res->d3 = __addc_cc(               res->d3,                 A12_lo);
  res->d4 = __addc_cc(               res->d4,                 A12_hi);
#ifndef SHORTCUT_75BIT  
  res->d5 = __addc   (__umul32hi(a.d2, a.d2),                      0);
#endif  
  
  res->d2 = __add_cc (               res->d2,                 A01_hi);
  res->d3 = __addc_cc(               res->d3,                 A12_lo);
  res->d4 = __addc_cc(               res->d4,                 A12_hi);
#ifndef SHORTCUT_75BIT  
  res->d5 = __addc   (               res->d5,                      0);
#endif
}


__device__ static void shl_192(int192 *a)
/* shiftleft a one bit */
{
  a->d0 = __add_cc (a->d0, a->d0);
  a->d1 = __addc_cc(a->d1, a->d1);
  a->d2 = __addc_cc(a->d2, a->d2);
  a->d3 = __addc_cc(a->d3, a->d3);
  a->d4 = __addc_cc(a->d4, a->d4);
#ifndef SHORTCUT_75BIT  
  a->d5 = __addc   (a->d5, a->d5);
#endif
}


#ifndef CHECKS_MODBASECASE
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
#ifdef CHECKS_MODBASECASE
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
#ifndef CHECKS_MODBASECASE  
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
#ifdef CHECKS_MODBASECASE
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
#ifndef CHECKS_MODBASECASE
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
#ifndef CHECKS_MODBASECASE  
  nn.d2 = __addc   (__umul32hi(n.d1, qi), __umul32(n.d2, qi));
#else
  nn.d2 = __addc_cc(__umul32hi(n.d1, qi), __umul32(n.d2, qi));
  nn.d3 = __addc   (__umul32hi(n.d2, qi),                  0);
#endif  

//  q = q - nn
  q.d0 = __sub_cc (q.d0, nn.d0);
  q.d1 = __subc_cc(q.d1, nn.d1);
#ifndef CHECKS_MODBASECASE
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
/*  if(cmp_96(*res,n)>0)
  {
    sub_96(&tmp96,*res,n);
    copy_96(res,tmp96);
  }*/
}


__global__ void
#ifdef SHORTCUT_75BIT
  #ifndef CHECKS_MODBASECASE
__launch_bounds__(THREADS_PER_BLOCK,2) mfakt_75(unsigned int exp, int96 k, unsigned int *k_tab, int shiftcount, int192 b, unsigned int *RES)
  #else
__launch_bounds__(THREADS_PER_BLOCK,2) mfakt_75(unsigned int exp, int96 k, unsigned int *k_tab, int shiftcount, int192 b, unsigned int *RES, unsigned int *modbasecase_debug)
  #endif
#else
  #ifndef CHECKS_MODBASECASE
__launch_bounds__(THREADS_PER_BLOCK,2) mfakt_95(unsigned int exp, int96 k, unsigned int *k_tab, int shiftcount, int192 b, unsigned int *RES)
  #else
__launch_bounds__(THREADS_PER_BLOCK,2) mfakt_95(unsigned int exp, int96 k, unsigned int *k_tab, int shiftcount, int192 b, unsigned int *RES, unsigned int *modbasecase_debug)
  #endif
#endif
/*
computes 2^exp mod f
shiftcount is used for precomputing without mod
a is precomputed on host ONCE. */
{
  int96 exp96,f;
  int96 a;
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  float ff;

  exp96.d2=0;exp96.d1=exp>>31;exp96.d0=exp<<1;	// exp96 = 2 * exp

  k.d0 = __add_cc (k.d0, __umul32  (k_tab[index], NUM_CLASSES));
  k.d1 = __addc   (k.d1, __umul32hi(k_tab[index], NUM_CLASSES));	/* k is limited to 2^64 -1 so there is no need for k.d2 */
        
//  mul_96(&f,k,exp96);				// f = 2 * k * exp
//  f.d0 += 1;					// f = 2 * k * exp + 1

  f.d0 = 1 +                                  __umul32(k.d0, exp96.d0);
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
  ff= ff * 4294967296.0f + __uint2float_rn(f.d1);
  ff= ff * 4294967296.0f + __uint2float_rn(f.d0);

//  ff=0.9999997f/ff;
//  ff=__int_as_float(0x3f7ffffc) / ff;	// just a little bit below 1.0f so we allways underestimate the quotient
  ff=__int_as_float(0x3f7ffffb) / ff;	// just a little bit below 1.0f so we allways underestimate the quotient
        
#ifndef CHECKS_MODBASECASE
  mod_192_96(&a,b,f,ff);			// a = b mod f
#else
  mod_192_96(&a,b,f,ff,modbasecase_debug);	// a = b mod f
#endif
  exp<<= 32 - shiftcount;
  while(exp)
  {
    square_96_192(&b,a);			// b = a^2
    if(exp&0x80000000)shl_192(&b);              // "optional multiply by 2" in Prime 95 documentation
#ifndef CHECKS_MODBASECASE
      mod_192_96(&a,b,f,ff);			// a = b mod f
#else
      mod_192_96(&a,b,f,ff,modbasecase_debug);	// a = b mod f
#endif
    exp<<=1;
  }
  if(cmp_96(a,f)>0)
  {
    sub_96(&exp96,a,f);
    copy_96(&a,exp96);
  }

/* finally check if we found a factor and write the factor to RES[] */
  if((a.d2|a.d1)==0 && a.d0==1)
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

#define TF_96BIT
#include "tf_common.cu"
#undef TF_96BIT
