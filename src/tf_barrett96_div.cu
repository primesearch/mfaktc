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


#ifndef CHECKS_MODBASECASE
  #ifdef DIV_160_96
__device__ static void div_160_96(int96 *res, int192 q, int96 n, float nf)
  #else
__device__ static void div_192_96(int96 *res, int192 q, int96 n, float nf)
  #endif
#else
  #ifdef DIV_160_96
__device__ static void div_160_96(int96 *res, int192 q, int96 n, float nf, unsigned int *modbasecase_debug)
  #else
__device__ static void div_192_96(int96 *res, int192 q, int96 n, float nf, unsigned int *modbasecase_debug)
  #endif
#endif
/* res = q / n (integer division) */
/* the code of div_160_96() is an EXACT COPY of div_192_96(), the only
difference is that the 160bit version ignores the most significant
word of q (q.d5) because it assumes it is 0. This is controlled by defining
DIV_160_96 here. */
{
  float qf;
  unsigned int qi;
  int192 nn;
  int96 tmp96;

/********** Step 1, Offset 2^75 (2*32 + 11) **********/
#ifndef DIV_160_96
  qf= __uint2float_rn(q.d5);
  qf= qf * 4294967296.0f + __uint2float_rn(q.d4);
#else
  #ifdef CHECKS_MODBASECASE
    q.d5 = 0;	// later checks in debug code will test if q.d5 is 0 or not but 160bit variant ignores q.d5
  #endif
  qf= __uint2float_rn(q.d4);
#endif  
  qf*= 2097152.0f;

  qi=__float2uint_rz(qf*nf);

  MODBASECASE_QI_ERROR(1<<22, 1, qi, 0);

#if __CUDA_ARCH__ >= KEPLER
  res->d2 = __umul32(qi, 2048);
#else
  res->d2 = qi << 11;
#endif

// nn = n * qi
  nn.d2 =                                 __umul32(n.d0, qi);
#if (__CUDA_ARCH__ >= KEPLER) && (CUDART_VERSION >= 4010) /* multiply-add with carry is not available on CC 1.x devices and before CUDA 4.1 */
  nn.d3 = __umad32hi_cc       (n.d0, qi,  __umul32(n.d1, qi));
  #ifndef DIV_160_96
  nn.d4 = __umad32hic_cc      (n.d1, qi,  __umul32(n.d2, qi));
  nn.d5 = __umad32hic         (n.d2, qi,                  0);
  #else
  nn.d4 = __umad32hic         (n.d1, qi,  __umul32(n.d2, qi));
  #endif
#else
  nn.d3 = __add_cc (__umul32hi(n.d0, qi), __umul32(n.d1, qi));
  #ifndef DIV_160_96
  nn.d4 = __addc_cc(__umul32hi(n.d1, qi), __umul32(n.d2, qi));
  nn.d5 = __addc   (__umul32hi(n.d2, qi),                  0);
  #else
  nn.d4 = __addc   (__umul32hi(n.d1, qi), __umul32(n.d2, qi));
  #endif
#endif

// shiftleft nn 11 bits
#if __CUDA_ARCH__ >= KEPLER
  #ifndef DIV_160_96
  nn.d5 = __umad32(nn.d5, 2048, (nn.d4 >> 21));
  #endif
  nn.d4 = __umad32(nn.d4, 2048, (nn.d3 >> 21));
  nn.d3 = __umad32(nn.d3, 2048, (nn.d2 >> 21));
  nn.d2 = __umul32(nn.d2, 2048);
#else
  #ifndef DIV_160_96
  nn.d5 = (nn.d5 << 11) + (nn.d4 >> 21);
  #endif
  nn.d4 = (nn.d4 << 11) + (nn.d3 >> 21);
  nn.d3 = (nn.d3 << 11) + (nn.d2 >> 21);
  nn.d2 =  nn.d2 << 11;
#endif

//  q = q - nn
  q.d2 = __sub_cc (q.d2, nn.d2);
  q.d3 = __subc_cc(q.d3, nn.d3);
  q.d4 = __subc_cc(q.d4, nn.d4);
#ifndef DIV_160_96
  q.d5 = __subc   (q.d5, nn.d5);
#endif
/********** Step 2, Offset 2^55 (1*32 + 23) **********/
#ifndef DIV_160_96
  qf= __uint2float_rn(q.d5);
  qf= qf * 4294967296.0f + __uint2float_rn(q.d4);
#else
  qf= __uint2float_rn(q.d4);
#endif
  qf= qf * 4294967296.0f + __uint2float_rn(q.d3);
  qf*= 512.0f;

  qi=__float2uint_rz(qf*nf);

  MODBASECASE_QI_ERROR(1<<22, 2, qi, 1);

#if __CUDA_ARCH__ >= KEPLER
  res->d1 =  __umul32(qi, 8388608);
#else
  res->d1 =  qi << 23;
#endif
  res->d2 += qi >>  9;

// nn = n * qi
  nn.d1 =                                 __umul32(n.d0, qi);
  nn.d2 = __add_cc (__umul32hi(n.d0, qi), __umul32(n.d1, qi));
  nn.d3 = __addc_cc(__umul32hi(n.d1, qi), __umul32(n.d2, qi));
  nn.d4 = __addc   (__umul32hi(n.d2, qi),                  0);

// shiftleft nn 23 bits
#ifdef CHECKS_MODBASECASE
  nn.d5 =                  nn.d4 >> 9;
#endif  
#if __CUDA_ARCH__ >= KEPLER
  nn.d4 = __umad32(nn.d4, 8388608, (nn.d3 >> 9));
  nn.d3 = __umad32(nn.d3, 8388608, (nn.d2 >> 9));
  nn.d2 = __umad32(nn.d2, 8388608, (nn.d1 >> 9));
  nn.d1 = __umul32(nn.d1, 8388608);
#else
  nn.d4 = (nn.d4 << 23) + (nn.d3 >> 9);
  nn.d3 = (nn.d3 << 23) + (nn.d2 >> 9);
  nn.d2 = (nn.d2 << 23) + (nn.d1 >> 9);
  nn.d1 =  nn.d1 << 23;
#endif

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
  qf*= 536870912.0f; // add (q.d1 >> 3) ???
//  qf*= 4294967296.0f; /* this includes the shiftleft of qi by 3 bits! */

  qi=__float2uint_rz(qf*nf);

  MODBASECASE_QI_ERROR(1<<22, 3, qi, 3);

#if __CUDA_ARCH__ >= KEPLER
  res->d1 = __add_cc(res->d1, __umul32(qi, 8) );
#else
  res->d1 = __add_cc(res->d1, qi << 3 );
#endif
  res->d2 = __addc  (res->d2, qi >> 29);

// shiftleft qi 3 bits to avoid "long shiftleft" after multiplication
#if __CUDA_ARCH__ >= KEPLER
  qi *= 8;
#else
  qi <<= 3;
#endif

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
  qf*= 131072.0f;
  
  qi=__float2uint_rz(qf*nf);

  MODBASECASE_QI_ERROR(1<<22, 4, qi, 5);

#if __CUDA_ARCH__ >= KEPLER
  res->d0 = __umul32(qi, 32768);
#else
  res->d0 = qi << 15;
#endif
  res->d1 = __add_cc(res->d1, qi >> 17);
  res->d2 = __addc  (res->d2, 0);
  
// nn = n * qi
  nn.d0 =                                 __umul32(n.d0, qi);
  nn.d1 = __add_cc (__umul32hi(n.d0, qi), __umul32(n.d1, qi));
  nn.d2 = __addc_cc(__umul32hi(n.d1, qi), __umul32(n.d2, qi));
  nn.d3 = __addc   (__umul32hi(n.d2, qi),                  0);

// shiftleft nn 15 bits
#ifdef CHECKS_MODBASECASE
  nn.d4 =                  nn.d3 >> 17;
#endif
#if __CUDA_ARCH__ >= KEPLER
  nn.d3 = __umad32(nn.d3, 32768, (nn.d2 >> 17));
  nn.d2 = __umad32(nn.d2, 32768, (nn.d1 >> 17));
  nn.d1 = __umad32(nn.d1, 32768, (nn.d0 >> 17));
  nn.d0 = __umul32(nn.d0, 32768);
#else
  nn.d3 = (nn.d3 << 15) + (nn.d2 >> 17);
  nn.d2 = (nn.d2 << 15) + (nn.d1 >> 17);
  nn.d1 = (nn.d1 << 15) + (nn.d0 >> 17);
  nn.d0 =  nn.d0 << 15;
#endif

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
  
  qi=__float2uint_rz(qf*nf);

  MODBASECASE_QI_ERROR(1<<20, 5, qi, 8);

  res->d0 = __add_cc (res->d0, qi);
  res->d1 = __addc_cc(res->d1,  0);
  res->d2 = __addc   (res->d2,  0);
  

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

//  res->d0=q.d0;
//  res->d1=q.d1;
//  res->d2=q.d2;
  tmp96.d0=q.d0;
  tmp96.d1=q.d1;
  tmp96.d2=q.d2;
  
  MODBASECASE_NONZERO_ERROR(q.d5, 6, 5, 9);
  MODBASECASE_NONZERO_ERROR(q.d4, 6, 4, 10);
  MODBASECASE_NONZERO_ERROR(q.d3, 6, 3, 11);

/*
qi is allways a little bit too small, this is OK for all steps except the last
one. Sometimes the result is a little bit bigger than n
*/
  if(cmp_ge_96(tmp96,n))
  {
    res->d0 = __add_cc (res->d0,  1);
    res->d1 = __addc_cc(res->d1,  0);
    res->d2 = __addc   (res->d2,  0);
  }
}
