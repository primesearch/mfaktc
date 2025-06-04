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

#ifndef DEBUG_GPU_MATH
#ifdef INV_160_96
__device__ static void inv_160_96(int96 *res, int96 n, float nf)
#else
__device__ static void div_192_96(int96 *res, int192 q, int96 n, float nf)
#endif
#else
#ifdef INV_160_96
__device__ static void inv_160_96(int96 *res, int96 n, float nf, unsigned int *modbasecase_debug)
#else
__device__ static void div_192_96(int96 *res, int192 q, int96 n, float nf, unsigned int *modbasecase_debug)
#endif
#endif
/* res = q / n (integer division) */
/* the code of inv_160_96() is an EXACT COPY of div_192_96(), the only
difference is that the 160bit version assumes q = 2^160. This is controlled
by defining INV_160_96 here.

Input assumptions:
- div_192_96()
  - q = 2^(2*ceil(log2(n))) (~ twize the size of n, if 2^74 < n < 2^75 than q = 2^150)
  - 2^64 < n < 2^96
  -> 2^64 < res < 2^96
  
- div_160_96()
  - q = 2^160 (implicit, not a real parameter)
  - 2^64 < n < 2^80
  -> 2^80 < res < 2^96
*/
{
    float qf;
    unsigned int qi;
    int192 nn;
#ifdef DEBUG_GPU_MATH
    int96 tmp96;
#endif

#ifdef INV_160_96
    int192 q;
    // set q to (nearly 2^160)
    q.d5 = 0;
    q.d4 = 0xFFFFFFFF;
    q.d3 = 0xFFFFFFFF;
    q.d2 = 0xFFFFFFFF;
    q.d1 = 0xFFFFFFFF;
    q.d0 = 0xFFFFFFFF;
#endif

/********** Step 1, Offset 2^75 (2*32 + 11) **********/
#ifndef INV_160_96
    qf = __uint2float_rn(q.d5);
    qf = qf * 4294967296.0f + __uint2float_rn(q.d4);
    qf *= 2097152.0f;
#else
    qf = 9007199254740992.0f; // 2^(32 + (32 - 11))
#endif

    qi = __float2uint_rz(qf * nf);

    // EDIT OW 2014-10-20: qi can't be > 2^21, otherwise result would be > 2^96... so check for 2^21 insteat of 2^22 here
    // this allows/avoid long shift after multiplication, we can do a simple shift before multiplication.
    MODBASECASE_QI_ERROR(1 << 21, 1, qi, 0);

#if __CUDA_ARCH__ >= KEPLER
    qi = __umul32(qi, 2048);
#else
    qi <<= 11;
#endif

    res->d2 = qi;

    // nn = n * qi
    // clang-format off
    nn.d2 =                            __umul32(n.d0, qi );
    nn.d3 = __umad32hi_cc  (n.d0, qi,  __umul32(n.d1, qi));
    #ifndef INV_160_96
    nn.d4 = __umad32hic_cc (n.d1, qi,  __umul32(n.d2, qi));
    nn.d5 = __umad32hic    (n.d2, qi,                   0);
    #else
    nn.d4 = __umad32hic    (n.d1, qi,  __umul32(n.d2, qi));
    #endif
    // clang-format on

    //  q = q - nn
    // clang-format off
    q.d2 = __sub_cc( q.d2, nn.d2);
    q.d3 = __subc_cc(q.d3, nn.d3);
#ifndef INV_160_96
    q.d4 = __subc_cc(q.d4, nn.d4);
    q.d5 = __subc(   q.d5, nn.d5);
#else
    q.d4 = __subc(   q.d4, nn.d4);
#endif
    // clang-format on

/********** Step 2, Offset 2^55 (1*32 + 23) **********/
#ifndef INV_160_96
    qf = __uint2float_rn(q.d5);
    qf = qf * 4294967296.0f + __uint2float_rn(q.d4);
#else
    qf = __uint2float_rn(q.d4);
#endif
    qf = qf * 4294967296.0f + __uint2float_rn(q.d3);
    qf *= 512.0f;

    qi = __float2uint_rz(qf * nf);

    MODBASECASE_QI_ERROR(1 << 22, 2, qi, 1);

#if __CUDA_ARCH__ >= KEPLER
    res->d1 = __umul32(qi, 8388608);
#else
    res->d1 = qi << 23;
#endif
    res->d2 += qi >> 9;

    // nn = n * qi
    // clang-format off
    nn.d1 =                                 __umul32(n.d0, qi );
    nn.d2 = __add_cc( __umul32hi(n.d0, qi), __umul32(n.d1, qi));
    nn.d3 = __addc_cc(__umul32hi(n.d1, qi), __umul32(n.d2, qi));
    nn.d4 = __addc(   __umul32hi(n.d2, qi),                  0);
    // clang-format on

// shiftleft nn 23 bits
#ifdef DEBUG_GPU_MATH
    nn.d5 = nn.d4 >> 9;
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
    nn.d1 = nn.d1 << 23;
#endif

    // q = q - nn
    // clang-format off
    q.d1 = __sub_cc (q.d1, nn.d1);
    q.d2 = __subc_cc(q.d2, nn.d2);
    q.d3 = __subc_cc(q.d3, nn.d3);
#ifndef DEBUG_GPU_MATH
    q.d4 = __subc   (q.d4, nn.d4);
#else
    q.d4 = __subc_cc(q.d4, nn.d4);
    q.d5 = __subc(   q.d5, nn.d5);
#endif
    // clang-format on

    /********** Step 3, Offset 2^35 (1*32 + 3) **********/
    MODBASECASE_NONZERO_ERROR(q.d5, 3, 5, 2);

    qf = __uint2float_rn(q.d4);
    qf = qf * 4294967296.0f + __uint2float_rn(q.d3);
    qf *= 536870912.0f; // add (q.d1 >> 3) ???
    //  qf*= 4294967296.0f; /* this includes the shiftleft of qi by 3 bits! */

    qi = __float2uint_rz(qf * nf);

    MODBASECASE_QI_ERROR(1 << 22, 3, qi, 3);

#if __CUDA_ARCH__ >= KEPLER
    res->d1 = __add_cc(res->d1, __umul32(qi, 8));
#else
    res->d1 = __add_cc(res->d1, qi << 3);
#endif
    res->d2 = __addc(res->d2, qi >> 29);

// shiftleft qi 3 bits to avoid "long shiftleft" after multiplication
#if __CUDA_ARCH__ >= KEPLER
    qi *= 8;
#else
    qi <<= 3;
#endif

    // nn = n * qi
    // clang-format off
    nn.d1 =                                 __umul32(n.d0, qi );
    nn.d2 = __add_cc( __umul32hi(n.d0, qi), __umul32(n.d1, qi));
    nn.d3 = __addc_cc(__umul32hi(n.d1, qi), __umul32(n.d2, qi));
    nn.d4 = __addc(   __umul32hi(n.d2, qi),                  0);

    //  q = q - nn
    q.d1 = __sub_cc( q.d1, nn.d1);
    q.d2 = __subc_cc(q.d2, nn.d2);
    q.d3 = __subc_cc(q.d3, nn.d3);
    q.d4 = __subc(   q.d4, nn.d4);
    // clang-format on

    /********** Step 4, Offset 2^15 (0*32 + 15) **********/
    MODBASECASE_NONZERO_ERROR(q.d5, 4, 5, 4);

    qf = __uint2float_rn(q.d4);
    qf = qf * 4294967296.0f + __uint2float_rn(q.d3);
    qf = qf * 4294967296.0f + __uint2float_rn(q.d2);
    qf *= 131072.0f;

    qi = __float2uint_rz(qf * nf);

    MODBASECASE_QI_ERROR(1 << 22, 4, qi, 5);

#if __CUDA_ARCH__ >= KEPLER
    res->d0 = __umul32(qi, 32768);
#else
    res->d0 = qi << 15;
#endif
    res->d1 = __add_cc(res->d1, qi >> 17);
    res->d2 = __addc(res->d2, 0);

    // nn = n * qi
    // clang-format off
    nn.d0 =                                 __umul32(n.d0, qi );
    nn.d1 = __add_cc( __umul32hi(n.d0, qi), __umul32(n.d1, qi));
    nn.d2 = __addc_cc(__umul32hi(n.d1, qi), __umul32(n.d2, qi));
    nn.d3 = __addc(   __umul32hi(n.d2, qi),                  0);
    // clang-format on

// shiftleft nn 15 bits
#ifdef DEBUG_GPU_MATH
    nn.d4 = nn.d3 >> 17;
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
    nn.d0 = nn.d0 << 15;
#endif

    //  q = q - nn
    q.d0 = __sub_cc(q.d0, nn.d0);
    q.d1 = __subc_cc(q.d1, nn.d1);
    q.d2 = __subc_cc(q.d2, nn.d2);
#ifndef DEBUG_GPU_MATH
    q.d3 = __subc(q.d3, nn.d3);
#else
    q.d3 = __subc_cc(q.d3, nn.d3);
    q.d4 = __subc(q.d4, nn.d4);
#endif

    /********** Step 5, Offset 2^0 (0*32 + 0) **********/
    MODBASECASE_NONZERO_ERROR(q.d5, 5, 5, 6);
    MODBASECASE_NONZERO_ERROR(q.d4, 5, 4, 7);

    qf = __uint2float_rn(q.d3);
    qf = qf * 4294967296.0f + __uint2float_rn(q.d2);
    qf = qf * 4294967296.0f + __uint2float_rn(q.d1);

    qi = __float2uint_rz(qf * nf);

    MODBASECASE_QI_ERROR(1 << 20, 5, qi, 8);

    res->d0 = __add_cc(res->d0, qi);
    res->d1 = __addc_cc(res->d1, 0);
    res->d2 = __addc(res->d2, 0);

#ifdef DEBUG_GPU_MATH
    /* compute to the end only in DEBUG_GPU_MATH mode */

    // nn = n * qi
    // clang-format off
    nn.d0 =                                 __umul32(n.d0, qi );
    nn.d1 = __add_cc( __umul32hi(n.d0, qi), __umul32(n.d1, qi));
    nn.d2 = __addc_cc(__umul32hi(n.d1, qi), __umul32(n.d2, qi));
    nn.d3 = __addc(   __umul32hi(n.d2, qi),                  0);

    //  q = q - nn
    q.d0 = __sub_cc( q.d0, nn.d0);
    q.d1 = __subc_cc(q.d1, nn.d1);
    q.d2 = __subc_cc(q.d2, nn.d2);
    q.d3 = __subc(   q.d3, nn.d3);
    // clang-format on

    tmp96.d0 = q.d0;
    tmp96.d1 = q.d1;
    tmp96.d2 = q.d2;

    if (n.d2) /* only care about errors is n >= 2^64 (see function description) */
    {
        MODBASECASE_NONZERO_ERROR(q.d5, 6, 5, 9);
        MODBASECASE_NONZERO_ERROR(q.d4, 6, 4, 10);
        MODBASECASE_NONZERO_ERROR(q.d3, 6, 3, 11);
    }

    /*
qi is allways a little bit too small, this is OK for all steps except the last
one. Sometimes the result is a little bit bigger than n
*/
    if (cmp_ge_96(tmp96, n)) {
        /*    res->d0 = __add_cc (res->d0,  1);
    res->d1 = __addc_cc(res->d1,  0);
    res->d2 = __addc   (res->d2,  0);*/
        // clang-format off
        tmp96.d0 = __sub_cc( tmp96.d0, n.d0);
        tmp96.d1 = __subc_cc(tmp96.d1, n.d1);
        tmp96.d2 = __subc(   tmp96.d2, n.d2);
        // clang-format off
    }
    if (cmp_ge_96(tmp96, n)) {
        printf("EEEEEK, final value of tmp96 is too big\n");
        printf("  tmp96 = 0x %08x %08x %08x\n", tmp96.d2, tmp96.d1, tmp96.d0);
        printf("  n =     0x %08x %08x %08x\n", n.d2, n.d1, n.d0);
    }
#endif
}
