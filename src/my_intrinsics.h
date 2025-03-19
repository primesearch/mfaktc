/*
This file is part of mfaktc.
Copyright (C) 2009, 2010, 2013, 2014  Oliver Weihe (o.weihe@t-online.de)

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

__device__ static unsigned int __umul24hi(unsigned int a, unsigned int b)
{
  unsigned int r;
  asm volatile("mul24.hi.u32 %0, %1, %2;" : "=r" (r) : "r" (a) , "r" (b));
  return r;
}


__device__ static unsigned int __umul32(unsigned int a, unsigned int b)
{
  unsigned int r;
  asm volatile("mul.lo.u32 %0, %1, %2;" : "=r" (r) : "r" (a) , "r" (b));
  return r;
}


__device__ static unsigned int __umul32hi(unsigned int a, unsigned int b)
{
/*  unsigned int r;
  asm volatile("mul.hi.u32 %0, %1, %2;" : "=r" (r) : "r" (a) , "r" (b));
  return r;*/
  return __umulhi(a,b);
}


__device__ static unsigned int __add_cc(unsigned int a, unsigned int b)
{
  unsigned int r;
  asm volatile("add.cc.u32 %0, %1, %2;" : "=r" (r) : "r" (a) , "r" (b));
  return r;
}


__device__ static unsigned int __addc_cc(unsigned int a, unsigned int b)
{
  unsigned int r;
  asm volatile("addc.cc.u32 %0, %1, %2;" : "=r" (r) : "r" (a) , "r" (b));
  return r;
}


__device__ static unsigned int __addc(unsigned int a, unsigned int b)
{
  unsigned int r;
  asm volatile("addc.u32 %0, %1, %2;" : "=r" (r) : "r" (a) , "r" (b));
  return r;
}


__device__ static unsigned int __sub_cc(unsigned int a, unsigned int b)
{
  unsigned int r;
  asm volatile("sub.cc.u32 %0, %1, %2;" : "=r" (r) : "r" (a) , "r" (b));
  return r;
}


__device__ static unsigned int __subc_cc(unsigned int a, unsigned int b)
{
  unsigned int r;
  asm volatile("subc.cc.u32 %0, %1, %2;" : "=r" (r) : "r" (a) , "r" (b));
  return r;
}


__device__ static unsigned int __subc(unsigned int a, unsigned int b)
{
  unsigned int r;
  asm volatile("subc.u32 %0, %1, %2;" : "=r" (r) : "r" (a) , "r" (b));
  return r;
}


__device__ static unsigned int __umad32(unsigned int a, unsigned int b, unsigned int c)
{
  unsigned int r;
  asm volatile("mad.lo.u32 %0, %1, %2, %3;" : "=r" (r) : "r" (a) , "r" (b), "r" (c));
  return r;
}


__device__ static unsigned int __umad32_cc(unsigned int a, unsigned int b, unsigned int c)
{
  unsigned int r;
  asm volatile("mad.lo.cc.u32 %0, %1, %2, %3;" : "=r" (r) : "r" (a) , "r" (b), "r" (c));
  return r;
}


__device__ static unsigned int __umad32c(unsigned int a, unsigned int b, unsigned int c)
{
  unsigned int r;
  asm volatile("madc.lo.u32 %0, %1, %2, %3;" : "=r" (r) : "r" (a) , "r" (b), "r" (c));
  return r;
}


__device__ static unsigned int __umad32c_cc(unsigned int a, unsigned int b, unsigned int c)
{
  unsigned int r;
  asm volatile("madc.lo.cc.u32 %0, %1, %2, %3;" : "=r" (r) : "r" (a) , "r" (b), "r" (c));
  return r;
}


__device__ static unsigned int __umad32hi(unsigned int a, unsigned int b, unsigned int c)
{
  unsigned int r;
  asm volatile("mad.hi.u32 %0, %1, %2, %3;" : "=r" (r) : "r" (a) , "r" (b), "r" (c));
  return r;
}


__device__ static unsigned int __umad32hi_cc(unsigned int a, unsigned int b, unsigned int c)
{
  unsigned int r;
  asm volatile("mad.hi.cc.u32 %0, %1, %2, %3;" : "=r" (r) : "r" (a) , "r" (b), "r" (c));
  return r;
}


__device__ static unsigned int __umad32hic(unsigned int a, unsigned int b, unsigned int c)
{
  unsigned int r;
  asm volatile("madc.hi.u32 %0, %1, %2, %3;" : "=r" (r) : "r" (a) , "r" (b), "r" (c));
  return r;
}


__device__ static unsigned int __umad32hic_cc(unsigned int a, unsigned int b, unsigned int c)
{
  unsigned int r;
  asm volatile("madc.hi.cc.u32 %0, %1, %2, %3;" : "=r" (r) : "r" (a) , "r" (b), "r" (c));
  return r;
}


__device__ static unsigned int __fshift_r(unsigned int a, unsigned int b, unsigned int c)
{
/* concatenates a and b and extract 32bits at given position
On input
 a has bits [0..31]
 b has bits [32..63]

0 <= c <= 32
Return value is bits [c..(32+c)] shifted inplace to the 32bit return value. */
#if (__CUDA_ARCH__ >= KEPLER_WITH_FUNNELSHIFT) && (CUDART_VERSION >= 5000)
/* needs CC >= 3.5 and CUDA 5.0
   needs CC >= 3.2 and CUDA 6.0
   there is no CC 3.2 in CUDA 5.x so CC >= 3.2 and CUDA >= 5.0 works fine here */
  return __funnelshift_r(a, b, c);
#else
  return (a >> c) + (b << (32 - c));
#endif
}
