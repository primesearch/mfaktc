//void print_hex72(int72 N)
//{
//	printf("0x%06X%06X%06X",N.d2,N.d1,N.d0);
//}

//void print_hex144(int144 N)
//{
//	printf("0x%06X %06X %06X %06X %06X %06X",N.d5,N.d4,N.d3,N.d2,N.d1,N.d0);
//}

__device__ void add_72(int72 *res, int72 a, int72 b)
/* res = a + b */
{
	unsigned int tmp;
	tmp = a.d0 + b.d0;
	res->d0 = tmp&0xFFFFFF;
	tmp >>= 24;
	tmp += a.d1 + b.d1;
	res->d1 = tmp&0xFFFFFF;
	tmp >>= 24;
	tmp += a.d2 + b.d2;
	res->d2 = tmp&0xFFFFFF;
}


__device__ void shl_144(int144 *a)
/* a = a << 1 */
{
	unsigned int tmp;
	
	a->d0 <<= 1;
	tmp = a->d0 >> 24;
	a->d0 &= 0xFFFFFF;
	
	a->d1 = (a->d1 << 1) + tmp;
	tmp = a->d1 >> 24;
	a->d1 &= 0xFFFFFF;

	a->d2 = (a->d2 << 1) + tmp;
	tmp = a->d2 >> 24;
	a->d2 &= 0xFFFFFF;

	a->d3 = (a->d3 << 1) + tmp;
	tmp = a->d3 >> 24;
	a->d3 &= 0xFFFFFF;

	a->d4 = (a->d4 << 1) + tmp;
	tmp = a->d4 >> 24;
	a->d4 &= 0xFFFFFF;

	a->d5 = (a->d5 << 1) + tmp;
	a->d5 &= 0xFFFFFF;
}

__device__ void shl_72(int72 *a)
/* a = a << 1 */
{
  unsigned int carry;
  
  a->d0 <<= 1;
  carry = a->d0 >> 24;
  a->d0 &= 0xFFFFFF;
  
  a->d1 = (a->d1 << 1) + carry;
  carry = a->d1 >> 24;
  a->d1 &= 0xFFFFFF;

  a->d2 = (a->d2 << 1) + carry;
  a->d2 &= 0xFFFFFF;
}


__device__ static void mul_96(int96 *res, int96 a, int96 b)
/* res = (a * b) mod (2^96) */
{
  res->d0  =          __umul32  (a.d0, b.d0);
  res->d1  = __add_cc(__umul32hi(a.d0, b.d0), __umul32  (a.d0, b.d1));
  res->d2  = __addc  (__umul32hi(a.d0, b.d1), __umul32hi(a.d1, b.d0));

  res->d1  = __add_cc(res->d1               , __umul32  (a.d1, b.d0));
  res->d2  = __addc  (res->d2               , __umul32  (a.d1, b.d1));
  
  res->d2 += __umul32(a.d2, b.d0);
  res->d2 += __umul32(a.d0, b.d2);
}


__device__ static void mad_96(int96 *res, int96 a, int96 b, int96 c)
/* res = a * b + c (only lower 96 bits of the result) */
{
#if (__CUDA_ARCH__ >= FERMI) && (CUDART_VERSION >= 4010) /* multiply-add with carry is not available on CC 1.x devices and before CUDA 4.1 */
  asm("{\n\t"
      "mad.lo.cc.u32  %0, %3, %6, %9;\n\t"   /* (a.d0 * b.d0).lo + c.d0 */
      "madc.hi.cc.u32 %1, %3, %6, %10;\n\t"  /* (a.d0 * b.d0).hi + c.d1 */
      "madc.lo.u32    %2, %5, %6, %11;\n\t"  /* (a.d2 * b.d0).lo + c.d2 */

      "mad.lo.cc.u32  %1, %4, %6, %1;\n\t"   /* (a.d1 * b.d0).lo */
      "madc.hi.u32    %2, %4, %6, %2;\n\t"   /* (a.d1 * b.d0).hi */

      "mad.lo.cc.u32  %1, %3, %7, %1;\n\t"   /* (a.d0 * b.d1).lo */
      "madc.hi.u32    %2, %3, %7, %2;\n\t"   /* (a.d0 * b.d1).hi */

      "mad.lo.u32     %2, %3, %8, %2;\n\t"   /* (a.d0 * b.d2).lo */

      "mad.lo.u32     %2, %4, %7, %2;\n\t"   /* (a.d1 * b.d1).lo */
      "}"
      : "=r" (res->d0), "=r" (res->d1), "=r" (res->d2)
      : "r" (a.d0), "r" (a.d1), "r" (a.d2), "r" (b.d0), "r" (b.d1), "r" (b.d2), "r" (c.d0), "r" (c.d1), "r" (c.d2));
#else
  res->d0 = __add_cc (__umul32  (a.d0, b.d0), c.d0);
  res->d1 = __addc_cc(__umul32hi(a.d0, b.d0), __umul32  (a.d1, b.d0));
  res->d2 = __addc   (__umul32  (a.d2, b.d0), __umul32hi(a.d1, b.d0));

  res->d1 = __add_cc (res->d1,                c.d1);
  res->d2 = __addc   (res->d2,                c.d2);
  
  res->d1 = __add_cc (res->d1,                __umul32  (a.d0, b.d1));
  res->d2 = __addc   (res->d2,                __umul32hi(a.d0, b.d1));

  res->d2+= __umul32  (a.d0, b.d2);

  res->d2+= __umul32  (a.d1, b.d1);
#endif
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


/* If no bit is set, CC 2.x returns 32, CC 1.x returns 31
Using trippel underscore because __clz() is used by CUDA toolkit */
__device__ static unsigned int ___clz (unsigned int a)
{
#if (__CUDA_ARCH__ >= FERMI) /* clz (count leading zeroes) is not available on CC 1.x devices */
	unsigned int r;
	asm("clz.b32 %0, %1;" : "=r" (r) : "r" (a));
	return r;
#else
	unsigned int r = 0;
	if ((a & 0xFFFF0000) == 0) r = 16, a <<= 16;
	if ((a & 0xFF000000) == 0) r += 8, a <<= 8;
	if ((a & 0xF0000000) == 0) r += 4, a <<= 4;
	if ((a & 0xC0000000) == 0) r += 2, a <<= 2;
	if ((a & 0x80000000) == 0) r += 1;
	return r;
#endif
}


__device__ static unsigned int __popcnt (unsigned int a)
{
#if (__CUDA_ARCH__ >= FERMI) /* popc (population count) is not available on CC 1.x devices */
	unsigned int r;
	asm("popc.b32 %0, %1;" : "=r" (r) : "r" (a));
	return r;
#else
	a = (a&0x55555555) + ((a>> 1)&0x55555555);  // Generate sixteen 2-bit sums
	a = (a&0x33333333) + ((a>> 2)&0x33333333);  // Generate eight 3-bit sums
	a = (a&0x07070707) + ((a>> 4)&0x07070707);  // Generate four 4-bit sums
	a = (a&0x000F000F) + ((a>> 8)&0x000F000F);  // Generate two 5-bit sums
	a = (a&0x0000001F) + ((a>>16)&0x0000001F);  // Generate one 6-bit sum
	return a;
#endif
}


