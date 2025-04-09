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


//__device__ static void mul_96_192(int192 *res, int96 a, int96 b)
/* res = a * b */
/*{
  res->d0 = __umul32  (a.d0, b.d0);
  res->d1 = __umul32hi(a.d0, b.d0);
  res->d2 = __umul32  (a.d2, b.d0);
  res->d3 = __umul32hi(a.d2, b.d0);
  
  res->d1 = __add_cc (res->d1, __umul32  (a.d1, b.d0));
  res->d2 = __addc_cc(res->d2, __umul32hi(a.d1, b.d0));
  res->d3 = __addc_cc(res->d3, __umul32  (a.d2, b.d1));
  res->d4 = __addc   (      0,                      0);
  
  res->d1 = __add_cc (res->d1, __umul32  (a.d0, b.d1));
  res->d2 = __addc_cc(res->d2, __umul32hi(a.d0, b.d1));
  res->d3 = __addc_cc(res->d3, __umul32  (a.d1, b.d2));
  res->d4 = __addc_cc(res->d4, __umul32hi(a.d1, b.d2));  
  res->d5 = __addc   (      0,                      0);

  res->d2 = __add_cc (res->d2, __umul32  (a.d0, b.d2));
  res->d3 = __addc_cc(res->d3, __umul32hi(a.d0, b.d2));
  res->d4 = __addc_cc(res->d4, __umul32  (a.d2, b.d2));
  res->d5 = __addc   (res->d5, __umul32hi(a.d2, b.d2));

  res->d2 = __add_cc (res->d2, __umul32  (a.d1, b.d1));
  res->d3 = __addc_cc(res->d3, __umul32hi(a.d1, b.d1));
  res->d4 = __addc_cc(res->d4, __umul32hi(a.d2, b.d1));
  res->d5 = __addc   (res->d5,                      0);
}*/


__device__ static void mad_96(int96 *res, int96 a, int96 b, int96 c)
/* res = a * b + c (only lower 96 bits of the result) */
{
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
Using triple underscore because __clz() is used by CUDA toolkit */
__device__ static unsigned int ___clz (unsigned int a)
{
	unsigned int r;
	asm("clz.b32 %0, %1;" : "=r" (r) : "r" (a));
	return r;
}


__device__ static unsigned int __popcnt (unsigned int a)
{
	unsigned int r;
	asm("popc.b32 %0, %1;" : "=r" (r) : "r" (a));
	return r;
}
