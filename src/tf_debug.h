/*
This file is part of mfaktc.
Copyright (C) 2009, 2010, 2012, 2015  Oliver Weihe (o.weihe@t-online.de)

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

#ifdef DEBUG_GPU_MATH

/*
A = limit for qi
B = step number
C = qi
D = index for modbasecase_debug[];
*/
  #define MODBASECASE_QI_ERROR(A, B, C, D) \
  if(C > (A)) \
  { \
    printf("EEEEEK, step %d qi = %u\n", B, C); \
    modbasecase_debug[D]++; \
  }


/*
A = q.dX
B = step number
C = number of q.dX
D = index for modbasecase_debug[];
*/
  #define MODBASECASE_NONZERO_ERROR(A, B, C, D) \
  if(A) \
  { \
    printf("EEEEEK, step %d q.d%d is nonzero: %u\n", B, C, A); \
    modbasecase_debug[D]++; \
  }


/*
A = limit
B = step number
C = nn
D = index for modbasecase_debug[];
*/
  #define MODBASECASE_VALUE_BIG_ERROR(A, NAME, B, C, D) \
  if(C > A) \
  { \
    printf("EEEEEK, step %d " NAME " is too big: %u\n", B, C); \
    modbasecase_debug[D]++; \
  }

#else

#define MODBASECASE_QI_ERROR(A, B, C, D)
#define MODBASECASE_NONZERO_ERROR(A, B, C, D)
#define MODBASECASE_VALUE_BIG_ERROR(A, NAME, B, C, D)

#endif


__device__ static void trace_96_textmsg(const char *filename, int line, int96 f, const char *textmsg)
{
#ifdef TRACE_FC
  if(f.d2 == TRACE_D2 && f.d1 == TRACE_D1 && f.d0 == TRACE_D0)
  {
    printf("%25s line %-5d %s\n", filename, line, textmsg);
  }
#endif
}


__device__ static void trace_96_32(const char *filename, int line, int96 f, const char *varname, unsigned int data)
{
#ifdef TRACE_FC
  if(f.d2 == TRACE_D2 && f.d1 == TRACE_D1 && f.d0 == TRACE_D0)
  {
    printf("%25s line %-5d %10s = 0x %08X\n", filename, line, varname, data);
  }
#endif
}


__device__ static void trace_96_96(const char *filename, int line, int96 f, const char *varname, int96 data)
{
#ifdef TRACE_FC
  if(f.d2 == TRACE_D2 && f.d1 == TRACE_D1 && f.d0 == TRACE_D0)
  {
    printf("%25s line %-5d %10s = 0x %08X %08X %08X\n", filename, line, varname, data.d2, data.d1, data.d0);
  }
#endif
}


__device__ static void trace_96_192(const char *filename, int line, int96 f, const char *varname, int192 data)
{
#ifdef TRACE_FC
  if(f.d2 == TRACE_D2 && f.d1 == TRACE_D1 && f.d0 == TRACE_D0)
  {
    printf("%25s line %-5d %10s = 0x %08X %08X %08X %08X %08X %08X\n", filename, line, varname, data.d5, data.d4, data.d3, data.d2, data.d1, data.d0);
  }
#endif
}
