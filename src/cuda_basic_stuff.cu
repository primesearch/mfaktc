/*
This file is part of mfaktc.
Copyright (C) 2015  Oliver Weihe (o.weihe@t-online.de)

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


__global__ void check_subcc_bug_GPU(unsigned int *RES)
{
  asm("{\n\t"
      ".reg .u32 tmp;\n\t"

      "add.u32       tmp,  0,  0;\n\t" /* set tmp to 0 */

      "sub.cc.u32    %0, %3, tmp;\n\t"
      "subc.cc.u32   %1, %4, tmp;\n\t"
      "subc.u32      %2, %5, tmp;\n\t"
      "}"
      : "=r" (RES[3]), "=r" (RES[4]), "=r" (RES[5])
      : "r" (RES[0]), "r" (RES[1]), "r" (RES[2]));
}


extern "C" __host__ int check_subcc_bug(mystuff_t *mystuff)
{
  int ret = 0;
  cudaError_t cuda_ret;

/* RES[5..3] will be overwritten (hopefully!) */
  mystuff->h_RES[5] = 0xdeadbeef;
  mystuff->h_RES[4] = 0xdeadbeef;
  mystuff->h_RES[3] = 0xdeadbeef;

/* RES[2..0] is test data input */
  mystuff->h_RES[2] = 0x33333333;
  mystuff->h_RES[1] = 0x22222222;
  mystuff->h_RES[0] = 0x11111111;

  cudaMemcpy(mystuff->d_RES, mystuff->h_RES, 6*sizeof(int), cudaMemcpyHostToDevice);

  check_subcc_bug_GPU<<<1, 1, 0, 0>>>(mystuff->d_RES);

  cuda_ret = cudaDeviceSynchronize();
  if(cuda_ret != cudaSuccess)printf("check_subcc_bug(): cudaDeviceSynchronize failed!\n");

  cudaMemcpy(mystuff->h_RES, mystuff->d_RES, 6*sizeof(int), cudaMemcpyDeviceToHost);

  if(mystuff->h_RES[5] != mystuff->h_RES[2] || \
     mystuff->h_RES[4] != mystuff->h_RES[1] || \
     mystuff->h_RES[3] != mystuff->h_RES[0]) ret = 1;

  if(ret != 0 || mystuff->verbosity >= 2)
  {
    printf("\n");
    printf("check_subcc_bug()\n");
    printf("  input:  mystuff->h_RES[2..0] = 0x%08X %08X %08X\n", mystuff->h_RES[2], mystuff->h_RES[1], mystuff->h_RES[0]);
    printf("  output: mystuff->h_RES[5..3] = 0x%08X %08X %08X\n", mystuff->h_RES[5], mystuff->h_RES[4], mystuff->h_RES[3]);
    if(ret == 0)printf("  passed, output == input\n");
    else
    {
      printf("  ERROR: output != input\n");
      printf("\n");
      printf("  could be caused by bad software environment (CUDA toolkit and/or graphics driver)\n");
      printf("  Known bad:\n");
      printf("    - CUDA 5.0.7RC + 302.06.03 with all supported GPUs\n");
      printf("      fixed by driver update after reported this issue to nvidia\n");
      printf("    - CUDA 7.0 + 346.47, 346.59, 346.72 and 349.16 346.72 with Maxwell GPUs\n");
    }
    printf("\n");
  }
  return ret;
}


#ifndef __CUDA_ARCH__
  #define __CUDA_ARCH__ 0
#endif


__global__ void get_CUDA_arch_(unsigned int *RES)
{
  RES[0] = __CUDA_ARCH__;
}


extern "C" __host__ void get_CUDA_arch(mystuff_t *mystuff)
{
  cudaError_t cuda_ret;

  mystuff->cuda_arch = -1;
  get_CUDA_arch_<<<1, 1, 0, 0>>>(mystuff->d_RES);

  cuda_ret = cudaDeviceSynchronize();
  if(cuda_ret != cudaSuccess)printf("get_CUDA_arch(): cudaDeviceSynchronize failed!\n");
  
  cudaMemcpy(&(mystuff->cuda_arch), mystuff->d_RES, sizeof(int), cudaMemcpyDeviceToHost);
}
