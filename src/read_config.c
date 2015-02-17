/*
This file is part of mfaktc.
Copyright (C) 2009, 2010, 2011, 2012  Oliver Weihe (o.weihe@t-online.de)
                                      Bertram Franz (bertramf@gmx.net)

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
#include <string.h>
#include <cuda_runtime.h>

#include "params.h"
#include "my_types.h"

int my_read_int(char *inifile, char *name, int *value)
{
  FILE *in;
  char buf[100];
  int found=0;

  in=fopen(inifile,"r");
  if(!in)return 1;
  while(fgets(buf,100,in) && !found)
  {
    if(!strncmp(buf,name,strlen(name)) && buf[strlen(name)]=='=')
    {
      if(sscanf(&(buf[strlen(name)+1]),"%d",value)==1)found=1;
    }
  }
  fclose(in);
  if(found)return 0;
  return 1;
}


int my_read_string(char *inifile, char *name, char *string, unsigned int len)
{
  FILE *in;
  char buf[256];
  unsigned int found = 0;
  unsigned int idx = strlen(name);

  if(len > 250) len = 250;
  
  in = fopen(inifile, "r");
  if(!in)return 1;
  while(fgets(buf, 250, in) && !found)
  {
    if(!strncmp(buf, name, idx) && buf[idx] == '=')
    {
      found = strlen(buf + idx + 1);
      found = (len > found ? found : len) - 1;
      if (found)
      {
        strncpy(string, buf + idx + 1, found);
        if(string[found - 1] == '\r') found--; //remove '\r' from string, this happens when reading a DOS/Windows formatted file on Linux
      }
      string[found] = '\0';
    }
  }  
  fclose(in);
  if(found >= 1)return 0;
  return 1;
}


int read_config(mystuff_t *mystuff)
{
  int i;
  if(mystuff->verbosity >= 1)printf("\nRuntime options\n");

  if(my_read_int("mfaktc.ini", "SievePrimes", &i))
  {
    printf("WARNING: Cannot read SievePrimes from mfaktc.ini, using default value (%d)\n",SIEVE_PRIMES_DEFAULT);
    i = SIEVE_PRIMES_DEFAULT;
  }
  else
  {
    if(i > SIEVE_PRIMES_MAX)
    {
      printf("WARNING: Read SievePrimes=%d from mfaktc.ini, using max value (%d)\n",i,SIEVE_PRIMES_MAX);
      i = SIEVE_PRIMES_MAX;
    }
    else if(i < SIEVE_PRIMES_MIN)
    {
      printf("WARNING: Read SievePrimes=%d from mfaktc.ini, using min value (%d)\n",i,SIEVE_PRIMES_MIN);
      i = SIEVE_PRIMES_MIN;
    }
  }
  if(mystuff->verbosity >= 1)printf("  SievePrimes               %d\n",i);
  mystuff->sieve_primes = i;

/*****************************************************************************/  

  if(my_read_int("mfaktc.ini", "SievePrimesAdjust", &i))
  {
    printf("WARNING: Cannot read SievePrimesAdjust from mfaktc.ini, using default value (1)\n");
    i = 1;
  }
  else if(i != 0 && i != 1)
  {
    printf("WARNING: SievePrimesAdjust must be 0 or 1, using default value (1)\n");
    i = 1;
  }
  if(mystuff->verbosity >= 1)printf("  SievePrimesAdjust         %d\n",i);
  mystuff->sieve_primes_adjust = i;

/*****************************************************************************/

  if(my_read_int("mfaktc.ini", "SievePrimesMin", &i))
  {
    printf("WARNING: Cannot read SievePrimesMin from mfaktc.ini, using min value (%d)\n",SIEVE_PRIMES_MIN);
    i = SIEVE_PRIMES_MIN;
  }
  else
  {
    if(i < SIEVE_PRIMES_MIN || i >= SIEVE_PRIMES_MAX || i > mystuff->sieve_primes)
    {
      printf("WARNING: Read SievePrimesMin=%d from mfaktc.ini, using min value (%d)\n",i,SIEVE_PRIMES_MIN);
      i = SIEVE_PRIMES_MIN;
    }
  }
  if(mystuff->verbosity >= 1)printf("  SievePrimesMin            %d\n",i);
  mystuff->sieve_primes_min = i;

/*****************************************************************************/

  if(my_read_int("mfaktc.ini", "SievePrimesMax", &i))
  {
    printf("WARNING: Cannot read SievePrimesMax from mfaktc.ini, using max value (%d)\n",SIEVE_PRIMES_MAX);
    i = SIEVE_PRIMES_MAX;
  }
  else
  {
    if(i <= SIEVE_PRIMES_MIN || i > SIEVE_PRIMES_MAX || i < mystuff->sieve_primes)
    {
      printf("WARNING: Read SievePrimesMax=%d from mfaktc.ini, using max value (%d)\n",i,SIEVE_PRIMES_MAX);
      i = SIEVE_PRIMES_MAX;
    }
  }
  if(mystuff->verbosity >= 1)printf("  SievePrimesMax            %d\n",i);
  mystuff->sieve_primes_max = i;

/*****************************************************************************/  

  if(my_read_int("mfaktc.ini", "NumStreams", &i))
  {
    printf("WARNING: Cannot read NumStreams from mfaktc.ini, using default value (%d)\n",NUM_STREAMS_DEFAULT);
    i = NUM_STREAMS_DEFAULT;
  }
  else
  {
    if(i > NUM_STREAMS_MAX)
    {
      printf("WARNING: Read NumStreams=%d from mfaktc.ini, using max value (%d)\n",i,NUM_STREAMS_MAX);
      i = NUM_STREAMS_MAX;
    }
    else if(i < NUM_STREAMS_MIN)
    {
      printf("WARNING: Read NumStreams=%d from mfaktc.ini, using min value (%d)\n",i,NUM_STREAMS_MIN);
      i = NUM_STREAMS_MIN;
    }
  }
  if(mystuff->verbosity >= 1)printf("  NumStreams                %d\n",i);
  mystuff->num_streams = i;

/*****************************************************************************/

  if(my_read_int("mfaktc.ini", "CPUStreams", &i))
  {
    printf("WARNING: Cannot read CPUStreams from mfaktc.ini, using default value (%d)\n",CPU_STREAMS_DEFAULT);
    i = CPU_STREAMS_DEFAULT;
  }
  else
  {
    if(i > CPU_STREAMS_MAX)
    {
      printf("WARNING: Read CPUStreams=%d from mfaktc.ini, using max value (%d)\n",i,CPU_STREAMS_MAX);
      i = CPU_STREAMS_MAX;
    }
    else if(i < CPU_STREAMS_MIN)
    {
      printf("WARNING: Read CPUStreams=%d from mfaktc.ini, using min value (%d)\n",i,CPU_STREAMS_MIN);
      i = CPU_STREAMS_MIN;
    }
  }
  if(mystuff->verbosity >= 1)printf("  CPUStreams                %d\n",i);
  mystuff->cpu_streams = i;

/*****************************************************************************/

  if(my_read_int("mfaktc.ini", "GridSize", &i))
  {
    printf("WARNING: Cannot read GridSize from mfaktc.ini, using default value (3)\n");
    i = 3;
  }
  else
  {
    if(i > 3)
    {
      printf("WARNING: Read GridSize=%d from mfaktc.ini, using max value (3)\n", i);
      i = 3;
    }
    else if(i < 0)
    {
      printf("WARNING: Read GridSize=%d from mfaktc.ini, using min value (0)\n", i);
      i = 0;
    }
  }
  if(mystuff->verbosity >= 1)printf("  GridSize                  %d\n",i);
       if(i == 0)  mystuff->threads_per_grid_max =  131072;
  else if(i == 1)  mystuff->threads_per_grid_max =  262144;
  else if(i == 2)  mystuff->threads_per_grid_max =  524288;
  else             mystuff->threads_per_grid_max = 1048576;

/*****************************************************************************/
  if(my_read_int("mfaktc.ini", "SieveOnGPU", &i))
  {
    printf("WARNING: Cannot read SieveOnGPU from mfaktc.ini, enabled by default\n");
    i = 1;
  }
  else
  {
    if(i < 0 || i > 1)
    {
      printf("WARNING: Read SieveOnGPU=%d from mfaktc.ini, enabled by default\n", i);
      i = 1;
    }
  }
  
  mystuff->gpu_sieving = i;

  if(mystuff->gpu_sieving) {

    if(mystuff->verbosity == 1)printf("  GPU Sieving               enabled\n");

/*****************************************************************************/

    if(my_read_int("mfaktc.ini", "GPUSievePrimes", &i))
    {
      printf("WARNING: Cannot read GPUSievePrimes from mfaktc.ini, using default value (%d)\n",GPU_SIEVE_PRIMES_DEFAULT);
      i = GPU_SIEVE_PRIMES_DEFAULT;
    }
    else
    {
      if(i > GPU_SIEVE_PRIMES_MAX)
      {
        printf("WARNING: Read GPUSievePrimes=%d from mfaktc.ini, using max value (%d)\n",i,GPU_SIEVE_PRIMES_MAX);
	i = GPU_SIEVE_PRIMES_MAX;
      }
      else if(i < GPU_SIEVE_PRIMES_MIN)
      {
        printf("WARNING: Read GPUSievePrimes=%d from mfaktc.ini, using min value (%d)\n",i,GPU_SIEVE_PRIMES_MIN);
	i = GPU_SIEVE_PRIMES_MIN;
      }
    }
    if(mystuff->verbosity >= 1)printf("  GPUSievePrimes            %d\n",i);
    mystuff->gpu_sieve_primes = i;

/*****************************************************************************/

    if(my_read_int("mfaktc.ini", "GPUSieveSize", &i))
    {
      printf("WARNING: Cannot read GPUSieveSize from mfaktc.ini, using default value (%d)\n",GPU_SIEVE_SIZE_DEFAULT);
      i = GPU_SIEVE_SIZE_DEFAULT;
    }
    else
    {
      if(i > GPU_SIEVE_SIZE_MAX)
      {
        printf("WARNING: Read GPUSieveSize=%d from mfaktc.ini, using max value (%d)\n",i,GPU_SIEVE_SIZE_MAX);
	i = GPU_SIEVE_SIZE_MAX;
      }
      else if(i < GPU_SIEVE_SIZE_MIN)
      {
        printf("WARNING: Read GPUSieveSize=%d from mfaktc.ini, using min value (%d)\n",i,GPU_SIEVE_SIZE_MIN);
	i = GPU_SIEVE_SIZE_MIN;
      }
    }
    if(mystuff->verbosity >= 1)printf("  GPUSieveSize              %dMi bits\n",i);
    mystuff->gpu_sieve_size = i * 1024 * 1024;

/*****************************************************************************/

    if(my_read_int("mfaktc.ini", "GPUSieveProcessSize", &i))
    {
      printf("WARNING: Cannot read GPUSieveProcessSize from mfaktc.ini, using default value (%d)\n",GPU_SIEVE_PROCESS_SIZE_DEFAULT);
      i = GPU_SIEVE_PROCESS_SIZE_DEFAULT;
    }
    else
    {
      if(i % 8 != 0)
      {
        printf("WARNING: GPUSieveProcessSize must be a multiple of 8\n");
        i &= 0xFFFFFFF0;
        if(i == 0)i = 8;
        printf("         --> changed GPUSieveProcessSize to %d\n", i);
      }
      if(i > GPU_SIEVE_PROCESS_SIZE_MAX)
      {
        printf("WARNING: Read GPUSieveProcessSize=%d from mfaktc.ini, using max value (%d)\n",i,GPU_SIEVE_PROCESS_SIZE_MAX);
	i = GPU_SIEVE_PROCESS_SIZE_MAX;
      }
      else if(i < GPU_SIEVE_PROCESS_SIZE_MIN)
      {
        printf("WARNING: Read GPUSieveProcessSize=%d from mfaktc.ini, using min value (%d)\n",i,GPU_SIEVE_PROCESS_SIZE_MIN);
	i = GPU_SIEVE_PROCESS_SIZE_MIN;
      }
      if(mystuff->gpu_sieve_size % (i * 1024) != 0)
      {
        printf("WARNING: GPUSieveSize must be a multiple of GPUSieveProcessSize, using default value (%d)!\n", GPU_SIEVE_PROCESS_SIZE_DEFAULT);
        i = GPU_SIEVE_PROCESS_SIZE_DEFAULT;
      }
    }
    if(mystuff->verbosity >= 1)printf("  GPUSieveProcessSize       %dKi bits\n",i);
    mystuff->gpu_sieve_processing_size = i * 1024;
  }

/*****************************************************************************/

  if(my_read_int("mfaktc.ini", "Checkpoints", &i))
  {
    printf("WARNING: Cannot read Checkpoints from mfaktc.ini, enabled by default\n");
    i = 1;
  }
  else if(i != 0 && i != 1)
  {
    printf("WARNING: Checkpoints must be 0 or 1, enabled by default\n");
    i = 1;
  }
  if(mystuff->verbosity >= 1)
  {
    if(i == 0)printf("  Checkpoints               disabled\n");
    else      printf("  Checkpoints               enabled\n");
  }
  mystuff->checkpoints = i;

/*****************************************************************************/

  if(my_read_int("mfaktc.ini", "CheckpointDelay", &i))
  {
    printf("WARNING: Cannot read CheckpointDelay from mfaktc.ini, set to 30s by default\n");
    i = 30;
  }
  if(i > 900)
  {
    printf("WARNING: Maximum value for CheckpointDelay is 900s\n");
    i = 900;
  }
  if(i < 0)
  {
    printf("WARNING: Minimum value for CheckpointDelay is 0s\n");
    i = 0;
  }
  if(mystuff->verbosity >= 1)printf("  CheckpointDelay           %ds\n", i);
  mystuff->checkpointdelay = i;

/*****************************************************************************/

  if(my_read_int("mfaktc.ini", "WorkFileAddDelay", &i))
  {
    printf("WARNING: Cannot read WorkFileAddDelay from mfaktc.ini, set to 600s by default\n");
    i = 600;
  }
  if(i > 3600)
  {
    printf("WARNING: Maximum value for WorkFileAddDelay is 3600s\n");
    i = 3600;
  }
  if(i != 0 && i < 30)
  {
    printf("WARNING: Minimum value for WorkFileAddDelay is 30s\n");
    i = 30;
  }
  if(mystuff->verbosity >= 1)
  {
    if(i > 0)printf("  WorkFileAddDelay          %ds\n", i);
    else     printf("  WorkFileAddDelay          disabled\n");
  }
  mystuff->addfiledelay = i;

/*****************************************************************************/

  if(my_read_int("mfaktc.ini", "Stages", &i))
  {
    printf("WARNING: Cannot read Stages from mfaktc.ini, enabled by default\n");
    i = 1;
  }
  else if(i != 0 && i != 1)
  {
    printf("WARNING: Stages must be 0 or 1, enabled by default\n");
    i = 1;
  }
  if(mystuff->verbosity >= 1)
  {
    if(i == 0)printf("  Stages                    disabled\n");
    else      printf("  Stages                    enabled\n");
  }
  mystuff->stages = i;

/*****************************************************************************/

  if(my_read_int("mfaktc.ini", "StopAfterFactor", &i))
  {
    printf("WARNING: Cannot read StopAfterFactor from mfaktc.ini, set to 1 by default\n");
    i = 1;
  }
  else if( (i < 0) || (i > 2) )
  {
    printf("WARNING: StopAfterFactor must be 0, 1 or 2, set to 1 by default\n");
    i = 1;
  }
  if(mystuff->verbosity >= 1)
  {
         if(i == 0)printf("  StopAfterFactor           disabled\n");
    else if(i == 1)printf("  StopAfterFactor           bitlevel\n");
    else if(i == 2)printf("  StopAfterFactor           class\n");
  }
  mystuff->stopafterfactor = i;

/*****************************************************************************/

  if(my_read_int("mfaktc.ini", "PrintMode", &i))
  {
    printf("WARNING: Cannot read PrintMode from mfaktc.ini, set to 0 by default\n");
    i = 0;
  }
  else if(i != 0 && i != 1)
  {
    printf("WARNING: PrintMode must be 0 or 1, set to 0 by default\n");
    i = 0;
  }
  if(mystuff->verbosity >= 1)
  {
    if(i == 0)printf("  PrintMode                 full\n");
    else      printf("  PrintMode                 compact\n");
  }
  mystuff->printmode = i;

/*****************************************************************************/

  if (my_read_string("mfaktc.ini", "V5UserID", mystuff->V5UserID, 50))
  {
    /* no problem, don't use any */
    if(mystuff->verbosity >= 1)printf("  V5UserID                  (none)\n");
    mystuff->V5UserID[0]='\0';
  }
  else
  {
    if(mystuff->verbosity >= 1)printf("  V5UserID                  %s\n", mystuff->V5UserID);
  }

/*****************************************************************************/

  if(my_read_string("mfaktc.ini", "ComputerID", mystuff->ComputerID, 50))
  {
    /* no problem, don't use any */
    if(mystuff->verbosity >= 1)printf("  ComputerID                (none)\n");
    mystuff->ComputerID[0]='\0';
  }
  else
  {   
    if(mystuff->verbosity >= 1)printf("  ComputerID                %s\n", mystuff->ComputerID);
  }

/*****************************************************************************/

  for(i = 0; i < 256; i++)mystuff->stats.progressheader[i] = 0;
  if(my_read_string("mfaktc.ini", "ProgressHeader", mystuff->stats.progressheader, 250))
  {
//    sprintf(mystuff->stats.progressheader, "    class | candidates |    time |    ETA | avg. rate | SievePrimes | CPU wait");
    sprintf(mystuff->stats.progressheader, "Date   Time     Pct    ETA | Exponent    Bits | GHz-d/day    Sieve     Wait");
    printf("WARNING, no ProgressHeader specified in mfaktc.ini, using default\n");
  }
  if(mystuff->verbosity >= 2)printf("  ProgressHeader            \"%s\"\n", mystuff->stats.progressheader);

/*****************************************************************************/

  for(i = 0; i < 256; i++)mystuff->stats.progressformat[i] = 0;
  if(my_read_string("mfaktc.ini", "ProgressFormat", mystuff->stats.progressformat, 250))
  {
//    sprintf(mystuff->stats.progressformat, "%%C/%4d |    %%n | %%ts | %%e | %%rM/s |     %%s |  %%W%%%%", NUM_CLASSES);
    sprintf(mystuff->stats.progressformat, "%%d %%T  %%p %%e | %%M %%l-%%u |   %%g  %%s  %%W%%%%");
    printf("WARNING, no ProgressFormat specified in mfaktc.ini, using default\n");
  }
  if(mystuff->verbosity >= 2)printf("  ProgressFormat            \"%s\"\n", mystuff->stats.progressformat);

/*****************************************************************************/

  if(my_read_int("mfaktc.ini", "AllowSleep", &i))
  {
    printf("WARNING: Cannot read AllowSleep from mfaktc.ini, set to 0 by default\n");
    i = 0;
  }
  else if(i < 0 || i > 1)
  {
    printf("WARNING: AllowSleep must be 0 or 1, set to 0 by default\n");
    i = 0;
  }
  if(mystuff->verbosity >= 1)
  {
    if(i == 0)printf("  AllowSleep                no\n");
    else      printf("  AllowSleep                yes\n");
  }
  mystuff->allowsleep = i;

/*****************************************************************************/

  if(my_read_int("mfaktc.ini", "TimeStampInResults", &i))
  {
    printf("WARNING: Cannot read TimeStampInResults from mfaktc.ini, set to 0 by default\n");
    i = 0;
  }
  else if(i < 0 || i > 1)
  {
    printf("WARNING: TimeStampInResults must be 0 or 1, set to 0 by default\n");
    i=0;
  }
  if(mystuff->verbosity >= 1)
  {
    if(i == 0)printf("  TimeStampInResults        no\n");
    else      printf("  TimeStampInResults        yes\n");
  }
  mystuff->print_timestamp = i;

  return 0;
}
