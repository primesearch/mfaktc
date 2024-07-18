/*
This file is part of mfaktc.
Copyright (C) 2009, 2010, 2011, 2012, 2013, 2014  Oliver Weihe (o.weihe@t-online.de)
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
#include <math.h>
#include <cuda_runtime.h>
#include <time.h>

#include "params.h"
#include "my_types.h"
#include "output.h"
#include "compatibility.h"


void print_help(char *string)
{
  printf("mfaktc v%s Copyright (C) 2009, 2010, 2011, 2012, 2013, 2014, 2015  Oliver Weihe (o.weihe@t-online.de)\n", MFAKTC_VERSION);
  printf("This program comes with ABSOLUTELY NO WARRANTY; for details see COPYING.\n");
  printf("This is free software, and you are welcome to redistribute it\n");
  printf("under certain conditions; see COPYING for details.\n\n\n");

  printf("Usage: %s [options]\n", string);
  printf("  -h                     display this help and exit\n");
  printf("  -d <device number>     specify the device number used by this program\n");
  printf("  -tf <exp> <min> <max>  trial factor %s<exp> from 2^<min> to 2^<max> and exit\n", NAME_NUMBERS);
  printf("                         instead of parsing the worktodo file\n");
  printf("  -st                    run builtin selftest and exit\n");
  printf("  -st2                   same as -st but extended range for k_min/m_max\n");
  printf("  -v <number>            set verbosity (min = 0, default = 1, more = 2, max/debug = 3)\n");
  printf("\n");
  printf("options for debuging purposes\n");
  printf("  --timertest            run test of timer functions and exit\n");
  printf("  --sleeptest            run test of sleep functions and exit\n");
}


/*
print_dezXXX(intXXX a, char *buf) writes "a" into "buf" in decimal
"buf" must be preallocated with enough space.
Enough space is
  23 bytes for print_dez72()  (2^72 -1  has 22 decimal digits)
  30 bytes for print_dez96()  (2^96 -1  has 29 decimal digits)
  45 bytes for print_dez144() (2^144 -1 has 44 decimal digits)
  59 bytes for print_dez192() (2^192 -1 has 58 decimal digits)

*/

void print_dez72(int72 a, char *buf)
{
  int192 tmp;

  tmp.d5 = 0;
  tmp.d4 = 0;
  tmp.d3 = 0;
  tmp.d2 =                 a.d2 >> 16;
  tmp.d1 = (a.d2 << 16) + (a.d1 >>  8);
  tmp.d0 = (a.d1 << 24) +  a.d0;
  
  print_dez192(tmp, buf);
}


void print_dez144(int144 a, char *buf)
{
  int192 tmp;

  tmp.d5 = 0;
  tmp.d4 =                 a.d5 >>  8;
  tmp.d3 = (a.d5 << 24) +  a.d4;
  tmp.d2 = (a.d3 <<  8) + (a.d2 >> 16);
  tmp.d1 = (a.d2 << 16) + (a.d1 >>  8);
  tmp.d0 = (a.d1 << 24) +  a.d0;
  
  print_dez192(tmp, buf);
}


void print_dez96(int96 a, char *buf)
{
  int192 tmp;

  tmp.d5 = 0;
  tmp.d4 = 0;
  tmp.d3 = 0;
  tmp.d2 = a.d2;
  tmp.d1 = a.d1;
  tmp.d0 = a.d0;

  print_dez192(tmp, buf);
}


void print_dez192(int192 a, char *buf)
{
  char digit[58];
  int digits=0,carry,i=0;
  long long int tmp;
  
  while((a.d0!=0 || a.d1!=0 || a.d2!=0 || a.d3!=0 || a.d4!=0 || a.d5!=0) && digits<58)
  {
                                                   carry = a.d5%10; a.d5 /= 10;
    tmp = a.d4; tmp += (long long int)carry << 32; carry = tmp%10;  a.d4 =  tmp/10;
    tmp = a.d3; tmp += (long long int)carry << 32; carry = tmp%10;  a.d3 =  tmp/10;
    tmp = a.d2; tmp += (long long int)carry << 32; carry = tmp%10;  a.d2 =  tmp/10;
    tmp = a.d1; tmp += (long long int)carry << 32; carry = tmp%10;  a.d1 =  tmp/10;
    tmp = a.d0; tmp += (long long int)carry << 32; carry = tmp%10;  a.d0 =  tmp/10;
    digit[digits++] = carry;
  }
  if(digits == 0)sprintf(buf, "0");
  else
  {
    digits--;
    while(digits >= 0)
    {
      sprintf(&(buf[i++]), "%1d", digit[digits--]);
    }
  }
}


void print_timestamp(FILE *outfile)
{
  time_t now;
  char *ptr;

  now = time(NULL);
  ptr = ctime(&now);
  ptr[24] = '\0'; // cut off the newline
  fprintf(outfile, "[%s]\n", ptr);
}


void print_status_line(mystuff_t *mystuff)
{
  unsigned long long int eta;
  int i = 0, max_class_number;
  char buffer[256];
  int index = 0;
  time_t now;
  struct tm *tm_now = NULL;
  int time_read = 0;
  double val;
                        

  if(mystuff->mode == MODE_SELFTEST_SHORT) return; /* no output during short selftest */
  
#ifdef MORE_CLASSES
  max_class_number = 960;
#else
  max_class_number = 96;
#endif


  if(mystuff->stats.output_counter == 0)
  {
    printf("%s\n", mystuff->stats.progressheader);
    mystuff->stats.output_counter = 20;
  }
  if(mystuff->printmode == 0)mystuff->stats.output_counter--;
  
  while(mystuff->stats.progressformat[i] && i < 250)
  {
    if(mystuff->stats.progressformat[i] != '%')
    {
      buffer[index++] = mystuff->stats.progressformat[i];
      i++;
    }
    else
    {
      if(mystuff->stats.progressformat[i+1] == 'C')
      {
        index += sprintf(buffer + index, "%4d", mystuff->stats.class_number);
      }
      else if(mystuff->stats.progressformat[i+1] == 'c')
      {
        index += sprintf(buffer + index, "%3d", mystuff->stats.class_counter);
      }
      else if(mystuff->stats.progressformat[i+1] == 'p')
      {
        index += sprintf(buffer + index, "%5.1f", (double)(mystuff->stats.class_counter * 100) / (double)max_class_number);
      }
      else if(mystuff->stats.progressformat[i+1] == 'g')
      {
        if(mystuff->mode == MODE_NORMAL)
          index += sprintf(buffer + index, "%7.2f", mystuff->stats.ghzdays * 86400000.0f / ((double)mystuff->stats.class_time * (double)max_class_number));
        else
          index += sprintf(buffer + index, "   n.a.");
      }
      else if(mystuff->stats.progressformat[i+1] == 't')
      {
             if(mystuff->stats.class_time < 100000ULL  )index += sprintf(buffer + index, "%6.3f", (double)mystuff->stats.class_time/1000.0);
        else if(mystuff->stats.class_time < 1000000ULL )index += sprintf(buffer + index, "%6.2f", (double)mystuff->stats.class_time/1000.0);
        else if(mystuff->stats.class_time < 10000000ULL)index += sprintf(buffer + index, "%6.1f", (double)mystuff->stats.class_time/1000.0);
        else                                            index += sprintf(buffer + index, "%6.0f", (double)mystuff->stats.class_time/1000.0);
      }
      else if(mystuff->stats.progressformat[i+1] == 'e')
      {
        if(mystuff->mode == MODE_NORMAL)
        {
          if(mystuff->stats.class_time > 250)
          {
            eta = (mystuff->stats.class_time * (max_class_number - mystuff->stats.class_counter) + 500)  / 1000;
                 if(eta < 3600) index += sprintf(buffer + index, "%2" PRIu64 "m%02" PRIu64 "s", eta / 60, eta % 60);
            else if(eta < 86400)index += sprintf(buffer + index, "%2" PRIu64 "h%02" PRIu64 "m", eta / 3600, (eta / 60) % 60);
            else                index += sprintf(buffer + index, "%2" PRIu64 "d%02" PRIu64 "h", eta / 86400, (eta / 3600) % 24);
          }
          else                  index += sprintf(buffer + index, "  n.a.");
        }
        else if(mystuff->mode == MODE_SELFTEST_FULL)index += sprintf(buffer + index, "  n.a.");
      }
      else if(mystuff->stats.progressformat[i+1] == 'n')
      {
        if (mystuff->stats.cpu_wait == -2.0f) {		// Hack to indicate GPU sieving kernel
	  if(mystuff->stats.grid_count < (1000000000 / mystuff->gpu_sieve_processing_size + 1))
	    index += sprintf(buffer + index, "%6.2fM", (double)mystuff->stats.grid_count * mystuff->gpu_sieve_processing_size / 1000000.0);
	  else
	    index += sprintf(buffer + index, "%6.2fG", (double)mystuff->stats.grid_count * mystuff->gpu_sieve_processing_size / 1000000000.0);
	} else {					// CPU sieving
	  if(((unsigned long long int)mystuff->threads_per_grid * (unsigned long long int)mystuff->stats.grid_count) < 1000000000ULL)
            index += sprintf(buffer + index, "%6.2fM", (double)mystuff->threads_per_grid * (double)mystuff->stats.grid_count / 1000000.0);
          else
	    index += sprintf(buffer + index, "%6.2fG", (double)mystuff->threads_per_grid * (double)mystuff->stats.grid_count / 1000000000.0);
	}
      }
      else if(mystuff->stats.progressformat[i+1] == 'r')
      {
        if (mystuff->stats.cpu_wait == -2.0f)		// Hack to indicate GPU sieving kernel
          val = (double)mystuff->stats.grid_count * mystuff->gpu_sieve_processing_size / ((double)mystuff->stats.class_time * 1000.0);
	else						// CPU sieving
          val = (double)mystuff->threads_per_grid * (double)mystuff->stats.grid_count / ((double)mystuff->stats.class_time * 1000.0);
        
        if(val <= 999.99f) index += sprintf(buffer + index, "%6.2f", val);
        else               index += sprintf(buffer + index, "%6.1f", val);
      }
      else if(mystuff->stats.progressformat[i+1] == 's')
      {
        if (mystuff->stats.cpu_wait == -2.0f)		// Hack to indicate GPU sieving kernel
	  index += sprintf(buffer + index, "%7d", mystuff->gpu_sieve_primes-1);  // Output number of odd primes sieved
	else						// CPU sieving
          index += sprintf(buffer + index, "%7d", mystuff->sieve_primes);
      }
      else if(mystuff->stats.progressformat[i+1] == 'w')
      {
        index += sprintf(buffer + index, "(n.a.)"); /* mfakto only */
      }
      else if(mystuff->stats.progressformat[i+1] == 'W')
      {
        if(mystuff->stats.cpu_wait >= 0.0f)index += sprintf(buffer + index, "%6.2f", mystuff->stats.cpu_wait);
        else                               index += sprintf(buffer + index, "  n.a.");
      }
      else if(mystuff->stats.progressformat[i+1] == 'd')
      {
        if(!time_read)
        {
          now = time(NULL);
          tm_now = localtime(&now);
          time_read = 1;
        }
        index += strftime(buffer + index, 7, "%b %d", tm_now);
      }
      else if(mystuff->stats.progressformat[i+1] == 'T')
      {
        if(!time_read)
        {
          now = time(NULL);
          tm_now = localtime(&now);
          time_read = 1;
        }
        index += strftime(buffer + index, 6, "%H:%M", tm_now);
      }
      else if(mystuff->stats.progressformat[i+1] == 'U')
      {
        index += sprintf(buffer + index, "%s", mystuff->V5UserID);
      }
      else if(mystuff->stats.progressformat[i+1] == 'H')
      {
        index += sprintf(buffer + index, "%s", mystuff->ComputerID);
      }
      else if(mystuff->stats.progressformat[i+1] == 'M')
      {
        index += sprintf(buffer + index, "%-10u", mystuff->exponent);
      }
      else if(mystuff->stats.progressformat[i+1] == 'l')
      {
        index += sprintf(buffer + index, "%2d", mystuff->bit_min);
      }
      else if(mystuff->stats.progressformat[i+1] == 'u')
      {
        index += sprintf(buffer + index, "%2d", mystuff->bit_max_stage);
      }
      else if(mystuff->stats.progressformat[i+1] == '%')
      {
        buffer[index++] = '%';
      }
      else /* '%' + unknown format character -> just print "%<character>" */
      {
        buffer[index++] = '%';
        buffer[index++] = mystuff->stats.progressformat[i+1];
      }

      i += 2;
    }
    if(index > 200) /* buffer has 256 bytes, single format strings are limited to 50 bytes */
    {
      buffer[index] = 0;
      printf("%s", buffer);
      index = 0;
    }
  }
  
  
  if(mystuff->mode == MODE_NORMAL)
  {
    if(mystuff->printmode == 1)index += sprintf(buffer + index, "\r");
    else                       index += sprintf(buffer + index, "\n");
  }
  if(mystuff->mode == MODE_SELFTEST_FULL && mystuff->printmode == 0)
  {
    index += sprintf(buffer + index, "\n");
  }

  buffer[index] = 0;
  printf("%s", buffer);
}


void print_result_line(mystuff_t *mystuff, int factorsfound)
/* printf the final result line (STDOUT and resultfile) */
{
  char UID[110]; /* 50 (V5UserID) + 50 (ComputerID) + 8 + spare */
  char aidjson[MAX_LINE_LENGTH+11];
  char userjson[61]; /* 50 (V5UserID) + 11 spare */
  char computerjson[65];  /* 50 (ComputerID) + 15 spare */
  char factorjson[513];
  char txtstring[200];
  char jsonstring[1000];
  
  FILE *txtresultfile=NULL;
  FILE *jsonresultfile=NULL;
   
  
  if(mystuff->V5UserID[0] && mystuff->ComputerID[0])
    sprintf(UID, "UID: %s/%s, ", mystuff->V5UserID, mystuff->ComputerID);
  else
    UID[0]=0;
  if (mystuff->assignment_key[0])
      sprintf(aidjson, ", \"aid\":\"%s\"", mystuff->assignment_key);
  else
      aidjson[0] = 0;
  if (mystuff->V5UserID[0])
      sprintf(userjson, ", \"user\":\"%s\"", mystuff->V5UserID);
  else
      userjson[0] = 0;
  if (mystuff->ComputerID[0])
      sprintf(computerjson, ", \"computer\":\"%s\"", mystuff->ComputerID);
  else
      computerjson[0] = 0;
  if (mystuff->factorsstring[0])
      sprintf(factorjson, ", \"factors\":[%s]", mystuff->factorsstring);
  else
      factorjson[0] = 0;
    
  if(mystuff->mode == MODE_NORMAL)
  {
    txtresultfile = fopen(mystuff->resultfile, "a");
    jsonresultfile = fopen(mystuff->jsonresultfile, "a");
    if(mystuff->print_timestamp == 1)print_timestamp(txtresultfile);
  }
#ifndef MORE_CLASSES
  bool partialresult = (mystuff->mode == MODE_NORMAL) && (mystuff->stats.class_counter < 96);
#else
  bool partialresult = (mystuff->mode == MODE_NORMAL) && (mystuff->stats.class_counter < 960);
#endif
  if (factorsfound)
  {
    sprintf(txtstring, "found %d factor%s for %s%u from 2^%2d to 2^%2d %s[mfaktc %s %s]", 
        factorsfound, (factorsfound > 1) ? "s" : "", NAME_NUMBERS, mystuff->exponent, mystuff->bit_min, mystuff->bit_max_stage, partialresult ? "(partially tested) " : "", MFAKTC_VERSION, mystuff->stats.kernelname);
  }
  else
  {
    sprintf(txtstring, "no factor for %s%u from 2^%d to 2^%d [mfaktc %s %s]", NAME_NUMBERS, mystuff->exponent, mystuff->bit_min, mystuff->bit_max_stage, MFAKTC_VERSION, mystuff->stats.kernelname);
  }
  sprintf(jsonstring, "{\"exponent\":%u, \"worktype\":\"TF\", \"status\":\"%s\", \"bitlo\":%2d, \"bithi\":%2d, \"rangecomplete\":%s%s, \"program\":{\"name\":\"mfaktc\", \"version\":\"%s\", \"subversion\":\"%s\"}%s%s%s}",
      mystuff->exponent, factorsfound > 0 ? "F" : "NF", mystuff->bit_min, mystuff->bit_max_stage, partialresult ? "false" : "true", factorjson, MFAKTC_VERSION, mystuff->stats.kernelname, aidjson, userjson, computerjson);
  // TODO timestamp

  if(mystuff->mode != MODE_SELFTEST_SHORT)
  {
    printf("%s\n", txtstring);
  }
  if(mystuff->mode == MODE_NORMAL)
  {
    fprintf(txtresultfile, "%s%s\n", UID, txtstring);
    fclose(txtresultfile);
    fprintf(jsonresultfile, "%s\n", jsonstring);
    fclose(jsonresultfile);
  }
}


void print_factor(mystuff_t *mystuff, int factor_number, char *factor)
{
  char UID[110]; /* 50 (V5UserID) + 50 (ComputerID) + 8 + spare */
  FILE *resultfile = NULL;

  if(mystuff->V5UserID[0] && mystuff->ComputerID[0])
    sprintf(UID, "UID: %s/%s, ", mystuff->V5UserID, mystuff->ComputerID);
  else
    UID[0]=0;
    

  if(mystuff->mode == MODE_NORMAL)
  {
    resultfile = fopen(mystuff->resultfile, "a");
    if(mystuff->print_timestamp == 1 && factor_number == 0)print_timestamp(resultfile);
  }

  if(factor_number < 10)
  {
    if(mystuff->mode != MODE_SELFTEST_SHORT)
    {
      if(mystuff->printmode == 1 && factor_number == 0)printf("\n");
      printf("%s%u has a factor: %s\n", NAME_NUMBERS, mystuff->exponent, factor);
    }
    if(mystuff->mode == MODE_NORMAL)
    {
#ifndef MORE_CLASSES      
      fprintf(resultfile, "%s%s%u has a factor: %s [TF:%d:%d%s:mfaktc %s %s]\n", UID, NAME_NUMBERS, mystuff->exponent, factor, mystuff->bit_min, mystuff->bit_max_stage, ((mystuff->stopafterfactor == 2) && (mystuff->stats.class_counter <  96)) ? "*" : "" , MFAKTC_VERSION, mystuff->stats.kernelname);
#else      
      fprintf(resultfile, "%s%s%u has a factor: %s [TF:%d:%d%s:mfaktc %s %s]\n", UID, NAME_NUMBERS, mystuff->exponent, factor, mystuff->bit_min, mystuff->bit_max_stage, ((mystuff->stopafterfactor == 2) && (mystuff->stats.class_counter < 960)) ? "*" : "" , MFAKTC_VERSION, mystuff->stats.kernelname);
#endif
    }
  }
  else /* factor_number >= 10 */
  {
    if(mystuff->mode != MODE_SELFTEST_SHORT)      printf("%s%u: %d additional factors not shown\n",      NAME_NUMBERS, mystuff->exponent, factor_number-10);
    if(mystuff->mode == MODE_NORMAL)fprintf(resultfile,"%s%s%u: %d additional factors not shown\n", UID, NAME_NUMBERS, mystuff->exponent, factor_number-10);
  }

  if(mystuff->mode == MODE_NORMAL)fclose(resultfile);
}


double primenet_ghzdays(unsigned int exp, int bit_min, int bit_max)
/* estimate the GHZ-days for the current job
GHz-days = <magic constant> * pow(2, $bitlevel - 48) * 1680 / $exponent
      
magic constant is 0.016968 for TF to 65-bit and above
magic constant is 0.017832 for 63-and 64-bit
magic constant is 0.011160 for 62-bit and below

example using M50,000,000 from 2^69-2^70:
 = 0.016968 * pow(2, 70 - 48) * 1680 / 50000000
 = 2.3912767291392 GHz-days*/
{
  double ghzdays = 0.0;
  bit_min++;

  while(bit_min <= bit_max && bit_min <= 62)
  {
    ghzdays += 0.011160 * pow(2.0, (double)bit_min - 48.0);
    bit_min++;
  }
  while(bit_min <= bit_max && bit_min <= 64)
  {
    ghzdays += 0.017832 * pow(2.0, (double)bit_min - 48.0);
    bit_min++;
  }
  while(bit_min <= bit_max)
  {
    ghzdays += 0.016968 * pow(2.0, (double)bit_min - 48.0);
    bit_min++;
  }

  ghzdays *= 1680.0 / (double)exp;

  return ghzdays;
}
