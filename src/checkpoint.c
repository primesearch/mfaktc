/*
This file is part of mfaktc.
Copyright (c) 2009-2011, 2013  Oliver Weihe (o.weihe@t-online.de)

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
#include <errno.h>

#include "params.h"

unsigned int checkpoint_checksum(char *string, int chars)
/* generates a CRC-32 like checksum of the string */
{
    unsigned int chksum = 0;
    int i, j;

    for (i = 0; i < chars; i++) {
        for (j = 7; j >= 0; j--) {
            if ((chksum >> 31) == (((unsigned int)(string[i] >> j)) & 1)) {
                chksum <<= 1;
            } else {
                chksum = (chksum << 1) ^ 0x04C11DB7;
            }
        }
    }
    return chksum;
}

void checkpoint_write(unsigned int exp, int bit_min, int bit_max, int cur_class, int num_factors, char *factors_string,
                      unsigned long long int bit_level_time)
/*
checkpoint_write() writes the checkpoint file.
*/
{
    FILE *f;
    char buffer[600], filename[20];
    unsigned int i;

    sprintf(filename, "%s%u.ckp", NAME_NUMBERS, exp);

    f = fopen(filename, "w");
    if (f == NULL) {
        printf("Warning: could not write checkpoint file \"%s\"\n", filename);
    } else {
        sprintf(buffer, "%s%u %d %d %d %s: %d %d %s %llu", NAME_NUMBERS, exp, bit_min, bit_max, NUM_CLASSES, MFAKTC_VERSION, cur_class,
                num_factors, strlen(factors_string) ? factors_string : "0", bit_level_time);
        i = checkpoint_checksum(buffer, strlen(buffer));
        fprintf(f, "%s%u %d %d %d %s: %d %d %s %llu %08X", NAME_NUMBERS, exp, bit_min, bit_max, NUM_CLASSES, MFAKTC_VERSION, cur_class,
                num_factors, strlen(factors_string) ? factors_string : "0", bit_level_time, i);
        fclose(f);
        f = NULL;
    }
}

int checkpoint_read(unsigned int exp, int bit_min, int bit_max, int *cur_class, int *num_factors, char *factors_string,
                    unsigned long long int *bit_level_time)
/*
checkpoint_read() reads the checkpoint file and compares values for exp,
bit_min, bit_max, NUM_CLASSES read from file with current values.
If these parameters are equal than it sets cur_class, num_factors,
factors_string, and class_time to the values from the checkpoint file.

returns 1 on success (valid checkpoint file)
returns 0 otherwise
*/
{
    FILE *f;
    int ret = 0, i, chksum;
    char buffer[600], buffer2[600], *ptr, filename[20];

    for (i = 0; i < 600; i++)
        buffer[i] = 0;

    *cur_class   = -1;
    *num_factors = 0;

    sprintf(filename, "%s%u.ckp", NAME_NUMBERS, exp);

    f = fopen(filename, "r");
    if (f == NULL) {
        return 0;
    }
    i = fread(buffer, sizeof(char), 599, f);
    sprintf(buffer2, "%s%u %d %d %d %s: ", NAME_NUMBERS, exp, bit_min, bit_max, NUM_CLASSES, MFAKTC_VERSION);
    ptr = strstr(buffer, buffer2);
    if (ptr == buffer) {
        i = strlen(buffer2);
        if (i < 70) {
            ptr = &(buffer[i]);
            sscanf(ptr, "%d %d %s %llu", cur_class, num_factors, factors_string, bit_level_time);
            sprintf(buffer2, "%s%u %d %d %d %s: %d %d %s %llu", NAME_NUMBERS, exp, bit_min, bit_max, NUM_CLASSES, MFAKTC_VERSION,
                    *cur_class, *num_factors, factors_string, *bit_level_time);
            chksum = checkpoint_checksum(buffer2, strlen(buffer2));
            sprintf(buffer2, "%s%u %d %d %d %s: %d %d %s %llu %08X", NAME_NUMBERS, exp, bit_min, bit_max, NUM_CLASSES, MFAKTC_VERSION,
                    *cur_class, *num_factors, factors_string, *bit_level_time, chksum);
            if (*cur_class >= 0 && *cur_class < NUM_CLASSES && *num_factors >= 0 && strlen(buffer) == strlen(buffer2) &&
                strstr(buffer, buffer2) == buffer &&
                ((*num_factors == 0 && strlen(factors_string) == 1) || (*num_factors >= 1 && strlen(factors_string) > 1))) {
                ret = 1;
            }
            if (factors_string[0] == '0') factors_string[0] = 0;
        }
    }
    fclose(f);
    f = NULL;
    return ret;
}

void checkpoint_delete(unsigned int exp)
/*
tries to delete the checkpoint file
*/
{
    char filename[20];
    sprintf(filename, "%s%u.ckp", NAME_NUMBERS, exp);

    if (remove(filename)) {
        if (errno != ENOENT) /* ENOENT = "No such file or directory" -> there was no checkpoint file */
        {
            printf("Warning: can't delete the checkpoint file \"%s\"\n", filename);
        }
    }
}
