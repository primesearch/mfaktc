/*
This file is part of mfaktc.
Copyright (C) 2009, 2010, 2011, 2013, 2015, 2024  Oliver Weihe (o.weihe@t-online.de)

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

#include <cuda_runtime.h>

#include "params.h"
#include "my_types.h"
#include "output.h"
#include "crc.h"
#include "compatibility.h"

void checkpoint_write(unsigned int exp, int bit_min, int bit_max, int cur_class, int num_factors, int96 factors[MAX_FACTORS_PER_JOB],
                      unsigned long long int bit_level_time)
/*
checkpoint_write() writes the checkpoint file.
*/
{
    FILE *f;
    const int MAX_FACTOR_BUFFER_LENGTH       = MAX_FACTORS_PER_JOB * MAX_DEZ_96_STRING_LENGTH;
    const int MAX_BUFFER_LENGTH              = MAX_FACTOR_BUFFER_LENGTH + 100;
    const int MAX_CHECKPOINT_FILENAME_LENGTH = 40;
    char buffer[MAX_BUFFER_LENGTH], filename[MAX_CHECKPOINT_FILENAME_LENGTH], factors_buffer[MAX_FACTOR_BUFFER_LENGTH];
    unsigned int i, factors_buffer_length;

    snprintf(filename, MAX_CHECKPOINT_FILENAME_LENGTH, "%s%u_%d-%d_%d.ckp", NAME_NUMBERS, exp, bit_min, bit_max, NUM_CLASSES);
    if (factors[0].d0 || factors[0].d1 || factors[0].d2) {
        i = 0;
        char factor[MAX_DEZ_96_STRING_LENGTH];
        print_dez96(factors[i++], factor);
        factors_buffer_length = sprintf(factors_buffer, "%s", factor);
        for (; i < MAX_FACTORS_PER_JOB; i++) {
            if (factors[i].d0 || factors[i].d1 || factors[i].d2) {
                print_dez96(factors[i], factor);
                factors_buffer_length += sprintf(factors_buffer + factors_buffer_length, ",%s", factor);
            }
        }
    } else {
        sprintf(factors_buffer, "0");
    }

    f = fopen(filename, "w");
    if (f == NULL) {
        printf("WARNING, could not write checkpoint file \"%s\"\n", filename);
    } else {
        sprintf(buffer, "%s%u %d %d %d %s: %d %d %s %llu", NAME_NUMBERS, exp, bit_min, bit_max, NUM_CLASSES, MFAKTC_CHECKPOINT_VERSION,
                cur_class, num_factors, strlen(factors_buffer) ? factors_buffer : "0", bit_level_time);
        i = crc32_checksum(buffer, strlen(buffer));
        fprintf(f, "%s%u %d %d %d %s: %d %d %s %llu %08X", NAME_NUMBERS, exp, bit_min, bit_max, NUM_CLASSES, MFAKTC_CHECKPOINT_VERSION,
                cur_class, num_factors, strlen(factors_buffer) ? factors_buffer : "0", bit_level_time, i);
        fclose(f);
        f = NULL;
    }
}

int checkpoint_read(unsigned int exp, int bit_min, int bit_max, int *cur_class, int *num_factors, int96 factors[MAX_FACTORS_PER_JOB],
                    unsigned long long int *bit_level_time)
/*
checkpoint_read() reads the checkpoint file and compares values for exp,
bit_min, bit_max, NUM_CLASSES read from file with current values.
If these parameters are equal than it sets cur_class, num_factors,
factors, and class_time to the values from the checkpoint file.

returns 1 on success (valid checkpoint file)
returns 0 otherwise
*/
{
    FILE *f;
    int ret = 0, i, chksum;
    char buffer[600], buffer2[600], *ptr, filename[40], factors_buffer[500];

    for (i = 0; i < 600; i++)
        buffer[i] = 0;

    *cur_class   = -1;
    *num_factors = 0;

    sprintf(filename, "%s%u_%d-%d_%d.ckp", NAME_NUMBERS, exp, bit_min, bit_max, NUM_CLASSES);

    f = fopen(filename, "r");
    if (f == NULL) {
        return 0;
    }
    i = fread(buffer, sizeof(char), 599, f);
    sprintf(buffer2, "%s%u %d %d %d %s: ", NAME_NUMBERS, exp, bit_min, bit_max, NUM_CLASSES, MFAKTC_CHECKPOINT_VERSION);

    ptr = strstr(buffer, buffer2);
    if (ptr == buffer) {
        i = strlen(buffer2);
        if (i < 70) {
            ptr = &(buffer[i]);
            sscanf(ptr, "%d %d %s %llu", cur_class, num_factors, factors_buffer, bit_level_time);
            sprintf(buffer2, "%s%u %d %d %d %s: %d %d %s %llu", NAME_NUMBERS, exp, bit_min, bit_max, NUM_CLASSES, MFAKTC_CHECKPOINT_VERSION,
                    *cur_class, *num_factors, factors_buffer, *bit_level_time);
            chksum = crc32_checksum(buffer2, strlen(buffer2));
            sprintf(buffer2, "%s%u %d %d %d %s: %d %d %s %llu %08X", NAME_NUMBERS, exp, bit_min, bit_max, NUM_CLASSES,
                    MFAKTC_CHECKPOINT_VERSION, *cur_class, *num_factors, factors_buffer, *bit_level_time, chksum);
            if (*cur_class >= 0 && *cur_class < NUM_CLASSES && *num_factors >= 0 && strlen(buffer) == strlen(buffer2) &&
                strstr(buffer, buffer2) == buffer &&
                ((*num_factors == 0 && strlen(factors_buffer) == 1) || (*num_factors >= 1 && strlen(factors_buffer) > 1))) {
                ret = 1;
            }

            // clang-format off
            if (factors_buffer[0] == '0') {
                for (i = 0; i < MAX_FACTORS_PER_JOB; i++) {
                    factors[i].d0 = 0;
                    factors[i].d1 = 0;
                    factors[i].d2 = 0;
                }
            } else {
                char *tok = strtok(factors_buffer, ",");
                for (i = 0; i < MAX_FACTORS_PER_JOB; i++) {
                    if (tok == NULL) {
                        factors[i].d0 = 0;
                        factors[i].d1 = 0;
                        factors[i].d2 = 0;
                    } else {
                        factors[i] = parse_dez96(tok);
                        tok = strtok(NULL, ",");
                    }
                }
            }
            // clang-format on
        }
    }
    fclose(f);
    f = NULL;
    return ret;
}

void checkpoint_delete(unsigned int exp, int bit_min, int bit_max)
/*
tries to delete the checkpoint file
*/
{
    char filename[40];
    sprintf(filename, "%s%u_%d-%d_%d.ckp", NAME_NUMBERS, exp, bit_min, bit_max, NUM_CLASSES);

    if (remove(filename)) {
        if (errno != ENOENT) /* ENOENT = "No such file or directory" -> there was no checkpoint file */
        {
            printf("WARNING: can't delete the checkpoint file \"%s\"\n", filename);
        }
    }
}
