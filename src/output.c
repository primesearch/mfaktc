/*
This file is part of mfaktc.
Copyright (C) 2009-2015, 2018, 2019, 2024  Oliver Weihe (o.weihe@t-online.de)
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
#include <stdarg.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <time.h>

#include "params.h"
#include "my_types.h"
#include "output.h"
#include "compatibility.h"
#include "crc.h"

/* Visual C++ introduced stdbool support in VS 2013 */
#if defined(_MSC_VER) && _MSC_VER < 1800
#define bool  int
#define true  1
#define false 0
#else
#include <stdbool.h>
#endif

void print_help(char *string)
{
    printf("mfaktc v%s Copyright (C) 2009-2015, 2018, 2019, 2024 Oliver Weihe (o.weihe@t-online.de)\n", MFAKTC_VERSION);
    printf("This program comes with ABSOLUTELY NO WARRANTY; for details see COPYING.\n");
    printf("This is free software, and you are welcome to redistribute it\n");
    printf("under certain conditions; see COPYING for details.\n\n\n");

    printf("Usage: %s [options]\n", string);
    printf("  -h                     display this help and exit\n");
    printf("  -d <device number>     specify the device number used by this program\n");
    printf("  -tf <exp> <min> <max>  trial factor %s<exp> from 2^<min> to 2^<max> and exit\n", NAME_NUMBERS);
    printf("                         instead of parsing the worktodo file\n");
    printf("  -st                    run built-in self-test and exit\n");
    printf("  -st2                   same as -st but extended range for k_min and k_max\n");
    printf("  -v <number>            set verbosity (min = 0, default = 1, more = 2, max = 3)\n");
    printf("\n");
    printf("options for debuging purposes\n");
    printf("  --timertest            run test of timer functions and exit\n");
    printf("  --sleeptest            run test of sleep functions and exit\n");
}

void logprintf(mystuff_t *mystuff, const char *fmt, ...)
{
    va_list args;

    va_start(args, fmt);
    int len = vfprintf(stdout, fmt, args);
    va_end(args);

    if (mystuff->logging == 1 && mystuff->logfileptr != NULL && len > 0) {
        if (mystuff->printmode == 1) {
            char *buffer = (char *)malloc(len + 1);
            va_start(args, fmt);
            vsnprintf(buffer, len + 1, fmt, args);
            va_end(args);

            // Replace to CR to LF if it's last char in the string when writing to logfile
            if (buffer[len - 1] == '\r') buffer[len - 1] = '\n';

            fprintf(mystuff->logfileptr, "%s", buffer);
            free(buffer);
        } else {
            va_start(args, fmt);
            vfprintf(mystuff->logfileptr, fmt, args);
            va_end(args);
        }
    }
}

/*
print_dezXXX(intXXX a, char *buf) writes "a" into "buf" in decimal
"buf" must be preallocated with enough space.
Enough space is
  30 bytes for print_dez96()  (2^96 -1  has 29 decimal digits)
  59 bytes for print_dez192() (2^192 -1 has 58 decimal digits)
*/

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
    int digits = 0, carry, i = 0;
    long long int tmp;

    // clang-format off
    while ((a.d0 != 0 || a.d1 != 0 || a.d2 != 0 || a.d3 != 0 || a.d4 != 0 || a.d5 != 0) && digits < 58) {
        carry = a.d5 % 10; a.d5 /= 10;
        tmp = a.d4; tmp += (long long int)carry << 32; carry = tmp % 10;  a.d4 = tmp / 10;
        tmp = a.d3; tmp += (long long int)carry << 32; carry = tmp % 10;  a.d3 = tmp / 10;
        tmp = a.d2; tmp += (long long int)carry << 32; carry = tmp % 10;  a.d2 = tmp / 10;
        tmp = a.d1; tmp += (long long int)carry << 32; carry = tmp % 10;  a.d1 = tmp / 10;
        tmp = a.d0; tmp += (long long int)carry << 32; carry = tmp % 10;  a.d0 = tmp / 10;
        digit[digits++] = carry;
    }
    // clang-format on
    if (digits == 0)
        sprintf(buf, "0");
    else {
        digits--;
        while (digits >= 0) {
            sprintf(&(buf[i++]), "%1d", digit[digits--]);
        }
    }
}

int96 parse_dez96(char *str)
{
    int96 result = { 0, 0, 0 };
    int len      = strlen(str);
    int i;
    while (*str == '0' && *(str + 1) != '\0') {
        str++;
        len--;
    }
    if (len == 0 || (len == 1 && *str == '0')) {
        return result;
    }
    for (i = 0; i < len; i++) {
        if (str[i] < '0' || str[i] > '9') {
            continue;
        }
        int digit = str[i] - '0';
        unsigned long long int carry;
        carry     = (unsigned long long int)result.d0 * 10 + digit;
        result.d0 = carry & 0xFFFFFFFF;
        carry >>= 32;
        carry += (unsigned long long int)result.d1 * 10;
        result.d1 = carry & 0xFFFFFFFF;
        carry >>= 32;
        carry += (unsigned long long int)result.d2 * 10;
        result.d2 = carry & 0xFFFFFFFF;
    }
    return result;
}

void print_timestamp(FILE *outfile)
{
    char *ptr;
    const time_t now = time(NULL);

    ptr     = asctime(gmtime(&now));
    ptr[24] = '\0'; // cut off the newline
    fprintf(outfile, "[%s]\n", ptr);
}

void print_status_line(mystuff_t *mystuff)
{
    unsigned long long int eta;
    unsigned long long int elapsed;
    int i = 0, max_class_number;
    char buffer[256];
    int index = 0;
    time_t now;
    struct tm *tm_now = NULL;
    int time_read     = 0;
    double val;

    if (mystuff->mode == MODE_SELFTEST_SHORT) return; /* no output during short selftest */

#ifdef MORE_CLASSES
    max_class_number = 960;
#else
    max_class_number = 96;
#endif

    if (mystuff->stats.output_counter == 0) {
        logprintf(mystuff, "%s\n", mystuff->stats.progressheader);
        mystuff->stats.output_counter = 20;
    }
    if (mystuff->printmode == 0) mystuff->stats.output_counter--;

    while (mystuff->stats.progressformat[i] && i < 250) {
        if (mystuff->stats.progressformat[i] != '%') {
            buffer[index++] = mystuff->stats.progressformat[i];
            i++;
        } else {
            if (mystuff->stats.progressformat[i + 1] == 'C') {
                index += sprintf(buffer + index, "%4d", mystuff->stats.class_number);
            } else if (mystuff->stats.progressformat[i + 1] == 'c') {
                index += sprintf(buffer + index, "%3d", mystuff->stats.class_counter);
            } else if (mystuff->stats.progressformat[i + 1] == 'p') {
                index += sprintf(buffer + index, "%5.1f", (double)(mystuff->stats.class_counter * 100) / (double)max_class_number);
            } else if (mystuff->stats.progressformat[i + 1] == 'g') {
                if (mystuff->mode == MODE_NORMAL)
                    index += sprintf(buffer + index, "%8.2f",
                                     mystuff->stats.ghzdays * 86400000.0f / ((double)mystuff->stats.class_time * (double)max_class_number));
                else
                    index += sprintf(buffer + index, "   n.a.");
            } else if (mystuff->stats.progressformat[i + 1] == 't') {
                if (mystuff->stats.class_time < 100000ULL)
                    index += sprintf(buffer + index, "%6.3f", (double)mystuff->stats.class_time / 1000.0);
                else if (mystuff->stats.class_time < 1000000ULL)
                    index += sprintf(buffer + index, "%6.2f", (double)mystuff->stats.class_time / 1000.0);
                else if (mystuff->stats.class_time < 10000000ULL)
                    index += sprintf(buffer + index, "%6.1f", (double)mystuff->stats.class_time / 1000.0);
                else
                    index += sprintf(buffer + index, "%6.0f", (double)mystuff->stats.class_time / 1000.0);
            } else if (mystuff->stats.progressformat[i + 1] == 'E') {
                if (mystuff->mode == MODE_NORMAL) {
                    elapsed = mystuff->stats.bit_level_time / 1000;
                    if (elapsed < 3600)
                        index += sprintf(buffer + index, "%2" PRIu64 "m%02" PRIu64 "s", elapsed / 60, elapsed % 60);
                    else if (elapsed < 86400)
                        index += sprintf(buffer + index, "%2" PRIu64 "h%02" PRIu64 "m", elapsed / 3600, (elapsed / 60) % 60);
                    else
                        index += sprintf(buffer + index, "%2" PRIu64 "d%02" PRIu64 "h", elapsed / 86400, (elapsed / 3600) % 24);
                } else if (mystuff->mode == MODE_SELFTEST_FULL)
                    index += sprintf(buffer + index, "  n.a.");
            } else if (mystuff->stats.progressformat[i + 1] == 'e') {
                if (mystuff->mode == MODE_NORMAL) {
                    if (mystuff->stats.class_time > 250) {
                        eta = (mystuff->stats.class_time * (max_class_number - mystuff->stats.class_counter) + 500) / 1000;
                        if (eta < 3600)
                            index += sprintf(buffer + index, "%2" PRIu64 "m%02" PRIu64 "s", eta / 60, eta % 60);
                        else if (eta < 86400)
                            index += sprintf(buffer + index, "%2" PRIu64 "h%02" PRIu64 "m", eta / 3600, (eta / 60) % 60);
                        else
                            index += sprintf(buffer + index, "%2" PRIu64 "d%02" PRIu64 "h", eta / 86400, (eta / 3600) % 24);
                    } else
                        index += sprintf(buffer + index, "  n.a.");
                } else if (mystuff->mode == MODE_SELFTEST_FULL)
                    index += sprintf(buffer + index, "  n.a.");
            } else if (mystuff->stats.progressformat[i + 1] == 'n') {
                if (mystuff->stats.cpu_wait == -2.0f) { // Hack to indicate GPU sieving kernel
                    if (mystuff->stats.grid_count < (1000000000 / mystuff->gpu_sieve_processing_size + 1))
                        index += sprintf(buffer + index, "%6.2fM",
                                         (double)mystuff->stats.grid_count * mystuff->gpu_sieve_processing_size / 1000000.0);
                    else
                        index += sprintf(buffer + index, "%6.2fG",
                                         (double)mystuff->stats.grid_count * mystuff->gpu_sieve_processing_size / 1000000000.0);
                } else { // CPU sieving
                    if (((unsigned long long int)mystuff->threads_per_grid * (unsigned long long int)mystuff->stats.grid_count) <
                        1000000000ULL)
                        index += sprintf(buffer + index, "%6.2fM",
                                         (double)mystuff->threads_per_grid * (double)mystuff->stats.grid_count / 1000000.0);
                    else
                        index += sprintf(buffer + index, "%6.2fG",
                                         (double)mystuff->threads_per_grid * (double)mystuff->stats.grid_count / 1000000000.0);
                }
            } else if (mystuff->stats.progressformat[i + 1] == 'r') {
                if (mystuff->stats.cpu_wait == -2.0f) // Hack to indicate GPU sieving kernel
                    val = (double)mystuff->stats.grid_count * mystuff->gpu_sieve_processing_size /
                          ((double)mystuff->stats.class_time * 1000.0);
                else // CPU sieving
                    val = (double)mystuff->threads_per_grid * (double)mystuff->stats.grid_count /
                          ((double)mystuff->stats.class_time * 1000.0);

                if (val <= 999.99f)
                    index += sprintf(buffer + index, "%6.2f", val);
                else
                    index += sprintf(buffer + index, "%6.1f", val);
            } else if (mystuff->stats.progressformat[i + 1] == 's') {
                if (mystuff->stats.cpu_wait == -2.0f) // Hack to indicate GPU sieving kernel
                    index += sprintf(buffer + index, "%7d", mystuff->gpu_sieve_primes - 1); // Output number of odd primes sieved
                else // CPU sieving
                    index += sprintf(buffer + index, "%7d", mystuff->sieve_primes);
            } else if (mystuff->stats.progressformat[i + 1] == 'w') {
                index += sprintf(buffer + index, "(n.a.)"); /* mfakto only */
            } else if (mystuff->stats.progressformat[i + 1] == 'W') {
                if (mystuff->stats.cpu_wait >= 0.0f)
                    index += sprintf(buffer + index, "%6.2f", mystuff->stats.cpu_wait);
                else
                    index += sprintf(buffer + index, "  n.a.");
            } else if (mystuff->stats.progressformat[i + 1] == 'd') {
                if (!time_read) {
                    now       = time(NULL);
                    tm_now    = localtime(&now);
                    time_read = 1;
                }
                index += strftime(buffer + index, 7, "%b %d", tm_now);
            } else if (mystuff->stats.progressformat[i + 1] == 'T') {
                if (!time_read) {
                    now       = time(NULL);
                    tm_now    = localtime(&now);
                    time_read = 1;
                }
                index += strftime(buffer + index, 6, "%H:%M", tm_now);
            } else if (mystuff->stats.progressformat[i + 1] == 'U') {
                index += sprintf(buffer + index, "%s", mystuff->V5UserID);
            } else if (mystuff->stats.progressformat[i + 1] == 'H') {
                index += sprintf(buffer + index, "%s", mystuff->ComputerID);
            } else if (mystuff->stats.progressformat[i + 1] == 'M') {
                index += sprintf(buffer + index, "%-10u", mystuff->exponent);
            } else if (mystuff->stats.progressformat[i + 1] == 'l') {
                index += sprintf(buffer + index, "%2d", mystuff->bit_min);
            } else if (mystuff->stats.progressformat[i + 1] == 'u') {
                index += sprintf(buffer + index, "%2d", mystuff->bit_max_stage);
            } else if (mystuff->stats.progressformat[i + 1] == '%') {
                buffer[index++] = '%';
            } else /* '%' + unknown format character -> just print "%<character>" */
            {
                buffer[index++] = '%';
                buffer[index++] = mystuff->stats.progressformat[i + 1];
            }

            i += 2;
        }
        if (index > 200) /* buffer has 256 bytes, single format strings are limited to 50 bytes */
        {
            buffer[index] = 0;
            logprintf(mystuff, "%s", buffer);
            index = 0;
        }
    }

    if (mystuff->mode == MODE_NORMAL) {
        if (mystuff->printmode == 1)
            index += sprintf(buffer + index, "\r");
        else
            index += sprintf(buffer + index, "\n");
    }
    if (mystuff->mode == MODE_SELFTEST_FULL && mystuff->printmode == 0) {
        index += sprintf(buffer + index, "\n");
    }

    buffer[index] = 0;
    logprintf(mystuff, "%s", buffer);
}

void get_utc_timestamp(char *timestamp)
{
    time_t now;
    struct tm *utc_time;

    time(&now);
    utc_time = gmtime(&now);
    strftime(timestamp, sizeof(char[50]), "%Y-%m-%d %H:%M:%S", utc_time);
}

const char *getArchitecture()
{
#if defined(__x86_64__) || defined(_M_X64)
    return "x86_64";
#elif defined(i386) || defined(__i386__) || defined(__i386) || defined(_M_IX86)
    return "x86_32";
#elif defined(__aarch64__) || defined(_M_ARM64)
    return "ARM64";
#else
    return "";
#endif
}

const char *getOS()
{
#if defined(_WIN32) || defined(_WIN64)
    return "Windows";
#elif defined(__APPLE__)
    return "Darwin";
#elif defined(__linux__)
    return "Linux";
#elif defined(__unix__)
    return "Unix";
#endif
}

void getOSJSON(char *string)
{
    sprintf(string, ", \"os\":{\"os\": \"%s\", \"architecture\": \"%s\"}", getOS(), getArchitecture());
}

static int cmp_int96(const void *p1, const void *p2)
{
    int96 *a = (int96 *)p1, *b = (int96 *)p2;

    // clang-format off
    if (a->d2 > b->d2)      return 1;
    else if (a->d2 < b->d2) return -1;
    else
        if (a->d1 > b->d1)      return 1;
        else if (a->d1 < b->d1) return -1;
        else
            if (a->d0 > b->d0)      return 1;
            else if (a->d0 < b->d0) return -1;
            else                    return 0;
    // clang-format on
}

void print_result_line(mystuff_t *mystuff, int factorsfound)
// printf the final result line to STDOUT and to resultfile if LegacyResultsTxt set to 1.
// Prints JSON string to the jsonresultfile for Mersenne numbers as well.
{
    char UID[110]; /* 50 (V5UserID) + 50 (ComputerID) + 8 + spare */
    int string_length = 0, factors_list_length = 0, factors_quote_list_length = 0, checksum, json_checksum;
    char aidjson[MAX_LINE_LENGTH + 11];
    char userjson[62]; /* 50 (V5UserID) + 11 spare + null character */
    char computerjson[66]; /* 50 (ComputerID) + 15 spare + null character */
    char factorjson[514];
    char factors_list[500];
    char factors_quote_list[500];
    char osjson[200];
    char details[50];
    char txtstring[200];
    char json_checksum_string[750];
    char timestamp[50];

    FILE *txtresultfile = NULL;

#ifndef WAGSTAFF
    char jsonstring[1350];
    FILE *jsonresultfile = NULL;
#endif

    if (mystuff->V5UserID[0] && mystuff->ComputerID[0])
        sprintf(UID, "UID: %s/%s, ", mystuff->V5UserID, mystuff->ComputerID);
    else
        UID[0] = 0;

    if (mystuff->assignment_key[0] && strspn(mystuff->assignment_key, "0123456789abcdefABCDEF") == 32 &&
        strlen(mystuff->assignment_key) == 32)
        sprintf(aidjson, ", \"aid\":\"%s\"", mystuff->assignment_key);
    else
        aidjson[0] = 0;

    if (mystuff->V5UserID[0])
        snprintf(userjson, sizeof(userjson), ", \"user\":\"%s\"", mystuff->V5UserID);
    else
        userjson[0] = 0;

    if (mystuff->ComputerID[0])
        snprintf(computerjson, sizeof(computerjson), ", \"computer\":\"%s\"", mystuff->ComputerID);
    else
        computerjson[0] = 0;

    if (factorsfound) {
        int i = 0;
        qsort(mystuff->factors, MAX_FACTORS_PER_JOB, sizeof(mystuff->factors[0]), cmp_int96);
        while (i < MAX_FACTORS_PER_JOB && mystuff->factors[i].d0 == 0 && mystuff->factors[i].d1 == 0 && mystuff->factors[i].d2 == 0) {
            i++;
        }
        char factor[MAX_DEZ_96_STRING_LENGTH];
        print_dez96(mystuff->factors[i++], factor);
        factors_list_length       = sprintf(factors_list, "%s", factor);
        factors_quote_list_length = sprintf(factors_quote_list, "\"%s\"", factor);
        for (; i < MAX_FACTORS_PER_JOB; i++) {
            if (mystuff->factors[i].d0 == 0 && mystuff->factors[i].d1 == 0 && mystuff->factors[i].d2 == 0) {
                continue;
            }
            print_dez96(mystuff->factors[i], factor);
            factors_list_length += sprintf(factors_list + factors_list_length, ",%s", factor);
            factors_quote_list_length += sprintf(factors_quote_list + factors_quote_list_length, ",\"%s\"", factor);
        }
    } else {
        factors_list[0]       = 0;
        factors_quote_list[0] = 0;
    }

    if (factors_quote_list[0])
        snprintf(factorjson, sizeof(factorjson), ", \"factors\":[%s]", factors_quote_list);
    else {
        factorjson[0] = 0;
    }

    getOSJSON(osjson);
    get_utc_timestamp(timestamp);

    if (mystuff->mode == MODE_NORMAL) {
#ifndef WAGSTAFF
        jsonresultfile = fopen(mystuff->jsonresultfile, "a");
#endif
        if (mystuff->legacy_results_txt == 1) {
            txtresultfile = fopen(mystuff->resultfile, "a");
            if (mystuff->print_timestamp == 1) print_timestamp(txtresultfile);
        }
    }
#ifndef MORE_CLASSES
    bool partialresult = (mystuff->mode == MODE_NORMAL) && (mystuff->stats.class_counter < 96);
#else
    bool partialresult = (mystuff->mode == MODE_NORMAL) && (mystuff->stats.class_counter < 960);
#endif
    if (factorsfound) {
        string_length = sprintf(txtstring, "found %d factor%s for %s%u from 2^%2d to 2^%2d %s", factorsfound, (factorsfound > 1) ? "s" : "",
                                NAME_NUMBERS, mystuff->exponent, mystuff->bit_min, mystuff->bit_max_stage,
                                partialresult ? "(partially tested) " : "");
    } else {
        string_length = sprintf(txtstring, "no factor for %s%u from 2^%d to 2^%d", NAME_NUMBERS, mystuff->exponent, mystuff->bit_min,
                                mystuff->bit_max_stage);
    }

    int cuda_major = mystuff->cuda_toolkit / 1000;
    int cuda_minor = (mystuff->cuda_toolkit % 1000) / 10;
    int arch_major = mystuff->cuda_arch / 100;
    int arch_minor = (mystuff->cuda_arch % 100) / 10;
    sprintf(details, "CUDA %d.%d arch %d.%d", cuda_major, cuda_minor, arch_major, arch_minor);

    string_length += sprintf(txtstring + string_length, " [mfaktc %s %s %s]", MFAKTC_VERSION, mystuff->stats.kernelname, details);

    checksum = crc32_checksum(txtstring, string_length);
    sprintf(txtstring + string_length, " %08X", checksum);
#ifndef WAGSTAFF
    snprintf(json_checksum_string, sizeof(json_checksum_string), "%u;TF;%s;;%d;%d;%u;;;mfaktc;%s;%s;%s;%s;%s;%s", mystuff->exponent,
             factors_list, mystuff->bit_min, mystuff->bit_max_stage, !partialresult, MFAKTC_VERSION, mystuff->stats.kernelname, details,
             getOS(), getArchitecture(), timestamp);
    json_checksum = crc32_checksum(json_checksum_string, strlen(json_checksum_string));
    snprintf(
        jsonstring, sizeof(jsonstring),
        "{\"exponent\":%u, \"worktype\":\"TF\", \"status\":\"%s\", \"bitlo\":%d, \"bithi\":%d, \"rangecomplete\":%s%s, \"program\":{\"name\":\"mfaktc\", \"version\":\"%s\", \"subversion\":\"%s\", \"details\":\"%s\"}, \"timestamp\":\"%s\"%s%s%s%s, \"checksum\":{\"version\":%u, \"checksum\":\"%08X\"}}",
        mystuff->exponent, factorsfound > 0 ? "F" : "NF", mystuff->bit_min, mystuff->bit_max_stage, partialresult ? "false" : "true",
        factorjson, MFAKTC_VERSION, mystuff->stats.kernelname, details, timestamp, userjson, computerjson, aidjson, osjson,
        MFAKTC_CHECKSUM_VERSION, json_checksum);
#endif
    if (mystuff->mode != MODE_SELFTEST_SHORT) {
        printf("%s\n", txtstring);
    }
    if (mystuff->mode == MODE_NORMAL) {
#ifndef WAGSTAFF
        fprintf(jsonresultfile, "%s\n", jsonstring);
        fclose(jsonresultfile);
        jsonresultfile = NULL;
#endif
        if (mystuff->legacy_results_txt == 1) {
            fprintf(txtresultfile, "%s%s\n", UID, txtstring);
            fclose(txtresultfile);
            txtresultfile = NULL;
        }
    }
}

void print_factor(mystuff_t *mystuff, int factor_number, char *factor)
{
    char UID[110]; /* 50 (V5UserID) + 50 (ComputerID) + 8 + spare */
    char string[200];
    int max_class_counter, string_length = 0, checksum;
    FILE *txtresultfile = NULL;

#ifndef MORE_CLASSES
    max_class_counter = 96;
#else
    max_class_counter = 960;
#endif

    if (mystuff->V5UserID[0] && mystuff->ComputerID[0])
        sprintf(UID, "UID: %s/%s, ", mystuff->V5UserID, mystuff->ComputerID);
    else
        UID[0] = 0;

    if (mystuff->mode == MODE_NORMAL && mystuff->legacy_results_txt == 1) {
        txtresultfile = fopen(mystuff->resultfile, "a");
        if (mystuff->print_timestamp == 1 && factor_number == 0) print_timestamp(txtresultfile);
    }

    if (factor_number < 10) {
        int cuda_major = mystuff->cuda_toolkit / 1000;
        int cuda_minor = (mystuff->cuda_toolkit % 1000) / 10;
        int arch_major = mystuff->cuda_arch / 100;
        int arch_minor = (mystuff->cuda_arch % 100) / 10;
        string_length  = sprintf(string, "%s%u has a factor: %s [TF:%d:%d%s:mfaktc %s %s CUDA %d.%d arch %d.%d]", NAME_NUMBERS,
                                 mystuff->exponent, factor, mystuff->bit_min, mystuff->bit_max_stage,
                                ((mystuff->stopafterfactor == 2) && (mystuff->stats.class_counter < max_class_counter)) ? "*" : "",
                                 MFAKTC_VERSION, mystuff->stats.kernelname, cuda_major, cuda_minor, arch_major, arch_minor);

        checksum = crc32_checksum(string, string_length);
        sprintf(string + string_length, " %08X", checksum);

        if (mystuff->mode != MODE_SELFTEST_SHORT) {
            if (mystuff->printmode == 1 && factor_number == 0) {
                printf("\n");
            }
            printf("%s\n", string);
        }
        if (mystuff->mode == MODE_NORMAL && mystuff->legacy_results_txt == 1) {
            fprintf(txtresultfile, "%s%s\n", UID, string);
        }
    } else /* factor_number >= 10 */
    {
        if (mystuff->mode != MODE_SELFTEST_SHORT)
            printf("%s%u: %d additional factors not shown\n", NAME_NUMBERS, mystuff->exponent, factor_number - 10);
        if (mystuff->mode == MODE_NORMAL && mystuff->legacy_results_txt == 1)
            fprintf(txtresultfile, "%s%s%u: %d additional factors not shown\n", UID, NAME_NUMBERS, mystuff->exponent, factor_number - 10);
    }

    if (mystuff->mode == MODE_NORMAL && mystuff->legacy_results_txt == 1) {
        fclose(txtresultfile);
    }
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

    while (bit_min <= bit_max && bit_min <= 62) {
        ghzdays += 0.011160 * pow(2.0, (double)bit_min - 48.0);
        bit_min++;
    }
    while (bit_min <= bit_max && bit_min <= 64) {
        ghzdays += 0.017832 * pow(2.0, (double)bit_min - 48.0);
        bit_min++;
    }
    while (bit_min <= bit_max) {
        ghzdays += 0.016968 * pow(2.0, (double)bit_min - 48.0);
        bit_min++;
    }

    ghzdays *= 1680.0 / (double)exp;

    return ghzdays;
}
