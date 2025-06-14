/*
This file is part of mfaktc.
Copyright (C) 2009, 2010, 2011, 2013, 2015  Oliver Weihe (o.weihe@t-online.de)

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

#include "crc.h"

/* uses a variant of CRC32 to generate the checksum of a string */
unsigned int crc32_checksum(char *string, int chars)
{
    unsigned int cur_char, chksum = 0xFFFFFFFF;
    int str_idx, cur_bit;

    for (str_idx = 0; str_idx < chars; str_idx++) {
        cur_char = string[str_idx];
        if (!cur_char) {
            printf("Error: failed to compute checksum due to invalid character at index %d\n", str_idx);
            break;
        }
        chksum ^= cur_char;
        for (cur_bit = 7; cur_bit >= 0; cur_bit--) {
            if (chksum & 1) {
                chksum = (chksum >> 1) ^ 0xEDB88320;
            }
            else {
                chksum >>= 1;
            }
        }
    }
    chksum ^= 0xFFFFFFFF;
    return chksum;
}
