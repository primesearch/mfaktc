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


#include "crc.h"

unsigned int crc32_checksum(char *string, int chars)
/* generates a CRC-32 like checksum of the string */
{
  unsigned int chksum=0;
  int i,j;
  
  for(i=0;i<chars;i++)
  {
    for(j=7;j>=0;j--)
    {
      if((chksum>>31) == (((unsigned int)(string[i]>>j))&1))
      {
        chksum<<=1;
      }
      else
      {
        chksum = (chksum<<1)^0x04C11DB7;
      }
    }
  }
  return chksum;
}
