/*
This file is part of mfaktc.
Copyright (C) 2009, 2010  Oliver Weihe (o.weihe@t-online.de)

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

#ifdef _MSC_VER
extern "C" {
#endif
void setui_96(int96 *res, unsigned long long int a);
#ifdef _MSC_VER
int tf_class_75(unsigned int exp, int bit_min, unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff);
int tf_class_95(unsigned int exp, int bit_min, unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff);
}
#else
extern int tf_class_75(unsigned int exp, int bit_min, unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff);
extern int tf_class_95(unsigned int exp, int bit_min, unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff);
#endif
