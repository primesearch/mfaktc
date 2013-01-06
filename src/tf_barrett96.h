/*
This file is part of mfaktc.
Copyright (C) 2009, 2010, 2012  Oliver Weihe (o.weihe@t-online.de)

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
       int tf_class_barrett76(unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff);
       int tf_class_barrett77(unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff);
       int tf_class_barrett79(unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff);
       int tf_class_barrett87(unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff);
       int tf_class_barrett88(unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff);
       int tf_class_barrett92(unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff);
       int tf_class_barrett76_gs(unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff);
       int tf_class_barrett77_gs(unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff);
       int tf_class_barrett79_gs(unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff);
       int tf_class_barrett87_gs(unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff);
       int tf_class_barrett88_gs(unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff);
       int tf_class_barrett92_gs(unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff);
}
#else
extern int tf_class_barrett76(unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff);
extern int tf_class_barrett77(unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff);
extern int tf_class_barrett79(unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff);
extern int tf_class_barrett87(unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff);
extern int tf_class_barrett88(unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff);
extern int tf_class_barrett92(unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff);
extern int tf_class_barrett76_gs(unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff);
extern int tf_class_barrett77_gs(unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff);
extern int tf_class_barrett79_gs(unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff);
extern int tf_class_barrett87_gs(unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff);
extern int tf_class_barrett88_gs(unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff);
extern int tf_class_barrett92_gs(unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff);
#endif
