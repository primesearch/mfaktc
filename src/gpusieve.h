/*
This file is part of mfaktc.
Copyright (C) 2009, 2010, 2011, 2015  Oliver Weihe (o.weihe@t-online.de)

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

#if defined(NVCC_EXTERN) || defined(_MSC_VER)
extern "C" {
#endif
void gpusieve_init (mystuff_t *mystuff);
#if defined(NVCC_EXTERN) || defined(_MSC_VER)
}
#endif
void gpusieve_init_exponent (mystuff_t *mystuff);
void gpusieve_init_class (mystuff_t *mystuff, unsigned long long k_min);
void gpusieve (mystuff_t *mystuff, unsigned long long num_k_remaining);
