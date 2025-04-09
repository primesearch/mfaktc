/*
This file is part of mfaktc.
Copyright (C) 2015  Oliver Weihe (o.weihe@t-online.de)

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
	int check_subcc_bug(mystuff_t *mystuff);
	void get_CUDA_arch(mystuff_t *mystuff);
};
#else
extern int check_subcc_bug(mystuff_t *mystuff);
void get_CUDA_arch(mystuff_t *mystuff);
#endif
