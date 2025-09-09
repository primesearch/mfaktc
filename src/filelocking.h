/*
This file is part of mfaktc (mfakto).
Copyright (C) 2009 - 2013  Oliver Weihe (o.weihe@t-online.de)
                           Bertram Franz (bertramf@gmx.net)

mfaktc (mfakto) is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

mfaktc (mfakto) is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with mfaktc (mfakto).  If not, see <http://www.gnu.org/licenses/>.
*/

#if defined(NVCC_EXTERN) && !defined(_MSC_VER)
extern "C" {
#endif

/**
 * @brief Creates a unique filename using the provided template.
 *
 * Windows (_MSC_VER or __MINGW32__) uses _mktemp_s to generate a unique
 * filename, and other platforms use mkstemp to create and open a unique file;
 * the file is closed afterwards.
 *
 * @param tpl A modifiable string containing the template for the temporary
 *            filename. It should contain a sequence of 'X's that will be
 *            replaced to generate a unique filename.
 * @return 0 on success, or an error code (e.g., EINVAL) on failure.
 */
int make_temp_file(char *tpl);

FILE *fopen_and_lock(const char *path, const char *mode);
int unlock_and_fclose(FILE *f);

#if defined(NVCC_EXTERN) && !defined(_MSC_VER)
}
#endif
