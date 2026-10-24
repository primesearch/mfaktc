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
 * @brief Creates a file with a unique name using the provided template and
 * opens a file handle, allowing it to be written to.
 *
 * Other platforms use mkstemp; Windows (_MSC_VER or __MINGW32__) uses
 * _mktemp_s to pick a name and creates the file with O_EXCL. The returned
 * stream uses the descriptor that created the file, so the newly created
 * file can't be replaced before it is opened.
 *
 * @param tpl A modifiable string containing the template for the temporary
 *            filename. Its last six characters must be "XXXXXX"; they are
 *            replaced to generate a unique filename.
 * @return The opened file handle, or NULL on failure.
 */
FILE *make_temp_file(char *tpl);

FILE *fopen_and_lock(const char *path, const char *mode);
int unlock_and_fclose(FILE *f);
int fclose_keep_lock(FILE *f);
int unlock_file(const char *path);
int sync_file(FILE *f);
int replace_file(const char *from, const char *to);
int lock_workfile(const char *workfile);

#if defined(NVCC_EXTERN) && !defined(_MSC_VER)
}
#endif
