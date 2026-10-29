/*
This file is part of mfaktc (mfakto).
Copyright (C) 2009 - 2014  Oliver Weihe (o.weihe@t-online.de)
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

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>
#include <string.h>
#include <time.h>

#if defined _MSC_VER || defined __MINGW32__
#include <Windows.h>
#include <io.h>
#undef open
#undef close
#define open     _open
#define close    _close
#define MODE     _S_IREAD | _S_IWRITE
#define O_RDONLY _O_RDONLY
#else
#include <unistd.h>
#include <sched.h>
#include <dirent.h>
#include <sys/file.h>
#define MODE       S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH
static void Sleep(unsigned int ms)
{
    struct timespec ts;
    ts.tv_sec  = (time_t)ms / 1000;
    ts.tv_nsec = (ms % 1000) * 1000000;
    nanosleep(&ts, NULL);
}
#endif

#define MAX_LOCKED_FILES 5

typedef struct _lockinfo {
    int lockfd;
    FILE *open_file;
    char lock_filename[256];
} lockinfo;

static unsigned int num_locked_files = 0;
static lockinfo locked_files[MAX_LOCKED_FILES];

/*
remove_lock_files() is called at exit and removes any currently held lock
files, such as those that remain when mfaktc exits because of an error or if
Ctrl + C is pressed a second time while a file is locked. Such files can still
be left behind by a crash or a power loss. fopen_and_lock() then waits until
they are deleted: it doesn't take over lock files it didn't create, as there
is no reliable way to tell whether the lock owner is still alive. Operating
systems recycle process IDs, and other programs such as AutoPrimeNet may share
the same directory as mfaktc.
*/
static void remove_lock_files(void)
{
    unsigned int i;

    for (i = 0; i < num_locked_files; i++) {
        close(locked_files[i].lockfd);
        remove(locked_files[i].lock_filename);
    }
    num_locked_files = 0;
}

/* whether name is "<base>.<pid>-<n>.write", a file made by fopen_new() */
static int is_new_file_name(const char *name, const char *base)
{
    size_t len = strlen(base), i;

    if (strncmp(name, base, len) != 0 || name[len] != '.') return 0;
    name += len + 1;
    for (i = 0; name[i] >= '0' && name[i] <= '9'; i++)
        ;
    if (i == 0 || name[i] != '-') return 0;
    name += i + 1;
    for (i = 0; name[i] >= '0' && name[i] <= '9'; i++)
        ;
    return i > 0 && strcmp(name + i, ".write") == 0;
}

/* the age after which a file made by fopen_new() has been left behind: writing
   a file takes milliseconds, so the writer is gone for sure */
#define STALE_NEW_FILE_SECONDS 3600

/* removes the files made by fopen_new(path) that a stopped program left
   behind. Newer ones are kept as another instance may be writing them. */
void remove_stale_new_files(const char *path)
{
    const char *base = path, *p;
    char prefix[256], name[512]; /* prefix is the directory of path with its trailing separator, or empty */
    size_t prefix_len;
#if defined _MSC_VER || defined __MINGW32__
    HANDLE h;
    WIN32_FIND_DATAA fd;
    FILETIME now;
    ULARGE_INTEGER now_t, file_t;

    for (p = path; *p; p++)
        if (*p == '/' || *p == '\\' || *p == ':') base = p + 1;
#else
    DIR *d;
    struct dirent *e;
    struct stat st;

    for (p = path; *p; p++)
        if (*p == '/') base = p + 1;
#endif
    prefix_len = (size_t)(base - path);
    if (prefix_len >= sizeof(prefix)) return;
    memcpy(prefix, path, prefix_len);
    prefix[prefix_len] = '\0';

#if defined _MSC_VER || defined __MINGW32__
    if (snprintf(name, sizeof(name), "%s*.write", prefix) >= (int)sizeof(name)) return;
    h = FindFirstFileA(name, &fd);
    if (h == INVALID_HANDLE_VALUE) return;
    GetSystemTimeAsFileTime(&now);
    now_t.LowPart  = now.dwLowDateTime;
    now_t.HighPart = now.dwHighDateTime;
    do {
        file_t.LowPart  = fd.ftLastWriteTime.dwLowDateTime;
        file_t.HighPart = fd.ftLastWriteTime.dwHighDateTime;
        /* FILETIME counts 100 ns */
        if (is_new_file_name(fd.cFileName, base) && now_t.QuadPart > file_t.QuadPart &&
            (now_t.QuadPart - file_t.QuadPart) / 10000000ULL > STALE_NEW_FILE_SECONDS &&
            snprintf(name, sizeof(name), "%s%s", prefix, fd.cFileName) < (int)sizeof(name))
            remove(name);
    } while (FindNextFileA(h, &fd));
    FindClose(h);
#else
    d = opendir(prefix_len ? prefix : ".");
    if (d == NULL) return;
    while ((e = readdir(d)) != NULL) {
        if (!is_new_file_name(e->d_name, base)) continue;
        if (snprintf(name, sizeof(name), "%s%s", prefix, e->d_name) >= (int)sizeof(name)) continue;
        if (stat(name, &st) == 0 && time(NULL) - st.st_mtime > STALE_NEW_FILE_SECONDS) remove(name);
    }
    closedir(d);
#endif
}

/* creates a new file with a unique name made of path, a number and ".write",
   and writes the name to buf. Instances writing the same file use different
   names, so neither can remove or replace the other one's new file, and
   unlike mkstemp() the file gets the permissions fopen() would give it.
   returns NULL on failure */
FILE *fopen_new(char *buf, size_t size, const char *path)
{
    static unsigned int counter = 0;
    unsigned long pid;
    unsigned int tries;
    int fd = -1;
    FILE *f;

#if defined _MSC_VER || defined __MINGW32__
    pid = (unsigned long)GetCurrentProcessId();
#else
    pid = (unsigned long)getpid();
#endif
    for (tries = 0; tries < 100; tries++) {
        /* the process ID keeps instances from trying the same names */
        if (snprintf(buf, size, "%s.%lu-%u.write", path, pid, counter++) >= (int)size) return NULL;
#if defined _MSC_VER || defined __MINGW32__
        fd = open(buf, _O_WRONLY | _O_CREAT | _O_EXCL, MODE);
#else
        /* the mode fopen() uses */
        fd = open(buf, O_WRONLY | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH);
#endif
        /* don't pick another name unless this one is taken */
        if (fd >= 0 || errno != EEXIST) break;
    }
    if (fd < 0) return NULL;
#if defined _MSC_VER || defined __MINGW32__
    f = _fdopen(fd, "w");
#else
    f = fdopen(fd, "w");
#endif
    if (f == NULL) {
        close(fd);
        remove(buf);
    }
    return f;
}

FILE *make_temp_file(char *tpl)
{
    int temp_fd;
    FILE *f;
#if defined _MSC_VER || defined __MINGW32__
    size_t len = strlen(tpl);
    int tries;

    /* create the file exclusively as _mktemp_s() only picks a name */
    for (tries = 0;; tries++) {
        memcpy(tpl + len - 6, "XXXXXX", 6);
        if (_mktemp_s(tpl, len + 1) != 0) {
            return NULL;
        }
        temp_fd = open(tpl, _O_WRONLY | _O_CREAT | _O_EXCL, MODE);
        /* don't pick another name unless the original one is taken */
        if (temp_fd >= 0 || errno != EEXIST || tries == 9) {
            break;
        }
    }
    if (temp_fd < 0) return NULL;
    f = _fdopen(temp_fd, "w");
#else
    temp_fd = mkstemp(tpl);
    if (temp_fd < 0) return NULL;
    f = fdopen(temp_fd, "w");
#endif
    if (f == NULL) {
        close(temp_fd);
        remove(tpl);
    }
    return f;
}

FILE *fopen_and_lock(const char *path, const char *mode)
{
    unsigned int i;
    int lockfd;
    FILE *f;
    static int remove_lock_files_registered = 0;

    if (!remove_lock_files_registered) {
        atexit(remove_lock_files);
        remove_lock_files_registered = 1;
    }

    if (!path) {
        fprintf(stderr, "fopen_and_lock() called with NULL passed as path argument.\n");
        return NULL;
     }

     if (!mode) {
         fprintf(stderr, "Cannot open %.250s: mode is NULL.\n", path);
         return NULL;
     }
    if (strlen(path) > 250) {
        fprintf(stderr, "Cannot open %.250s: Name too long.\n", path);
        return NULL;
    }

    if (num_locked_files >= MAX_LOCKED_FILES) {
        fprintf(stderr, "Cannot open %.250s: Too many locked files.\n", path);
        return NULL;
    }

    sprintf(locked_files[num_locked_files].lock_filename, "%.250s.lck", path);

    for (i = 0;;) {
        if ((lockfd = open(locked_files[num_locked_files].lock_filename, O_EXCL | O_CREAT, MODE)) < 0) {
            if (errno == EEXIST) {
                if (i == 0) {
                    fprintf(stderr, "%.250s already exists, waiting ...\n", locked_files[num_locked_files].lock_filename);
                    fprintf(
                        stderr,
                        "If no other program (such as AutoPrimeNet or another mfaktc instance) is using %.250s, the lock file was left behind by an unexpected exit and can be deleted.\n",
                        path);
                }
                if (i < 1000) i++; // slowly increase sleep time up to 1 sec
                Sleep(i);
                continue;
            } else {
                perror("Cannot open lock file");
                break;
            }
        }
        break;
    }

    if (lockfd < 0) {
        /* The lock file could not be created for a reason other than
           contention; do not pretend the file is locked. */
        return NULL;
    }

    locked_files[num_locked_files].lockfd = lockfd;

    if (i > 0) {
        printf("Locked %.250s\n", path);
    }

    f = fopen(path, mode);
    if (f) {
        locked_files[num_locked_files++].open_file = f;
    } else {
        if (close(locked_files[num_locked_files].lockfd) != 0) perror("Failed to close lock file");
        if (remove(locked_files[num_locked_files].lock_filename) != 0) perror("Failed to delete lock file");
    }

    return f;
}

/* closes the lock file of locked_files[i] and removes the entry */
static void release_lock(unsigned int i)
{
    unsigned int j;

    if (close(locked_files[i].lockfd) != 0) perror("Failed to close lock file");
    if (remove(locked_files[i].lock_filename) != 0) perror("Failed to delete lock file");
    //      printf("release_lock(%s)\n", locked_files[i].lock_filename);
    for (j = i + 1; j < num_locked_files; j++) {
        locked_files[j - 1].lockfd    = locked_files[j].lockfd;
        locked_files[j - 1].open_file = locked_files[j].open_file;
        strcpy(locked_files[j - 1].lock_filename, locked_files[j].lock_filename);
    }
    num_locked_files--;
}

int unlock_and_fclose(FILE *f)
{
    unsigned int i;
    int ret;

    if (f == NULL) {
        return -1;
    }

    for (i = 0; i < num_locked_files; i++) {
        if (locked_files[i].open_file == f) {
            ret = fclose(f);
            f   = NULL;
            release_lock(i);
            break;
        }
    }
    if (f) {
        ret = fclose(f);
    }
    return ret;
}

/* closes a file opened with fopen_and_lock() but keeps it locked until unlock_file(path) */
int fclose_keep_lock(FILE *f)
{
    unsigned int i;

    if (f == NULL) {
        return -1;
    }
    for (i = 0; i < num_locked_files; i++) {
        if (locked_files[i].open_file == f) {
            locked_files[i].open_file = NULL;
            break;
        }
    }
    return fclose(f);
}

/* releases the lock of a file closed with fclose_keep_lock() */
int unlock_file(const char *path)
{
    unsigned int i;
    char lock_filename[256];

    snprintf(lock_filename, sizeof(lock_filename), "%.250s.lck", path);
    for (i = 0; i < num_locked_files; i++) {
        if (locked_files[i].open_file == NULL && strcmp(locked_files[i].lock_filename, lock_filename) == 0) {
            release_lock(i);
            return 0;
        }
    }
    return -1;
}

/* writes the buffered data of f and syncs the file to the disk so that a file
   that replaces another one is complete even after a power loss. returns 0 on
   success */
int sync_file(FILE *f)
{
    if (fflush(f) != 0) return -1;
#if defined _MSC_VER || defined __MINGW32__
    return _commit(_fileno(f)) == 0 ? 0 : -1;
#else
    return fsync(fileno(f)) == 0 ? 0 : -1;
#endif
}

#if !(defined _MSC_VER || defined __MINGW32__)
/* syncs the file or directory path to the disk */
static int sync_path(const char *path)
{
    int fd = open(path, O_RDONLY), ret, err;

    if (fd < 0) return -1;
    ret = fsync(fd);
    err = errno;
    close(fd);
    /* for the caller's warning */
    errno = err;
    /* returns 0 on success */
    return ret == 0 ? 0 : -1;
}

/* syncs the directory containing path so that a rename in it survives a power
   loss */
static void sync_directory(const char *path)
{
    const char *slash = strrchr(path, '/');
    char *dir         = slash == NULL ? strdup(".") : strndup(path, slash == path ? 1 : (size_t)(slash - path));

    /* only display a warning as the file has been replaced already */
    if (dir == NULL || sync_path(dir) != 0)
        fprintf(stderr, "Warning: could not sync the directory of \"%s\", its new version may not survive a power loss (%s)\n", path,
                strerror(errno));
    free(dir);
}
#endif

/* replaces the file "to" with the file "from" */
int replace_file(const char *from, const char *to)
{
#if defined _MSC_VER || defined __MINGW32__
    /* MOVEFILE_WRITE_THROUGH: return only when the move is on the disk */
    return MoveFileExA(from, to, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) ? 0 : -1;
#else
    /* rename the file on POSIX systems atomically */
    struct stat st;

    /* keep the group and the permissions of the replaced file (as mkstemp()
       creates files with mode 0600, and the new file may get another group).
       If that fails, replace it anyway as not removing a finished assignment
       from the work file would be worse. */
    if (stat(to, &st) == 0) {
        if (chown(from, (uid_t)-1, st.st_gid) != 0) {
            fprintf(stderr, "Warning: could not keep the group of \"%s\" (%s)\n", to, strerror(errno));
            /* don't give the group permissions to another group */
            st.st_mode &= ~(mode_t)S_IRWXG;
        }
        if (chmod(from, st.st_mode & 07777) != 0) {
            fprintf(stderr, "Warning: could not keep the permissions of \"%s\" (%s)\n", to, strerror(errno));
        }
        /* the caller synced the contents, sync the new group and permissions as well */
        if (sync_path(from) != 0) {
            fprintf(stderr, "Warning: could not sync the group and permissions of \"%s\" (%s)\n", to, strerror(errno));
        }
    }
    if (rename(from, to) != 0) return -1;
    /* the rename itself must survive a power loss too */
    sync_directory(to);
    /* returns 0 on success */
    return 0;
#endif
}

/*
lock_workfile() prevents multiple instances from working on the same
worktodo.txt file, which would make them process the same set of assignments.
It takes an exclusive lock on "<work file>.pid" and keeps it until the
process exits; the operating system releases the lock when the process ends
for any reason (such as a crash or power loss) so that mfaktc can be restarted
even if a file is left behind. The file contains the process ID only for
informational purposes.

return values:
0 = the lock was acquired
1 = another process holds the lock
-1 = could not create or lock the lock file; mfaktc will print the reason

In the absense of a lock file, mfaktc cannot tell whether the worktodo.txt
file is in use by another mfaktc instance.
*/
int lock_workfile(const char *workfile)
{
    char filename[256];
    char pid[32];

    snprintf(filename, sizeof(filename), "%.250s.pid", workfile);
#if defined _MSC_VER || defined __MINGW32__
    static HANDLE lock_handle = INVALID_HANDLE_VALUE;
    OVERLAPPED ov;
    DWORD written;

    lock_handle = CreateFileA(filename, GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_ALWAYS,
                              FILE_ATTRIBUTE_NORMAL, NULL);
    if (lock_handle == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "Could not create \"%s\" (error %lu)\n", filename, (unsigned long)GetLastError());
        return -1;
    }
    memset(&ov, 0, sizeof(ov));
    if (!LockFileEx(lock_handle, LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY, 0, 1, 0, &ov)) {
        DWORD err = GetLastError();
        CloseHandle(lock_handle);
        lock_handle = INVALID_HANDLE_VALUE;
        if (err == ERROR_LOCK_VIOLATION) return 1;
        fprintf(stderr, "Could not lock \"%s\" (error %lu)\n", filename, (unsigned long)err);
        return -1;
    }
    snprintf(pid, sizeof(pid), "%lu\n", (unsigned long)GetCurrentProcessId());
    SetEndOfFile(lock_handle);
    WriteFile(lock_handle, pid, (DWORD)strlen(pid), &written, NULL);
    /* lock_handle stays open until the process exits */
#else
    static int lock_fd = -1;

    lock_fd = open(filename, O_RDWR | O_CREAT, MODE);
    if (lock_fd < 0) {
        fprintf(stderr, "Could not create \"%s\" (%s)\n", filename, strerror(errno));
        return -1;
    }
    if (flock(lock_fd, LOCK_EX | LOCK_NB) != 0) {
        int err = errno;
        close(lock_fd);
        lock_fd = -1;
        if (err == EWOULDBLOCK) return 1;
        fprintf(stderr, "Could not lock \"%s\" (%s)\n", filename, strerror(err));
        return -1;
    }
    snprintf(pid, sizeof(pid), "%lu\n", (unsigned long)getpid());
    if (ftruncate(lock_fd, 0) != 0 || write(lock_fd, pid, strlen(pid)) < 0) {
        /* the lock itself is what matters, the PID is for information only */
    }
    /* lock_fd stays open until the process exits */
#endif
    return 0;
}
