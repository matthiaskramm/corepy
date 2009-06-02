/* Copyright (c) 2006-2009 The Trustees of Indiana University.
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * - Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 * 
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * 
 * - Neither the Indiana University nor the names of its contributors may be
 *   used to endorse or promote products derived from this software without
 *   specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef ALLOC_H
#define ALLOC_H

#define _XOPEN_SOURCE 600
//#define _DEBUG

#include <Python.h>
#include <stdlib.h>


//Only want huge page support on linux
#ifdef __linux__
#include <stdio.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <string.h>
#include <mntent.h>
#include <errno.h>


// Huge page code derived from:
// http://www.cellperformance.com/public/attachments/cp_hugemem.c
struct _hugerec
{
    int fd;
    void* addr;
    size_t length;
};


static char _hugefs_mnt[PATH_MAX - 1] = {0};
static struct _hugerec* _hugerecs = NULL;
static int _hugerecs_len = 0;


static int _hugefs_find_mnt(void)
{
    FILE* mount_table;
    struct mntent* mount_entry;
    int mount_table_retry_max = 512;
    int mount_table_retry_count = 0;

    while(mount_table_retry_count < mount_table_retry_max) {
        mount_table = setmntent(_PATH_MOUNTED, "r");

        if(!mount_table) {
           if(errno == EACCES || errno == EAGAIN || errno == ENFILE) {
               usleep(100000);
               mount_table_retry_count++;
               continue;
           } 
          
           fprintf(stderr,"ERROR Could not obtain mount table lock\n");
           return 0;
        }

        break;
    }

    if(mount_table_retry_count == mount_table_retry_max) {
        fprintf(stderr,"ERROR Could not obtain mount table lock\n");
        return 0;
    }

    mount_entry  = getmntent(mount_table);
    while(mount_entry) {
        if(strcmp(mount_entry->mnt_type, "hugetlbfs") == 0) {
            if(strlen(mount_entry->mnt_dir) >= PATH_MAX - 7) {
                fprintf(stderr, "ERROR: mount point name length too long\n");
            }

            strncpy(_hugefs_mnt, mount_entry->mnt_dir, PATH_MAX - 7);
            strcat(_hugefs_mnt, "/XXXXXX");
            endmntent(mount_table);
            return 1;
        }

        mount_entry = getmntent(mount_table);
    }

    endmntent(mount_table);

    fprintf(stderr,"ERROR: No hugetlbfs entry in the mount table\n");
    return 0;
}


static void* alloc_hugemem(int size)
{
    void* addr;
    char  filename[PATH_MAX + 1];
    int   fd;

#if 0
    if(_hugefs_mnt[0] == '\0' && !_hugefs_find_mnt()) {
        return 0;
    }
#endif

    strcpy(filename, _hugefs_mnt);
    fd = mkstemp(filename);
    if(fd == -1) {
        fprintf(stderr,"ERROR: Couldn't create file %s\n", filename);
        return 0;
    }

    unlink(filename);

    addr = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if(addr == MAP_FAILED) {
        fprintf(stderr,
                "ERROR: Couldn't mmap huge page file: %s\n", strerror(errno));
        return 0;
    }

    _hugerecs = (struct _hugerec*)realloc(_hugerecs,
            sizeof(struct _hugerec) * (_hugerecs_len + 1));
   
    _hugerecs[_hugerecs_len].fd = fd;
    _hugerecs[_hugerecs_len].addr = addr;
    _hugerecs[_hugerecs_len].length = size;
    _hugerecs_len++;

    return addr;
}


static void free_hugemem(void* addr)
{
    int i;

    for(i = 0; i < _hugerecs_len; i++) {
        if(_hugerecs[i].addr == addr) {
            munmap((void*)_hugerecs[i].addr, _hugerecs[i].length);
            close(_hugerecs[i].fd);

            _hugerecs[i] = _hugerecs[--_hugerecs_len];
            _hugerecs = (struct _hugerec*)realloc(_hugerecs,
                    sizeof(struct _hugerec) * _hugerecs_len);
            return;
        }
    }
}


static void* realloc_hugemem(void* mem, Py_ssize_t oldsize, Py_ssize_t newsize)
{
    void* oldaddr = (void*)mem;
    void* newaddr;

#ifdef _DEBUG
    if(oldaddr != NULL) {
        //TODO - make this throw a python warning?
        puts("WARNING realloc'ing hugepages might fail");
    }
#endif

    newaddr = (void*)alloc_hugemem(newsize);
    memcpy(newaddr, oldaddr, oldsize < newsize ? oldsize : newsize);
    free_hugemem(oldaddr);
    return newaddr;
}


static int get_hugepage_size(void)
{
    //TODO - do this right..
    // do a system("cat /proc/meminfo |grep Hugepagesize") and cache it
    return 16 * 1024 * 1024;
}


static int has_huge_pages(void)
{
    //Can a mount path be found?
    if(_hugefs_mnt[0] == '\0' && !_hugefs_find_mnt()) {
        return 0;
    }

    return 1;
}

#else // not __linux__

//Huge pages not supported on this platform
static int has_huge_pages(void)
{
    return 0;
}

static void* alloc_hugemem(int size)
{
    return 0;
}

static void free_hugemem(void* addr)
{
}

static void* realloc_hugemem(void* mem, Py_ssize_t oldsize, Py_ssize_t newsize)
{
    return 0;
}

static int get_hugepage_size(void)
{
    return 0;
}

#endif //__linux__


static int get_page_size(void)
{
    return sysconf(_SC_PAGESIZE);
}


static void* alloc_mem(int size)
{
#ifdef __MACH__
    return (void*)valloc(size);
#else
    void* addr;

    int rc = posix_memalign(&addr, sysconf(_SC_PAGESIZE), size);
    return addr;
    rc = rc + 1;
#endif
}


static void* realloc_mem(void* mem, Py_ssize_t oldsize, Py_ssize_t newsize)
{
    void* oldaddr = mem;
    void* newaddr = (void*)alloc_mem(newsize);

    memcpy(newaddr, oldaddr, oldsize < newsize ? oldsize : newsize);
    free(oldaddr);
    return newaddr;
}


static void free_mem(void* addr)
{
    free(addr);
}

#endif //ALLOC_H
