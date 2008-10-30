/* Copyright (c) 2006-2008 The Trustees of Indiana University.
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

#ifndef SWIG
#define _XOPEN_SOURCE 600
#define _DEBUG
#endif

#include <stdlib.h>


const int size_char = sizeof(char);
const int size_short = sizeof(short);
const int size_int = sizeof(int);
const int size_long = sizeof(long);
const int size_float = sizeof(float);
const int size_double = sizeof(double);

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


//SWIG doesn't need to know about internal structs/vars/fns
#ifndef SWIG

// Huge page code derived from:
// http://www.cellperformance.com/public/attachments/cp_hugemem.c
struct _hugerec
{
    int fd;
    unsigned long addr;
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
#endif //SWIG

unsigned long alloc_hugemem(size_t size)
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
    _hugerecs[_hugerecs_len].addr = (unsigned long)addr;
    _hugerecs[_hugerecs_len].length = size;
    _hugerecs_len++;

    return (unsigned long)addr;
}


void free_hugemem(unsigned long addr)
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


unsigned long realloc_hugemem(unsigned long mem, size_t oldsize, size_t newsize)
{
    void* oldaddr = (void*)mem;
    void* newaddr;

#ifdef _DEBUG
    if(oldaddr != NULL) {
        puts("WARNING realloc'ing hugepages isn't smart and might fail");
    }
#endif

    newaddr = (void*)alloc_hugemem(newsize);
    memcpy(newaddr, oldaddr, oldsize < newsize ? oldsize : newsize);
    free_hugemem((unsigned long)oldaddr);
    return (unsigned long)newaddr;
}


int get_hugepage_size(void)
{
    //TODO - do this right..
    return 16 * 1024 * 1024;
}


int has_huge_pages(void)
{
    //Can a mount path be found?
    if(_hugefs_mnt[0] == '\0' && !_hugefs_find_mnt()) {
        return 0;
    }

    return 1;
}

#else // not __linux__

//Huge pages not supported on this platform
int has_huge_pages(void)
{
    return 0;
}

unsigned long alloc_hugemem(size_t size)
{
    return 0;
}

void free_hugemem(unsigned long addr)
{
}

unsigned long realloc_hugemem(unsigned long mem, size_t oldsize, size_t newsize)
{
    return 0;
}

int get_hugepage_size(void)
{
    return 0;
}

#endif //__linux__


int get_page_size(void)
{
    return sysconf(_SC_PAGESIZE);
}


void synchronize(void)
{
// TODO - other architectures
#ifdef __powerpc__
  asm("lwsync");
#else
#ifndef  SWIG
//#error "No sync primitives for this platform"
#endif
#endif
}

unsigned long alloc_mem(size_t size)
{
#ifdef __MACH__
#if 0
    //From http://stackoverflow.com/questions/196329?sort=votes
    int pg_sz = sysconf(_SC_PAGESIZE);

    void* mem = malloc(size + (pg_sz - 1) + sizeof(void*));
    char* amem = ((char*)mem) + sizeof(void*);
    amem += pg_sz - ((uintptr_t)amem & (pg_sz - 1));
    ((void**)amem)[-1] = mem;
    return (unsigned long)amem;
#endif
    unsigned long addr = (unsigned long)valloc(size);
    return addr;
    //return (unsigned long)valloc(size);
#else
    void* addr;

    int rc = posix_memalign(&addr, sysconf(_SC_PAGESIZE), size);
    return (unsigned long)addr;
    return rc;
#endif
}



unsigned long realloc_mem(unsigned long mem, size_t oldsize, size_t newsize)
{
    void* oldaddr = (void*)mem;
    void* newaddr = (void*)alloc_mem(newsize);
    //posix_memalign(&newaddr, sysconf(_SC_PAGESIZE), newsize);
    memcpy(newaddr, oldaddr, oldsize < newsize ? oldsize : newsize);
    free(oldaddr);
    return (unsigned long)newaddr;
}


void free_mem(unsigned long addr)
{
#ifdef __MACH__
    //free(((void**)addr)[-1]);
    free((void*)addr);
#else
    free((void*)addr);
#endif
}

void zero_mem(unsigned long addr, size_t size)
{
    memset((void*)addr, 0, size);
}


void copy_direct(unsigned long dst, char* src, int len)
{
    memcpy((void*)dst, src, len);
}


void byteswap_2(unsigned long mem, int elems)
{
    unsigned short* addr = (unsigned short*)mem;
    int i;

    for(i = 0; i < elems; ++i) {
        addr[i] = (addr[i] << 8) | (addr[i] >> 8);
    }
}

void byteswap_4(unsigned long mem, int elems)
{
    unsigned int* addr = (unsigned int*)mem;
    int i;

    for(i = 0; i < elems; ++i) {
        addr[i] = (addr[i] << 24) | ((addr[i] >> 8) & 0xFF00) |
                ((addr[i] & 0xFF00) << 8) | (addr[i] >> 24);
    }
}

void byteswap_8(unsigned long mem, int elems)
{
    unsigned long long int* addr = (unsigned long long int*)mem;
    int i;

    for(i = 0; i < elems; ++i) {
        addr[i] = (addr[i] << 56) | ((addr[i] >> 48) & 0xFF00) |
                ((addr[i] >> 24) & 0xFF0000) | ((addr[i] >> 8) & 0xFF000000) |
                ((addr[i] & 0xFF000000) << 8) | ((addr[i] & 0xFF0000) << 24) |
                ((addr[i] & 0xFF00) << 48) | (addr[i] >> 56);
    }
}


void setitem_schar(unsigned long mem, int ind, char val)
{
    ((char*)mem)[ind] = val;
}

void setitem_uchar(unsigned long mem, int ind, unsigned char val)
{
    ((unsigned char*)mem)[ind] = val;
}

void setitem_sshort(unsigned long mem, int ind, short val)
{
    ((short*)mem)[ind] = val;
}

void setitem_ushort(unsigned long mem, int ind, unsigned short val)
{
    ((unsigned short*)mem)[ind] = val;
}

void setitem_sint(unsigned long mem, int ind, int val)
{
    ((int*)mem)[ind] = val;
}

void setitem_uint(unsigned long mem, int ind, unsigned int val)
{
    ((unsigned int*)mem)[ind] = val;
}

void setitem_slong(unsigned long mem, int ind, long val)
{
    ((long*)mem)[ind] = val;
}

void setitem_ulong(unsigned long mem, int ind, unsigned long val)
{
    ((unsigned long*)mem)[ind] = val;
}

void setitem_float(unsigned long mem, int ind, float val)
{
    ((float*)mem)[ind] = val;
}

void setitem_double(unsigned long mem, int ind, double val)
{
    ((double*)mem)[ind] = val;
}


char getitem_schar(unsigned long mem, int ind)
{
    return ((char*)mem)[ind];
}

unsigned char getitem_uchar(unsigned long mem, int ind)
{
    return ((unsigned char*)mem)[ind];
}

short getitem_sshort(unsigned long mem, int ind)
{
    return ((short*)mem)[ind];
}

unsigned short getitem_ushort(unsigned long mem, int ind)
{
    return ((short*)mem)[ind];
}

int getitem_sint(unsigned long mem, int ind)
{
    return ((int*)mem)[ind];
}

unsigned int getitem_uint(unsigned long mem, int ind)
{
    return ((unsigned int*)mem)[ind];
}

long getitem_slong(unsigned long mem, int ind)
{
    return ((long*)mem)[ind];
}

unsigned long getitem_ulong(unsigned long mem, int ind)
{
    return ((unsigned long*)mem)[ind];
}

float getitem_float(unsigned long mem, int ind)
{
    return ((float*)mem)[ind];
}

double getitem_double(unsigned long mem, int ind)
{
    return ((double*)mem)[ind];
}


#endif //ALLOC_H
