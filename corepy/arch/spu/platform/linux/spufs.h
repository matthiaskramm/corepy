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

/* Utility routines to make it easier to use SPUFS */

#ifndef SPUFS_H
#define SPUFS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <fcntl.h>


//Define _DEBUG to get informational debug output
//#define _DEBUG

#ifdef _DEBUG
#define DEBUG(args) printf args
#else
#define DEBUG(args)
#endif

#define SPUFS_PATH "/spu"
#define SPULS_SIZE (256 * 1024)
#define PSMAP_SIZE (128 * 1024)

// SPU status codes from spu_run(2) man page
#define SPU_STOP_SIGNAL 0x02
#define SPU_HALT        0x04
#define SPU_WAIT_CHAN   0x08
#define SPU_SINGLE_STEP 0x10
#define SPU_INVAL_INST  0x20
#define SPU_INVAL_CHAN  0x40

#define SPU_CODE_MASK   0xFFFF
#define SPU_STOP_MASK   0x3FFF0000

// MFC DMA commands from 'Cell Broadband Engine Architecture' v1.01
// (CBE_Architecture_v101.pdf)
#define MFC_PUT     0x20
#define MFC_PUTB    0x21
#define MFC_PUTF    0x22
#define MFC_GET     0x40
#define MFC_GETB    0x41
#define MFC_GETF    0x42


// ------------------------------------------------------------
// Typedefs
// ------------------------------------------------------------


struct spufs_context {
  void* mem_ptr;
  void* psmap_ptr;

  int spu_fd;
  int mem_fd;
  int psmap_fd;
  int regs_fd;
  int ibox_fd;
};


// MFC command struct, taken from spufs(7)
struct mfc_dma_command {
  uint32_t pad;
  uint32_t lsa;
  uint64_t ea;
  uint16_t size;
  uint16_t tag;
  uint16_t xclass;
  uint16_t cmd;
};


// ------------------------------------------------------------
// Code
// ------------------------------------------------------------


/* Open a new SPU execution context and return the file descriptor */
static struct spufs_context* spufs_open_context(const char* basename) {
  struct spufs_context* ctx;
  char pathname[256] = {0};
  //struct timeval tv_start = {0};
  //struct timeval tv_stop = {0};
  //float usec;

  snprintf(pathname, 256, "%s/%s-%d-%d",
      SPUFS_PATH, basename, getpid(), rand());
  DEBUG(("SPUFS context path: %s\n", pathname));

  ctx = (struct spufs_context*)malloc(sizeof(struct spufs_context));
  if(ctx == NULL) {
    perror("spufs_open_context() malloc");
    return NULL;
  }

#if 0
  gettimeofday(&tv_start, NULL);
#endif

  ctx->spu_fd = syscall(SYS_spu_create, pathname, 0, S_IRWXU);
  if(ctx->spu_fd == -1) {
    perror("spufs_open_context() spu_create");
    goto fail;
  }

  ctx->mem_fd = openat(ctx->spu_fd, "mem", O_RDWR);
  if(ctx->mem_fd == -1) {
    perror("spufs_open_context() open mem");
    goto fail_mem;
  }

  ctx->mem_ptr = mmap(NULL, SPULS_SIZE,
      PROT_READ | PROT_WRITE, MAP_SHARED, ctx->mem_fd, 0);
  if((void*)ctx->mem_ptr == MAP_FAILED) {
    perror("spufs_open_context() mmap mem");
    goto fail_mem_mmap;
  }

  ctx->psmap_fd = openat(ctx->spu_fd, "psmap", O_RDWR);
  if(ctx->psmap_fd == -1) {
    perror("spufs_open_context() open psmap");
    goto fail_psmap;
  }

  ctx->psmap_ptr = mmap(NULL, PSMAP_SIZE,
      PROT_READ | PROT_WRITE, MAP_SHARED, ctx->psmap_fd, 0);
  if((void*)ctx->psmap_ptr == MAP_FAILED) {
    perror("spufs_open_context() mmap psmap");
    goto fail_psmap_mmap;
  }

  ctx->regs_fd = openat(ctx->spu_fd, "regs", O_RDWR);
  if(ctx->regs_fd == -1) {
    perror("spufs_open_context() open regs");
    goto fail_regs;
  }

  ctx->ibox_fd = openat(ctx->spu_fd, "ibox", O_RDONLY);
  if(ctx->ibox_fd == -1) {
    perror("spufs_open_context() open ibox");
    goto fail_ibox;
  }

#if 0
  gettimeofday(&tv_stop, NULL);

  usec = (tv_stop.tv_sec - tv_start.tv_sec) * 1000000.0;
  usec += tv_stop.tv_usec - tv_start.tv_usec;
  printf("time %0.5f us %0.5f ms\n", usec, usec / 1000.0);
#endif
  return ctx;

fail_ibox:
  close(ctx->regs_fd);
fail_regs:
  munmap(ctx->psmap_ptr, PSMAP_SIZE);
fail_psmap_mmap:
  close(ctx->psmap_fd);
fail_psmap:
  munmap(ctx->mem_ptr, SPULS_SIZE);
fail_mem_mmap:
  close(ctx->mem_fd);
fail_mem:
  close(ctx->spu_fd);
fail:
  free(ctx);
  return NULL;
}


static void spufs_close_context(struct spufs_context* ctx) {
  close(ctx->ibox_fd);
  close(ctx->regs_fd);
  munmap(ctx->psmap_ptr, PSMAP_SIZE);
  close(ctx->psmap_fd);
  munmap(ctx->mem_ptr, SPULS_SIZE);
  close(ctx->mem_fd);
  close(ctx->spu_fd);
  free(ctx);
}


static int spufs_get_phys_id(struct spufs_context* ctx)
{
  int fd = openat(ctx->spu_fd, "phys-id", O_RDONLY);
  char buf[16] = {0};
  int len;

  if(fd == -1) {
    perror("spufs open phys-id");
  }

  len = read(fd, buf, 16);
  buf[len - 1] = 0;
  //printf("phys-id %s psmap_ptr %p mem_ptr %p\n", buf, ctx->psmap_ptr, ctx->mem_ptr);

  return strtol(buf, NULL, 16);
}


static inline int spufs_run(struct spufs_context* ctx, unsigned int* lsa) {
  return syscall(SYS_spu_run, ctx->spu_fd, lsa, NULL);
}


static void spufs_set_signal_mode(struct spufs_context* ctx, int which, int mode) {
  int fd;
  char* file;
  char* buf;

  if(which == 1) {
    file = "signal1_type";
  } else {
    file = "signal2_type";
  }

  fd = openat(ctx->spu_fd, file, O_WRONLY);
  if(fd == -1) {
    perror("spufs_set_signal_mode() open signal type");
    return;
  }

  if(mode == 0) {
    buf = "0\n";
  } else {
    buf = "1\n";
  }

  if(write(fd, buf, strlen(buf)) != strlen(buf)) {
    perror("spufs_set_signal_mode write");
  }

  close(fd);
}

#ifdef __cplusplus
}
#endif

#endif // SPUFS_H

