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
#define MFC_GET     0x40
#define MFC_GETB    0x41


// ------------------------------------------------------------
// Typedefs
// ------------------------------------------------------------


struct spufs_context {
  int spu_fd;
  int mem_fd;
  void* mem_ptr;
  int regs_fd;
  int mbox_fd;
  int ibox_fd;
  int wbox_fd;
  int mbox_stat_fd;
  int ibox_stat_fd;
  int wbox_stat_fd;
  int signal1_fd;
  int signal2_fd;
  int mfc_fd;
};


// MFC command struct, taken from spufs(7)
struct mfc_dma_command {
  int32_t pad;
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

  snprintf(pathname, 256, "%s/%s-%d-%d",
      SPUFS_PATH, basename, getpid(), rand());
  DEBUG(("SPUFS context path: %s\n", pathname));

  ctx = (struct spufs_context*)malloc(sizeof(struct spufs_context));
  if(ctx == NULL) {
    perror("spufs_open_context() malloc");
    return NULL;
  }

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

  ctx->regs_fd = openat(ctx->spu_fd, "regs", O_RDWR);
  if(ctx->regs_fd == -1) {
    perror("spufs_open_context() open regs");
    goto fail_regs;
  }

  ctx->mbox_fd = openat(ctx->spu_fd, "mbox", O_RDONLY);
  if(ctx->mbox_fd == -1) {
    perror("spufs_open_context() open mbox");
    goto fail_mbox;
  }

  ctx->ibox_fd = openat(ctx->spu_fd, "ibox", O_RDONLY);
  if(ctx->ibox_fd == -1) {
    perror("spufs_open_context() open ibox");
    goto fail_ibox;
  }

  ctx->wbox_fd = openat(ctx->spu_fd, "wbox", O_WRONLY);
  if(ctx->wbox_fd == -1) {
    perror("spufs_open_context() open wbox");
    goto fail_wbox;
  }

  ctx->mbox_stat_fd = openat(ctx->spu_fd, "mbox_stat", O_RDONLY);
  if(ctx->mbox_stat_fd == -1) {
    perror("spufs_open_context() open mbox_stat");
    goto fail_mbox_stat;
  }

  ctx->ibox_stat_fd = openat(ctx->spu_fd, "ibox_stat", O_RDONLY);
  if(ctx->ibox_stat_fd == -1) {
    perror("spufs_open_context() open ibox_stat");
    goto fail_ibox_stat;
  }

  ctx->wbox_stat_fd = openat(ctx->spu_fd, "wbox_stat", O_RDONLY);
  if(ctx->wbox_stat_fd == -1) {
    perror("spufs_open_context() open wbox_stat");
    goto fail_wbox_stat;
  }

  ctx->signal1_fd = openat(ctx->spu_fd, "signal1", O_RDWR);
  if(ctx->signal1_fd == -1) {
    perror("spufs_open_context() open signal1");
    goto fail_signal1;
  }

  ctx->signal2_fd = openat(ctx->spu_fd, "signal2", O_RDWR);
  if(ctx->signal2_fd == -1) {
    perror("spufs_open_context() open signal2");
    goto fail_signal2;
  }

  ctx->mfc_fd = openat(ctx->spu_fd, "mfc", O_RDWR);
  if(ctx->mfc_fd == -1) {
    perror("spufs_open_context() open mfc");
    goto fail_mfc;
  }


  return ctx;

fail_mfc:
  close(ctx->signal2_fd);
fail_signal2:
  close(ctx->signal1_fd);
fail_signal1:
  close(ctx->wbox_stat_fd);
fail_wbox_stat:
  close(ctx->ibox_stat_fd);
fail_ibox_stat:
  close(ctx->mbox_stat_fd);
fail_mbox_stat:
  close(ctx->wbox_fd);
fail_wbox:
  close(ctx->ibox_fd);
fail_ibox:
  close(ctx->mbox_fd);
fail_mbox:
  close(ctx->regs_fd);
fail_regs:
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
  close(ctx->mfc_fd);
  close(ctx->signal2_fd);
  close(ctx->signal1_fd);
  close(ctx->wbox_stat_fd);
  close(ctx->ibox_stat_fd);
  close(ctx->mbox_stat_fd);
  close(ctx->wbox_fd);
  close(ctx->ibox_fd);
  close(ctx->mbox_fd);
  close(ctx->regs_fd);
  munmap(ctx->mem_ptr, SPULS_SIZE);
  close(ctx->mem_fd);
  close(ctx->spu_fd);
  free(ctx);
}


static inline int spufs_run(struct spufs_context* ctx, unsigned int* lsa) {
  return syscall(SYS_spu_run, ctx->spu_fd, lsa, NULL);
}

#ifdef __cplusplus
}
#endif

#endif // SPUFS_H

