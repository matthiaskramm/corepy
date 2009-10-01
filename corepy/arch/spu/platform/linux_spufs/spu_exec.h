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

// Native code for executing instruction streams on Linux.

#ifndef SPU_EXEC_H
#define SPU_EXEC_H

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <malloc.h>
#include <pthread.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <signal.h>

#include <sched.h>
#include "spufs.h"

#ifdef HAS_LIBSPE2
#include <libspe2.h>
#endif

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

//Base address type -- integer that holds memory addresses
typedef unsigned long addr_t;


// Structure for putting parameters in the preferred slots of the
// command parameters. The un-numbered field is the preferred slot 
// for the parameter passed registers $r3-r5.

struct ExecParams {
  // $r3
  addr_t addr;   // address of syn code
  unsigned int p1;  
  unsigned int p2;  
  unsigned int p3;

  // $r4
  unsigned int size;       // size of syn code
  unsigned int p4;
  unsigned int p5;
  unsigned int p6;

  // $r5
  unsigned int p7;
  unsigned int p8;
  unsigned int p9;
  unsigned int p10;
};


// Information structure returned when executing in async mode
struct ThreadInfo {
  pthread_t th;
  struct spufs_context* spu_ctx;
  addr_t spuls;        //Local store address 
  addr_t spups;        //Problem state address

  int spu_run_ret;

  struct ExecParams params;

  int mode;   //Execution mode; allows python to determine return type
  int stop;   //Whether stop code should be included with return value
};


#ifndef SWIG
struct ThreadParams {
  struct ThreadInfo* ti;    //Execution Context
  addr_t addr;           //Main memory addr of associated stream
  int len;                  //Length of stream in bytes
  addr_t code_lsa;       //Local store addr of associated stream
  addr_t exec_lsa;       //Local store addr to start execution
};  
#endif


// Function pointers for different return values
typedef long  (*Stream_func_int)(struct ExecParams);
typedef double (*Stream_func_double)(struct ExecParams);


#if 0
class aligned_memory {
 private:
  unsigned int size;
  unsigned int alignment;
  char *data;

 public:
  aligned_memory(unsigned int size, unsigned int alignment) {
    puts("NOTICE:  aligned_memory is deprecated; consider using extarray instead");
    this->data = (char*)memalign(alignment, size);
    this->size = size;
    this->alignment = alignment;
  };

  ~aligned_memory() {
    free(this->data);
  };

  unsigned long get_addr() {
    return (unsigned long)(this->data);
  };

  unsigned long get_size() {
    return (unsigned long)(this->size);
  };

  unsigned long get_alignment() {
    return (unsigned long)(this->alignment);
  };
  
  void copy_to(unsigned int source, unsigned int size) {
    memcpy((void*)this->data, (void*)source, size);
  };

  void copy_from(unsigned int dest, unsigned int size) {
    memcpy((void*)dest, (void*)this->data, size);
  };

  unsigned int word_at(unsigned int offset) {
    return *(unsigned int*)(this->data + offset);
  };

  int signed_word_at(unsigned int offset) {
    return *(int*)(this->data + offset);
  };

  void print_memory() {
    unsigned int i = 0;
    for(; i < this->size; ++i) {
      if((i % 8) == 0)
        printf(" ");
      if((i % 64) == 0)
        printf("\n");
      printf("%02X", this->data[i]);
    }
    printf("\n");
  };
};
#endif


// ------------------------------------------------------------
// Code
// ------------------------------------------------------------

// ------------------------------------------------------------
// Function: make_executable
// Arguments:
//   addr - the address of the code stream
//   size - the number of instructions in the stream
// Return: void
// Description:
//   Make an instruction stream executable.  The stream must 
//   be a contiguous sequence of word sized instructions.
// ------------------------------------------------------------

// unsigned int __code_size;

int make_executable(addr_t addr, int size) {
  // Note: At some point in the future, memory protection will 
  // be an issue on PPC Linux.  The commented out code sets
  // the execute bit on the memory pages.

  /* 
  if(mprotect((void *)(addr & 0xFFFF0000), 
              size * 4 + (addr & 0xFFFF), 
              PROT_READ | PROT_WRITE | PROT_EXEC) == -1)
    perror("Error setting memory protections");
  */

  return 0;
}


// ------------------------------------------------------------
// Function: cancel_async
// Arguments:
//   spe_id - spe id to cancel
// Return: 0 on success, -1 on failure
// Description:
//   The native interface for cancelling execution of an spu
//   Note that all three spe_kill calls are available using
//    cancel, suspend, and resume.
// ------------------------------------------------------------

int cancel_async(struct ThreadInfo* ti) {
  return pthread_cancel(ti->th);
}


// ------------------------------------------------------------
// Function: suspend_async
// Arguments:
//   spe_id - spe id to suspend
// Return: 0 on success, -1 on failure
// Description:
//   The native interface for suspending execution of an spu.
// ------------------------------------------------------------

int suspend_async(addr_t arg) {
  return -1;
}


// ------------------------------------------------------------
// Function: resume_async
// Arguments:
//   t - thread id to resume
// Return: 0 on success, -1 on failure
// Description:
//   The native interface for resuming execution of an spu.
// ------------------------------------------------------------

int resume_async(addr_t arg) {
  return -1;
}


// ------------------------------------------------------------
// Function: alloc_context
// Arguments: none
// Return:
//  struct ThreadInfo* - pointer to execution context
// Description:
//  Allocate and initialize a new SPU execution context.
// ------------------------------------------------------------

struct ThreadInfo* alloc_context(void) {
  struct ThreadInfo* ti;

  ti = (struct ThreadInfo*)malloc(sizeof(struct ThreadInfo));

  ti->spu_ctx = spufs_open_context("corepy-spu");
  ti->spuls = (addr_t)ti->spu_ctx->mem_ptr;
  ti->spups = (addr_t)ti->spu_ctx->psmap_ptr;

  return ti;
}


// ------------------------------------------------------------
// Function: free_context
// Arguments:
//   *ti - pointer to execution context
// Return: void
// Description:
//  Free an allocated SPU context
// ------------------------------------------------------------

void free_context(struct ThreadInfo* ti) {
  spufs_close_context(ti->spu_ctx);
  free(ti);
}


// ------------------------------------------------------------
// Function: run_stream
// Arguments:
//   *ti   Execution context to run
//   addr     Address to load code from, NULL if none
//   len      Length of code in bytes
//   code_lsa Local store address to load code at
//   exec_lsa Local store address to start execution at
// Return: LSA address following last SPU instruction executed
// Description:
//  Start SPU execution at the given LSA address, blocking until
//  the SPU stops execution.  Assumes len and code_lsa are
//  16-byte aligned.
// ------------------------------------------------------------

addr_t run_stream(struct ThreadInfo* ti, addr_t addr, int len,
                        addr_t code_lsa, addr_t exec_lsa) {

  unsigned char* spuls = (unsigned char*)ti->spuls;

  if(addr != 0) {
    //Memcpy the code from main memory to the SPU local store
    DEBUG(("Copying code from EA %p to LSA %p (%x) %d bytes\n",
        addr, &spuls[code_lsa], code_lsa, len));
    memcpy(&spuls[code_lsa], (unsigned char*)addr, len);
  }

  DEBUG(("Starting SPU %p\n", ti->spu_ctx));
  ti->spu_run_ret = spufs_run(ti->spu_ctx, (unsigned int*)&exec_lsa);
  DEBUG(("SPU %p finished executing, code %d\n", ti->spu_ctx, ti->spu_run_ret));

  return exec_lsa;
}


// ------------------------------------------------------------
// Function: run_stream_async
// Arguments:
//   *ti     Execution context to run
//   addr     Address to load code from, NULL if none
//   len      Length of code in bytes
//   code_lsa Local store address to load code at
//   exec_lsa Local store address to start execution at
// Return: 0 if successful, -1 if error
// Description:
//  Start SPU execution at the given LSA address, returning as
//  soon as execution begins.  Assumes len and code_lsa are
//  16-byte aligned.
// ------------------------------------------------------------

// Internal thread 'main' function for async execution
#ifndef SWIG
void *run_stream_thread(void* arg) {
  struct ThreadParams* tp = (struct ThreadParams*)arg;
  unsigned int rc;

  rc = run_stream(tp->ti, tp->addr, tp->len, tp->code_lsa, tp->exec_lsa);

  free(tp);
  return (void*)rc;
}
#endif


int run_stream_async(struct ThreadInfo* ti, addr_t addr, int len,
                     addr_t code_lsa, addr_t exec_lsa) {
  struct ThreadParams* tp;
  int rc;

  tp = (struct ThreadParams*)malloc(sizeof(struct ThreadParams));

  tp->ti = ti;
  tp->addr = addr;
  tp->len = len;
  tp->code_lsa = code_lsa;
  tp->exec_lsa = exec_lsa;

  rc = pthread_create(&ti->th, NULL, run_stream_thread, (void*)tp);
  if(rc) {
    printf("Error creating async stream: %d\n", rc);
    free(tp);
  }

  return rc;
}


// ------------------------------------------------------------
// Function: wait_stream
// Arguments:
//   *ti     Execution context to wait on
// Return: LSA address following last SPU instruction executed
// Description:
//  Wait for an asynchronous SPU execution thread to complete.
// ------------------------------------------------------------

addr_t wait_stream(struct ThreadInfo* ti)
{
  addr_t rc;

  pthread_join(ti->th, (void**)&rc);
  return rc;
}


// ------------------------------------------------------------
// Function: execute_{void, int, fp}_async
// Arguments:
//   addr   - address of the instruction stream
//   params - parameters to pass to the instruction stream
// Return: a new thread id
// Description:
//   The native interface for executing a code stream as a 
//   thread.  make_executable must be called first.
// ------------------------------------------------------------

int get_result(struct ThreadInfo* ti) {
  switch(ti->spu_run_ret & SPU_CODE_MASK) {
  case SPU_STOP_SIGNAL:
    return (ti->spu_run_ret & SPU_STOP_MASK) >> 16;
  case SPU_HALT:
    return 0;
  default:
    DEBUG(("SPU unexpected stop reason %x\n", ti->spu_run_ret));
  }

  return 0;
}


#ifndef SWIG
void put_spu_params(struct ThreadInfo* ti);
#endif

long execute_int(addr_t addr, struct ExecParams params) {
  struct ThreadInfo* ti;
  int len;
  addr_t lsa;
  long result;

  ti = alloc_context();
  ti->params = params;

  put_spu_params(ti);

  len = params.size;
  if(len % 16) {
    len += (16 - len % 16);
  }

  lsa = 0x40000 - len;
  run_stream(ti, addr, len, lsa, lsa);

  result = get_result(ti);
  free_context(ti);
  return result;
}


struct ThreadInfo* execute_int_async(addr_t addr,
                                     struct ExecParams params) {
  struct ThreadInfo* ti;
  int len;
  addr_t lsa;

  ti = alloc_context();
  ti->params = params;

  put_spu_params(ti);

  len = params.size;
  if(len % 16) {
    len += (16 - len % 16);
  }

  lsa = 0x40000 - len;
  run_stream_async(ti, addr, len, lsa, lsa);

  return ti;
}


//Block on a running SPU thread until it completes, freeing resources when
// it does.
long join_int(struct ThreadInfo* ti) {
  long result;

  wait_stream(ti);

  result = get_result(ti);
  free_context(ti);
  return result;
}


pthread_t execute_fp_async(long addr, struct ExecParams params) {
  pthread_t t = (pthread_t)NULL;

  printf("Warning: execute_fp is not implemented for SPUs");
  return t;
}


double join_fp(pthread_t t) {
  printf("Warning: join_fp is not implemented for SPUs");
  return 0.0;
}


double execute_fp(long addr, struct ExecParams params) {
  printf("Warning: join_fp is not implemented for SPUs");
  return 0.0;
}


// ------------------------------------------------------------
// Utility Functions
//------------------------------------------------------------


// Return the number of SPUs available for use
int get_num_avail_spus(void)
{
  // TODO - count the entries in /sys/devices/system/spu
  return 6;
}


void get_spu_registers(struct ThreadInfo* ti, unsigned int data) {
  if(read(ti->spu_ctx->regs_fd, (void*)data, 128 * 16) != 128 * 16) {
    perror("get_spu_registers read");
  }

  lseek(ti->spu_ctx->regs_fd, 0, SEEK_SET);
}


void put_spu_registers(struct ThreadInfo* ti, unsigned int data) {
  if(write(ti->spu_ctx->regs_fd, (void*)data, 128 * 16) != 128 * 16) {
    perror("put_spu_registers write");
  }

  lseek(ti->spu_ctx->regs_fd, 0, SEEK_SET);
}


void put_spu_params(struct ThreadInfo* ti) {
  // SPUFS is *RETARDED*!! 12 bytes into the file means seek to pos 3.
  lseek(ti->spu_ctx->regs_fd, 3, SEEK_SET);

  if(write(ti->spu_ctx->regs_fd, (void*)&ti->params, 12 * 4) != 12 * 4) {
    perror("put_spu_params write");
  }

  lseek(ti->spu_ctx->regs_fd, 0, SEEK_SET);
}


int get_phys_id(struct ThreadInfo* ti) {
  return spufs_get_phys_id(ti->spu_ctx);
}


unsigned int read_out_mbox(struct ThreadInfo* ti) {
  unsigned int* addr = (unsigned int*)(ti->spups + 0x4004);
  return *addr;
}


unsigned int stat_out_mbox(struct ThreadInfo* ti) {
  unsigned int* addr = (unsigned int*)(ti->spups + 0x4014);
  return *addr & 0x1;
}


unsigned int stat_read_out_mbox(struct ThreadInfo* ti) {
  volatile unsigned int* addr = (volatile unsigned int*)(ti->spups + 0x4014);
  volatile unsigned int status;

  //__asm__("eieio");

  //Poll the status register until a message is available
  do {
    status = *addr;
  } while((status & 0xFF) == 0);

  //Manual says to do this, is it necessary?
  __asm__("eieio");

  addr = (volatile unsigned int*)(ti->spups + 0x4004);
  return *((volatile unsigned int*)addr);
}

unsigned int poll_out_mbox(struct ThreadInfo* ti) {
  unsigned int* addr = (unsigned int*)(ti->spups + 0x4014);
  int i;

  //Poll the status register until a message is available
  //while(*addr & 0x1 == 0);
  for(i = 0; (*addr & 0x1) == 0; i++);
  __asm__("eieio");
  return i;
}


unsigned int read_out_ibox(struct ThreadInfo* ti) {
  //Interrupt Mailbox register is Privilege level 2, meaning we cant read it
  //unsigned int* addr = (unsigned int*)(ti->spups + 0x4000);
  //return *addr;
  unsigned int data = 0;

  if(read(ti->spu_ctx->ibox_fd, &data, 4) != 4) {
    perror("read_out_mbox read");
  }

  return data;
}


unsigned int stat_out_ibox(struct ThreadInfo* ti) {
  unsigned int* addr = (unsigned int*)(ti->spups + 0x4014);
  return ((*addr & 0x10000) >> 16);
}


void write_in_mbox(struct ThreadInfo* ti, unsigned int data) {
  unsigned int* addr = (unsigned int*)(ti->spups + 0x400C);
  *addr = data;
}


//Write the same data to a list of SPUs
void write_in_mbox_list(struct ThreadInfo* ti[], int len, unsigned int data) {
  int i;

  for(i = 0; i < len; i++) {
    unsigned int* addr = (unsigned int*)(ti[i]->spups + 0x400C);
    *addr = data;
  }
}


unsigned int stat_in_mbox(struct ThreadInfo* ti) {
  unsigned int* addr = (unsigned int*)(ti->spups + 0x4014);
  return ((*addr & 0x700) >> 8);
}


void write_signal(struct ThreadInfo* ti, int which, unsigned int data) {
  unsigned int* addr;

  if(which == 1) {
    addr = (unsigned int*)(ti->spups + 0x1400C);
  } else { //if(which == 2) {
    addr = (unsigned int*)(ti->spups + 0x1C00C);
  }

  *addr = data;
}


void set_signal_mode(struct ThreadInfo* ti, int which, int mode) {
  spufs_set_signal_mode(ti->spu_ctx, which , mode);
}


// MFC Proxy DMA Functions

#ifndef SWIG
static void write_mfc_cmd(struct ThreadInfo* ti, struct mfc_dma_command* cmd)
{
  unsigned int data;
  unsigned int* addr = (unsigned int*)(ti->spups + 0x3000);
  memcpy(addr, cmd, sizeof(struct mfc_dma_command));

  //Have to read the command status register to start the DMA
  addr = (unsigned int*)(ti->spups + 0x3014);
  data = *addr;
} 
#endif


void spu_put(struct ThreadInfo* ti, unsigned int lsa, unsigned long ea,
        unsigned int size, unsigned int tag, unsigned int tid,
        unsigned int rid) {
  struct mfc_dma_command cmd;

  cmd.lsa = lsa;
  cmd.ea = ea;
  cmd.size = size;
  cmd.tag = tag;
  cmd.xclass = (tid << 8) | rid;
  cmd.cmd = MFC_PUT;

  write_mfc_cmd(ti, &cmd);
}


void spu_putb(struct ThreadInfo* ti, unsigned int lsa, unsigned long ea,
        unsigned int size, unsigned int tag, unsigned int tid,
        unsigned int rid) {
  struct mfc_dma_command cmd;

  cmd.lsa = lsa;
  cmd.ea = ea;
  cmd.size = size;
  cmd.tag = tag;
  cmd.xclass = (tid << 8) | rid;
  cmd.cmd = MFC_PUTB;

  write_mfc_cmd(ti, &cmd);
}


void spu_putf(struct ThreadInfo* ti, unsigned int lsa, unsigned long ea,
        unsigned int size, unsigned int tag, unsigned int tid,
        unsigned int rid) {
  struct mfc_dma_command cmd;

  cmd.lsa = lsa;
  cmd.ea = ea;
  cmd.size = size;
  cmd.tag = tag;
  cmd.xclass = (tid << 8) | rid;
  cmd.cmd = MFC_PUTF;

  write_mfc_cmd(ti, &cmd);
}


void spu_get(struct ThreadInfo* ti, unsigned int lsa, unsigned long ea,
        unsigned int size, unsigned int tag, unsigned int tid,
        unsigned int rid) {
  struct mfc_dma_command cmd;

  cmd.lsa = lsa;
  cmd.ea = ea;
  cmd.size = size;
  cmd.tag = tag;
  cmd.xclass = (tid << 8) | rid;
  cmd.cmd = MFC_GET;

  write_mfc_cmd(ti, &cmd);
}


void spu_getb(struct ThreadInfo* ti, unsigned int lsa, unsigned long ea,
        unsigned int size, unsigned int tag, unsigned int tid,
        unsigned int rid) {
  struct mfc_dma_command cmd;

  cmd.lsa = lsa;
  cmd.ea = ea;
  cmd.size = size;
  cmd.tag = tag;
  cmd.xclass = (tid << 8) | rid;
  cmd.cmd = MFC_GETB;

  write_mfc_cmd(ti, &cmd);
}


void spu_getf(struct ThreadInfo* ti, unsigned int lsa, unsigned long ea,
        unsigned int size, unsigned int tag, unsigned int tid,
        unsigned int rid) {
  struct mfc_dma_command cmd;

  cmd.lsa = lsa;
  cmd.ea = ea;
  cmd.size = size;
  cmd.tag = tag;
  cmd.xclass = (tid << 8) | rid;
  cmd.cmd = MFC_GETF;

  write_mfc_cmd(ti, &cmd);
}


unsigned int poll_tag_status(struct ThreadInfo* ti, unsigned int mask) {
  unsigned int* addr = (unsigned int*)(ti->spups + 0x321C);
  *addr = mask;

  addr = (unsigned int*)(ti->spups + 0x322C);
  return *addr;
}


unsigned int read_tag_status(struct ThreadInfo* ti, unsigned int mask) {
  unsigned int status;
  unsigned int* addr = (unsigned int*)(ti->spups + 0x321C);
  *addr = mask;

  addr = (unsigned int*)(ti->spups + 0x322C);
  status = *addr;
  while(1) {
    if(status != 0 && (status ^ mask) == 0) {
      return status;
    }
  }

  return 0;
}


// SPE Native Code execution
// This allows a user to execute an SPE program in a separate binary via CorePy
#ifdef HAS_LIBSPE2

int run_native_code(char* filename, struct ExecParams params)
{
  spe_program_handle_t* prgm;
  spe_context_ptr_t ctx;
  spe_stop_info_t stopinfo;
  unsigned int entry = SPE_DEFAULT_ENTRY;

  ctx = spe_context_create(0, NULL);
  if(ctx == NULL) {
    perror("run_native_code spe_context_create");
  }

  prgm = spe_image_open(filename);
  if(prgm == NULL) {
    perror("run_native_code spe_image_open");
  }

  if(spe_program_load(ctx, prgm) == -1) {
    perror("run_native_code spe_program_load");
  }

  spe_context_run(ctx, &entry, 0, (void*)&params, NULL, &stopinfo);

  spe_image_close(prgm);  
  spe_context_destroy(ctx);

  return stopinfo.result.spe_exit_code;
}


struct NativeParams {
  char* filename;
  struct ExecParams params;
};


void *run_native_code_thread(void* arg) {
  struct NativeParams* np = arg;

  int rc = run_native_code(np->filename, np->params);

  return (void*)rc;
}


pthread_t* run_native_code_async(char* filename, struct ExecParams params)
{
  pthread_t* th = malloc(sizeof(pthread_t));
  struct NativeParams* np;
  int rc;

  np = (struct NativeParams*)malloc(sizeof(struct NativeParams));

  np->filename = filename;
  np->params = params;

  rc = pthread_create(th, NULL, run_native_code_thread, (void*)np);
  if(rc) {
    printf("Error creating async stream: %d\n", rc);
    free(np);
    free(th);
    return NULL;
  }

  return th;
}


int join_native_code(pthread_t* th)
{
  int rc;

  pthread_join(*th, (void**)&rc);
  free(th);
  return rc;
}


#else


void run_native_code(char* filename, struct ExecParams params)
{
  puts("ERROR run_native_code() not available; compile with libspe2 support");
}


pthread_t* run_native_code_async(char* filename, struct ExecParams params)
{
  puts("ERROR run_native_code_async() not available; compile with libspe2 support");
  return NULL;
}


int join_native_code(pthread_t* th)
{
  puts("ERROR join_native_code() not available; compile with libspe2 support");
  return 0;
}


#endif //HAS_LIBSPE2

#endif // SPU_EXEC_H

