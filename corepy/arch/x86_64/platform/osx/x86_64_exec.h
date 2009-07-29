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

#ifndef X86_64_EXEC_H
#define X86_64_EXEC_H

//#include <libkern/OSCacheControl.h>
#include <stdio.h>
#include <sys/mman.h>
#include <errno.h>
#include <pthread.h>


// ------------------------------------------------------------
// Typedefs
// ------------------------------------------------------------

//Base address type -- integer that holds memory addresses
typedef unsigned long addr_t;


// Parameter passing structures
/*rax             return value, free
  rbx             used??
  rcx             4th int arg, free
  rdx             3rd int arg, free
  rbp             used
  rsi             2nd int arg, free
  rdi             1st int arg, free
  r8              5th int arg, free
  r9              6th int arg, free
  r10             free
  r11             free
  r12             used
  r13             used
  r14             used
  r15             used
*/

struct ExecParams {
  unsigned long p1; //rdi
  unsigned long p2; //rsi
  unsigned long p3; //rdx
  unsigned long p4; //rcx
  unsigned long p5; //r8
  unsigned long p6; //r9
  //unsigned long p7;
  //unsigned long p8;
};


struct ThreadParams {
  addr_t addr;
  struct ExecParams params;
  union {
    long l;
    float d;
  } ret;
};


struct ThreadInfo {
  pthread_t th;
  int mode;
};


// Function pointers for different return values
typedef long (*Stream_func_int)(struct ExecParams);
typedef long (*Stream_func_int_reg)(unsigned long p1, unsigned long p2, unsigned long p3, unsigned long p4, unsigned long p5, unsigned long p6);
typedef float (*Stream_func_fp)(struct ExecParams);
typedef float (*Stream_func_fp_reg)(unsigned long p1, unsigned long p2, unsigned long p3, unsigned long p4, unsigned long p5, unsigned long p6);


// ------------------------------------------------------------
// Code
// ------------------------------------------------------------

// ------------------------------------------------------------
// Function: make_executable
// Arguments:
//   addr - the address of the code stream
//   size - the size of the code stream in bytes
// Return: void
// Description:
//   Make an instruction stream executable.  The stream must 
//   be a contiguous sequence of bytes, forming instructions.
// ------------------------------------------------------------

int make_executable(addr_t addr, long size) {
  // TODO - AWF - should query for the page size instead of just masking
  //sys_icache_invalidate((char *)addr, size * 4);
  if(mprotect((void *)addr, size, 
        PROT_READ | PROT_WRITE | PROT_EXEC) == -1) {
  //if(mprotect((void *)(addr & 0xFFFFF000), size + (addr & 0xFFF), 
  //      PROT_READ | PROT_WRITE | PROT_EXEC) == -1) {
    perror("make_executeable");
  }

  return 0;
}


// ------------------------------------------------------------
// Function: cancel_async
// Arguments:
//   t - thread id cancel
// Return: 0 on success, -1 on failure
// Description:
//   The native interface for cancelling execution of a thread.
// ------------------------------------------------------------

int cancel_async(struct ThreadInfo* tinfo) {
  return pthread_cancel(tinfo->th);
}


// ------------------------------------------------------------
// Function: suspend_async
// Arguments:
//   t - thread id to cancle
// Return: 0 on success, -1 on failure
// Description:
//   Not currently implemented
// ------------------------------------------------------------

int suspend_async(struct ThreadInfo* tinfo) {
  return -1;
}


// ------------------------------------------------------------
// Function: resume_async
// Arguments:
//   t - thread id to resume
// Return: 0 on success, -1 on failure
// Description:
//   Not currently implemented
// ------------------------------------------------------------

int resume_async(struct ThreadInfo* tinfo) {
  return -1;
}


// ------------------------------------------------------------
// Function: run_stream
// Arguments:
//   *addr - pointer to thread data 
// Return: undefined
// Description:
//   The thread execution function.  The code stream is passed
//   as the argument via pthreads and executed here.
// ------------------------------------------------------------

void cleanup(void* params) {
    struct ThreadParams *p = (struct ThreadParams*)params;
    //delete p;
    free(p);
}


void *run_stream_int(void *params) {
  struct ThreadParams *p = (struct ThreadParams *)params;
  pthread_cleanup_push(cleanup, params);

  //p->ret.l = ((Stream_func_int)p->addr)(p->params);
  p->ret.l = ((Stream_func_int_reg)p->addr)(p->params.p1, p->params.p2, p->params.p3, p->params.p4, p->params.p5, p->params.p6);

  pthread_cleanup_pop(0);
  return params;
}


void *run_stream_fp(void *params) {
  struct ThreadParams *p = (struct ThreadParams *)params;
  pthread_cleanup_push(cleanup, params);

  //p->ret.d = ((Stream_func_fp)p->addr)(p->params);
  p->ret.d = ((Stream_func_fp_reg)p->addr)(p->params.p1, p->params.p2, p->params.p3, p->params.p4, p->params.p5, p->params.p6);

  pthread_cleanup_pop(0);
  return params;
}


// ------------------------------------------------------------
// Function: execute_{int, fp}_async
// Arguments:
//   addr   - address of the instruction stream
//   params - parameters to pass to the instruction stream
// Return: a new thread id
// Description:
//   The native interface for executing a code stream as a 
//   thread.  make_executable must be called first.
// ------------------------------------------------------------


struct ThreadInfo* execute_int_async(addr_t addr, struct ExecParams params) {
  long rc;

  struct ThreadInfo* tinfo = malloc(sizeof(struct ThreadInfo));
  struct ThreadParams* tparams = malloc(sizeof(struct ThreadParams));

  tparams->addr = addr;
  tparams->params = params;

  rc = pthread_create(&tinfo->th, NULL, run_stream_int, (void *)tparams);
  if(rc) {
    printf("Error creating async stream: %d\n", rc);
  }

  return tinfo;
}


struct ThreadInfo* execute_fp_async(addr_t addr, struct ExecParams params) {
  long rc;

  struct ThreadInfo* tinfo = malloc(sizeof(struct ThreadInfo));
  struct ThreadParams* tparams = malloc(sizeof(struct ThreadParams));

  tparams->addr = addr;
  tparams->params = params;

  rc = pthread_create(&tinfo->th, NULL, run_stream_fp, (void *)tparams);
  if(rc) {
    printf("Error creating async stream: %d\n", rc);
  }

  return tinfo;
}


long join_int(struct ThreadInfo* tinfo) {
  struct ThreadParams *p;

  pthread_join(tinfo->th, (void**)&p);

  long result = p->ret.l;

  free(tinfo);
  free(p);
  return result;
}


float join_fp(struct ThreadInfo* tinfo) {
  struct ThreadParams *p;

  pthread_join(tinfo->th, (void**)&p);

  float result = p->ret.d;

  free(tinfo);
  free(p);
  return result;
}


long execute_int(addr_t addr, struct ExecParams params) {
  return ((Stream_func_int_reg)addr)(params.p1, params.p2, params.p3, params.p4, params.p5, params.p6);
  //return ((Stream_func_int)addr)(params);
}


float execute_fp(addr_t addr, struct ExecParams params) {
  return ((Stream_func_fp_reg)addr)(params.p1, params.p2, params.p3, params.p4, params.p5, params.p6);
  //return ((Stream_func_fp)addr)(params);
}

#endif // X86_64_EXEC_H
