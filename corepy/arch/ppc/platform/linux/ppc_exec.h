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

#ifndef PPC_EXEC_H
#define PPC_EXEC_H

#include <stdio.h>
#include <stdint.h>
#include <sys/mman.h>
#include <errno.h>

#include <pthread.h>


// ------------------------------------------------------------
// Typedefs
// ------------------------------------------------------------

//Base address type -- integer that holds memory addresses
typedef unsigned long addr_t;


struct ExecParams {
  unsigned long p1;  // r3
  unsigned long p2;  // r4
  unsigned long p3;  // r5
  unsigned long p4;  // r6
  unsigned long p5;  // r7
  unsigned long p6;  // r8
  unsigned long p7;  // r9
  unsigned long p8;  // r10
};


struct ThreadParams {
  addr_t addr;
  struct ExecParams params;
  union {
    long l;
    double d;
  } ret;
};


//Returned to python in async mode
struct ThreadInfo {
  pthread_t th;
  int mode;
};


// Function pointers for different return values
typedef long (*Stream_func_int)(struct ExecParams);
typedef double (*Stream_func_fp)(struct ExecParams);


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

int make_executable(addr_t addr, int size) {
  // -- Note: For SELinux installations, the ifdef'd code may be necessary
  //          It hasn't been tested, but is kept for reference in case
  //          it's needed in the future.
#ifdef COREPY_SE_LINUX
  if(mprotect((void *)(addr & 0xFFFFF000), size * 4 + (addr & 0xFFF), 
              PROT_READ | PROT_EXEC) == -1)
    perror("make_executeable");
#endif // COREPY_SE_LINUX
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
//   Not currently implemented for PPC cores on Cell
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
//   Not currently implemented for PPC cores on Cell
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
    free(p);
}


void *run_stream_int(void *params) {
  struct ThreadParams *p = (struct ThreadParams *)params;
  pthread_cleanup_push(cleanup, params);

  p->ret.l = ((Stream_func_int)p->addr)(p->params);

  pthread_cleanup_pop(0);
  return params;
}


void *run_stream_fp(void *params) {
  struct ThreadParams *p = (struct ThreadParams *)params;
  pthread_cleanup_push(cleanup, params);

  p->ret.d = ((Stream_func_fp)p->addr)(p->params);

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
  int rc;

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


struct ThreadInfo*  execute_fp_async(addr_t addr, struct ExecParams params) {
  int rc;

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


double join_fp(struct ThreadInfo* tinfo) {
  struct ThreadParams *p;

  pthread_join(tinfo->th, (void**)&p);

  double result = p->ret.d;

  free(tinfo);
  free(p);
  return result;
}


long execute_int(addr_t addr, struct ExecParams params) {
  return ((Stream_func_int)addr)(params);
}


double execute_fp(addr_t addr, struct ExecParams params) {
  return ((Stream_func_fp)addr)(params);
}

#endif // PPC_EXEC_H
