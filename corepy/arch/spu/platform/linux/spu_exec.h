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

// Native code for executing instruction streams on Linux.

#ifndef SPU_EXEC_H
#define SPU_EXEC_H

#include <stdio.h>
#include <string.h>
#include <malloc.h>
#include <pthread.h>

#include <sched.h>
#include <libspe2.h>

//Define _DEBUG to get informational debug output
//#define _DEBUG

#ifdef _DEBUG
#define DEBUG(args) printf args
#else
#define DEBUG(args)
#endif


// ------------------------------------------------------------
// Typedefs
// ------------------------------------------------------------

// Structure for putting parameters in the preferred slots of the
// command parameters. The un-numbered field is the preferred slot 
// for the parameter passed registers $r3-r5.

struct ExecParams {
  // $r3
  unsigned int addr;   // address of syn code
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
  spe_context_ptr_t spe;
  unsigned char* spels; 

  //spe_event_handler_ptr_t handler;
  //spe_spu_control_area_t* spe_ctrl_area;

  spe_stop_info_t stopinfo;
  struct ExecParams params;

  int mode;
};


#ifndef SWIG
struct ThreadParams {
  struct ThreadInfo* ti;  //Execution Context
  unsigned long addr;       //Main memory addr of associated stream
  int len;                  //Length of stream in bytes
  unsigned int code_lsa;    //Local store addr of associated stream
  unsigned int exec_lsa;    //Local store addr to start execution
};  
#endif


// Function pointers for different return values
typedef long  (*Stream_func_int)(struct ExecParams);
typedef double (*Stream_func_double)(struct ExecParams);


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
    int i = 0;
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

int make_executable(unsigned int addr, int size) {
  // Note: At some point in the future, memory protection will 
  // be an issue on PPC Linux.  The commented out code sets
  // the execute bit on the memory pages.

  /* 
  if(mprotect((void *)(addr & 0xFFFF0000), 
              size * 4 + (addr & 0xFFFF), 
              PROT_READ | PROT_WRITE | PROT_EXEC) == -1)
    perror("Error setting memory protections");
  */

  // TODO: THIS IS A BAD BAD BAD HACK 
  //       !!! FIX THE INTERFACE TO MAKE SURE SIZE MAKES IT TO EXEC !!!
  // __code_size = (unsigned int)size;
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

int suspend_async(void* arg) {
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

int resume_async(void* arg) {
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

  //Create an SPE context
  ti->spe = spe_context_create(SPE_EVENTS_ENABLE | SPE_MAP_PS, NULL);
  if(NULL == ti->spe) {
    perror("alloc_context spe_context_create");
    free(ti);
    return NULL;
  }

#if 0
  ti->handler = spe_event_handler_create();
  if(NULL == ti->handler) {
    perror("execute_int_async spe_context_create");
    spe_context_destroy(ti->spe);
    free(ti);
    return NULL;
  }

  //TODO - support other events, like interrupecng mbox
  event.events = SPE_EVENT_SPE_STOPPED;
  event.spe = ti->spe,

  rc = spe_event_handler_register(ti->handler, &event);
  if(rc) {
    perror("execute_int_async spe_context_create");
    spe_event_handler_destroy(ti->handler);
    spe_context_destroy(ti->spe);
    free(ti);
    return NULL;
  }
#endif
  

  //Memory map the local store and problem state area
  ti->spels = (unsigned char*)spe_ls_area_get(ti->spe);
  if(NULL == ti->spels) {
    perror("execute_int_async spe_ls_area_get");
    free(ti);
    return NULL;
  }

#if 0
  ti->spe_ctrl_area = (spe_spu_control_area_t*)spe_ps_area_get(ti->spe, SPE_CONTROL_AREA);
  if(NULL == ti->spe_ctrl_area) {
    perror("execute_int_async spe_ps_area_get");
    free(ti);
    return NULL;
  }
#endif

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
  //spe_event_handler_destroy(ti->handler);
  spe_context_destroy(ti->spe);
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
// Return: void
// Description:
//  Start SPU execution at the given LSA address, blocking until
//  the SPU stops execution.  Assumes len and code_lsa are
//  16-byte aligned.
// ------------------------------------------------------------

void run_stream(struct ThreadInfo* ti, unsigned long addr, int len,
               unsigned int code_lsa, unsigned int exec_lsa) {

  if(addr != 0) {
    //Memcpy the code from main memory to the SPU local store
    DEBUG(("Copying code from EA %p to LSA %p (%x) %d bytes\n",
        addr, &ti->spels[code_lsa], code_lsa, len));
    memcpy(&ti->spels[code_lsa], (unsigned char*)addr, len);
  }

  DEBUG(("Starting SPU %p\n", ti->spe));
  spe_context_run(ti->spe, &exec_lsa,
      SPE_RUN_USER_REGS, &ti->params, NULL, &ti->stopinfo);
  DEBUG(("SPU %p finished executing\n", ti->spe));
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
  struct ThreadParams* ti = (struct ThreadParams*)arg;
  int rc;

  run_stream(ti->ti, ti->addr, ti->len, ti->code_lsa, ti->exec_lsa);

  free(ti);
  return NULL;
}
#endif


int run_stream_async(struct ThreadInfo* ti, unsigned long addr, int len,
                     unsigned int code_lsa, unsigned int exec_lsa) {
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
// Function: run_stream_async
// Arguments:
//   *ti     Execution context to run
// Return: void
// Description:
//  Wait for an asynchronous SPU execution thread to complete.
// ------------------------------------------------------------

void wait_stream(struct ThreadInfo* ti)
{
  pthread_join(ti->th, NULL);
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

long execute_int(unsigned long addr, struct ExecParams params) {
  struct ThreadInfo* ti;
  int len, lsa;
  long result;

  ti = alloc_context();
  ti->params = params;

  len = params.size;
  if(len % 16) {
    len += (16 - len % 16);
  }

  lsa = 0x40000 - len;
  run_stream(ti, addr, len, lsa, lsa);

  if(ti->stopinfo.stop_reason == SPE_STOP_AND_SIGNAL) {
    result = ti->stopinfo.result.spe_signal_code;
  } else if(ti->stopinfo.stop_reason == SPE_EXIT) {
    result = ti->stopinfo.result.spe_exit_code;
  } else {
    printf("SPU unexpected stop reason %d result %x\n",
        ti->stopinfo.stop_reason, ti->stopinfo.result.spe_exit_code);
    result = ti->stopinfo.result.spe_exit_code;
  }

  free_context(ti);
  return result;
}


struct ThreadInfo* execute_int_async(unsigned long addr, struct ExecParams params) {
  struct ThreadInfo* ti;
  int len, lsa;

  ti = alloc_context();
  ti->params = params;

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

  if(ti->stopinfo.stop_reason == SPE_STOP_AND_SIGNAL) {
    result = ti->stopinfo.result.spe_signal_code;
  } else if(ti->stopinfo.stop_reason == SPE_EXIT) {
    result = ti->stopinfo.result.spe_exit_code;
  } else {
    printf("SPU unexpected stop reason %d result %x\n",
        ti->stopinfo.stop_reason, ti->stopinfo.result.spe_exit_code);
    result = ti->stopinfo.result.spe_exit_code;
  }

  free_context(ti);
}


pthread_t execute_fp_async(long addr, struct ExecParams params) {
  pthread_t t;

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
  DEBUG(("%d of %d SPUs on %d processors available\n",
      spe_cpu_info_get(SPE_COUNT_USABLE_SPES, -1),
      spe_cpu_info_get(SPE_COUNT_PHYSICAL_SPES, -1),
      spe_cpu_info_get(SPE_COUNT_PHYSICAL_CPU_NODES, -1)));

  return spe_cpu_info_get(SPE_COUNT_USABLE_SPES, -1);
}


unsigned long get_ls_area(struct ThreadInfo* ti) {
  return (unsigned long)ti->spels;
}


unsigned int read_out_mbox(struct ThreadInfo* ti) {
  unsigned int data;

  if(spe_out_mbox_read(ti->spe, &data, 1)) {
    perror("read_out_mbox spe_read_out_mbox");
  }

  return data;
}


unsigned int stat_out_mbox(struct ThreadInfo* ti) {
  return spe_out_mbox_status(ti->spe);
}


unsigned int write_in_mbox(struct ThreadInfo* ti, unsigned int data) {
  return spe_in_mbox_write(ti->spe, &data, 1, SPE_MBOX_ALL_BLOCKING);
}


unsigned int stat_in_mbox(struct ThreadInfo* ti) {
  return spe_in_mbox_status(ti->spe);
}


int write_signal(struct ThreadInfo* ti,
                 unsigned int signal_reg, unsigned int data) {
  return spe_signal_write(ti->spe, signal_reg, data);
}


// MFC DMA Functions
 
int spu_putb(struct ThreadInfo* ti, unsigned int lsa, unsigned long ea,
        unsigned int size, unsigned int tag, unsigned int tid,
        unsigned int rid) {
  return spe_mfcio_putb(ti->spe, lsa, (void*)ea, size, tag, tid, rid);
}


int spu_getb(struct ThreadInfo* ti, unsigned int lsa, unsigned long ea,
        unsigned int size, unsigned int tag, unsigned int tid,
        unsigned int rid) {
  //printf("%X %X %d %d %d %d\n", ls, ea, size, tag, tid, rid);
  return spe_mfcio_getb(ti->spe, lsa, (void*)ea, size, tag, tid, rid);
}


unsigned int read_tag_status_all(struct ThreadInfo* ti, unsigned int mask) {
  unsigned int status;

  spe_mfcio_tag_status_read(ti->spe, mask, SPE_TAG_ALL, &status);
  return status;
}

#endif // SPU_EXEC_H

