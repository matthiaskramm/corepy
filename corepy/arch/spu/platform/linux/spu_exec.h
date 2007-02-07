// Copyright 2006 The Trustees of Indiana University.

// This software is available for evaluation purposes only.  It may not be
// redistirubted or used for any other purposes without express written
// permission from the authors.

// Author:
//   Christopher Mueller

// Native code for executing instruction streams on OS X.
// Compile with -DDEBUG_PRINT to enable additional debugging code

#ifndef SPU_EXEC_H
#define SPU_EXEC_H

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <sys/mman.h>
#include <signal.h>
#include <errno.h>

#include <malloc.h>
 
#include <vector>

#include <pthread.h>
//#include <signal.h>

extern "C" {
#include <sched.h>
#include <libspe.h>


};

#include "spu_constants.h"

// ------------------------------------------------------------
// Typedefs
// ------------------------------------------------------------

// Function pointers for different return values
typedef void (*Stream_func_void)();
typedef int  (*Stream_func_int)();
typedef double (*Stream_func_double)();

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

// Object file that contains the SPU bootstrap code
static char* spu_bootstrap_path = NULL;

void set_bootstrap_path(char* path) {
  if(spu_bootstrap_path != NULL)
    free(spu_bootstrap_path);
  spu_bootstrap_path = strdup(path);
  
  return;
}


class aligned_memory {
 private:
  unsigned int size;
  unsigned int alignment;
  char *data;

 public:
  aligned_memory(unsigned int size, unsigned int alignment) {
    this->data = (char*)memalign(alignment, size);
    this->size = size;
    this->alignment = alignment;
  };

  ~aligned_memory() {
    free(this->data);
  };

  unsigned int get_addr() {
    return (unsigned int)(this->data);
  };

  unsigned int get_size() {
    return (unsigned int)(this->size);
  };

  unsigned int get_alignment() {
    return (unsigned int)(this->alignment);
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

int cancel_async(speid_t spe_id) {
  return spe_kill(spe_id, SIGKILL);
}

// ------------------------------------------------------------
// Function: suspend_async
// Arguments:
//   spe_id - spe id to suspend
// Return: 0 on success, -1 on failure
// Description:
//   The native interface for suspending execution of an spu.
// ------------------------------------------------------------

int suspend_async(speid_t spe_id) {
  return spe_kill(spe_id, SIGSTOP);
}

// ------------------------------------------------------------
// Function: resume_async
// Arguments:
//   t - thread id to resume
// Return: 0 on success, -1 on failure
// Description:
//   The native interface for resuming execution of an spu.
// ------------------------------------------------------------

int resume_async(speid_t spe_id) {
  return spe_kill(spe_id, SIGCONT);
}

// ------------------------------------------------------------
// Function: wait/join_async
// Arguments:
//   spe_id - spu to wait for
//   result - pointer to an integer for the result value.
// Return: 0 on success, -1 on failure
// Description:
//   The native interface for spe wait, more flexible than 
//   join_async.
// ------------------------------------------------------------

int wait_async(speid_t spe_id, int *result) {
  int err = 0;

  if ((err = spe_wait(spe_id, result, 0 /*WUNTRACED*/)) != 0) {
    perror("spe_wait");
    printf("Error waiting for thread: %d\n", err);
  }

  return err;
}

int join_async(speid_t spe_id) {
  return wait_async(spe_id, NULL);
}



// ------------------------------------------------------------
// Functions: spu execute methods
// Arguments:
//   addr - instruction stream address
// Return: 
//   _void - nothing
//   _int  - the 8-bit value returned by the stop instruction.
// Description:
//   The native interfaces for executing instruction streams 
//   on SPUs.  All functions are are based on execute_async
//   and that mode should be the primary mode used by applications.  
// ------------------------------------------------------------


// ------------------------------------------------------------
// Function: execute_async
// Arguments:
//   addr  - address of the instruction stream
//   p0-p2 - parameters placed in the four word slots in $r5
//           (3 for now to keep the interface consistent, add
//            a 4th to ppc...)
// Return: a new spu id
// Description:
//   The native interface for executing a code stream on an SPU.  
//   All other SPU exec functions use it.
//   make_executable must be called first.
// ------------------------------------------------------------

// New interface - this should be more flexible, esp. for parallel execution
speid_t execute_param_async(unsigned int addr, ExecParams params) {
  speid_t   spe_id = 0;
  spe_gid_t grp_id = 0;
  int err = 0;
  struct spe_event ready_event;

  // This is set in Python
  assert(params.addr == (unsigned int)addr);
  // params.addr  = (unsigned int)addr;
  // params.size  = (unsigned short)__code_size * 4;

  // printf("Addr: 0x%x (0x%x, %d)\n", addr, &params, __code_size);

  // Create a thread group 
  grp_id = spe_create_group(SCHED_OTHER, 0, 1);

  if (grp_id == 0) {
    perror("spu_create_group");
  }

  spe_program_handle_t *bootstrap = spe_open_image(spu_bootstrap_path);
  
  if(bootstrap == NULL) {
    perror("spu_open_image");
  }

  // Create the spe thread
  spe_id = spe_create_thread(grp_id, bootstrap , (void*)&params, 
                             NULL, -1, SPE_USER_REGS); // | SPE_MAP_PS);
  if (spe_id == 0) {
    perror("spu_create_thread");
  }

  // Setup the event handler and wait for the ready event
  ready_event.gid    = grp_id;
  ready_event.events = SPE_EVENT_STOP;
  err = spe_get_event(&ready_event, 1, -1);
  
  if(err == -1) {
    perror("spu_get_event (ready)");
  } 
  // else {
  // printf("Ready event: %d %ld\n", ready_event.revents, ready_event.data);
  // }

  // Copy the synthetic program over
  unsigned int lsa = (0x3FFFF - params.size) & 0xFFF80;
  unsigned int size = params.size + (16 - params.size % 16);
  unsigned int tag = 4;

  // printf("Transferring %d bytes to %X\n", size, lsa);
  spe_mfc_getb(spe_id, lsa, (void *)params.addr, size, tag, 0, 0);  
  spe_mfc_read_tag_status_all(spe_id, 1 << tag);

  // Restart the spu
  spe_kill(spe_id, SIGCONT);
  
  return spe_id;
}

speid_t execute_async(unsigned int addr) {
  ExecParams params;
  params.addr = addr;
  return execute_param_async(addr, params);
}


int execute_int(unsigned int addr) {
  speid_t spe_id = 0;
  int result = 0;

  spe_id = execute_async(addr);

  // Wait for the thread to finish
  wait_async(spe_id, &result);

  printf("execute_int: %X (%d)\n", result, result);

  // Shift the return value to extract the 8-bit user value
  return (result >> 8);
}

int execute_param_int(unsigned int addr, ExecParams params) {
  speid_t spe_id = 0;
  int result = 0;

  spe_id = execute_param_async(addr, params);

  // Wait for the thread to finish
  wait_async(spe_id, &result);

  // printf("execute_int: %X (%d)\n", result, result);

  // Shift the return value to extract the 8-bit user value
  return (result >> 8);
}

void execute_void(unsigned int addr) {
  speid_t spe_id = 0;

  spe_id = execute_async(addr);
  wait_async(spe_id, NULL);

  return;
}

void execute_void(unsigned int addr,  ExecParams params) {
  speid_t spe_id = 0;

  spe_id = execute_param_async(addr, params);
  wait_async(spe_id, NULL);

  return;
}

double execute_fp(unsigned int addr) {
  double result = 0.0;

  printf("Warning: execute_fp is not implemented for SPUs");

  return result;
}

// ------------------------------------------------------------
// MFC Functions
//------------------------------------------------------------

unsigned int read_out_mbox(speid_t spe_id) {
  return spe_read_out_mbox(spe_id);
}

unsigned int stat_out_mbox(speid_t spe_id) {
  return spe_stat_out_mbox(spe_id);
}


unsigned int write_in_mbox(speid_t spe_id, unsigned int data) {
  return spe_write_in_mbox(spe_id, data);
}

unsigned int stat_in_mbox(speid_t spe_id) {
  return spe_stat_in_mbox(spe_id);
}

int write_signal(speid_t spe_id, unsigned int signal_reg, unsigned int data) {
  return spe_write_signal(spe_id, signal_reg, data);
}

unsigned long wait_stop_event(speid_t spe_id) {
  int err;
  struct spe_event event;

  event.gid    = spe_get_group(spe_id);
  event.events = SPE_EVENT_STOP;
  err = spe_get_event(&event, 1, -1);
  
  return event.data;
}

// MFC Put Functions
// int spe_mfc_put(speid_t speid, unsigned int ls, void *ea, unsigned int size, unsigned int tag, unsigned int tid, unsigned int rid) 
 
// int spe_mfc_putb(speid_t speid, unsigned int ls, void *ea, unsigned int size, unsigned int tag, unsigned int tid, unsigned int rid) 
int spu_putb(speid_t speid, unsigned int ls, unsigned long ea, unsigned int size, unsigned int tag, unsigned int tid, unsigned int rid) {
  return spe_mfc_putb(speid, ls, (void *)ea, size, tag, tid, rid);
  }

// int spe_mfc_putf(speid_t speid, unsigned int ls, void *ea, unsigned int size, unsigned int tag, unsigned int tid, unsigned int rid) 


// MFC Get Functions
// int mfc_get(speid_t speid, unsigned int ls, void *ea, unsigned int size, unsigned int tag, unsigned int tid, unsigned int rid);

int spu_getb(speid_t speid, unsigned int ls, unsigned long ea, unsigned int size, unsigned int tag, unsigned int tid, unsigned int rid) {
  printf("%X %X %d %d %d %d\n", ls, ea, size, tag, tid, rid);
  return spe_mfc_getb(speid, ls, (void *)ea, size, tag, tid, rid);
}

int read_tag_status_all(speid_t speid, unsigned int mask) {
  return spe_mfc_read_tag_status_all(speid, mask);
}

// int mfc_getf(speid_t speid, unsigned int ls, void *ea, unsigned int size, unsigned int tag, unsigned int tid, unsigned int rid);

#endif // SPU_EXEC_H
