// Copyright 2006 The Trustees of Indiana University.

// This software is available for evaluation purposes only.  It may not be
// redistirubted or used for any other purposes without express written
// permission from the authors.

// Author:
//   Christopher Mueller

// Native code for executing instruction streams on OS X.
// Compile with -DDEBUG_PRINT to enable additional debugging code

#ifndef CELL_EXEC_H
#define CELL_EXEC_H

#include <stdio.h>
#include <stdint.h>
#include <sys/mman.h>
#include <errno.h>

#include <vector>

#include <pthread.h>
//#include <signal.h>

// #define DEBUG_PRINT 1

// ------------------------------------------------------------
// Typedefs
// ------------------------------------------------------------

// Function pointers for different return values
typedef void (*Stream_func_void)();
typedef int  (*Stream_func_int)();
typedef double (*Stream_func_double)();

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

int make_executable(int addr, int size) {
//   //  MakeDataExecutable((void *)addr, size * 4);
//   if(mprotect((void *)(addr & 0xFFFFF000), size * 4 + (addr & 0xFFF), 
//   	      PROT_READ | PROT_EXEC) == -1)
//     perror("make_executeable");
//   else
//     printf("Memory protection changed: %d\n", size * 4 + (addr & 0xFFF));
  return 0;
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
 
void *run_stream(void *addr) {
  ((Stream_func_void)addr)();
  pthread_exit(NULL);
  return NULL;
}

// ------------------------------------------------------------
// Function: execute_async
// Arguments:
//   addr - address of the instruction stream
// Return: a new thread id
// Description:
//   The native interface for executing a code stream as a 
//   thread.  make_executable must be called first.
// ------------------------------------------------------------

pthread_t execute_async(int addr) {
  pthread_t t;
  int rc;
  
  rc = pthread_create(&t, NULL, run_stream, (void *)addr);
  if(rc) {
    printf("Error creating async stream: %d\n", rc);
  }
  return t;
}

// ------------------------------------------------------------
// Function: cancel_async
// Arguments:
//   t - thread id cancel
// Return: 0 on success, -1 on failure
// Description:
//   The native interface for cancelling execution of a thread.
// ------------------------------------------------------------

int cancel_async(pthread_t t) {
  return pthread_cancel(t);
}

// ------------------------------------------------------------
// Function: suspend_async
// Arguments:
//   t - thread id to cancle
// Return: 0 on success, -1 on failure
// Description:
//   The native interface for suspending execution of a thread.
//   Becuase there is no pthread interface for thread 
//   susped/resume, this suspends the underlying OS X mach 
//   thread.
// ------------------------------------------------------------

int suspend_async(pthread_t t) {
  return -1;
}

// ------------------------------------------------------------
// Function: resume_async
// Arguments:
//   t - thread id to resume
// Return: 0 on success, -1 on failure
// Description:
//   The native interface for resuming execution of a thread.
//   Becuase there is no pthread interface for thread 
//   susped/resume, this suspends the underlying OS X mach 
//   thread.
// ------------------------------------------------------------

int resume_async(pthread_t t) {
  return -1;
}

// ------------------------------------------------------------
// Functions: execute_{void,int,fp}
// Arguments:
//   addr - instruction stream address
// Return: 
//   _void - nothing
//   _int  - the value in regsiter gp_return (r3)
//   _fp   - the value in regsiter fp_return (fp1)
// Description:
//   The native interfaces for executing instruction streams.
//   Each fucntion casts the stream to a function of the 
//   appropriate type and calls it. If DEBUG_PRINT is set, the 
//   address and result are printed. 
// ------------------------------------------------------------

void execute_void(int addr) {
#ifdef DEBUG_PRINT
  printf("addr (native): %d\n", addr);
#endif // DEBUG_PRINT

  ((Stream_func_void)addr)();

  return;
}

int execute_int(int addr) {
  int result = 0;

#ifdef DEBUG_PRINT
  printf("addr (native): 0x%X\n", addr);
#endif // DEBUG_PRINT

  result = ((Stream_func_int)addr)();

#ifdef DEBUG_PRINT     
  printf("result: %d\n", result);
#endif // DEBUG_PRINT
  return result;
}

double execute_fp(int addr) {
  double result = 0.0;

#ifdef DEBUG_PRINT
  printf("addr (native): %d\n", addr);
#endif // DEBUG_PRINT

  result = ((Stream_func_double)addr)();

#ifdef DEBUG_PRINT     
  printf("result: %d\n", result);
#endif // DEBUG_PRINT
  return result;
}

// ------------------------------------------------------------
// Parameter passing execute functions
// ------------------------------------------------------------

struct ThreadParams {
  int addr;
  int p1;
  int p2;
  int p3;
};

typedef int  (*Stream_param_func_int)(int, int, int);

void *run_param_stream(void *params) {
  ThreadParams *p = (ThreadParams*)params;

  int addr = p->addr;
  int p1 = p->p1;
  int p2 = p->p2;
  int p3 = p->p3;
  int r = 0;

  r = ((Stream_param_func_int)(addr))(p1, p2, p3);

  pthread_exit(NULL);
  return NULL;
}

pthread_t execute_param_async(int addr, int p1, int p2, int p3) {
  pthread_t t;
  int rc;
  
  ThreadParams *params = new ThreadParams();
  params->addr = addr;
  params->p1 = p1;
  params->p2 = p2;
  params->p3 = p3;

  rc = pthread_create(&t, NULL, run_param_stream, (void *)params);
  if(rc) {
    printf("Error creating async stream: %d\n", rc);
  }

  return t;
}

int join_async(pthread_t t) {
  return pthread_join(t, NULL);
}


int execute_param_int(int addr, int p1, int p2, int p3) {
  int result = 0;
  result = ((Stream_param_func_int)addr)(p1, p2, p3);

  return result;
}

#endif // CELL_EXEC_H
