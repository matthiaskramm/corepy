// Copyright 2006-2007 The Trustees of Indiana University.

// This software is available for evaluation purposes only.  It may not be
// redistirubted or used for any other purposes without express written
// permission from the authors.

// Authors:
//   Christopher Mueller (chemuell@cs.indiana.edu)
//   Andrew Lumsdaine    (lums@cs.indiana.edu)

#include <stdio.h>
#include <spu_intrinsics.h>
#include <spu_mfcio.h>

#define SPU_EXIT_FAILURE 0x2001
#define SPU_READY 0x000D

typedef void (*Stream_func_void)( vector unsigned int v1, 
                                  vector unsigned int v2,
                                  vector unsigned int v3);


int main(unsigned long long id) {

  // Vector parameters - passed on to the synthetic program
  register vector unsigned int v1 asm("$3");
  register vector unsigned int v2 asm("$4");
  register vector unsigned int v3 asm("$5");
  
  // Stop with the ready signal
  asm("stop 0x000D");
  
  printf("Hello, CorePy\n");
  printf("Hello, CorePy\n");  
  printf("Hello, CorePy\n");
  printf("Hello, CorePy\n");
  printf("Hello, CorePy\n");

  printf("Hello, CorePy\n");
  printf("Hello, CorePy\n");  
  printf("Hello, CorePy\n");
  printf("Hello, CorePy\n");
  printf("Hello, CorePy\n");

  // Return 12
  return retVal;
}
