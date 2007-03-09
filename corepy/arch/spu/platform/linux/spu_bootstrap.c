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

  // Preferred slot parameters
  register unsigned int size asm("$4");

  // Vector parameters - passed on to the synthetic program
  register vector unsigned int v1 asm("$3");
  register vector unsigned int v2 asm("$4");
  register vector unsigned int v3 asm("$5");

  // Instructions are stored at the end of memory (overwrite the stack),
  // rounded down to 128 byte alignment 
  unsigned int lsa = (0x3FFFF - size) & 0xFFF80;

  // Place a stop instruction immediately before the synthetic
  // program to ensure the program counter points to the start of the
  // program when the SPU is restarted.
  unsigned int stop = SPU_READY;
  *((unsigned int*)(lsa - 4)) = stop;
  
  // Branch to the stop instruction 
  ((Stream_func_void)(lsa - 4))(v1, v2, v3);

  // This should never be executed
  return SPU_EXIT_FAILURE;
}
