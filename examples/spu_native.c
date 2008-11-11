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


// Example SPU C program to run via CorePy using the native code functionality;
//  demonstrates how to access parameters passed from CorePy.
// See spu_native.py for the related CorePy code example.

//Compile with:
// spu-gcc spu_native.c -o spu_native  spu_native.c -L/opt/cell/sdk/usr/spu/lib -I/opt/cell/sdk/usr/spu/include -Wl,-N -lmisc

#include <stdio.h>
#include <spu_intrinsics.h>
#include <spu_mfcio.h>
#include <simdmath.h>
#include <libmisc.h>

struct ExecParams
{
  unsigned int addr;
  unsigned int p1;
  unsigned int p2;
  unsigned int p3;

  unsigned int size;
  unsigned int p4;
  unsigned int p5;
  unsigned int p6;

  unsigned int p7;
  unsigned int p8;
  unsigned int p9;
  unsigned int p10;
} __attribute__((aligned(16)));


struct ExecParams* getExecParams(uint64_t argp)
{
  struct ExecParams* params = malloc_align(sizeof(struct ExecParams), 4);
  int tag = mfc_tag_reserve();

  mfc_get(params, argp, sizeof(struct ExecParams), tag, 0, 0);
  mfc_write_tag_mask(1<<tag);
  mfc_read_tag_status_all();
  mfc_tag_release(tag);

  return params;
}


int main(int id, uint64_t argp)
{
  struct ExecParams* params = getExecParams(argp);

  printf("p1 %d\n", params->p1);
  printf("p2 %d\n", params->p2);
  printf("p3 %d\n", params->p3);
  printf("p4 %d\n", params->p4);
  printf("p5 %d\n", params->p5);
  printf("p6 %d\n", params->p6);
  printf("p7 %d\n", params->p7);
  printf("p8 %d\n", params->p8);
  printf("p9 %d\n", params->p9);
  printf("p10 %d\n", params->p10);

  free_align(params);
  return 0;
}
