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

// 128-bit bit vector population count

#include <stdio.h>
#include <spu_intrinsics.h>
#include <spu_mfcio.h>

#include "corepy.h"

vector unsigned int
popc(vector unsigned int x) {
  vector unsigned int count;
  
  count = (vector unsigned int)spu_cntb((vector unsigned char)x);
  count = (vector unsigned int)spu_sumb((vector unsigned char)count, (vector unsigned char)0);

  return count;
}

vector unsigned int
reduce_word(vector unsigned int x) {
  vector unsigned int result;
  int i = 0;

  for(i; i < 4; i++) {
    result = result + x;
    x = si_rotqbyi(x, 4);
  }

  return result;
}


int 
main(unsigned long long id) {
  vector unsigned int x = get_vector_param_3();
  vector unsigned int count  = (vector unsigned int){0,0,0,0};
  vector unsigned int result = (vector unsigned int){0,0,0,0};

  spu_ready();

  count  = popc(x);
  result = reduce_word(count);
  
  spu_write_out_mbox(spu_extract(result, 0));

  return SPU_SUCCESS;
}

