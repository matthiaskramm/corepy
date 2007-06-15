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

