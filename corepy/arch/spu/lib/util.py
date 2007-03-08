
import corepy.arch.spu.isa as spu

def load_word(code, r_target, word, clear = False, zero = True):
  # If r0 != 0, set zero to false
  if zero and (-512 < word < 511):
    code.add(spu.ai(r_target, 0, word))
  elif zero and (word & 0x7FFF) == word:
    code.add(spu.il(r_target, word))
  elif zero and (word & 0x3FFFF) == word:
    code.add(spu.ila(r_target, word))
  else:
    code.add(spu.ilhu(r_target, (word & 0xFFFF0000) >> 16))
    code.add(spu.iohl(r_target, (word & 0xFFFF)))

  if clear:
    code.add(spu.shlqbyi(r_target, r_target, 12))
  return


def vector_from_array(code, r_target, a):
  """
  Generate the instructions to fill a vector register with the values
  from an array.
  """
  r0 = r_target

  r1 = code.acquire_register()
  r2 = code.acquire_register()
  r3 = code.acquire_register()
  
  load_word(code, r0, a[0], True)
  load_word(code, r1, a[1], True)
  code.add(spu.rotqbyi(r1, r1, 12)) # rotate qw by bytes

  load_word(code, r2, a[2], True)
  code.add(spu.rotqbyi(r2, r2, 8))

  load_word(code, r3, a[3], True)
  code.add(spu.rotqbyi(r3, r3, 4))

  code.add(spu.a(r0, r0, r1))
  code.add(spu.a(r0, r0, r2))
  code.add(spu.a(r0, r0, r3)) 

  code.release_register(r1)
  code.release_register(r2)
  code.release_register(r3)
  
  return




# Copyright 2006 The Trustees of Indiana University.

# This software is available for evaluation purposes only.  It may not be
# redistirubted or used for any other purposes without express written
# permission from the authors.

# Authors:
#   Christopher Mueller (chemuell@cs.indiana.edu)
#   Andrew Lumsdaine    (lums@cs.indiana.edu)

