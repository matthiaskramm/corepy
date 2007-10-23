# Copyright 2006-2007 The Trustees of Indiana University.

# This software is available for evaluation purposes only.  It may not be
# redistirubted or used for any other purposes without express written
# permission from the authors.

# Authors:
#   Christopher Mueller (chemuell@cs.indiana.edu)
#   Andrew Lumsdaine    (lums@cs.indiana.edu)

import corepy.spre.spe as spe
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


def set_slot_value(code, reg, slot, value):
  """
  Set the value in reg[slot] with value.  If value is a register, use
  the value from the preferred slot (value[0]).  If value is a
  constant, load it into reg[slot], preserving the values in the other
  slots. 
  """
  if slot not in [0,1,2,3]:
    raise Exception("Invalid SIMD slot: " + slot)

  mask = code.acquire_register()
  vector_from_array(code, mask, [0xFFFFFFFF, 0, 0, 0])
  
  if not issubclass(type(value), (spe.Register, spe.Variable)):
    r_value = code.acquire_register()
    load_word(code, r_value, value)
  else:
    r_value = value

  code.add(spu.rotqbyi(reg, reg, slot * 4))
  code.add(spu.selb(reg, reg, r_value, mask))
  code.add(spu.rotqbyi(reg, reg, (4 - slot) * 4))
  
  code.release_register(mask)
  if not issubclass(type(value), (spe.Register, spe.Variable)):
    code.release_register(r_value)
  return


def TestSetSlotValue():
  import corepy.arch.spu.platform as synspu
  import corepy.arch.spu.isa as spu
  import corepy.arch.spu.types.spu_types as var
  import corepy.arch.spu.lib.dma as dma

  code = synspu.InstructionStream()
  proc = synspu.Processor()
  spu.set_active_code(code)
  a = var.SignedWord(0x11)
  b = var.SignedWord(0x13)
  r = var.SignedWord(0xFFFFFFFF)

  set_slot_value(code, r, 0, 0x10)
  set_slot_value(code, r, 1, a)
  set_slot_value(code, r, 2, 0x12)
  set_slot_value(code, r, 3, b)      

  for i in range(4):
    spu.wrch(r, dma.SPU_WrOutMbox)
    spu.rotqbyi(r, r, 4)
  
  spe_id = proc.execute(code, mode = 'async')

  for i in range(4):
    while synspu.spu_exec.stat_out_mbox(spe_id) == 0: pass
    result = synspu.spu_exec.read_out_mbox(spe_id)
    assert(result == (i + 0x10))

  proc.join(spe_id)

  return
  

if __name__=='__main__':
  TestSetSlotValue()
