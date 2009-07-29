# Copyright (c) 2006-2009 The Trustees of Indiana University.                   
# All rights reserved.                                                          
#                                                                               
# Redistribution and use in source and binary forms, with or without            
# modification, are permitted provided that the following conditions are met:   
#                                                                               
# - Redistributions of source code must retain the above copyright notice, this 
#   list of conditions and the following disclaimer.                            
#                                                                               
# - Redistributions in binary form must reproduce the above copyright notice,   
#   this list of conditions and the following disclaimer in the documentation   
#   and/or other materials provided with the distribution.                      
#                                                                               
# - Neither the Indiana University nor the names of its contributors may be used
#   to endorse or promote products derived from this software without specific  
#   prior written permission.                                                   
#                                                                               
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"   
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE     
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE   
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL    
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR    
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER    
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, 
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.          

import corepy.spre.spe as spe
import corepy.arch.spu.isa as spu
import corepy.lib.extarray as extarray

def load_word(code, r_target, word, clear = False, zero = True):
  """If r0 is not set to 0, the zero parameter should be set to False"""

  #if zero and (-512 < word < 511):
  #  code.add(spu.ai(r_target, code.r_zero, word))
  #elif (word & 0x7FFF) == word:
  #  code.add(spu.il(r_target, word))
  if (word & 0x3FFFF) == word:
    code.add(spu.ila(r_target, word))
  else:
    code.add(spu.ilhu(r_target, (word & 0xFFFF0000) >> 16))
    if word & 0xFFFF != 0:
      code.add(spu.iohl(r_target, (word & 0xFFFF)))

  if clear:
    code.add(spu.shlqbyi(r_target, r_target, 12))
  return


def load_float(code, reg, val):
  data = extarray.extarray('f', (val,))
  data.change_type('I')

  return load_word(code, reg, data[0])


def vector_from_array(code, r_target, a):
  """
  Generate the instructions to fill a vector register with the values
  from an array.
  """
  prgm = code.prgm
  r0 = r_target

  r1 = prgm.acquire_register()
  r2 = prgm.acquire_register()
  r3 = prgm.acquire_register()

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

  prgm.release_register(r1)
  prgm.release_register(r2)
  prgm.release_register(r3)

  return


def set_slot_value(code, reg, slot, value):
  """
  Set the value in reg[slot] with value.  If value is a register, use
  the value from the preferred slot (value[0]).  If value is a
  constant, load it into reg[slot], preserving the values in the other
  slots.
  """
  prgm = code.prgm

  if slot not in [0,1,2,3]:
    raise Exception("Invalid SIMD slot: " + slot)

  mask = prgm.acquire_register()
  vector_from_array(code, mask, [0xFFFFFFFF, 0, 0, 0])

  if not issubclass(type(value), (spe.Register, spe.Variable)):
    r_value = prgm.acquire_register()
    load_word(code, r_value, value)
  else:
    r_value = value

  code.add(spu.rotqbyi(reg, reg, slot * 4))
  code.add(spu.selb(reg, reg, r_value, mask))
  code.add(spu.rotqbyi(reg, reg, (4 - slot) * 4))

  prgm.release_register(mask)
  if not issubclass(type(value), (spe.Register, spe.Variable)):
    prgm.release_register(r_value)
  return



def get_param_reg(code, param, dict, copy = True):
  """ Take a parameter given to a function, which may be a value or a
      register containing that value, and return a register containing the
      value.

      If copy is True, a new register is always returned.  Otherwise if a
      register was passed in, that register is returned unchanged. 

      dict is a dictionary used internally between get_param_reg() and
      put_param_reg() to keep track of whether registers have been allocated for
      parameters.  A function should use one (initially empty) dictionary for
      all of its parameters.
  """

  reg = None

  if isinstance(param, (spe.Register, spe.Variable)):
    if copy == True:
      # TODO - behave differently if at an even/odd spot
      reg = code.prgm.acquire_register()
      code.add(spu.ori(reg, param, 0))
      dict[reg] = True
    else:
      reg = param
      dict[reg] = False
  else: # TODO - check types?
    reg = code.prgm.acquire_register()
    load_word(code, reg, param)
    dict[reg] = True

  return reg


def put_param_reg(code, reg, dict):
  """Check a register containing a parameter, release the register if the
     provided dictionary indicates it was acquired by get_param_reg()/
  """
  if dict[reg] == True:
    code.prgm.release_register(reg)


# ------------------------------------------------------------
# Unit Test Code
# ------------------------------------------------------------

def TestSetSlotValue():
  import corepy.arch.spu.platform as synspu
  import corepy.arch.spu.types.spu_types as var
  import corepy.arch.spu.lib.dma as dma

  prgm = synspu.Program()
  code = prgm.get_stream()
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

  prgm.add(code)
  spe_id = proc.execute(prgm, async = True)

  for i in range(4):
    while synspu.spu_exec.stat_out_mbox(spe_id) == 0: pass
    result = synspu.spu_exec.read_out_mbox(spe_id)
    assert(result == (i + 0x10))

  proc.join(spe_id)

  return


if __name__=='__main__':
  TestSetSlotValue()
