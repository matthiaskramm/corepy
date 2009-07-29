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

import corepy.arch.spu.isa as spu
import corepy.spre.spe as spe

from corepy.arch.spu.lib.util import load_word

__doc__="""
SPU Extended Instructions
"""

class SPUExt(spe.ExtendedInstruction):
  isa_module = spu

class shr(SPUExt):
  """
  Shift-right word.

  Shift the value in register a to the right by the number of bits
  specified by the value in register b.  Store the result in register
  d. 
  """
  def block(self, d, a, b):
    # Based on example on p133 of SPU ISA manual
    code = self.get_active_code()
    temp = code.prgm.acquire_register()
    spu.sfi(temp, b, 0)
    spu.rotm(d, a, temp)
    code.prgm.release_register(temp)
    return

class cneq(SPUExt):
  """
  Compare the word values in registers a and b.  If the operands are
  not equal, register d contains all ones.
  """
  def block(self, d, a, b):
    spu.ceq(d, a, a)
    spu.nor(d, d, d)
    return


class cge(SPUExt):
  """
  Word compare greater than equal.
  """

  def block(self, d, a, b):
    code = self.get_active_code()
    temp = code.prgm.acquire_register()
    spu.cgt(temp, a, b)
    spu.ceq(d, a, b)
    spu.or_(d, d, temp)
    code.prgm.release_register(temp)
    return

class cgei(SPUExt):
  """
  Word compare greater than equal immediate.
  """

  def block(self, d, a, b):
    code = self.get_active_code()
    temp = code.prgm.acquire_register()
    spu.cgti(temp, a, b)
    spu.ceqi(d, a, b)
    spu.or_(d, d, temp)
    code.prgm.release_register(temp)
    return


class lt(SPUExt):
  """
  Word compare less than
  """

  def block(self, d, a, b):
    spu.cgt(d, b, a)
    return

class lti(SPUExt):
  """
  Word compare less than
  """

  def block(self, d, a, b):
    code = self.get_active_code()    
    temp = code.prgm.acquire_register()
    spu.cgti(temp, a, b)
    spu.ceqi(d, a, b)
    spu.nor(d, d, temp)
    code.prgm.release_register(temp)
    
    return

class sub(SPUExt):
  """
  Subtract
  """

  def block(self, d, a, b):
    spu.sf(d, b, a)
    return

class subi(SPUExt):
  """
  Subtract immediate
  """

  def block(self, d, a, value):
    code = self.get_active_code()    
    temp = code.prgm.acquire_register()

    load_word(code, temp, value)
    # RD = RB - RA    
    spu.sf(d, temp, a)
    code.prgm.release_register(temp)

    return

class extended_I10(SPUExt):
  """
  Take an instruction and its immediate form and dispatch to the
  immediate form if the value fits into the I10 range.  Otherwise,
  load the value into a temporary register and use the original form.

  The instruction and the immediate form are set using the inst and
  insti class variables.  For example, the add instruction is formed
  using:
  
    inst  = spu.a
    insti = spu.ai
  
  """

  inst  = None
  insti = None

  def block(self, d, a, value):
    """
    Dispatch to the proper form of the instruction.
    """

    if (-512 < value < 512):
      self.insti(d, a, value)
    else:
      code = self.get_active_code()      
      temp = code.prgm.acquire_register()

      load_word(code, temp, value)
      self.inst(d, a, temp)

      code.prgm.release_register(temp)
    return

class ah_immediate(extended_I10):
  inst  = spu.ah
  insti = spu.ahi
  
class a_immediate(extended_I10):
  inst  = spu.a
  insti = spu.ai
  
class sfh_immediate(extended_I10):
  inst  = spu.sfh
  insti = spu.sfhi
  
class sf_immediate(extended_I10):
  inst  = spu.sf
  insti = spu.sfi
  
class mpy_immediate(extended_I10):
  inst  = spu.mpy
  insti = spu.mpyi
  
class mpyu_immediate(extended_I10):
  inst  = spu.mpyu
  insti = spu.mpyui
  
class and_immediate(extended_I10):
  inst  = spu.and_
  insti = spu.andi
  
class or_immediate(extended_I10):
  inst  = spu.or_
  insti = spu.ori
  
class xor_immediate(extended_I10):
  inst  = spu.xor
  insti = spu.xori
  
class heq_immediate(extended_I10):
  inst  = spu.heq
  insti = spu.heqi
  
class hgt_immediate(extended_I10):
  inst  = spu.hgt
  insti = spu.hgti
  
class hlgt_immediate(extended_I10):
  inst  = spu.hlgt
  insti = spu.hlgti
  
class ceqb_immediate(extended_I10):
  inst  = spu.ceqb
  insti = spu.ceqbi
  
class ceqh_immediate(extended_I10):
  inst  = spu.ceqh
  insti = spu.ceqhi
  
class ceq_immediate(extended_I10):
  inst  = spu.ceq
  insti = spu.ceqi
  
class cgtb_immediate(extended_I10):
  inst  = spu.cgtb
  insti = spu.cgtbi
  
class cgth_immediate(extended_I10):
  inst  = spu.cgth
  insti = spu.cgthi
  
class cgt_immediate(extended_I10):
  inst  = spu.cgt
  insti = spu.cgti
  
class clgtb_immediate(extended_I10):
  inst  = spu.clgtb
  insti = spu.clgtbi
  
class clgth_immediate(extended_I10):
  inst  = spu.clgth
  insti = spu.clgthi
  
class clgt_immediate(extended_I10):
  inst  = spu.clgt
  insti = spu.clgti
  

# !!! STOPPED HERE !!!
# !!! TODO: Do the other immediate versions !!!

    
# ------------------------------------------------------------
# Unit Tests
# ------------------------------------------------------------

def TestAll():
  import corepy.arch.spu.platform as env

  prgm = env.Program()
  code = prgm.get_stream()
  spu.set_active_code(code)

  a = code.prgm.acquire_register()
  b = code.prgm.acquire_register()
  c = code.prgm.acquire_register()
  
  shr(c, a, b)
  cneq(c, a, b)
  cge(c, a, b)
  cgei(c, a, 10)
  lt(c, a, b)
  lti(c, a, 10)  

  a_immediate(c, a, 10)
  a_immediate(c, a, 10000)  
  sf_immediate(c, a, 10000)
  

  prgm.add(code)
  prgm.print_code()

  proc = env.Processor()
  proc.execute(prgm)
  return

if __name__=='__main__':
  TestAll()
