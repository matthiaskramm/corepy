# Copyright (c) 2006-2008 The Trustees of Indiana University.                   
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

__doc__="""
SPE for the x86 processor family.
"""

import corepy.spre.spe as spe
import x86_64_exec

import corepy.arch.x86_64.isa as x86
from   corepy.arch.x86_64.types.registers import *
from   corepy.arch.x86_64.lib.util import load_word

ExecParams = x86_64_exec.ExecParams


# ------------------------------
# Constants
# ------------------------------

HWORD_TYPE = 'B'          # half word (byte)
HWORD_SIZE = 1
HWORD_BITS = HWORD_SIZE * 8
WORD_TYPE = 'H'           # array type that corresponds to 1 word
WORD_SIZE = 2             # size in bytes of one word
WORD_BITS = WORD_SIZE * 8 # number of bits in a word


# ------------------------------------------------------------
# InstructionStream
# ------------------------------------------------------------

class InstructionStream(spe.InstructionStream):
  """
  x86_64 OSX ABI 
  """

  default_register_type = GPRegister64
  instruction_type  = 'B'
  exec_module = x86_64_exec

  gp_return = rax
  fp_return = xmm0
 
  # ------------------------------
  # Execute/ABI support
  # ------------------------------

  def _synthesize_prologue(self):
    """
    Create the prologue.

    This manages the register preservation requirements from the ABI.
    """

    # Reset the prologue
    self._prologue = [self.lbl_prologue]

    # Set up the call frame and push callee-save registers
    # Note the stack is expected to remain 16-byte aligned, which is true here.
    self._prologue.append(x86.mov(rax, rsp, ignore_active = True))
    self._prologue.append(x86.push(rbp, ignore_active = True))
    self._prologue.append(x86.mov(rbp, rsp, ignore_active = True))
    self._prologue.append(x86.push(r15, ignore_active = True))
    self._prologue.append(x86.push(r14, ignore_active = True))
    self._prologue.append(x86.push(r13, ignore_active = True))
    self._prologue.append(x86.push(r12, ignore_active = True))
    #self._prologue.append(x86.push(rdi, ignore_active = True))
    #self._prologue.append(x86.push(rsi, ignore_active = True))
    self._prologue.append(x86.push(rbx, ignore_active = True))

    return

  def _synthesize_epilogue(self):
    """
    Save the caller-save registers
    """

    # Reset the epilogue
    self._epilogue = [self.lbl_epilogue]

    # Pop callee-save regs and clean up the stack frame
    self._epilogue.append(x86.pop(rbx, ignore_active = True))
    #self._epilogue.append(x86.pop(rsi, ignore_active = True))
    #self._epilogue.append(x86.pop(rdi, ignore_active = True))
    self._epilogue.append(x86.pop(r12, ignore_active = True))
    self._epilogue.append(x86.pop(r13, ignore_active = True))
    self._epilogue.append(x86.pop(r14, ignore_active = True))
    self._epilogue.append(x86.pop(r15, ignore_active = True))
    self._epilogue.append(x86.leave(ignore_active = True))
    self._epilogue.append(x86.ret(ignore_active = True))
    return


  def make_executable(self):
    self.exec_module.make_executable(self.render_code.buffer_info()[0], len(self.render_code))

  def create_register_files(self):
      self._register_files[GPRegister8] = spe.RegisterFile(gp8_array, "gp8")
      self._register_files[GPRegister16] = spe.RegisterFile(gp16_array, "gp16")
      self._register_files[GPRegister32] = spe.RegisterFile(gp32_array, "gp32")
      self._register_files[GPRegister64] = spe.RegisterFile(gp64_array, "gp64")
      self._register_files[FPRegister] = spe.RegisterFile(st_array, "st")
      self._register_files[MMXRegister] = spe.RegisterFile(mm_array, "mm")
      self._register_files[XMMRegister] = spe.RegisterFile(xmm_array, "xmm")
      self._reg_type["gp8"] = GPRegister8
      self._reg_type["gp16"] = GPRegister16
      self._reg_type["gp32"] = GPRegister32
      self._reg_type["gp64"] = GPRegister64
      self._reg_type["st"] = FPRegister
      self._reg_type["mm"] = MMXRegister
      self._reg_type["xmm"] = XMMRegister


class Processor(spe.Processor):
  exec_module = x86_64_exec
  

# ------------------------------------------------------------
# Unit tests
# ------------------------------------------------------------

def TestInt():
  code = InstructionStream()
  proc = Processor()
  params = x86_64_exec.ExecParams()
  params.p1 = 32

  code.add(x86.mov(eax, spe.MemRef(ebp, 8)))
  #code.add(x86.xor(code.eax, code.eax))
  code.add(x86.add(eax, 1200))

  code.print_code(pro = False, epi = False, binary = True)
  r = proc.execute(code, debug = True, params = params)
  print 'int result:', r
  assert(r == 1232)
  return

# AWF - no float for now
#def TestFloat():
#  code = InstructionStream()
#  proc = Processor()
#  a = array.array('d', [3.14])

#  load_word(code, gp_return, a.buffer_info()[0])
#  code.add(ppc.lfd(fp_return, gp_return, 0))

#  r = proc.execute(code, mode='fp')
#  assert(r == 3.14)
#  print 'float result:', r
#  return


if __name__ == '__main__':
  TestInt()
#  TestFloat()

