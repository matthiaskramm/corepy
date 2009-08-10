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

__doc__="""
SPE for the x86_64 processor family.
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

WORD_TYPE = 'L'           # array type that corresponds to 1 word
WORD_SIZE = 8             # size in bytes of one word
WORD_BITS = WORD_SIZE * 8 # number of bits in a word


# ------------------------------------------------------------
# InstructionStream
# ------------------------------------------------------------

class InstructionStream(spe.InstructionStream): pass


class Program(spe.Program):
  exec_module = x86_64_exec
  default_register_type = GPRegister64
  instruction_type  = 'B'

  gp_return = rax
  fp_return = xmm0

  stream_type = InstructionStream

  def __init__(self):
    spe.Program.__init__(self, None)
    return

  def create_register_files(self):
    # Certain registers should never be available to be acquired
    # rax/rcx/rdx/rsp/rbp are used implicitly, so users always need them
    # available.  Don't include them in the register pool.
    self._register_avail_bins = [#(rax, eax, ax, al),
                                 (rbx, ebx, bx, bl),
                                 #(rcx, ecx, cx, cl),
                                 #(rdx, edx, dx, dl),
                                 #(rsp, esp, sp, spl),
                                 #(rbp, ebp, bp, bpl),
                                 (rsi, esi, si, sil),
                                 (rdi, edi, di, dil),
                                 (r8, r8d, r8w, r8b),
                                 (r9, r9d, r9w, r9b),
                                 (r10, r10d, r10w, r10b),
                                 (r11, r11d, r11w, r11b),
                                 (r12, r12d, r12w, r12b),
                                 (r13, r13d, r13w, r13b),
                                 (r14, r14d, r14w, r14b),
                                 (r15, r15d, r15w, r15b)]

    # Map GP register types to position in the bin tuples above
    self._reg_map = {GPRegister64:0, GPRegister32:1,
                     GPRegister16:2, GPRegister8:3}

    # FP/MMX/XMM regs can be treated just like other archs
    # Skip st0, it's a special register that should always be used explicitly,
    # not via acquire/release.
    self._register_files[FPRegister] = [FPRegister("st%d" % i) for i in xrange(1, 8)]
    self._register_files[MMXRegister] = [MMXRegister("mm%d" % i) for i in xrange(0, 8)]
    self._register_files[XMMRegister] = [XMMRegister("xmm%d" % i) for i in xrange(0, 16)]

    RegisterFiles = (('gp8', GPRegister8),   ('gp16', GPRegister16),
                     ('gp32', GPRegister32), ('gp64', GPRegister64),
                     ('st', FPRegister),     ('mm', MMXRegister),
                     ('xmm', XMMRegister))

    for (reg_type, cls) in RegisterFiles:
      self._reg_type[reg_type] = cls

    return


  def _align_stream(self, length, align):
    return [x86.nop() for i in xrange(0, align - (length % align))]


  # ------------------------------
  # Execute/ABI support
  # ------------------------------

  def make_executable(self):
    self.exec_module.make_executable(self.render_code.buffer_info()[0], len(self.render_code))

  def _synthesize_prologue(self):
    """
    Create the prologue.

    This manages the register preservation requirements from the ABI.
    """

    # Set up the call frame and push callee-save registers
    # Note the stack is expected to remain 16-byte aligned, which is true here.
    self._prologue = [self.lbl_prologue,
                      x86.push(rbp, ignore_active = True),
                      x86.mov(rbp, rsp, ignore_active = True),
                      x86.push(r15, ignore_active = True),
                      x86.push(r14, ignore_active = True),
                      x86.push(r13, ignore_active = True),
                      x86.push(r12, ignore_active = True),
                      x86.push(rbx, ignore_active = True)]
    return


  def _synthesize_epilogue(self):
    """
    Restore the callee-save registers
    """

    # Pop callee-save regs and clean up the stack frame
    self._epilogue = [self.lbl_epilogue,
                      x86.pop(rbx, ignore_active = True),
                      x86.pop(r12, ignore_active = True),
                      x86.pop(r13, ignore_active = True),
                      x86.pop(r14, ignore_active = True),
                      x86.pop(r15, ignore_active = True),
                      x86.leave(ignore_active = True),
                      x86.ret(ignore_active = True)]
    return


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

