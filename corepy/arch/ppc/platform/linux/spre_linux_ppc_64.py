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
SPE for the Cell PPU.
"""

import array
import corepy.lib.extarray as extarray
import corepy.spre.spe as spe
import ppc_exec

import corepy.arch.ppc.isa as ppc
import corepy.arch.vmx.isa as vmx
from   corepy.arch.ppc.lib.util import load_word

ExecParams = ppc_exec.ExecParams

# ------------------------------
# Registers
# ------------------------------

class GPRegister(spe.Register):
  def __init__(self, reg, code):
    spe.Register.__init__(self, reg, code, prefix = 'r')

class FPRegister(spe.Register):
  def __init__(self, reg, code):
    spe.Register.__init__(self, reg, code, prefix = 'f')

class VMXRegister(spe.Register):
  def __init__(self, reg, code):
    spe.Register.__init__(self, reg, code, prefix = 'v')



# ------------------------------
# Constants
# ------------------------------

WORD_TYPE = 'I'           # array type that corresponds to 1 word
WORD_SIZE = 4             # size in bytes of one word
WORD_BITS = WORD_SIZE * 8 # number of bits in a word

# Parameter Registers
gp_param_1 = 3
gp_param_2 = 4
gp_param_3 = 5

# Return registers
# TODO - AWF - whats going with these?
#fp_return = 1
#vx_return = 1
#gp_return = 3

# Callee save registers
gp_save = [GPRegister(i, None) for i in range(14, 31)]
fp_save = [FPRegister(i, None) for i in range(14, 32)]
vx_save = [VMXRegister(i, None) for i in range(20, 32)]  # !!! NOT SURE ABOUT VMX !!!
#vx_save = [VMXRegister(i, None) for i in range(0, 32)]  # !!! NOT SURE ABOUT VMX !!!

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def copy_param(code, target, param):
  """
  Copy a parameter to the taget register.
  """
  if param not in (gp_param_1, gp_param_2, gp_param_3):
    raise Exception('Invalid parameter id: ' + str(param))
  code.add(ppc.addi(target, param, 0))
  return


# ------------------------------------------------------------
# InstructionStream
# ------------------------------------------------------------

class InstructionStream(spe.InstructionStream):
  """
  PPC Linux/Cell ABI 
  """

  # Class attributes
  RegisterFiles = (('gp', GPRegister, range(3,31)),
                   ('fp', FPRegister, range(0,32)),
                   ('vector', VMXRegister, range(0,32)))

  default_register_type = GPRegister
  exec_module   = ppc_exec
  instruction_type  = WORD_TYPE
  
  def __init__(self):
    spe.InstructionStream.__init__(self)
    
    # Memory buffers for saved registers
    self._saved_gp_registers = None
    self._saved_fp_registers = None
    self._saved_vx_registers = None

    # Return Register 'Constants'
    #   *_return can be used with a return register is needed.

    #   Note that these do not reserve the register, but only identify
    #   the registers.  To reserve a return register, use:
    #     code.acquire_register(reg = code.gp_return)
    self.fp_return = FPRegister(1, self)
    self.vx_return = VMXRegister(1, self)
    self.gp_return = GPRegister(3, self)
    self._vrsave = GPRegister(31, None)

    return

  def make_executable(self):
    self.exec_module.make_executable(self.render_code.buffer_info()[0], len(self.render_code))
    return 

  def create_register_files(self):
    # Each declarative RegisterFiles entry is:
    #   (file_id, register class, valid values)
    for reg_type, cls, values in self.RegisterFiles:
      regs = [cls(value, self) for value in values]
      self._register_files[cls] = spe.RegisterFile(regs, reg_type)
      self._reg_type[reg_type] = cls
      for reg in regs:
        reg.code = self
    
    return
  
  # ------------------------------
  # Execute/ABI support
  # ------------------------------

  def _load_word(self, array, reg, word):
    """Load an immediate value into a register w/o using load_word(); instead
       append the instruction objects to an array.
       Used when synthesizing the prologue/epilogue."""
    array.append(ppc.addi(reg, 0, word & 0xFFFF, ignore_active = True))
    if (word & 0xFFFF) != word:
      array.append(ppc.addis(reg, reg, ((word + 32768) >> 16) & 0xFFFF, ignore_active = True))
    return

  def _synthesize_prologue(self):
    """
    Create the prologue. (see PPC ABI p41)

    This manages the register preservation requirements from the ABI.

    TODO: CR2-4 need to be preserved.
    """

    # Reset the prologue
    self._prologue = [self.lbl_prologue]

    # Get the lists of registers to save
    save_gp = [reg for reg in self._register_files[GPRegister].get_used() if reg in gp_save]
    save_fp = [reg for reg in self._register_files[FPRegister].get_used() if reg in fp_save]
    save_vx = [reg for reg in self._register_files[VMXRegister].get_used() if reg in vx_save]    
    
    self._saved_gp_registers = array.array('I', range(len(save_gp)))
    self._saved_fp_registers = array.array('d', range(len(save_fp)))
    self._saved_vx_registers = array.array('I', range(len(save_vx)*4))

    # Add the instructions to save the registers

    # Store the value in register 2 in the red zone
    #  r1 is the stackpointer, -4(r1) is in the red zone

    r_addr = GPRegister(13, None) # Only available volatile register
    r_idx = GPRegister(14, None)  # Non-volatile; safe to use before restoring

    # TODO - AWF - don't want to push things on the stack, that changes the
    # relative location of the passed-in arguments
    # However, we could just use the stack to save all the registers, and use
    # a frame pointer to give access to the arguments
    # self._prologue.add(ppc.stwu(r_addr, r_addr, -WORD_SIZE))
    
    self._load_word(self._prologue, r_addr, self._saved_gp_registers.buffer_info()[0])

    for i, reg in enumerate(save_gp):
      #print 'saving gp:', reg, r_addr, i * WORD_SIZE
      self._prologue.append(ppc.stw(reg, r_addr, i * WORD_SIZE, ignore_active = True))

    self._load_word(self._prologue, r_addr, self._saved_fp_registers.buffer_info()[0])
    
    for i, reg in enumerate(save_fp):
      #print 'saving fp:', reg, r_addr, i * WORD_SIZE
      self._prologue.append(ppc.stfd(reg, r_addr, i * WORD_SIZE * 2, ignore_active = True))

    self._load_word(self._prologue, r_addr, self._saved_vx_registers.buffer_info()[0])
   
    for i, reg in enumerate(save_vx):
      #print 'saving vx:', reg, r_addr, i * WORD_SIZE * 4
      self._load_word(self._prologue, r_idx, i * WORD_SIZE * 4)
      self._prologue.append(vmx.stvx(reg, r_idx, r_addr, ignore_active = True))

    # Set up VRSAVE
    # Currently, we save the old value of VRSAVE in r31.
    # On the G4, someone stomps on registers < 20 ... save them all for now.

    # Save vrsave and put our value in it
    self._prologue.append(ppc.mfvrsave(self._vrsave, ignore_active = True))
    self._load_word(self._prologue, r_addr, 0xFFFFFFFF)
    self._prologue.append(ppc.mtvrsave(r_addr, ignore_active = True))    
    return


  def _synthesize_epilogue(self):
    """
    Save the values in some registers (see PPC ABI p41)
    """

    # Reset the epilogue
    self._epilogue = [self.lbl_epilogue]

    # Restore vrsave
    self._epilogue.append(ppc.mtvrsave(self._vrsave))

    # Get the list of saved registers
    save_gp = [reg for reg in self._register_files[GPRegister].get_used() if reg in gp_save]
    save_fp = [reg for reg in self._register_files[FPRegister].get_used() if reg in fp_save]
    save_vx = [reg for reg in self._register_files[VMXRegister].get_used() if reg in vx_save]    

    r_addr = GPRegister(13, None) # Only available volatile register
    r_idx = GPRegister(14, None)  # Non-volatile; safe to use before restoring

    self._load_word(self._epilogue, r_addr, self._saved_vx_registers.buffer_info()[0])

    for i, reg in enumerate(save_vx):
      #print 'restoring vx:', reg, r_addr, i * WORD_SIZE * 4
      self._load_word(self._epilogue, r_idx, i * WORD_SIZE * 4)
      self._epilogue.append(vmx.lvx(reg, r_idx, r_addr, ignore_active = True))

    self._load_word(self._epilogue, r_addr, self._saved_fp_registers.buffer_info()[0])

    for i, reg in enumerate(save_fp):
      # print 'restoring fp:', reg, r_addr, i * WORD_SIZE
      self._epilogue.append(ppc.lfd(reg, r_addr, i * WORD_SIZE * 2, ignore_active = True))

    self._load_word(self._epilogue, r_addr, self._saved_gp_registers.buffer_info()[0])

    for i, reg in enumerate(save_gp):
      # print 'restoring gp:', reg, r_addr, i * WORD_SIZE
      self._epilogue.append(ppc.lwz(reg, r_addr, i * WORD_SIZE, ignore_active = True))

    self._epilogue.append(ppc.blr(ignore_active = True))
    return


class Processor(spe.Processor):
  exec_module = ppc_exec
  

# ------------------------------------------------------------
# Unit tests
# ------------------------------------------------------------

def TestInt():
  code = InstructionStream()
  proc = Processor()

  code.add(ppc.addi(gp_return, 0, 12))

  r = proc.execute(code)
  assert(r == 12)
  print 'int result:', r
  return

def TestFloat():
  code = InstructionStream()
  proc = Processor()
  a = array.array('d', [3.14])

  load_word(code, gp_return, a.buffer_info()[0])
  code.add(ppc.lfd(fp_return, gp_return, 0))

  r = proc.execute(code, mode='fp')
  assert(r == 3.14)
  print 'float result:', r
  return


if __name__ == '__main__':
  TestInt()
  TestFloat()
