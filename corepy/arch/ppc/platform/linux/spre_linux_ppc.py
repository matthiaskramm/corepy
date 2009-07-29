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

# TODO - make this code not be duplicated?
class GPRegister(spe.Register):
  def __init__(self, name):
    if isinstance(name, int):
      self.reg = name
      self.name = "r%d" % name
    elif isinstance(name, str):
      self.name = name
      try:
        self.reg = int(name[1:])
      except ValueError:
        raise Exception("Invalid register name %s" % name)

      if name[0] != 'r':
        raise Exception("Invalid register name %s" % name)
    else:
      raise Exception("Invalid register name %s" % str(name))
    return


class FPRegister(spe.Register):
  def __init__(self, name):
    if isinstance(name, int):
      self.reg = name
      self.name = "f%d" % name
    elif isinstance(name, str):
      self.name = name
      try:
        self.reg = int(name[1:])
      except ValueError:
        raise Exception("Invalid register name %s" % name)

      if name[0] != 'f':
        raise Exception("Invalid register name %s" % name)
    else:
      raise Exception("Invalid register name %s" % str(name))
    return


class VMXRegister(spe.Register):
  def __init__(self, name):
    if isinstance(name, int):
      self.reg = name
      self.name = "v%d" % name
    elif isinstance(name, str):
      self.name = name
      try:
        self.reg = int(name[1:])
      except ValueError:
        raise Exception("Invalid register name %s" % name)

      if name[0] != 'v':
        raise Exception("Invalid register name %s" % name)
    else:
      raise Exception("Invalid register name %s" % str(name))
    return


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
gp_save = ["r%d" % i for i in xrange(14, 31)]
fp_save = ["f%d" % i for i in xrange(14, 32)]
vx_save = ["v%d" % i for i in xrange(20, 32)]  # !!! NOT SURE ABOUT VMX !!!
#vx_save = [VMXRegister(i) for i in range(0, 32)]  # !!! NOT SURE ABOUT VMX !!!

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def copy_param(code, target, param):
  """
  Copy a parameter to the taget register.
  """
  if param not in (gp_param_1, gp_param_2, gp_param_3):
    raise Exception('Invalid parameter id: ' + str(param))
  code.add(ppc.addi(target, GPRegister(param), 0))
  return


# ------------------------------------------------------------
# InstructionStream
# ------------------------------------------------------------

class InstructionStream(spe.InstructionStream):
  """
  PPC Linux/Cell ABI 
  """

#  def __init__(self, prgm = None):
#    if not isinstance(prgm, Program):
#      raise TypeError("ERROR: A Program must be passed to InstructionStream.  Have you updated your code for the code composition changes?")
#    spe.InstructionStream.__init__(self, prgm)

#    return


class Program(spe.Program):
  default_register_type = GPRegister
  instruction_type  = WORD_TYPE
  stream_type = InstructionStream

  def __init__(self, enable_vmx = True, debug = False):
    spe.Program.__init__(self, debug)

    # Disable the default LRU-style register allocation, and instead use a
    # stack-based MRU style that minimizes number of registers changed, reducing
    # the number of registers that are saved/restored in the prolog/epilog.
    self.lru_reg_allocation = False

    self._enable_vmx = enable_vmx

    # Memory buffers for saved registers
    self._saved_gp_registers = None
    self._saved_fp_registers = None
    if self._enable_vmx:
      self._saved_vx_registers = None

    # Return Register 'Constants'
    #   *_return can be used with a return register is needed.

    #   Note that these do not reserve the register, but only identify
    #   the registers.  To reserve a return register, use:
    #     code.acquire_register(reg = code.gp_return)
    self.gp_return = GPRegister(3)
    self.fp_return = FPRegister(1)

    if self._enable_vmx:
      self.vx_return = VMXRegister(1)
      self._vrsave = GPRegister(31)
    return


  def make_executable(self):
    ppc_exec.make_executable(self.render_code.buffer_info()[0],
        len(self.render_code))
    return 

  def create_register_files(self):
    self._register_files[GPRegister] = [GPRegister(i) for i in xrange(3,31)]
    self._reg_type['gp'] = GPRegister

    self._register_files[FPRegister] = [FPRegister(i) for i in xrange(0,32)]
    self._reg_type['fp'] = FPRegister

    self._register_files[VMXRegister] = [VMXRegister(i) for i in xrange(0,32)]
    self._reg_type['vector'] = VMXRegister
    return

 
  # ------------------------------
  # Execute/ABI support
  # ------------------------------

  def _load_word(self, array, reg, word):
    """Load an immediate value into a register w/o using load_word(); instead
       append the instruction objects to an array.
       Used when synthesizing the prologue/epilogue."""
    array.append(ppc.addi(reg, 0, word & 0xFFFF, ignore_active = True))

    uw = (word >> 16) & 0xFFFF
    msb = word & 0x8000

    if msb != 0:
      # lower 16-bit MSB is set, upper 16 bits are 1, adjust uw
      # If all upper 16 bits are 1, that is the value -1, so add 1 back in.
      uw = (uw + 1) & 0xFFFF

    if uw != 0:
      array.append(ppc.addis(reg, reg, uw, ignore_active = True))
    return

  def _synthesize_prologue(self):
    """
    Create the prologue. (see PPC ABI p41)

    This manages the register preservation requirements from the ABI.

    TODO: CR2-4 need to be preserved.
    """

    # Reset the prologue
    self._prologue = [self.lbl_prologue]

    # Add the instructions to save the registers

    r_addr = GPRegister(13) # Only available volatile register
    r_idx = GPRegister(14)  # Non-volatile; safe to use before restoring

    # TODO - AWF - don't want to push things on the stack, that changes the
    # relative location of the passed-in arguments
    # However, we could just use the stack to save all the registers, and use
    # a frame pointer to give access to the arguments

    save_gp = [r for r in self._used_registers[GPRegister] if r in gp_save]

    if len(save_gp) > 0:    
      # Save GP registers
      self._saved_gp_registers = array.array('I', xrange(len(save_gp)))

      self._load_word(self._prologue, r_addr,
          self._saved_gp_registers.buffer_info()[0])

      for i, reg in enumerate(save_gp):
        self._prologue.append(
            ppc.stw(reg, r_addr, i * WORD_SIZE, ignore_active = True))


    save_fp = [r for r in self._used_registers[FPRegister] if r in fp_save]

    if len(save_fp) > 0:
      # Save FP registers
      self._saved_fp_registers = array.array('d', xrange(len(save_fp)))

      self._load_word(self._prologue, r_addr,
          self._saved_fp_registers.buffer_info()[0])

      for i, reg in enumerate(save_fp):
        self._prologue.append(
            ppc.stfd(reg, r_addr, i * WORD_SIZE * 2, ignore_active = True))


    if self._enable_vmx:
      save_vx = [r for r in self._used_registers[VMXRegister] if r in vx_save]

      if len(save_vx) > 0:
        # Save VMX registers
        self._saved_vx_registers = extarray.extarray('I', range(len(save_vx)*4))

        self._load_word(self._prologue, r_addr,
            self._saved_vx_registers.buffer_info()[0])

        for i, reg in enumerate(save_vx):
          self._load_word(self._prologue, r_idx, i * WORD_SIZE * 4)
          self._prologue.append(
              vmx.stvx(reg, r_idx, r_addr, ignore_active = True))

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

    r_addr = GPRegister(13) # Only available volatile register
    r_idx = GPRegister(14)  # Non-volatile; safe to use before restoring

    # Get the list of saved registers
    save_gp = [r for r in self._used_registers[GPRegister] if r in gp_save]
    save_fp = [r for r in self._used_registers[FPRegister] if r in fp_save]

    if self._enable_vmx:
      save_vx = [r for r in self._used_registers[VMXRegister] if r in vx_save]

      if len(save_vx) > 0:
        # Restore vrsave
        self._epilogue.append(ppc.mtvrsave(self._vrsave, ignore_active = True))

        # Restore saved VMX registers
        self._load_word(self._epilogue, r_addr,
            self._saved_vx_registers.buffer_info()[0])

        for i, reg in enumerate(save_vx):
          self._load_word(self._epilogue, r_idx, i * WORD_SIZE * 4)
          self._epilogue.append(
              vmx.lvx(reg, r_idx, r_addr, ignore_active = True))

    if len(save_fp) > 0:
      # Restore saved FP registers
      self._load_word(self._epilogue, r_addr,
          self._saved_fp_registers.buffer_info()[0])

      for i, reg in enumerate(save_fp):
        self._epilogue.append(
            ppc.lfd(reg, r_addr, i * WORD_SIZE * 2, ignore_active = True))

    if len(save_gp) > 0:
      # Restore saved GP registers
      self._load_word(self._epilogue, r_addr,
          self._saved_gp_registers.buffer_info()[0])

      for i, reg in enumerate(save_gp):
        self._epilogue.append(
            ppc.lwz(reg, r_addr, i * WORD_SIZE, ignore_active = True))

    self._epilogue.append(ppc.blr(ignore_active = True))
    return


class Processor(spe.Processor):
  exec_module = ppc_exec
  

# ------------------------------------------------------------
# Unit tests
# ------------------------------------------------------------

def TestInt():
  prgm = Program()
  code = prgm.get_stream()
  prgm += code
  proc = Processor()

  code.add(ppc.addi(prgm.gp_return, 0, 12))

  r = proc.execute(prgm)
  assert(r == 12)
  print 'int result:', r
  return

def TestFloat():
  prgm = Program()
  code = prgm.get_stream()
  prgm += code
  proc = Processor()
  a = array.array('d', [3.14])

  load_word(code, prgm.gp_return, a.buffer_info()[0])
  code.add(ppc.lfd(prgm.fp_return, prgm.gp_return, 0))

  r = proc.execute(prgm, mode='fp')
  assert(r == 3.14)
  print 'float result:', r
  return


if __name__ == '__main__':
  TestInt()
  TestFloat()
