# Copyright 2006 The Trustees of Indiana University.

# This software is available for evaluation purposes only.  It may not be
# redistirubted or used for any other purposes without express written
# permission from the authors.

# Author:
#   Christopher Mueller

__doc__="""
SPE for the Cell PPU.
"""

import array
import corepy.spre.spe as spe
import cell_exec

# from isa_syn import Instruction
import corepy.arch.ppc.isa as ppc
import corepy.arch.vmx.isa as vmx
from   corepy.arch.ppc.lib.util import load_word


# ------------------------------
# Registers
# ------------------------------

class GPRegister(spe.Register): pass
class FPRegister(spe.Register): pass
class VMXRegister(spe.Register): pass


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
fp_return = 1
vx_return = 1
gp_return = 3

# Callee save registers
gp_save = [GPRegister(i, None) for i in range(13, 32)]
fp_save = [FPRegister(i, None) for i in range(13, 32)]
vx_save = [VMXRegister(i, None) for i in range(20, 32)]  # !!! NOT SURE ABOUT VMX !!!

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
  RegisterFiles = (('gp', GPRegister, range(3,32)),
                   ('fp', FPRegister, range(0,32)),
                   ('vector', VMXRegister, range(0,32)))

  default_register_type = GPRegister
  exec_module   = cell_exec
  align         = 0
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

    return

  # ------------------------------
  # Execute/ABI support
  # ------------------------------

  def _synthesize_prologue(self):
    """
    Create the prologue. (see PPC ABI p41)

    This manages the register preservation requirements from the ABI.

    TODO: CR2-4 need to be preserved.
    """

    # Reset the prologue
    self._prologue = InstructionStream() # array.array(WORD_TYPE)

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

    r_addr = GPRegister(3, None) # use r2 - not on PPC Linux!!!  Use a volitile register for this, say R3
    # self._prologue.add(ppc.stwu(r_addr, r_addr, -WORD_SIZE))
    
    load_word(self._prologue, r_addr, self._saved_gp_registers.buffer_info()[0])

    for i, reg in enumerate(save_gp):
      # print 'saving gp:', reg, r_addr, i * WORD_SIZE
      self._prologue.add(ppc.stw(reg, r_addr, i * WORD_SIZE))

    load_word(self._prologue, r_addr, self._saved_fp_registers.buffer_info()[0])
    
    for i, reg in enumerate(save_fp):
      # print 'saving fp:', reg, r_addr, i * WORD_SIZE
      self._prologue.add(ppc.stfd(reg, r_addr, i * WORD_SIZE * 2))

    load_word(self._prologue, r_addr, self._saved_vx_registers.buffer_info()[0])
    
    for i, reg in enumerate(save_vx):
      # print 'saving vx:', reg, r_addr, i * WORD_SIZE
      self._prologue.add(vmx.stvx(reg, i * WORD_SIZE * 4, r_addr))

    # Restore r2
    # self._prologue.add(ppc.lwz(r_addr, 1, -4))
    
    # Set up VRSAVE
    # Currently, we save the old value of VRSAVE in r31.
    vx_bits = 0l

    # Fill in the bits for the used vector registers

    # On the G4, someone stomps on registers < 20 ... save them all for now.
    for vx in range(32): #save_vx:
      vx_bits |= (1l << vx)
      
    # Save vrsave and put our value in it
    self._prologue.add(ppc.mfvrsave(31))
    self._prologue.add(ppc.addi(5, 0, vx_bits))
    self._prologue.add(ppc.mtvrsave(5))    

    return

  def _synthesize_epilogue(self):
    """
    Save the values in some registers (see PPC ABI p41)
    """

    # Reset the epilogue
    self._epilogue = InstructionStream() # array.array(WORD_TYPE)

    # Restore vrsave
    self._epilogue.add(ppc.mtvrsave(31))

    # Get the list of saved registers
    save_gp = [reg for reg in self._register_files[GPRegister].get_used() if reg in gp_save]
    save_fp = [reg for reg in self._register_files[FPRegister].get_used() if reg in fp_save]
    save_vx = [reg for reg in self._register_files[VMXRegister].get_used() if reg in vx_save]    

    r_addr = 4
    load_word(self._epilogue, r_addr, self._saved_gp_registers.buffer_info()[0])    
    
    for i, reg in enumerate(save_gp):
      # print 'restoring gp:', reg, r_addr, i * WORD_SIZE
      self._epilogue.add(ppc.lwz(reg, r_addr, i * WORD_SIZE))

    load_word(self._epilogue, r_addr, self._saved_fp_registers.buffer_info()[0])    
    
    for i, reg in enumerate(save_fp):
      # print 'restoring fp:', reg, r_addr, i * WORD_SIZE
      self._epilogue.add(ppc.lfd(reg, r_addr, i * WORD_SIZE * 2))

    load_word(self._epilogue, r_addr, self._saved_vx_registers.buffer_info()[0])

    for i, reg in enumerate(save_vx):
      # print 'saving vx:', reg, r_addr, i * WORD_SIZE
      self._epilogue.add(vmx.lvx(reg, i * WORD_SIZE * 4, r_addr))

    return


  def add_return(self):
    """
    Add the architecture dependent code to return from a function.
    Used by cache_code to have the epilogue return.
    """
    self.add(ppc.blr())
    return

  def add_jump(self, addr, reg):
    """
    Add the architecture dependent code to jump to a new instruction.
    Used by cache_code to chain the prologue, code, and epilogue.
    """

    if (addr & 0x7FFFFF) == addr:
      self.add(ppc.ba(addr >> 2))
    else:
      r_addr = reg
      load_word(self, r_addr, addr)
      self.add(ppc.mtctr(r_addr))
      self.add(ppc.bcctrx(0x14, 0))

    return
  

class Processor(spe.Processor):
  exec_module = cell_exec
  

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
