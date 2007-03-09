# Copyright 2006-2007 The Trustees of Indiana University.

# This software is available for evaluation purposes only.  It may not be
# redistirubted or used for any other purposes without express written
# permission from the authors.

# Authors:
#   Christopher Mueller (chemuell@cs.indiana.edu)
#   Andrew Lumsdaine    (lums@cs.indiana.edu)


__doc__="""
An implementation of InstructionStream that conforms to the OS X
ABI on PowerPC/AltiVec processors - G4/G5 or IBM PPC 7400/7410/970. 
"""

import array
import sys

import corepy.spre.spe as spe
import ppc_exec


import corepy.arch.ppc.isa as ppc
import corepy.arch.vmx.isa as vmx
from   corepy.arch.ppc.lib.util import load_word

ExecParams = ppc_exec.ExecParams


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
# fp_return = FPRegister(1)
# vx_return = VMXRegister(1)
# gp_return = GPRegister(3)

# Callee save registers
gp_save = [GPRegister(i, None) for i in range(14, 32)]
fp_save = [FPRegister(i, None) for i in range(14, 32)]
vx_save = [VMXRegister(i, None) for i in range(20, 32)]


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
  This implementation of InstructionStream conforms to the Mac OS X
  ABI for  32-bit PowerPC processors.
  
  An InstructionStream is the main abstraction for a sequence of
  instructions and the processor resources it uses.  The user can add
  arbitrary instructions to a stream using the add() method and
  execute then with the Processor.execute() method.  Instructions are
  executed starting with the first instruction added by the user.  An 
  instruction stream can return an integer or floating point value by
  placing the result in gp_return or fp_return and calling execute()
  with the appropriate return mode.  Any other values passed between
  the calling environment and the InstructionStream should be passed
  through memory.

  InstructionStream manages register allocation and also tracks heap
  storage to by its instructions.  Registers are 'allocated' to the
  user via requests to acquire_register(). When the user is done with
  the register, it must be released using release_register(). If not,
  it is unavailable for future use.  Advanced register allocation is
  left to the user.  If all available registers have been acquired, an
  exception is thrown.  Note that the return registers are included in
  the collection of available registers.

  If instructions use heap allocated memory (e.g., an array() to cache
  values during execution), they can pass a reference to
  InstructionStream using add_storage().  This assures that the memory
  will not be garbage collected until the InstructionStream has been
  cleared.  (note that this is an alternative to using the stack for
  temporary variables)

  Internally, the instruction stream is broken into three sections: a
  prologue, the code, and an epilogue.  The code section contains the
  user supplied instructions.  The prologue and epilogue manage
  register saves and any other ABI considerations. They are stored in
  separate memory locations and called immediately before (prologue)
  and after (epilogue) the user code.
  """

  # Class attributes

  # Register file descriptor: ('file id', register class, valid values)
  # These are used during instanciation to create register files for 
  # the InstructionStream instance.
  RegisterFiles = (('gp', GPRegister, range(2,10) + range(14, 31)),
                   ('fp', FPRegister, range(1,32)),
                   ('vector', VMXRegister, range(0,32)))

  default_register_type = GPRegister
  exec_module   = ppc_exec
  align         = 4
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
    r_addr = 2 # use r2
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
      #print 'saving vx:', reg, r_addr, i * WORD_SIZE
      self._prologue.add(vmx.stvx(reg, i * WORD_SIZE * 4, r_addr))
      # print 'TODO: VMX Support'
      
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
    self._prologue.add(ppc.addi(r_addr, 0, vx_bits))
    self._prologue.add(ppc.mtvrsave(r_addr))    

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
      # print 'TODO: VMX Support'
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

    # On the G4/G5, bx seems to not append the two 00 bits as specified
    # in the PEM.  Instead, just use the whole address.
    if (addr & 0xFFFFFF) == addr:
      self.add(ppc.ba(addr))
    else:
      r_addr = reg
      load_word(self, r_addr, addr)
      self.add(ppc.mtctr(r_addr))
      self.add(ppc.bcctrx(0x14, 0))

    return
      
class Processor(spe.Processor):
  exec_module = ppc_exec
  

# ------------------------------------------------------------
# Unit tests
# ------------------------------------------------------------

def TestInt():
  code = InstructionStream()
  proc = Processor()

  code.add(ppc.addi(code.gp_return, 0, 12))

  r = proc.execute(code, debug=True)
  assert(r == 12)
  print 'int result:', r
  return

def TestFloat():
  code = InstructionStream()
  proc = Processor()
  a = array.array('d', [3.14])

  load_word(code, gp_return, a.buffer_info()[0])
  code.add(ppc.lfd(code.fp_return, code.gp_return, 0))

  r = proc.execute(code, mode='fp', debug=True)
  assert(r == 3.14)
  print 'float result:', r
  return

def TestExtended():

  class Add10(spe.ExtendedInstruction):
    isa_module = ppc
    def __init__(self, d, value):
      self.d = d
      self.value = value

      spe.ExtendedInstruction.__init__(self)
      
      return

    def block(self):
      for i in range(10):
        ppc.addi(self.d, self.d, self.value)
      return
  
  code = InstructionStream()
  proc = Processor()

  # Using code.add 
  code.add(ppc.addi(code.gp_return, 0, 0))
  code.add(Add10(code.gp_return, 1))

  Add10.ex(1).eval(code, reg = code.gp_return)
  
  code.print_code()
  r = proc.execute(code)
  print r
  assert(r == 20)

  # Using active code
  code.reset()
  ppc.set_active_code(code)

  ppc.addi(code.gp_return, 0, 0)  
  Add10(code.gp_return, 1)

  Add10.ex(1).eval(ppc.get_active_code(), reg = code.gp_return)
  
  code.print_code()
  r = proc.execute(code)
  print r
  assert(r == 20)

  
  return


def TestCodedCall():
  code = InstructionStream()
  proc = Processor()
  
  a = array.array('d', [3.14])

  load_word(code, code.gp_return, a.buffer_info()[0])

  ppc.set_active_code(code)
  ppc.lfd(code.fp_return, code.gp_return, 0)
  code.print_code()
  r = proc.execute(code, mode='fp', debug=True)
  assert(r == 3.14)
  print 'float result:', r
  return


if __name__ == '__main__':
  # TestInt()
  # TestFloat()
  # TestCodedCall()
  TestExtended()
