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
import corepy.arch.x86_64.platform as env
import corepy.arch.x86_64.isa as x86
import corepy.arch.x86_64.lib.memory as memory
import corepy.arch.x86_64.types.registers as regs

# I need to override a variety of things here.
# First, Program objects need to be subclassed, with my own acquire/release
# register methods.
# Second, an ISA needs to be implementated that accepts synthetic registers
# as operands and does the right type checking.
#  This work will probably require subclassing Instruction and
#  DispatchInstruction.


#
# Synthetic Register
#

class SynRegister(spe.Register):
  reg_type = None
  def __init__(self, name):
    # No physical register assigned yet
    self.reg = None
    self.rex = None

    spe.Register.__init__(self, name)
    return


class SynGPRegister32(SynRegister):
  reg_type = regs.GPRegister32
class SynGPRegister64(SynRegister):
  reg_type = regs.GPRegister64
class SynXMMRegister(SynRegister):
  reg_type = regs.XMMRegister


#
# Synthetic Instruction
#

class SynRegisterOperand(spe.InstructionOperand):
  def __init__(self, reg_type):
    self.reg_type = reg_type
    return

  def check(self, value):
    return isinstance(value, self.reg_type)

  
class SynDispatchInstruction(spe.DispatchInstruction):
  syn_dispatch = ()

  def __init__(self, *operands, **koperands):
    self.syn_signature = None

    for signature in self.syn_dispatch:
      # zip through the operands and the signature
      match = True
      for op, sig_op in zip(operands, signature):
        #print "op", op, sig_op
        #print "check", sig_op.check(op)
        if not sig_op.check(op):
          match = False
          break

      if match:
        self.syn_signature = signature
        break

    if self.syn_signature is None:
      raise TypeError("Instruction %s does not support operands (%s)" % (
        type(self), ', '.join([str(op) for op in operands],)))


    self.syn_operands = operands
    self.syn_koperands = koperands

    # If active code is present, add ourself to it and remember that
    # we did so.  active_code_used is checked by InstructionStream
    # to avoid double adds from code.add(inst(...)) when active_code
    # is set.
    self.active_code_used = None    

    # Allow the user to create an instruction without adding it to active code.
    ignore_active = False
    if koperands.has_key('ignore_active'):
      ignore_active = koperands['ignore_active']
      del koperands['ignore_active']

    if self.active_code is not None and not ignore_active:
      self.active_code.add(self)
      self.active_code_used = self.active_code

    return


sgpreg32_t = SynRegisterOperand(SynGPRegister32)
sgpreg64_t = SynRegisterOperand(SynGPRegister64)
sxmmreg_t = SynRegisterOperand(SynXMMRegister)


#
# Program
#

class Program(env.Program):
  default_register_type = SynGPRegister64

  def __init__(self):
    self.__sr_count = 0
    env.Program.__init__(self)
    return

  def acquire_register(self, reg_type = None):
    if reg_type is None:
      reg_type = self.default_register_type

    name = "sr" + str(self.__sr_count)
    self.__sr_count += 1
    return reg_type(name)

  def release_register(self, reg):
    return


  # TODO - need to overload cache_code and do a pass to assign phys regs.
  def cache_code(self):
    # Go through the instructions and replace any syn regs with physical regs
    #  Do something dumb initially, but we want to do real register allocation
    #  Deal with:
    #   Running out of registers
    #   Certain instructions needing specific registers (div)

    regs = {}

    for stream in self.objects:
      for obj in stream:
        if isinstance(obj, SynDispatchInstruction):
          operands = []
          for op in obj.syn_operands:
            if isinstance(op, SynRegister):
              # Need to replace the synregs with real regs.
              if not regs.has_key(op):
                # Need a new physical register
                regs[op] = env.Program.acquire_register(self, reg_type = op.reg_type)
                print "assigned reg", op, regs[op]
              operands.append(regs[op])
            else:
              operands.append(op)

          #spe.DispatchInstruction.__init__(obj, operands)
          obj.inst.__init__(obj, *operands)

    print "CALLING SPE CACHE CODE"
    return env.Program.cache_code(self)


#
# ISA Definition
#

class add(SynDispatchInstruction, x86.add):
  inst = x86.add
  syn_dispatch = (
    (sgpreg64_t, x86.imm32_t),
    (sgpreg64_t, sgpreg64_t),
    )
 
class mov(SynDispatchInstruction, x86.mov):
  inst = x86.mov
  syn_dispatch = (
    (sgpreg64_t, x86.imm32_t),
    (sgpreg64_t, sgpreg64_t),
    )

