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

from corepy.spre.spe import InstructionOperand, Label
from corepy.arch.x86_64.lib.memory import MemoryReference
#import corepy.arch.x86_64.types.registers as regs

# ------------------------------
# x86 Operands
# ------------------------------

class x86InstructionOperand(InstructionOperand):
  def __eq__(self, other):
    return type(self) == type(other)


class FixedRegisterOperand(x86InstructionOperand):
  def __init__(self, name, gp_equiv, gp_reg):
    self.gp_equiv = gp_equiv
    self.gp_reg = gp_reg
    x86InstructionOperand.__init__(self, name)
    return

  def __eq__(self, other):
    # Fixed registers (ie 'eax') are equivalent to themselves, but are also
    # equivalent to x86RegisterOperands of the same gp_equiv type.
    if isinstance(other, FixedRegisterOperand):
      return self.name == other.name and self.gp_equiv == other.gp_equiv
    elif isinstance(other, x86RegisterOperand):
      return self.gp_equiv == other.gp_equiv
    return False

  def check(self, value):
    return isinstance(value, self.gp_equiv) and self.name == value.name

  def render(value):
    raise Exception('Fixed register (e.g., ax, al, eax) operands cannot be rendered')


class x86ConstantOperand(x86InstructionOperand):
  def __init__(self, name, const):
    self.const = const
    x86InstructionOperand.__init__(self, name)
    return

  def check(self, value):
    return type(self.const) == type(value) and self.const == value
  
  def __eq__(self, other):
    return (type(self) == type(other) and self.const == other.const) or (isinstance(other, x86ImmediateOperand) and other.fits(self.const))


class x86ImmediateOperand(x86InstructionOperand):
  def __init__(self, name, range):
    self.range = range
    x86InstructionOperand.__init__(self, name)
    return

  # This so instances with different names can be created, i.e. reg8_t('rd')
  def __call__(self, name):
    return self.__class__(name, self.range)

  def check(self, value):
    return self.fits(value)

  def fits(self, value):
    return isinstance(value, (int, long)) and (self.range[0] <= value and value < self.range[1])

  def __eq__(self, other):
    # Carefully written so that an immediate with a larger range is equal.
    return isinstance(other, x86ImmediateOperand) and self.range[0] >= other.range[0] and self.range[1] <= other.range[1]


class x86LabelOperand(x86InstructionOperand):
  relative_op = True

  def __init__(self, name, range):
    x86InstructionOperand.__init__(self, name)
    self.range = range
    return

  def check(self, value):
    return self.fits(value)

  def fits(self, value):
    return isinstance(value, Label)


class x86RegisterOperand(x86InstructionOperand):
  def __init__(self, name, gp_equiv):
    self.gp_equiv = gp_equiv
    x86InstructionOperand.__init__(self, name)
    return

  # This so instances with different names can be created, i.e. reg8_t('rd')
  def __call__(self, name):
    return self.__class__(name, self.gp_equiv)

  def __eq__(self, other):
    return type(self) == type(other) and self.gp_equiv == other.gp_equiv

  def check(self, value):
    return isinstance(value, self.gp_equiv)


class x86MemoryOperand(x86InstructionOperand):
  def __init__(self, name, size):
    self.size = size
    x86InstructionOperand.__init__(self, name)
    return

  def __eq__(self, other):
    return type(self) == type(other) and self.size == other.size

  def check(self, value):
    # Careful to support 'mem' operand with no size
    if isinstance(value, MemoryReference):
      return self.size == value.data_size
    return False


class x86PrefixOperand(x86InstructionOperand):
  def __init__(self, name, value):
    self.value = value
    x86InstructionOperand.__init__(self, name)

  def check(self, value):
    return True


  
# Immediate values

class Rel8off(x86ImmediateOperand):
  relative_op = True

class Rel32off(x86ImmediateOperand):
  relative_op = True

class Imm8(x86ImmediateOperand): pass
class Imm16(x86ImmediateOperand): pass
class Imm32(x86ImmediateOperand): pass
class Imm64(x86ImmediateOperand): pass

