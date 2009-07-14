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

# Field types for Cell SPU

from corepy.spre.spe import Register, InstructionOperand, Variable, Label

bit_mask = [
  0x0, 0x1, 0x3, 0x7, 0xF, 0x1F, 0x3F, 0x7F, 0xFF,
  0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF, 0x3FFF, 0x7FFF, 0xFFFF,
  0x1FFFF, 0x3FFFF, 0x7FFFF, 0xFFFFF, 0x1FFFFF, 0x3FFFFF, 0x7FFFFF, 0xFFFFFF,
  0x1FFFFFF, 0x3FFFFFF, 0x7FFFFFF, 0xFFFFFFF, 0x1FFFFFFF, 0x3FFFFFFF, 0x7FFFFFFF]


class SPUField(InstructionOperand):
  def __init__(self, name, position, default = None):
    if type(position) == int:
      position = (position, position)
      
    self.shift = 31 - position[1]
    self.width = position[1] - position[0] + 1
    self.position = position

    self.bit_mask = bit_mask[self.width]
    InstructionOperand.__init__(self, name, default)
    return

  def render(self, value):
    return (long(value) & self.bit_mask)  << self.shift

  #def __eq__(self, other):
  #  return type(self) == type(other)


class RegisterField(SPUField):
  # TODO - AWF - really still want ints as registers?
  def check(self, value):
    if isinstance(value, (Register, Variable)):
      return True
    elif isinstance(value, int):
      return value >= 0 and value < 128 # Number of registers
    return False

  def render(self, value):
    if isinstance(value, Register):
      return (long(value.reg) & self.bit_mask) << self.shift
    elif isinstance(value, Variable):
      return (long(value.reg.reg) & self.bit_mask) << self.shift
    else:
      return (long(value) & self.bit_mask) << self.shift

  #def __eq__(self, other):
  #  return isinstance(other, (RegisterField, Immediate7))


class ImmediateField(SPUField):
  def __init__(self, name, position, range, default = None):
    self.range = range
    SPUField.__init__(self, name, position, default = default)
    return

  def check(self, value):
    return isinstance(value, (int, long)) and self.range[0] <= value and value < self.range[1]

  def fits(self, value):
    return isinstance(value, (int, long)) and (self.range[0] <= value and value < self.range[1])


class LabelField(SPUField):
  def __init__(self, name, range):
    self.range = range
    SPUField.__init__(self, name, 0)
    return

  def check(self, value):
    return isinstance(value, Label)
  

# Special split field (relative offset) for hbr instruction
class ROField(InstructionOperand):
  def __init__(self, positions, range):
    self.range = range
    self.shifts = (31 - positions[0][1], 31 - positions[1][1])
    self.widths = (positions[0][1] - positions[0][0] + 1, positions[1][1] - positions[1][0] + 1)
    self.positions = positions
    self.bit_masks = (bit_mask[self.widths[0]], bit_mask[self.widths[1]])

    InstructionOperand.__init__(self, "RO", None)
    return

  def render(self, value):
    return ((long(value) >> self.widths[1] & self.bit_masks[0]) << self.shifts[0]) | ((long(value) & self.bit_masks[1]) << self.shifts[1])

  def check(self, value):
    return isinstance(value, (int, long)) and self.range[0] <= value and value < self.range[1]


#class Immediate7(ImmediateField): pass
#  def __eq__(self, other):
#    return isinstance(other, (Immediate7, ROField, Immediate8, Immediate10, Immediate16, Immediate18, RegisterField))


#class Immediate8(ImmediateField): pass
#  def __eq__(self, other):
#    return isinstance(other, (ROField, Immediate8, Immediate10, Immediate16, Immediate18, RegisterField))


#class Immediate10(ImmediateField): pass
#  def __eq__(self, other):
#    return isinstance(other, (Immediate10, Immediate16, Immediate18, RegisterField))


#class Immediate16(ImmediateField): pass
#  def __eq__(self, other):
#    return isinstance(other, (Immediate16, Immediate18, RegisterField))


#class Immediate18(ImmediateField): pass


# TODO - AWF - are the bit ranges right?
# Opcode fields with varying bit widths
OPCD16 = SPUField("OPCD16", (0,15))
OPCD11 = SPUField("OPCD11", (0,10))
OPCD10 = SPUField("OPCD10", (0,9))
OPCD9 = SPUField("OPCD9", (0,8))
OPCD8 = SPUField("OPCD8", (0,7))
OPCD7 = SPUField("OPCD7", (0,6))
OPCD4 = SPUField("OPCD4", (0,3))

# Standard/common register fields
A = RegisterField("A", (18,24))
B = RegisterField("B", (11,17))
C = RegisterField("C", (25,31))
T3 = RegisterField("T3", (25,31))
T4 = RegisterField("T4", (4,10))

# Special fields
SA = RegisterField("SA", (18,24), 0)            # For mtspr/mfspr
STOP_SIG = ImmediateField("STOP_SIG", (18, 31), (-8192, 16384))  # Stop signal
RO = ROField(((16,17), (25,31)), (-512,512))    # Relative offset
ROA = ROField(((7,8), (25,31)), (-512,512))     # Relative offset a-form

# Feature flags
D = ImmediateField("D", (12,12), (0, 2), 0)     # Interrupt Disable
E = ImmediateField("E", (13,13), (0, 2), 0)     # Interrupt Enable
P = ImmediateField("P", (11,11), (0, 2), 0)     # Prefetch
CF = ImmediateField("CF", (11,11), (0, 2), 0)   # Channel synchronization

# Immediate operand fields
# TODO - AWF - some instructions have tighter immediate ranges
I7 = ImmediateField("I7", (11,17), (-64, 128))
I8 = ImmediateField("I8", (10,17), (0, 256))
I10 = ImmediateField("I10", (8, 17), (-512, 1024))
I16 = ImmediateField("I16", (9, 24), (-32768, 65536))
I18 = ImmediateField("I18", (7, 24), (-131072, 262144))

# Label operand fields
LBL9 = LabelField("LBL9", (-512, 512))
LBL16 = LabelField("LBL16", (-32768, 32768))

