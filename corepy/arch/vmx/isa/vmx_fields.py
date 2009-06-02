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

# Field types for VMX

from corepy.spre.spe import Register, InstructionOperand, Variable

bit_mask = [
  0x0, 0x1, 0x3, 0x7, 0xF, 0x1F, 0x3F, 0x7F, 0xFF,
  0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF, 0x3FFF, 0x7FFF, 0xFFFF,
  0x1FFFF, 0x3FFFF, 0x7FFFF, 0xFFFFF, 0x1FFFFF, 0x3FFFFF, 0x7FFFFF, 0xFFFFFF,
  0x1FFFFFF, 0x3FFFFFF, 0x7FFFFFF, 0xFFFFFFF, 0x1FFFFFFF, 0x3FFFFFFF, 0x7FFFFFFF]


class VMXField(InstructionOperand):
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


class RegisterField(VMXField):
  # TODO - AWF - really still want ints as registers?
  def check(self, value):
    if isinstance(value, (Register, Variable)):
      return True
    elif isinstance(value, int):
      return value >= 0 and value < 32
    return False

  def render(self, value):
    if isinstance(value, Register):
      return (long(value.reg) & self.bit_mask) << self.shift
    elif isinstance(value, Variable):
      return (long(value.reg.reg) & self.bit_mask) << self.shift
    else:
      return (long(value) & self.bit_mask) << self.shift


class ImmediateField(VMXField):
  def __init__(self, name, position, range, default = None):
    self.range = range
    VMXField.__init__(self, name, position, default = default)

  def check(self, value):
    return isinstance(value, (int, long)) and self.range[0] <= value and value < self.range[1]


OPCD = VMXField("OPCD", (0, 5))
V_XO = VMXField("VA_XO", (26, 31))
VX_XO = VMXField("VX_XO", (21, 31))
X_XO = VMXField("X_XO", (21, 30))
VXR_XO = VMXField("VXX_XO", (22, 31))

vA = RegisterField("vA", (11,15))
vB = RegisterField("vB", (16,20))
vC = RegisterField("vC", (21,25))
vD = RegisterField("vD", (6,10))
A = RegisterField("A", (11,15))
B = RegisterField("B", (16,20))

SH = ImmediateField("SH", (22,25), (0, 16))
UIMM = ImmediateField("UIMM", (11,15), (0, 32))
SIMM = ImmediateField("SIMM", (11,15), (-16, 16))
STRM = ImmediateField("STRM", (9, 10), (0, 4))
T = ImmediateField("T", (6, 6), (0, 2))
Rc = ImmediateField("RC", (21, 21), (0, 2))

