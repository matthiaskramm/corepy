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

# Field types for PowerPC

from corepy.spre.spe import Register, InstructionOperand, Variable, Label

bit_mask = [
  0x0, 0x1, 0x3, 0x7, 0xF, 0x1F, 0x3F, 0x7F, 0xFF,
  0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF, 0x3FFF, 0x7FFF, 0xFFFF,
  0x1FFFF, 0x3FFFF, 0x7FFFF, 0xFFFFF, 0x1FFFFF, 0x3FFFFF, 0x7FFFFF, 0xFFFFFF,
  0x1FFFFFF, 0x3FFFFFF, 0x7FFFFFF, 0xFFFFFFF, 0x1FFFFFFF, 0x3FFFFFFF, 0x7FFFFFFF]


class PPCField(InstructionOperand):
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


# TODO: Separate these out by type a little more like the x86 versions
class ConstantField(PPCField):
  def __init__(self, name, position, const, default = None):
    self.const = const
    PPCField.__init__(self, name, position, default = default)
    return

  def check(self, value):
    return True

  def render(self, value):
    return (long(self.const) & self.bit_mask)  << self.shift


class RegisterField(PPCField):
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


class SplitField(PPCField):
  def check(self, value):
    return isinstance(value, (int, long)) and value >= 0 and value < (1 << 10)

  def render(self, value):  
    return (((long(value) & 0x3E0) >> 5) | ((long(value) & 0x1F) << 5)) << self.shift


class ImmediateField(PPCField):
  def __init__(self, name, position, range, default = None):
    self.range = range
    PPCField.__init__(self, name, position, default = default)

  def check(self, value):
    return isinstance(value, (int, long)) and self.range[0] <= value and value < self.range[1]


class TruncatedField(ImmediateField):
  """
  Remove bits from the right side of the field using a right-shift before masking and rendering.
  """
  def __init__(self, name, position, truncate, default = None):
    self.truncate = truncate
    r = 1 << (position[1] - position[0] + truncate)
    ImmediateField.__init__(self, name, position,  (-r, r << 1), default = default)
    return

  def render(self, value):
    return ((long(value) >> self.truncate)  & self.bit_mask)  << self.shift


class LabelField(TruncatedField):
  def check(self, value):
    return isinstance(value, Label)


# TODO - AWF - huh???
OPCD = PPCField("A", (11,15))
OPCD = PPCField(OPCD, (0, 5))
#XO_1 = PPCField("XO_1", (21,30))
#XO_2 = PPCField("XO_2", (22,30))
#XO_3 = PPCField("XO_3", (26,30))

A = RegisterField("A", (11,15))
B = RegisterField("B", (16,20))
C = RegisterField("C", (21,25))
D = RegisterField("D", (6,10))
S = RegisterField("S", (6,10))

# Other fields
AA   = ImmediateField("AA",  (30), (0,2), 0)
BD   = TruncatedField("BD",  (16,29), 2)
BDLBL = LabelField("BDLBL",  (16,29), 2)
BI   = ImmediateField("BI",  (11,15), (0, 32))
BO   = ImmediateField("BO",  (6,10), (0, 32))
crbA = RegisterField("crbA", (11,15))
crbB = RegisterField("crbB", (16,20))
crbD = RegisterField("crbD", (6,10))
crfD = RegisterField("crfD", (6,8))
crfS = RegisterField("crfS", (11,13))
CRM  = ImmediateField("CRM", (12,19), (0, 256))
d    = ImmediateField("d",(16,31), (-32768, 65536))
FM   = ImmediateField("FM",  (7,14), (0, 256))
frA  = RegisterField("frA",  (11,15))
frB  = RegisterField("frB",  (16,20))
frC  = RegisterField("frC",  (21,25))
frD  = RegisterField("frD",  (6,10))
frS  = RegisterField("frS",  (6,10))
IMM  = ImmediateField("IMM", (16,19), (0, 16))
L    = ConstantField("L",    (10), 1)
LI   = TruncatedField("LI",  (6,29), 2)
LILBL = LabelField("LILBL",  (6,29), 2)
LK   = ImmediateField("LK",  (31), (0, 2), 0)
MB   = ImmediateField("MB",  (21,25), (0, 32))
ME   = ImmediateField("ME",  (26,30), (0, 32))
NB   = ImmediateField("NB",  (16,20), (0, 32))
OE   = ImmediateField("OE",  (21), (0, 2), 0)
#CD   = PPCField("CD",   (0,5)) # TODO - AWF - what is this?
rA   = RegisterField("rA",   (11,15))
rB   = RegisterField("rB",   (16,20))
Rc   = RegisterField("Rc",   (31), 0)
rD   = RegisterField("rD",   (6,10))
rS   = RegisterField("rS",   (6,10))
SH   = ImmediateField("SH",  (16,20), (0, 32)) # is this signed?
SIMM = ImmediateField("SIMM",(16,31), (-32768, 65536))
spr  = SplitField("spr",     (11,20))  # split field 
SR   = ImmediateField("SR",  (12,15), (0, 16))
tbr  = ImmediateField("tbr", (11,20), (268,270))       # Note: this may need specialization AWF - needs testing too
TO   = ImmediateField("TO",  (6,10), (0, 32))  
UIMM = ImmediateField("UIMM",(16,31), (0, 65536))
XO_1 = ImmediateField("XO_1",(21,30), (0, 1024))
XO_2 = ImmediateField("XO_2",(22,30), (0, 512))
XO_3 = ImmediateField("XO_3",(26,30), (0, 32))
SC_ONE = ConstantField("SC_ONE", (30), 1)  # Custom field for the '1' bit that's set in sc
STWCX_ONE = ConstantField("STWCX_ONE", (31), 1)  # Custom field for the '1' bit that's set in stwxc.


  
  
