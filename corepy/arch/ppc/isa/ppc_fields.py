# Field types for PowerPC

# from isa2 import InstructionOperand
from corepy.spre.spe import Register, InstructionOperand

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
    self.postion = position

    self.bit_mask = bit_mask[self.width]
    InstructionOperand.__init__(self, name, default)
    
    return

  def render(self, value):
    return (long(value) & self.bit_mask)  << self.shift

# TODO: Separate these out by type a little more like the x86 versions
class RegisterField(PPCField):
  def check(self, value):
    return isinstance(value, (Register, int))

  def render(self, value):
    if isinstance(value, Register):
      return (long(value.reg) & self.bit_mask) << self.shift
    else:
      return (long(value) & self.bit_mask) << self.shift

OPCD = PPCField("A", (11,15))

OPCD = PPCField(OPCD, (0, 5))
XO_1 = PPCField("XO_1", (21,30))
XO_2 = PPCField("XO_2", (22,30))
XO_3 = PPCField("XO_3", (26,30))

A = RegisterField("A", (11,15))
B = RegisterField("B", (16,20))
C = RegisterField("C", (21,25))
D = RegisterField("D", (6,10))
S = RegisterField("S", (6,10))

# Other fields
AA   = PPCField("AA",   (30), 0)
BD   = PPCField("BD",   (16,29))
BI   = PPCField("BI",   (11,15))
BO   = PPCField("BO",   (6,10))
crbA = RegisterField("crbA", (11,15))
crbB = RegisterField("crbB", (16,20))
crbD = RegisterField("crbD", (6,10))
crfD = RegisterField("crfD", (6,8))
crfS = RegisterField("crfS", (11,13))
CRM  = PPCField("CRM",  (12,19))
d    = RegisterField("d",    (16,31))
FM   = PPCField("FM",   (7,14))
frA  = RegisterField("frA",  (11,15))
frB  = RegisterField("frB",  (16,20))
frC  = RegisterField("frC",  (21,25))
frD  = RegisterField("frD",  (6,10))
frS  = RegisterField("frS",  (6,10))
IMM  = PPCField("IMM",  (16,19))
L    = PPCField("L",    (10))
LI   = PPCField("LI",   (6,29))
LK   = PPCField("LK",   (31), 0)
MB   = PPCField("MB",   (21,25))
ME   = PPCField("ME",   (26,30))
NB   = PPCField("NB",   (16,20))
OE   = PPCField("OE",   (21), 0)
CD   = PPCField("CD",   (0,5))
rA   = RegisterField("rA",   (11,15))
rB   = RegisterField("rB",   (16,20))
Rc   = RegisterField("Rc",   (31), 0)
rD   = RegisterField("rD",   (6,10))
rS   = RegisterField("rS",   (6,10))
SH   = PPCField("SH",   (16,20))
SIMM = PPCField("SIMM", (16,31))
spr  = PPCField("spr",  (11,20))  # split field 
SR   = PPCField("SR",   (12,15))
tbr  = PPCField("tbr",  (11,20))       # Note: this may need specialization
TO   = PPCField("TO",   (6,10))  
UIMM = PPCField("UIMM", (16,31))
XO_1 = PPCField("XO_1", (21,30))
XO_2 = PPCField("XO_2", (22,30))
XO_3 = PPCField("XO_3", (26,30))
SC_ONE = PPCField("SC_ONE", (30))  # Custom field for the '1' bit that's set in sc
STWCX_ONE = PPCField("STWCX_ONE", (31))  # Custom field for the '1' bit that's set in stwxc.


  
  
