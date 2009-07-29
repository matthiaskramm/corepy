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


# ------------------------------
# Registers
# ------------------------------

gp8 =  {"al": (0, 0),   "bl": (3, 0),   "cl": (1, 0),   "dl": (2, 0),
        "ah": (4, 0),   "bh": (7, 0),   "ch": (5, 0),   "dh": (6, 0),
        "spl": (4, 0),  "bpl": (5, 0),  "sil": (6, 0),  "dil": (7, 0),
        "r8b": (0, 1),  "r9b": (1, 1),  "r10b": (2, 1), "r11b": (3, 1),
        "r12b": (4, 1), "r13b": (5, 1), "r14b": (6, 1), "r15b": (7, 1)}
gp16 = {"ax": (0, 0),   "bx": (3, 0),   "cx": (1, 0),   "dx": (2, 0),
        "sp": (4, 0),   "bp": (5, 0),   "si": (6, 0),   "di": (7, 0),
        "r8w": (0, 1),  "r9w": (1, 1),  "r10w": (2, 1), "r11w": (3, 1),
        "r12w": (4, 1), "r13w": (5, 1), "r14w": (6, 1), "r15w": (7, 1)}
gp32 = {"eax": (0, 0),  "ebx": (3, 0),  "ecx": (1, 0),  "edx": (2, 0),
        "esp": (4, 0),  "ebp": (5, 0),  "esi": (6, 0),  "edi": (7, 0),
        "r8d": (0, 1),  "r9d": (1, 1),  "r10d": (2, 1), "r11d": (3, 1),
        "r12d": (4, 1), "r13d": (5, 1), "r14d": (6, 1), "r15d": (7, 1)}
gp64 = {"rax": (0, 0),  "rbx": (3, 0),  "rcx": (1, 0),  "rdx": (2, 0),
        "rsp": (4, 0),  "rbp": (5, 0),  "rsi": (6, 0),  "rdi": (7, 0),
        "r8":  (0, 1),   "r9": (1, 1),  "r10": (2, 1),  "r11": (3, 1),
        "r12": (4, 1),  "r13": (5, 1),  "r14": (6, 1),  "r15": (7, 1)}


class x86_64Register(spe.Register):
  def __init__(self, name):
    try:
      (self.reg, self.rex) = self._reg_dict[name]
    except KeyError:
      raise Exception("Invalid register name %s" % str(name))

    spe.Register.__init__(self, name)
    return

  def __eq__(self, other):
    if isinstance(other, str):
      return self.name == other
    return type(self) == type(other) and self.reg == other.reg and self.rex == other.rex and self.name == other.name


class GPRegister8(x86_64Register):
  _complex_reg = True
  _reg_dict = gp8
class GPRegister16(x86_64Register):
  _complex_reg = True
  _reg_dict = gp16
class GPRegister32(x86_64Register):
  _complex_reg = True
  _reg_dict = gp32
class GPRegister64(x86_64Register):
  _complex_reg = True
  _reg_dict = gp64

class FPRegister(x86_64Register):
  def __init__(self, name):
    if not isinstance(name, str) or name[0:2] != "st":
      raise Exception("Invalid register name %s" % str(name))

    try:
      self.reg = int(name[2:])
    except ValueError:
      raise Exception("Invalid register name %s" % str(name))

    if self.reg < 0 or self.reg >= 8:
      raise Exception("Invalid register name %s" % str(name))

    self.rex = None
    spe.Register.__init__(self, name)
    return

class MMXRegister(x86_64Register):
  def __init__(self, name):
    if not isinstance(name, str) or name[0:2] != "mm":
      raise Exception("Invalid register name %s" % str(name))

    try:
      self.reg = int(name[2:])
    except ValueError:
      raise Exception("Invalid register name %s" % str(name))

    if self.reg < 0 or self.reg >= 8:
      raise Exception("Invalid register name %s" % str(name))

    self.rex = None
    spe.Register.__init__(self, name)
    return

class XMMRegister(x86_64Register):
  def __init__(self, name):
    if not isinstance(name, str) or name[0:3] != "xmm":
      raise Exception("Invalid register name %s" % str(name))

    try:
      self.reg = int(name[3:])
    except ValueError:
      raise Exception("Invalid register name %s" % str(name))

    if self.reg < 0 or self.reg >= 16:
      raise Exception("Invalid register name %s" % str(name))

    if self.reg >= 8:
      self.reg -= 8
      self.rex = 1
    else:
      self.rex = 0
    spe.Register.__init__(self, name)
    return

class IPRegister(x86_64Register):
  def __init__(self, name):
    if "rip" == name:
      self.rex = 1
    elif "eip" == name:
      self.rex = 0
    else:
      raise Exception("Invalid register name %s" % str(name))

    self.reg = 8
    spe.Register.__init__(self, name)
    return


GPRegisterType = (GPRegister8, GPRegister16, GPRegister32, GPRegister64)

# Set up an instance for each register
# TODO - ah-dh registers are accessible only w/o a REX,
# and sil/dil/bpl/spl are accessible only WITH a REX.  How do I enforce this?
# encoding wise, a REX value of 0x40 is needed for si/dil/bpl.

# Use a bool for indicating whether an REX prefix is needed.  This doesn't mean
# there are bits in the REX to set, just that one is needed.  Render methods
# can then conditionally add the REX prefix based on this boolean.  Registers
# can use their 4-bit values, but still in the comparison checks for gp8 the
# REX boolean will need to be considered.
# Is there a way to do this without the REX prefix?
# Slightly cleaner might be to use 0/1/None, with the same idea above.
# 0 or 1 indicates the value of the bit to go in the REX prefix.  This
# simplifies the bitwise logic to create the REX prefix, no need to extract the
# bit from the register value.  Would then need to always compare the REX bit
# when doing register comparisons.

# All the GP and XMM regs need to have a rex field, FP and MMX do not.

#gp8 =  ((0, 0, "al"),   (3, 0, "bl"),   (1, 0, "cl"),   (2, 0, "dl"),
#        (4, 0, "ah"),   (7, 0, "bh"),   (5, 0, "ch"),   (6, 0, "dh"),
#        (6, 0, "sil"),  (7, 0, "dil"),  (5, 0, "bpl"),  (4, 0, "spl"),
#        (0, 1, "r8b"),  (1, 1, "r9b"),  (2, 1, "r10b"), (3, 1, "r11b"),
#        (4, 1, "r12b"), (5, 1, "r13b"), (6, 1, "r14b"), (7, 1, "r15b"))
#gp16 = ((0, 0, "ax"),   (3, 0, "bx"),   (1, 0, "cx"),   (2, 0, "dx"),
#        (4, 0, "sp"),   (5, 0, "bp"),   (6, 0, "si"),   (7, 0, "di"),
#        (0, 1, "r8w"),  (1, 1, "r9w"),  (2, 1, "r10w"), (3, 1, "r11w"),
#        (4, 1, "r12w"), (5, 1, "r13w"), (6, 1, "r14w"), (7, 1, "r15w"))
#gp32 = ((0, 0, "eax"),  (3, 0, "ebx"),  (1, 0, "ecx"),  (2, 0, "edx"),
#        (4, 0, "esp"),  (5, 0, "ebp"),  (6, 0, "esi"),  (7, 0, "edi"),
#        (0, 1, "r8d"),  (1, 1, "r9d"),  (2, 1, "r10d"), (3, 1, "r11d"),
#        (4, 1, "r12d"), (5, 1, "r13d"), (6, 1, "r14d"), (7, 1, "r15d"))
#gp64 = ((0, 0, "rax"),  (3, 0, "rbx"),  (1, 0, "rcx"),  (2, 0, "rdx"),
#        (4, 0, "rsp"),  (5, 0, "rbp"),  (6, 0, "rsi"),  (7, 0, "rdi"),
#        (0, 1, "r8"),   (1, 1, "r9"),   (2, 1, "r10"),  (3, 1, "r11"),
#        (4, 1, "r12"),  (5, 1, "r13"),  (6, 1, "r14"),  (7, 1, "r15"))

gp8_array = []
gp16_array = []
gp32_array = []
gp64_array = []
st_array = []
mm_array = []
xmm_array = []


# Set up RIP register.  This register is only useable with a displacement in a
# memory reference, and nowhere else.
#globals()["rip"] = IPRegister(8, 1, "rip")
#globals()["eip"] = IPRegister(8, 0, "eip")
globals()["rip"] = IPRegister("rip")
globals()["eip"] = IPRegister("eip")

# Set up GP registers
for (regs, cls, arr) in ((gp8, GPRegister8, gp8_array), (gp16, GPRegister16, gp16_array), (gp32, GPRegister32, gp32_array), (gp64, GPRegister64, gp64_array)):
  for name in regs.keys():
    globals()[name] = cls(name)
    arr.append(globals()[name])

# Set up x87, MMX, and SSE registers
for i in range(0, 8):
  stri = str(i)

  name = "st" + stri
  globals()[name] = FPRegister(name)
  st_array.append(globals()[name])

  name = "mm" + stri
  globals()[name] = MMXRegister(name)
  mm_array.append(globals()[name])

  name = "xmm" + stri
  globals()[name] = XMMRegister(name)
  xmm_array.append(globals()[name])

  # Set up 8 more SSE registers, with REX = 1
  name = "xmm" + str(i + 8)
  globals()[name] = XMMRegister(name)
  xmm_array.append(globals()[name])
  
