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

gp8 =  {"al": 0,   "bl": 3,   "cl": 1,   "dl": 2,
        "ah": 4,   "bh": 7,   "ch": 5,   "dh": 6}
gp16 = {"ax": 0,   "bx": 3,   "cx": 1,   "dx": 2,
        "sp": 4,   "bp": 5,   "si": 6,   "di": 7}
gp32 = {"eax": 0,  "ebx": 3,  "ecx": 1,  "edx": 2,
        "esp": 4,  "ebp": 5,  "esi": 6,  "edi": 7}


class x86Register(spe.Register):
  def __init__(self, name):
    try:
      self.reg = self._reg_dict[name]
    except KeyError:
      raise Exception("Invalid register name %s" % str(name))

    spe.Register.__init__(self, name)
    return

  def __eq__(self, other):
    if isinstance(other, str):
      return self.name == other
    return type(self) == type(other) and self.reg == other.reg

class GPRegister8(x86Register):
  _complex_reg = True
  _reg_dict = gp8
class GPRegister16(x86Register):
  _complex_reg = True
  _reg_dict = gp16
class GPRegister32(x86Register):
  _complex_reg = True
  _reg_dict = gp32
class FPRegister(x86Register):
  def __init__(self, name):
    if not isinstance(name, str) or name[0:2] != "st":
      raise Exception("Invalid register name %s" % str(name))

    try:
      self.reg = int(name[2:])
    except ValueError:
      raise Exception("Invalid register name %s" % str(name))

    if self.reg < 0 or self.reg >= 8:
      raise Exception("Invalid register name %s" % str(name))

    spe.Register.__init__(self, name)
    return

class MMXRegister(x86Register):
  def __init__(self, name):
    if not isinstance(name, str) or name[0:2] != "mm":
      raise Exception("Invalid register name %s" % str(name))

    try:
      self.reg = int(name[2:])
    except ValueError:
      raise Exception("Invalid register name %s" % str(name))

    if self.reg < 0 or self.reg >= 8:
      raise Exception("Invalid register name %s" % str(name))

    spe.Register.__init__(self, name)
    return

class XMMRegister(x86Register):
  def __init__(self, name):
    if not isinstance(name, str) or name[0:3] != "xmm":
      raise Exception("Invalid register name %s" % str(name))

    try:
      self.reg = int(name[3:])
    except ValueError:
      raise Exception("Invalid register name %s" % str(name))

    if self.reg < 0 or self.reg >= 8:
      raise Exception("Invalid register name %s" % str(name))

    spe.Register.__init__(self, name)
    return


GPRegisterType = (GPRegister8, GPRegister16, GPRegister32)

# Set up an instance for each register
#gp8 = ((0, "al"), (3, "bl"), (1, "cl"), (2, "dl"),
#        (4, "ah"), (7, "bh"), (5, "ch"), (6, "dh"))
#gp16 = ((0, "ax"), (3, "bx"), (1, "cx"), (2, "dx"),
#        (4, "sp"), (5, "bp"), (6, "si"), (7, "di"))
#gp32 = ((0, "eax"), (3, "ebx"), (1, "ecx"), (2, "edx"),
#        (4, "esp"), (5, "ebp"), (6, "esi"), (7, "edi"))

gp8_array = []
gp16_array = []
gp32_array = []
st_array = []
mm_array = []
xmm_array = []

# Set up GP and FP registers
for (regs, cls, arr) in ((gp8, GPRegister8, gp8_array), (gp16, GPRegister16, gp16_array), (gp32, GPRegister32, gp32_array)):
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


