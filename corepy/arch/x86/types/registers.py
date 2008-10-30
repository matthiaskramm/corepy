# Copyright (c) 2006-2008 The Trustees of Indiana University.                   
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

class x86Register(spe.Register):
  def __eq__(self, other):
    return type(self) == type(other) and self.reg == other.reg

class GPRegister8(x86Register): pass
class GPRegister16(x86Register): pass
class GPRegister32(x86Register): pass
class FPRegister(x86Register): pass
class MMXRegister(x86Register): pass
class XMMRegister(x86Register): pass


# Set up an instance for each register
gp8 = ((0, "al"), (3, "bl"), (1, "cl"), (2, "dl"),
        (4, "ah"), (7, "bh"), (5, "ch"), (6, "dh"))
gp16 = ((0, "ax"), (3, "bx"), (1, "cx"), (2, "dx"),
        (4, "sp"), (5, "bp"), (6, "si"), (7, "di"))
gp32 = ((0, "eax"), (3, "ebx"), (1, "ecx"), (2, "edx"),
        (4, "esp"), (5, "ebp"), (6, "esi"), (7, "edi"))

gp8_array = []
gp16_array = []
gp32_array = []
st_array = []
mm_array = []
xmm_array = []

# Set up GP and FP registers
for (regs, cls, arr) in ((gp8, GPRegister8, gp8_array), (gp16, GPRegister16, gp16_array), (gp32, GPRegister32, gp32_array)):
  for (reg, name) in regs:
    globals()[name] = cls(reg, name = name)
    arr.append(globals()[name])

# Set up x87, MMX, and SSE registers
for i in range(0, 8):
  stri = str(i)

  name = "st" + stri
  globals()[name] = FPRegister(i, name = name)
  st_array.append(globals()[name])
  name = "mm" + stri
  globals()[name] = MMXRegister(i, name = name)
  mm_array.append(globals()[name])
  name = "xmm" + stri
  globals()[name] = XMMRegister(i, name = name)
  xmm_array.append(globals()[name])


