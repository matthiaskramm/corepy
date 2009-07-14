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


# import platform_conf
from ppc_isa import *

# Nothing to see here, move along... ;)
__active_code = None

def set_active_code(code):
  global __active_code

  if __active_code is not None:
    __active_code.set_active_callback(None)

  __active_code = code

  if code is not None:
    code.set_active_callback(set_active_code)
  return

# Property version
def __get_active_code(self):
  global __active_code
  return __active_code

# Free function version
def get_active_code():
  global __active_code
  return __active_code

# _ppc_active_code_prop = property(get_active_code)

# Build the instructions
for l in locals().values():
  if isinstance(l, type) and issubclass(l, (Instruction, DispatchInstruction)):
    l.active_code = property(__get_active_code) 


# ------------------------------
# Mnemonics
# ------------------------------

# TODO: Find a better place for these...
def add(D, A, SIMM, **koperands): return addx(D, A, SIMM, **koperands)

def b(LI, **koperands):   return bx(LI, **koperands)
#class b(bx):
#  def __init__(self, LI, **koperands):
#    bx.__init__(self, LI, AA=0, LK=0, **koperands)

def ba(LI, **koperands):   return bx(LI, AA=1, **koperands)
def bl(LI, **koperands):   return bx(LI, LK=1, **koperands)
def bla(LI, **koperands):  return bx(LI, AA=1, LK=1, **koperands)

def bdnz(BD, **koperands): return bcx(0x10, 0, BD, **koperands)
# bo = 011zy -> 01101 branch if true (> 0), likely to be taken
def bgt(BD, **koperands):  return bcx(0x0D, 1, BD, **koperands)
# bo = 011zy -> 01101 branch if true (> 0), likely to be taken
def blt(BD, **koperands):  return bcx(0x0D, 0, BD, **koperands)
def bne(BD, **koperands):  return bcx(4, 2, BD, **koperands)
def beq(BD, **koperands):  return bcx(12, 2, BD, **koperands)

# def blr(): return (19 << 26) | (20 << 21) | (0 << 16) | (0 << 11) | (16 << 1)
def blr(**koperands): return bclrx(20, 0, **koperands)

def cmpw(crfD, A, B, **koperands): return cmp_(crfD, 0, A, B, **koperands)
def divw(D, A, B, **koperands): return divwx(D, A, B, **koperands)
def li(D, SIMM, **koperands): return addi(D, 0, SIMM, **koperands)
def mftbl(D, **koperands): return mftb(D, 268, **koperands)
def mftbu(D, **koperands): return mftb(D, 269, **koperands)
def mullw(D, A, B, **koperands): return mullwx(D, A, B, **koperands)
def mtctr(S, **koperands): return mtspr(9, S, **koperands)
def mtvrsave(S, **koperands): return mtspr(256, S, **koperands)
def mfvrsave(S, **koperands): return mfspr(S, 256, **koperands)
def subf(D, A, B, **koperands): return subfx(D, A, B, **koperands)


# preferred PPC noop (CWG p14)
def noop(**koperands): return ori(0,0,0, **koperands)

def Illegal(): return 0;

