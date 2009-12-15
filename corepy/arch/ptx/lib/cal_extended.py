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

import corepy.arch.cal.isa as cal
import corepy.spre.spe as spe

from corepy.arch.cal.lib.util import load_word

__doc__="""
CAL Extended Instructions
"""

class CALExt(spe.ExtendedInstruction):
  isa_module = cal

class iaddi(CALExt):
  """
  Add immediate
  """
  def block(self, d, a, value):
    code = self.get_active_code()    
    temp = code.prgm.acquire_register((value, value, value, value))
    code.add(cal.iadd(d, a, temp))
    code.prgm.release_register(temp)

    return

class isub(CALExt):
  """
  Subtract
  """
  def block(self, d, a, b):
    code = self.get_active_code()   
    code.add(cal.iadd(d, a, b.reg(neg=('x', 'y', 'z', 'w'))))
    return

class isubi(CALExt):
  """
  Subtract immediate
  """
  def block(self, d, a, value):
    code = self.get_active_code()    
    temp = code.prgm.acquire_register((-1*value, -1*value, -1*value, -1*value))
    code.add(cal.iadd(d, a, temp))
    code.prgm.release_register(temp)

    return

class isubf(CALExt):
  """
  Subtract
  """
  def block(self, d, a, b):
    code = self.get_active_code()
    code.add(cal.iadd(d, b, a.reg(neg=('x', 'y', 'z', 'w'))))
    return

class isubfi(CALExt):
  """
  Subtract immediate
  """
  def block(self, d, value, a):
    code = self.get_active_code()    
    temp = code.prgm.acquire_register((value, value, value, value))
    code.add(isub(d, temp, a))
    code.prgm.release_register(temp)

    return

class umodi(CALExt):
  """
  Unsigned mod immediate
  """
  def block(self, d, a, value):
    code = self.get_active_code()    
    temp = code.prgm.acquire_register((value, value, value, value))
    code.add(cal.umod(d, a, temp))
    code.prgm.release_register(temp)

    return

class umuli(CALExt):
  """
  Unsigned multiply immediate
  """
  def block(self, d, a, value):
    code = self.get_active_code()    
    temp = code.prgm.acquire_register((value, value, value, value))
    code.add(cal.umul(d, a, temp))
    code.prgm.release_register(temp)

    return

class imuli(CALExt):
  """
  Signed multiply immediate
  """
  def block(self, d, a, value):
    code = self.get_active_code()    
    temp = code.prgm.acquire_register((value, value, value, value))
    code.add(cal.imul(d, a, temp))
    code.prgm.release_register(temp)

    return

class ishli(CALExt):
  """
  Shift left immediate
  """
  def block(self, d, a, value):
    code = self.get_active_code()    
    temp = code.prgm.acquire_register((value, value, value, value))
    code.add(cal.ishl(d, a, temp)) # documentation says temp is 'scalar' with all components the same
    code.prgm.release_register(temp)

    return

class ishri(CALExt):
  """
  Shift right immediate
  """
  def block(self, d, a, value):
    code = self.get_active_code()    
    temp = code.prgm.acquire_register((value, value, value, value))
    code.add(cal.ishr(d, a, temp))
    code.prgm.release_register(temp)

    return

class ushri(CALExt):
  """
  Unsigned shift right immediate
  """
  def block(self, d, a, value):
    code = self.get_active_code()    
    temp = code.prgm.acquire_register((value, value, value, value))
    code.add(cal.ushr(d, a, temp))
    code.prgm.release_register(temp)

    return

class radd(CALExt):
  """
  Floating point add immediate
  """
  def block(self, d, a, b):
    code = self.get_active_code()    
    code.add(cal.add(d, b, a))

    return

class raddi(CALExt):
  """
  Floating point add immediate
  """
  def block(self, d, value, a):
    code = self.get_active_code()    
    temp = code.prgm.acquire_register((value, value, value, value))
    code.add(cal.add(d, temp, a))
    code.prgm.release_register(temp)

    return

class addi(CALExt):
  """
  Floating point add immediate
  """
  def block(self, d, a, value):
    code = self.get_active_code()    
    temp = code.prgm.acquire_register((value, value, value, value))
    code.add(cal.add(d, a, temp))
    code.prgm.release_register(temp)

    return

class subi(CALExt):
  """
  Floating point add immediate
  """
  def block(self, d, a, value):
    code = self.get_active_code()    
    temp = code.prgm.acquire_register((value, value, value, value))
    code.add(cal.sub(d, a, temp))
    code.prgm.release_register(temp)

    return

class subf(CALExt):
  """
  Floating point add immediate
  """
  def block(self, d, a, b):
    code = self.get_active_code()    
    code.add(cal.sub(d, b, a))

    return

class subfi(CALExt):
  """
  Floating point add immediate
  """
  def block(self, d, a, value):
    code = self.get_active_code()    
    temp = code.prgm.acquire_register((value, value, value, value))
    code.add(cal.sub(d, temp, a))
    code.prgm.release_register(temp)

    return

class muli(CALExt):
  """
  Floating point multiply immediate
  """
  def block(self, d, a, value):
    code = self.get_active_code()    
    temp = code.prgm.acquire_register((value, value, value, value))
    code.add(cal.mul(d, a, temp))
    code.prgm.release_register(temp)

    return

class divi(CALExt):
  """
  Floating point divide immediate
  """
  def block(self, d, a, value):
    code = self.get_active_code()    
    temp = code.prgm.acquire_register((value, value, value, value))
    code.add(cal.div(d, a, temp))
    code.prgm.release_register(temp)

    return

class rdiv(CALExt):
  """
  Floating point divide immediate reverse
  """
  def block(self, d, a, b):
    code = self.get_active_code()   
    code.add(cal.div(d, b, a))

    return

class rdivi(CALExt):
  """
  Floating point divide immediate reverse
  """
  def block(self, d, a, value):
    code = self.get_active_code()    
    temp = code.prgm.acquire_register((value, value, value, value))
    code.add(cal.div(d, temp, a))
    code.prgm.release_register(temp)

    return

class daddi(CALExt):
  """
  Double add immediate
  """
  def block(self, d, a, value):
    code = self.get_active_code()    
    temp = code.prgm.acquire_register((value, value, value, value))
    code.add(cal.dadd(d, a, temp))
    code.prgm.release_register(temp)

    return

class dsub(CALExt):
  """
  Double sub
  """
  def block(self, d, a, b):
    code = self.get_active_code()
    code.add(cal.dadd(d, a, b(neg=('x', 'y', 'z', 'w'))))

    return

class dsubi(CALExt):
  """
  Double sub immediate
  """
  def block(self, d, a, value):
    code = self.get_active_code()    
    temp = code.prgm.acquire_register((-1.0*value, -1.0*value, -1.0*value, 1.0*value))
    code.add(cal.dadd(d, a, temp))
    code.prgm.release_register(temp)

    return

class dmuli(CALExt):
  """
  Double multiply immediate
  """
  def block(self, d, a, value):
    code = self.get_active_code()    
    temp = code.prgm.acquire_register((value, value, value, value))
    code.add(cal.dmul(d, a, temp))
    code.prgm.release_register(temp)

    return

class ddivi(CALExt):
  """
  Double divide immediate
  """
  def block(self, d, a, value):
    code = self.get_active_code()    
    temp = code.prgm.acquire_register((value, value, value, value))
    code.add(cal.ddiv(d, a, temp))
    code.prgm.release_register(temp)

    return

class drdiv(CALExt):
  """
  Double divide reverse
  """
  def block(self, d, a, b):
    code = self.get_active_code()    
    code.add(cal.ddiv(d, b, a))

    return

class drdivi(CALExt):
  """
  Double divide immediate reverse
  """
  def block(self, d, a, value):
    code = self.get_active_code()    
    temp = code.prgm.acquire_register((value, value, value, value))
    code.add(cal.ddiv(d, temp, a))
    code.prgm.release_register(temp)

    return
