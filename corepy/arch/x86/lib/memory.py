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

import corepy.arch.x86.types.registers as regs

class MemoryReference:
  def __init__(self, val, disp = None, index = None, scale = 1, data_size = 32, addr_size = None):
    if not data_size in (8, 16, 32, 64, 80, 128, 228, 752, 4096, None):
      raise Exception('Memory reference data size must be 8, 16, 32, 64, 80, 128, 228, 752, or 4096 bits')
    self.data_size = data_size

    if isinstance(val, regs.GPRegister32):
      # TODO - validate that disp/index are correct
      self.base = val
      self.addr = None
      self.disp = disp
      self.index = index

      if index != None and type(index) != type(self.base):
        raise Exception('When specified, index register must be same type register as the base register')

      if addr_size == None:
        # No size, set based on register sizes.
        self.addr_size = 32
      elif addr_size == 32 or addr_size == 16:
        self.addr_size = addr_size
      else:
        raise Exception('Address size must be either 32 or 16 bits')

      # Doing a little pre-computing can speed up rendering
      if scale == 1:
        self.scale_sib = 0   # 0 << 6
      elif scale == 2:
        self.scale_sib = 64  # 1 << 6
      elif scale == 4:
        self.scale_sib = 128 # 2 << 6
      elif scale == 8:
        self.scale_sib = 192 # 3 << 6
      else:
        raise Exception('Invalid scale value %s must be 1,2,4,8' % (str(scale)))
      self.scale = scale

    elif isinstance(val, (int, long)):
      self.base = None
      self.addr = val
      self.disp = None
      self.index = None

      if addr_size == None:
        self.addr_size = 32
      elif addr_size == 32 or addr_size == 16:
        self.addr_size = addr_size
      else:
        raise Exception('Address size must be either 32 or 16 bits')
    else:
      raise Exception("Invalid memory value, must be an address or register")
    return

  def __str__(self):
    data_size = "data_size = %s" % str(self.data_size)
    if self.base != None:
      if self.disp != None:
        if self.index != None:
          return "MemRef(%s, %d, %s, %d, %s)" % (str(self.base), self.disp, str(self.index), self.scale, data_size)
        return "MemRef(%s, %d, %s)" % (str(self.base), self.disp, data_size)
      elif self.index != None:
        return "MemRef(%s, index = %s, scale = %d, %s)" % (str(self.base), str(self.index), self.scale, data_size)
      return "MemRef(%s, %s)" % (str(self.base), data_size)
    elif self.addr != None:
      return "MemRef(0x%x, %s)" % (self.addr, data_size)
    else:
      return "INVALID MemoryReference"


class MemRef(MemoryReference): pass

