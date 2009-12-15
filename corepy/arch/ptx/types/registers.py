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

class ptxVariable(spe.Register):
  def __init__(self, space = None, rtype = None, name = '', v_comps = 1):
    spe.Register.__init__(self, name)

    if space == None:
      space = self.space
    if rtype == None:
      rtype = self.rtype_str

    if space in ('reg', 'sreg', 'const', 'global', 'local', 'param', 'shared', 'surf', 'tex'):
      self.space = space
    else:
      raise "Not a valid state space for a PTX register/variable."
    if rtype in ('b8', 'b16', 'b32', 'b64', 'u8', 'u16', 'u32', 'u64', 's8', 's16', 's32', 's64', 'f16', 'f32', 'f64', 'pred'):
      self.rtype_str = rtype
      self.rtype = rtype[0]
      if rtype == 'pred':
        self.rwidth = 1
      else:
        self.rwidth = int(rtype[1:])
    else:
      raise Exception("Not a valid type for a PTX register/variable.")

    if v_comps in (1, 2, 4):
      self.v_comps = v_comps
    else:
      raise "PTX register/variables must have vector length 1, 2, or 4."

    # TODO: generate name if none given BDM
    self.reg = name # TODO: Is this right??? BDM

    self.type = self.rtype_str

    return

#   def __add__(self, other):
#     if type(other) == int or type(other) == long or type(other) == float:
#       return Address(self, other)
#     else:
#       raise "Can't do that with a CALRegister."

#   def __radd__(self, other):
#     if type(other) == int or type(other) == long or type(other) == float:
#       return Address(self, other)
#     else:
#       raise "Can't do that with a CALRegister."

#   def __eq__(self, other):
#     # TODO - AWF - allow string names like "i0" to be equal?
#     return type(self) == type(other) and self.name == other.name

  #def __getattribute__(self, name):
  #  try:
  #    return object.__getattribute__(self, name)
  #  except:
  #    valid = True
  #    if len(name) == 1 or len(name) == 2 or len(name) == 4:
  #      for swizzle_comp in name:
  #        if not swizzle_comp in ['x', 'y', 'z', 'w', '_', '0', '1', 'r', 'g', 'b', 'a']:
  #          valid = False
  #    if valid:
  #      return QualifiedCALRegister(self, tuple(name))
  #
  #  raise AttributeError

  def render(self):
    return self.name

class ptxSpecialRegister(ptxVariable):
  def __init__(self, rtype = 'u16', name='', v_comps = 1):
    ptxVariable.__init__(self, space='sreg', rtype=rtype, name=name, v_comps=v_comps)

class ptxSpecialRegisterVector(ptxVariable):
  def __init__(self, rtype = 'u16', name='', v_comps = 4):
    ptxVariable.__init__(self, space='sreg', rtype=rtype, name=name, v_comps=v_comps)
    self.x = ptxSpecialRegister(name=name+'.x')
    self.y = ptxSpecialRegister(name=name+'.y')
    self.z = ptxSpecialRegister(name=name+'.z')
    self.w = ptxSpecialRegister(name=name+'.w')

tid = ptxSpecialRegisterVector(name='%tid')
ntid = ptxSpecialRegisterVector(name='%ntid')
ctaid = ptxSpecialRegisterVector(name='%ctaid')
nctaid = ptxSpecialRegisterVector(name='%nctaid')
gridid = ptxSpecialRegister(name='%gridid')
clock = ptxSpecialRegister(rtype='u32', name='%clock')

class ptxRegister(ptxVariable):
  _complex_reg = True
  space = 'reg'
  def __init__(self, rtype = 'b32', name = '', v_comps = 1):
    ptxVariable.__init__(self, 'reg', rtype, name, v_comps)
    return

class ptxRegister_b8(ptxRegister):
  rtype_str = 'b8'

class ptxRegister_b16(ptxRegister):
  rtype_str = 'b16'

class ptxRegister_b32(ptxRegister):
  rtype_str = 'b32'

class ptxRegister_b64(ptxRegister):
  rtype_str = 'b64'

class ptxRegister_u8(ptxRegister):
  rtype_str = 'u8'

class ptxRegister_u16(ptxRegister):
  rtype_str = 'u16'

class ptxRegister_u32(ptxRegister):
  rtype_str = 'u32'

class ptxRegister_u64(ptxRegister):
  rtype_str = 'u64'

class ptxRegister_s8(ptxRegister):
  rtype_str = 's8'

class ptxRegister_s16(ptxRegister):
  rtype_str = 's16'

class ptxRegister_s32(ptxRegister):
  rtype_str = 's32'

class ptxRegister_s64(ptxRegister):
  rtype_str = 's64'

class ptxRegister_f16(ptxRegister):
  rtype_str = 'f16'

class ptxRegister_f32(ptxRegister):
  rtype_str = 'f32'

class ptxRegister_f64(ptxRegister):
  rtype_str = 'f64'

class ptxRegister_pred(ptxRegister):
  rtype_str = 'pred'

class ptxAddress(ptxVariable):
  def __init__(self, base, offset = 0, name = ''):
    self.name = name
    if isinstance(base, (spe.Variable, ptxVariable, int, long)):
      self.base = base
      if isinstance(offset, (int, long)):
        self.offset = offset
      else:
        print type(offset)
        raise Exception("Invalid offset for address - must be an integer/long.")
    else:
      raise Exception("Invalid base for address - must be a synthetic variable, ptxVariable, or an integer/long.")

  def render(self):
    if isinstance(self.base, spe.Variable):
      base_str = self.base.reg.render()
    elif isinstance(self.base, ptxVariable):
      base_str = self.base.render()
    else:
      base_str = str(self.base)
    if self.offset != 0:
      offset_str = ' + ' + str(self.offset)
    else:
      offset_str = ''
    return '[' + base_str + offset_str + ']'

# TODO: support arrays BDM

# TODO: Change this back!
num_regs = 32
num_regs = 128

_rtypes = ('b8', 'b16', 'b32', 'b64', 'u8', 'u16', 'u32', 'u64', 's8', 's16', 's32', 's64', 'f16', 'f32', 'f64', 'pred')
_rclasses = (ptxRegister_b8, ptxRegister_b16, ptxRegister_b32, ptxRegister_b64,
             ptxRegister_u8, ptxRegister_u16, ptxRegister_u32, ptxRegister_u64,
             ptxRegister_s8, ptxRegister_s16, ptxRegister_s32, ptxRegister_s64, 
             ptxRegister_f16, ptxRegister_f32, ptxRegister_f64, ptxRegister_pred)

_reg_names = [['r' + str(i) + '_' + rtype for rtype in _rtypes] for i in xrange(num_regs)]

for _i in xrange(num_regs):
  for _rtype, _rclass  in zip(_rtypes, _rclasses):
    _reg_name = 'r' + str(_i) + '_' + _rtype
    globals()[_reg_name] = _rclass(rtype = _rtype, name = _reg_name)

