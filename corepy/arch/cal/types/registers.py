
# Copyright 2006-2007 The Trustees of Indiana University.

# This software is available for evaluation purposes only.  It may not be
# redistirubted or used for any other purposes without express written
# permission from the authors.

# Authors:
#   Andrew Friedley     (afriedle@cs.indiana.edu)
#   Christopher Mueller (chemuell@cs.indiana.edu)
#   Andrew Lumsdaine    (lums@cs.indiana.edu)

import corepy.spre.spe as spe

# ------------------------------
# Registers
# ------------------------------

class mods:
  abs = 'abs'
  bias = 'bias'
  bx2 = 'bx2'
  invert = 'invert'
  sign = 'sign'
  x2 = 'x2'

class divcomp:
  x = 'x'
  y = 'y'
  z = 'z'
  w = 'w'

class Address(object):
  def __init__(self, base, offset):
    if not isinstance(base, QualifiedCALRegister) and not isinstance(CALRegister):
      raise "base must be a register."
    if type(offset) != int and type(offset) != long and type(offset) != float:
      raise "offset must be a numeric type"
    self.base = base
    self.offset = offset

  def render(self):
    return self.base.render() + ' + ' + str(self.offset)

  def __str__(self):
    return self.render()

class CALRegister(spe.Register):
  def __init__(self, reg, name = None):
    spe.Register.__init__(self, reg, name = name)
    return

  def __add__(self, other):
    if type(other) == int or type(other) == long or type(other) == float:
      return Address(self, other)
    else:
      raise "Can't do that with a CALRegister."

  def __radd__(self, other):
    if type(other) == int or type(other) == long or type(other) == float:
      return Address(self, other)
    else:
      raise "Can't do that with a CALRegister."

  def __eq__(self, other):
    return type(self) == type(other) and self.reg == other.reg and self.name == other.name

  def __call__(self, swizzle_str = '', abs = False, bias = False, bx2 = False, invert = False, sign = False, x2 = False, neg = '', divcomp = None):
    """
    swizzle_str should be a string representing the swizzle or mask, such as 'xxxx' or 'x___'.
    neg should be a string of components to be negated such as ('xw')
    divcomp should be one of registers.divcomp such as divcomp.x
    """
    return QualifiedCALRegister(self, tuple(swizzle_str), abs=abs, bias=bias, bx2=bx2, invert=invert, sign=sign, x2=x2, neg=tuple(neg), divcomp=divcomp)

  def __getattribute__(self, name):
    try:
      return object.__getattribute__(self, name)
    except:
      valid = True
      if len(name) == 1 or len(name) == 2 or len(name) == 4:
        for swizzle_comp in name:
          if not swizzle_comp in ['x', 'y', 'z', 'w', '_', '0', '1', 'r', 'g', 'b', 'a']:
            valid = False
      if valid:
        return QualifiedCALRegister(self, tuple(name))

    raise AttributeError

  def render(self):
    return self.name
    
# This is the register type that will be seen by the ISA WHENEVER any source modifiers or swizzles 
# are present. It's job is basically to generate a string based on all of the modifiers.
# Note that these registers are TEMPORARY - for all practical purposes for now, we can view them as
# tied to a particular instruction. The user can cache them by naming them, though, and this should be ok.
class QualifiedCALRegister(CALRegister):
  def __init__(self, cal_reg, swizzle = (), abs=False, bias=False, bx2=False, invert=False, sign=False, x2=False, neg = (), divcomp = None):
    if type(cal_reg) == QualifiedCALRegister:
      cal_reg = cal_reg.GetBaseRegister()
    CALRegister.__init__(self, cal_reg.reg)
    self.cal_reg = cal_reg
    self.reg = cal_reg.reg
    self.swizzle = swizzle
    self.abs = abs
    self.bias = bias
    self.bx2 = bx2
    self.invert = invert
    self.sign = sign
    self.x2 = x2
    self.neg = neg
    self.divcomp = divcomp
    self.name = self.render()

  def check(self):
    valid = True
    if len(self.swizzle) == 1 or len(self.swizzle) == 2 or len(self.swizzle) == 4:
      for swizzle_comp in self.swizzle:
        if not swizzle_comp in ['x', 'y', 'z', 'w', '_', '0', '1', 'r', 'g', 'b', 'a']:
          valid = False
    else:
      valid = False
    if self.divcomp != None and self.divcomp not in ['y', 'z', 'w', 'unkown']:
      valid = False
    if neg != ():
      for neg_comp in self.neg:
        if neg_comp not in ['x', 'y', 'z', 'w', 'r', 'g', 'b', 'a']:
          valid = False
    return valid

  def render(self):
    render_str = ''
    render_str += self.GetBaseRegister().reg
    if self.abs == True:
      render_str += '_abs'
    if self.bias == True:
      render_str += '_bias'
    if self.bx2 == True:
      render_str += '_bx2'
    if self.invert == True:
      render_str += '_invert'
    if self.sign == True:
      render_str += '_sign'
    if self.x2 == True:
      render_str += '_x2'
    if self.neg != () and self.neg != None:
      render_str += '_neg('
      for neg_comp in self.neg:
        render_str += neg_comp
      render_str += ')'
    if self.divcomp != None:
      render_str += '_divcomp(' + divcomp + ')'
    if self.swizzle != ():
      render_str += '.'
      for swizzle_comp in self.swizzle:
        render_str += swizzle_comp
    return render_str

  def __call__(self, swizzle_str = None, abs = None, bias = None, bx2 = None, invert = None, sign = None, x2 = None, neg = None, divcomp = None):
    """
    swizzle_str should be a string representing the swizzle or mask, such as 'xxxx' or 'x___'.
    neg should be a string of components to be negated such as ('xw')
    divcomp should be one of registers.divcomp such as divcomp.x
    """
    retval = self.copy()
    if swizzle_str != None:
      valid = True
      if len(name) == 1 or len(name) == 2 or len(name) == 4:
        for swizzle_comp in name:
          if not swizzle_comp in ['x', 'y', 'z', 'w', '_', '0', '1', 'r', 'g', 'b', 'a']:
            valid = False
      if valid:
        retval.swizzle = tuple(swizzle_str)
      else:
        raise "Bad swizzle value (must be string of length 1, 2 or 4 composed of ['x', 'y', 'z', 'w', '-', '0', '1', 'r', 'g', 'b', 'a']"
    if abs != None:
      retval.abs = abs
    if bias != None:
      retval.bias = bias
    if bx2 != None:
      retval.bx2 = bx2
    if invert != None:
      retval.invert = invert
    if sign != None:
      retval.sign = sign
    if x2 != None:
      retval.x2 = x2
    if neg != None:
      retval.neg = neg
    if divcomp != None:
      retval.divcomp = divcomp
  
    return retval

  def __getattribute__(self, name):
    try:
      return object.__getattribute__(self, name)
    except:
      return object.__getattribute__(self, '__call__')(swizzle_str=name)
      #valid = True
      #if len(name) == 1 or len(name) == 2 or len(name) == 4:
      #  for swizzle_comp in name:
      #    if not swizzle_comp in ['x', 'y', 'z', 'w', '_', '0', '1', 'r', 'g', 'b', 'a']:
      #      valid = False
      #if valid:
      #  retval = object.__getattribute__(self, 'copy')()
      #  retval.swizzle = tuple(name)
    raise AttributeError

  def __str__(self):
    return self.render()

  def copy(self):
    return QualifiedCALRegister(self, swizzle=self.swizzle, abs=self.abs, bias=self.bias, bx2=self.bx2, invert=self.invert, sign=self.sign, x2=self.x2, neg=self.neg, divcomp=self.divcomp)

  def GetBaseRegister(self):
    return self.cal_reg

class CALBuffer:
  def __init__(self, buffer, name, rel_addressable=False):
    # right now, buffer and name should match
    self.buffer = buffer
    self.name = name
    self.rel_addressable = rel_addressable

  def __getitem__(self, i):

    if type(i) != int and type(i) != long and type(i) != float and not isinstance(i, QualifiedCALRegister) and not isinstance(i, Address):
      if self.rel_addressable == True:
        if type(i) != str:
          raise "Index must be numeric or a register"
        else:
          raise "Index must be numeric"

    if isinstance(i, QualifiedCALRegister) or isinstance(i, Address):
      if self.rel_addressable == False:
        raise "This register/buffer type is not relatively addressable"
      else:
        i_str = i.render()
    else:
      i_str = str(i)
    name = self.name + '[' + i_str + ']'
    if name not in globals():
      globals()[name] = CALRegister(name, name=name)
    return globals()[name]

class TempRegister(CALRegister): pass
class LiteralRegister(CALRegister): pass

r = []
l = []

for i in range(256, -1, -1): # reverse order so acquire_register starts at 0
  stri = str(i)

  name = "r" + stri
  globals()[name] = TempRegister(name, name = name)
  r.append(globals()[name])

  name = "l" + stri
  globals()[name] = LiteralRegister(name, name = name)
  l.append(globals()[name])

globals()['vWinCoord0'] = CALRegister('vWinCoord0', name='vWinCoord0')
globals()['v0'] = CALRegister('v0', name='v0')
globals()['a0'] = CALRegister('a0', name='a0')
globals()['g'] = CALBuffer('g', name='g', rel_addressable=True)


for i in range(0, 16):
  stri = str(i)

  name = "i" + stri
  globals()[name] = CALBuffer(name, name=name, rel_addressable=False)

  name = "cb" + stri
  globals()[name] = CALBuffer(name, name=name, rel_addressable=True)
  
  name = "o" + stri
  globals()[name] = CALRegister(name, name=name)

  name = "x" + stri
  globals()[name] = CALBuffer(name, name = name, rel_addressable=True)



def TestRelativeAddressing():
  import corepy.arch.cal.platform as env
  import corepy.arch.cal.isa as cal
  
  proc = env.Processor(0)
  
  input_mem = proc.alloc_remote('I', 4, 16, 1)
  output_mem = proc.alloc_remote('I', 4, 1, 1)
  
  for i in range(16*1*4):
    for j in range(4):
      input_mem[i*4 + j] = i
    
  code = env.InstructionStream()
  cal.set_active_code(code)
    
  cal.dcl_output(o0, USAGE=cal.usage.generic)
  cal.dcl_literal(l0, 1, 1, 1, 1)
  cal.dcl_literal(l1, 16, 16, 16, 16)
  cal.mov(r0, r0('0000'))
  cal.mov(r1, r1('0000'))
  

  cal.whileloop()
  cal.iadd(r1, r1, g[r0.x])
  cal.iadd(r0, r0, l0)
  cal.breakc(cal.relop.ge, r0, l1)
  cal.endloop()

  cal.mov(o0, r1)
  
  code.set_remote_binding('g[]', input_mem)
  code.set_remote_binding('o0', output_mem)
  
  domain = (0, 0, 128, 128)
  proc.execute(code, domain)
  
  # code.cache_code()
  # print code.render_string
 
  if output_mem[0] == 120:
    print "Passed relative addressing test"
  else:
    print "Failed relative addressing test"

  proc.free_remote(input_mem)
  proc.free_remote(output_mem)


def TestRelativeAddressing2():
  import corepy.arch.cal.platform as env
  import corepy.arch.cal.isa as cal
  
  proc = env.Processor(0)
  
  input_mem = proc.alloc_remote('I', 4, 16, 1)
  output_mem = proc.alloc_remote('I', 4, 1, 1)
  
  for i in range(17*1*4):
    for j in range(4):
      input_mem[i*4 + j] = i
    
  code = env.InstructionStream()
  cal.set_active_code(code)
    
  cal.dcl_output(o0, USAGE=cal.usage.generic)
  cal.dcl_literal(l0, 1, 1, 1, 1)
  cal.dcl_literal(l1, 16, 16, 16, 16)
  cal.mov(r0, r0('0000'))
  cal.mov(r1, r1('0000'))
  

  cal.whileloop()
  cal.iadd(r1, r1, g[1 + r0.x])
  cal.iadd(r0, r0, l0)
  cal.breakc(cal.relop.ge, r0, l1)
  cal.endloop()

  cal.mov(o0, r1)

  #code.cache_code()
  #print code.render_string
  
  code.set_remote_binding('g[]', input_mem)
  code.set_remote_binding('o0', output_mem)
  
  domain = (0, 0, 128, 128)
  proc.execute(code, domain)
   
  if output_mem[0] == 136:
    print "Passed relative addressing with offset test"
  else:
    print "Failed relative addressing with offset test"
  # print output_mem
 
  proc.free_remote(input_mem)
  proc.free_remote(output_mem)


if __name__ == '__main__':
  TestRelativeAddressing()
  TestRelativeAddressing2()
