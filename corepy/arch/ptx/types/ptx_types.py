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

import array

import registers as regs
import corepy.arch.ptx.isa as ptx
import corepy.spre.spe as spe
import corepy.arch.ptx.lib.util as util
  
__doc__ = """
"""

# ------------------------------------------------------------
# 'Type' Classes
# ------------------------------------------------------------

# Type classes implement the operator overloads for a type and hold
# other type-specific information, such as register types and valid
# literal types.

# They are separate from the type so they can be used as mix-ins in
# different contexts, e.g. Variables and Expressions subclasses can
# both share operator semantics by subclassing the same operator
# class.  

# Operator classes also provide static interfaces to typed versions of
# the operations.  

# Operator methods return an Expression of an appropriate type for the
# operation.

# To always return the same type:
#  return SignedWord.expr_class(inst, *(self, other))

# To upcast to the type of the first operand:
#  return self.expr_class(inst, *(self, other))

# To upcast to the type of the second operand:
#  return other.expr_class(inst, *(self, other))

# Upcasting can be useful for two types of different specificity are
# used in expressions and the more specific type should be 
# preserved type the expressions.  For instance, the logical
# operators are the base classes of all integer-like types.  A logical 
# operation, e.g. (a & b), should preserve the most specific type of a
# and b.

def _most_specific(a, b, default = None):
  """
  If a and b are from the same hierarcy, return the more specific of
  [type(a), type(b)], or the default type if they are from different
  hierarchies. If default is None, return type(a), or type(b) if a
  does not have a type_cls
  """
  if (hasattr(a, 'type_cls') and hasattr(b, 'type_cls')):
    if issubclass(b.type_cls, a.type_cls):
      return type(b)
    elif issubclass(a.type_cls, b.type_cls):
      return type(a)
  elif default is None:
    if hasattr(a, 'type_cls'):
      return type(a)
    elif hasattr(b, 'type_cls'):
      return type(b)
    
  return default
  
_int_literals = (spe.Immediate, int, long)

class PTXType(spe.Type):
  def _get_active_code(self):
    return ptx.get_active_code()

  def _set_active_code(self, code):
    return ptx.set_active_code(code)
  active_code = property(_get_active_code, _set_active_code)

# ------------------------------------------------------------
# General Purpose Register Types
# ------------------------------------------------------------

class BitType(PTXType):
  register_type_id = 'b32'
  literal_types = (int,long)

  def _upcast(self, other, inst):
    return inst.ex(self, other, type_cls = _most_specific(self, other))

  def __and__(self, other):
    if isinstance(other, BitType):
      return self._upcast(other, ptx.and_)
    elif isinstance(other, _int_literals):
      return ptx.and_.ex(self, other, type_cls = self.var_cls)
    raise Exception('__and__ not implemented for %s and %s' % (type(self), type(other)))    

  and_ = staticmethod(__and__)

  def __lshift__(self, other):
    if isinstance(other, BitType):
      return ptx.shl.ex(self, other, type_cls = self.var_cls)
    raise Exception('__lshift__ not implemented for %s and %s' % (type(self), type(other)))    
  lshift = staticmethod(__lshift__)

  def __rshift__(self, other):
    if isinstance(other, BitType):
      return ptx.shr.ex(self, other, type_cls = self.var_cls)
    raise Exception('__rshift__ not implemented for %s and %s' % (type(self), type(other)))    
  rshift = staticmethod(__rshift__)

  def __or__(self, other):
    if isinstance(other, BitType):
      return self._upcast(other, ptx.orx)
    elif isinstance(other, _int_literals):
      return ptx.or_.ex(self, other, type_cls = self.var_cls)
    raise Exception('__or__ not implemented for %s and %s' % (type(self), type(other)))    
  or_ = staticmethod(__or__)

  def __xor__(self, other):
    if isinstance(other, BitType):
      return self._upcast(other, ptx.xorx)
    elif isinstance(other, _int_literals):
      return ptx.xor.ex(self, other, type_cls = self.var_cls)
    raise Exception('__xor__ not implemented for %s and %s' % (type(self), type(other)))    
  xor = staticmethod(__xor__)

  def _set_literal_value(self, value):
    ## Put the lower 16 bits into r-temp
    #self.code.add(ptx.addi(self.reg, 0, value & 0xFFFF))
  
    ## Addis r-temp with the upper 16 bits (shifted add immediate) and
    ## put the result in r-target
    #if (value & 0x7FFF) != value:
    #  self.code.add(ptx.addis(self.reg, self.reg, ((value + 32768) >> 16)))

    self.code.add(ptx.add(self.reg, self.reg, value))
    return

  def copy_register(self, other):
    return self.code.add(ptx.mov(self, other))
    #return self.code.add(ptx.add(self, other, 0))

  def load(self, addr, offset = 0, space='global'):
    if isinstance(offset, (int, long)):
      self.code.add(ptx.ld(space, self, regs.ptxAddress(addr, offset)))
    else:

      temp = self.code.prgm.acquire_register('u64')
      self.code.add(ptx.add(temp, addr, offset))
      self.code.add(ptx.ld(space, self, regs.ptxAddress(temp)))
      self.code.prgm.release_register(temp)

  def store(self, addr, offset = 0, space='global'):
    if isinstance(offset, (int, long)):
      self.code.add(ptx.st(space, regs.ptxAddress(addr, offset), self))
    else:

      temp = self.code.prgm.acquire_register('u64')
      self.code.add(ptx.add(temp, addr, offset))
      self.code.add(ptx.st(space, regs.ptxAddress(temp), self))
      self.code.prgm.release_register(temp)


# ------------------------------
# Integer Types
# ------------------------------

class UnsignedWordType(BitType):
  register_type_id = 'u64'

  def __add__(self, other):
    if isinstance(other, UnsignedWordType):
      return ptx.add.ex(self, other, type_cls = self.var_cls)
    elif isinstance(other, (spe.Immediate, int)):
      return self.expr_cls(ptx.add, *(self, other))
    raise Exception('__add__ not implemented for %s and %s' % (type(self), type(other)))    
  add = staticmethod(__add__)
  
  def __div__(self, other):
    if isinstance(other, SignedWordType):
      return self.expr_cls(ptx.div, *(self, other))
    raise NotImplemented
  div = staticmethod(__div__)

  def __mul__(self, other):
    if isinstance(other, UnsignedWordType):
      return self.expr_cls(ptx.mul, *(self, other))
    elif isinstance(other, (spe.Immediate, int)):
      return self.expr_cls(ptx.mul, *(self, other))      
    raise Exception('__mul__ not implemented for %s and %s' % (type(self), type(other)))          
  div = staticmethod(__mul__)

class SignedWordType(BitType):
  register_type_id = 's32'

  def __add__(self, other):
    if isinstance(other, SignedWordType):
      return ptx.add.ex(self, other, type_cls = self.var_cls)
    elif isinstance(other, (spe.Immediate, int)):
      return self.expr_cls(ptx.add, *(self, other))
    raise Exception('__add__ not implemented for %s and %s' % (type(self), type(other)))    
  add = staticmethod(__add__)
  
  def __div__(self, other):
    if isinstance(other, SignedWordType):
      return self.expr_cls(ptx.div, *(self, other))
    raise Exception('__div__ not implemented for %s and %s' % (type(self), type(other)))      
  div = staticmethod(__div__)

  def __mul__(self, other):
    if isinstance(other, SignedWordType):
      return self.expr_cls(ptx.mul, *(self, other))
    elif isinstance(other, (spe.Immediate, int)):
      return self.expr_cls(ptx.mul, *(self, other))      
    raise Exception('__mul__ not implemented for %s and %s' % (type(self), type(other)))          
  div = staticmethod(__div__)

  def __neg__(self):
    return ptx.negx(self, type_cls = self.var_cls)

  def __sub__(self, other):
    if isinstance(other, SignedWordType):
      return self.expr_cls(ptx.sub, self, other) # swap a and b
    raise Exception('__add__ not implemented for %s and %s' % (type(self), type(other)))    
  sub = staticmethod(__sub__)

  
# ------------------------------------------------------------
# Floating Point Register Types
# ------------------------------------------------------------

class SingleFloatType(PTXType):
  register_type_id = 'f32'
  literal_types = (float,)

  def __abs__(self):
    return ptx.abs.ex(self, type_cls = self.var_cls)
  abs = staticmethod(__abs__)
  
  def __add__(self, other):
    if isinstance(other, SingleFloatType):
      return ptx.add.ex(self, other, type_cls = self.var_cls)
    raise Exception('__add__ not implemented for %s and %s' % (type(self), type(other)))        
  add = staticmethod(__add__)
  
  def __div__(self, other):
    if isinstance(other, SingleFloatType):
      return ptx.div.ex(self, other, type_cls = self.var_cls)
    raise Exception('__div__ not implemented for %s and %s' % (type(self), type(other)))    
  div = staticmethod(__div__)

  def __mul__(self, other):
    if isinstance(other, SingleFloatType):
      return ptx.mul.ex(self, other, type_cls = self.var_cls)
    raise Exception('__mul__ not implemented for %s and %s' % (type(self), type(other)))
  mul = staticmethod(__mul__)

  def __neg__(self):
    return ptx.neg.ex(self, type_cls = self.var_cls)
  neg = staticmethod(__neg__)

  def __sub__(self, other):
    if isinstance(other, SingleFloatType):
      return ptx.sub.ex(self, other, type_cls = self.var_cls)
    raise Exception('__sub__ not implemented for %s and %s' % (type(self), type(other)))
  sub = staticmethod(__sub__)

#   def _set_literal_value(self, value):
#     storage = array.array('f', (float(value),))
#     self.code.prgm.add_storageOA(storage)

#     self.load(storage.buffer_info()[0])

#     #storage = array.array('f', (float(self.value),))
#     #self.code.prgm.add_storage(storage)
    
#     #r_storage = self.code.prgm.acquire_register()
#     #addr = Bits(storage.buffer_info()[0], reg = r_storage)
#     #self.code.add(ptx.lfs(self.reg, addr.reg, 0))
#     #self.code.prgm.release_register(r_storage)

#     return

  def copy_register(self, other):
    return self.code.add(ptx.mov(self, other))
    #return self.code.add(ptx.add(self, other, 0))
  
  def load(self, addr, offset = 0, space='global'):
    if isinstance(offset, (int, long)):
      self.code.add(ptx.ld(space, self, regs.ptxAddress(addr, offset)))
    else:

      temp = self.code.prgm.acquire_register('u64')
      self.code.add(ptx.add(temp, addr, offset))
      self.code.add(ptx.ld(space, self, regs.ptxAddress(temp)))
      self.code.prgm.release_register(temp)

#     # If addr is a constant, create a variable and store the value
#     if not issubclass(type(addr), spe.Type):
#       r_storage = self.code.prgm.acquire_register()
#       addr = Bits(addr, reg = r_storage)
#     else:
#       r_storage = None

#     # If offset is a constant, use lfd, otherwise use lfdx
#     if issubclass(type(offset), spe.Type):
#       self.code.add(ptx.lfsx(self, addr, offset))
#     else:
#       # TODO: Check size of offset to ensure it fits in the immediate field 
#       self.code.add(ptx.lfs(self, addr, offset))

#     if r_storage is not None:
#       self.code.prgm.release_register(r_storage)

#     return

  def store(self, addr, offset = 0, space='global'):
    if isinstance(offset, (int, long)):
      self.code.add(ptx.st(space, regs.ptxAddress(addr, offset), self))
    else:

      temp = self.code.prgm.acquire_register('u64')
      self.code.add(ptx.add(temp, addr, offset))
      self.code.add(ptx.st(space, regs.ptxAddress(temp), self))
      self.code.prgm.release_register(temp)

#     # If addr is a constant, create a variable and store the value
#     if not issubclass(type(addr), spe.Type):
#       r_storage = self.code.prgm.acquire_register()
#       addr = Bits(addr, reg = r_storage)
#     else:
#       r_storage = None

#     # If offset is a constant, use lfd, otherwise use lfdx
#     if issubclass(type(offset), spe.Type):
#       self.code.add(ptx.stfsx(self, addr, offset))
#     else:
#       # TODO: Check size of offset to ensure it fits in the immediate field 
#       self.code.add(ptx.stfs(self, addr, offset))

#     if r_storage is not None:
#       self.code.prgm.release_register(r_storage)

#     return


class DoubleFloatType(PTXType):
  register_type_id = 'f64'
  literal_types = (float,)

  def __abs__(self):
    return ptx.abs.ex(self, type_cls = self.var_cls)
  abs = staticmethod(__abs__)
  
  def __add__(self, other):
    if isinstance(other, DoubleFloatType):
      return ptx.add.ex(self, other, type_cls = self.var_cls)
    raise Exception('__add__ not implemented for %s and %s' % (type(self), type(other)))
  add = staticmethod(__add__)
  
  def __div__(self, other):
    if isinstance(other, DoubleFloatType):
      return ptx.div.ex(self, other, type_cls = self.var_cls)
    raise Exception('__div__ not implemented for %s and %s' % (type(self), type(other)))
  div = staticmethod(__div__)

  def __mul__(self, other):
    if isinstance(other, DoubleFloatType):
      return ptx.mul.ex(self, other, type_cls = self.var_cls)
    raise Exception('__mul__ not implemented for %s and %s' % (type(self), type(other)))
  mul = staticmethod(__mul__)

  def __neg__(self):
    return ptx.neg.ex(self, type_cls = self.var_cls)
  neg = staticmethod(__neg__)
    
  def __sub__(self, other):
    if isinstance(other, DoubleFloatType):
      return ptx.sub.ex(self, other, type_cls = self.var_cls)
    raise Exception('__sub__ not implemented for %s and %s' % (type(self), type(other)))
  sub = staticmethod(__sub__)
    
#   def _set_literal_value(self, value):
#     storage = array.array('d', (float(value),))
#     self.code.prgm.add_storage(storage)

#     self.load(storage.buffer_info()[0])
# #     r_storage = self.code.prgm.acquire_register()
# #     addr = Bits(storage.buffer_info()[0], reg = r_storage)
# #     self.code.add(ptx.lfd(self.reg, addr.reg, 0))
# #     self.code.prgm.release_register(r_storage)

#     return

  def copy_register(self, other):
    return self.code.add(ptx.mov(self, other))
    #return self.code.add(ptx.add(self, other, 0))

  def load(self, addr, offset = 0, space='global'):
    if isinstance(offset, (int, long)):
      self.code.add(ptx.ld(space, self, regs.ptxAddress(addr, offset)))
    else:
      temp = self.code.prgm.acquire_register('u64')
      self.code.add(ptx.add(temp, addr, offset))
      self.code.add(ptx.ld(space, self, regs.ptxAddress(temp)))
      self.code.prgm.release_register(temp)

#     # If addr is a constant, create a variable and store the value
#     if not issubclass(type(addr), spe.Type):
#       r_storage = self.code.prgm.acquire_register()
#       addr = Bits(addr, reg = r_storage)
#     else:
#       r_storage = None

#     # If offset is a constant, use lfd, otherwise use lfdx
#     if issubclass(type(offset), spe.Type):
#       self.code.add(ptx.lfdx(self, addr, offset))
#     else:
#       # TODO: Check size of offset to ensure it fits in the immediate field 
#       self.code.add(ptx.lfd(self, addr, offset))

#     if r_storage is not None:
#       self.code.prgm.release_register(r_storage)

#     return

  def store(self, addr, offset = 0, space='global'):
    if isinstance(offset, (int, long)):
      self.code.add(ptx.st(space, regs.ptxAddress(addr, offset), self))
    else:

      temp = self.code.prgm.acquire_register('u64')
      self.code.add(ptx.add(temp, addr, offset))
      self.code.add(ptx.st(space, regs.ptxAddress(temp), self))
      self.code.prgm.release_register(temp)

#     # If addr is a constant, create a variable and store the value
#     if not issubclass(type(addr), spe.Type):
#       r_storage = self.code.prgm.acquire_register()
#       addr = Bits(addr, reg = r_storage)
#     else:
#       r_storage = None

#     # If offset is a constant, use lfd, otherwise use lfdx
#     if issubclass(type(offset), spe.Type):
#       self.code.add(ptx.stfdx(self, addr, offset))
#     else:
#       # TODO: Check size of offset to ensure it fits in the immediate field 
#       self.code.add(ptx.stfd(self, addr, offset))

#     if r_storage is not None:
#       self.code.prgm.release_register(r_storage)

#     return


# ------------------------------
# Floating Point Free Functions
# ------------------------------

class _float_function(object):
  """
  Callable object that performs basic type checking and dispatch for
  floating point operations.
  """

  def __init__(self, name, single_func, double_func):
    self.name = name
    self.single_func = single_func
    self.double_func = double_func
    return

  def __call__(self, *operands, **koperands):
    a = operands[0]
    for op in operands[1:]:
      if op.var_cls != a.var_cls:
        raise Exception('Types for all operands must be the same')
      
    if isinstance(a, SingleFloatType):
      return self.single_func.ex(*operands, **{'type_cls': SingleFloat})
    elif isinstance(a, DoubleFloatType):
      return self.double_func.ex(*operands, **{'type_cls': DoubleFloat})    

    raise Exception(self.name + ' is not implemeneted for ' + str(type(a)))

    
fmadd = _float_function('fmadd', ptx.mad, ptx.mad)
#fmsub = _float_function('fmsub', ptx.fmsubsx, ptx.fmsubx)
#fnmadd = _float_function('fnmadd', ptx.fnmaddsx, ptx.fnmaddx)
#fnmsub = _float_function('fnmsub', ptx.fnmsubsx, ptx.fnmsubx)
fsqrt = _float_function('fsqrt', ptx.sqrt, ptx.sqrt)

# ------------------------------------------------------------
# User Types
# ------------------------------------------------------------

# Type classes are mixed-in with Variables and Expressions to form the
# final user types.  

def make_user_type(name, type_cls, g = None):
  """
  Create a Variable class and an Expression class for a type class.

  This is equivalent to creating two classes and updating the type
  class (except that the Expression class is not added to the global 
  namespace):

    class [name](spe.Variable, type_cls):
      type_cls = type_cls
    class [name]Ex(spe.Expression, type_cls):
      type_cls = type_cls    
    type_class.var_cls = [name]
    type_class.expr_cls = [name]Ex

  type_cls is added to help determine type precedence among Variables
  and Expressions.

  (note: there's probably a better way to model these hierarchies that
   avoids the type_cls, var_cls, expr_cls references.  But, this works
   and keeping explicit references avoids tricky introspection
   operations) 
  """

  # Create the sublasses of Varaible and Expression
  var_cls = type(name, (spe.Variable, type_cls), {'type_cls': type_cls})
  expr_cls = type(name + 'Ex', (spe.Expression, type_cls), {'type_cls': type_cls})

  # Update the type class with references to the variable and
  # expression classes 
  type_cls.var_cls = var_cls
  type_cls.expr_cls = expr_cls

  # Add the Variable class to the global namespace
  if g is None: g = globals()
  g[name] = var_cls

  return


_user_types = ( # name, type class
  ('Bits', BitType),
  ('UnsignedWord', UnsignedWordType),
  ('SignedWord', SignedWordType),
  ('SingleFloat', SingleFloatType),
  ('DoubleFloat', DoubleFloatType)
  )

for t in _user_types:
  make_user_type(*(t + (globals(),)))


# ------------------------------------------------------------
# Unit Tests
# ------------------------------------------------------------

def SimpleTest():
  """
  Just make sure things are working...
  """
  from corepy.arch.ptx.platform import Processor, InstructionStream

  code = InstructionStream()
  proc = Processor()

  # Without active code
  a = SignedWord(11, code)
  b = SignedWord(31, code, reg = code.prgm.acquire_register())
  c = SignedWord(code = code, reg = code.gp_return)

  byte_mask = Bits(0xFF, code)
  code.add(ptx.addi(code.gp_return, 0, 31))

  # c.v = a + SignedWord.cast(b & byte_mask) + 12
  c.v = a + (byte_mask & b) + 12

  if True:
    r = proc.execute(code)
    assert(r == (42 + 12))
  
  # With active code
  code.reset()

  ptx.set_active_code(code)
  
  a = SignedWord(11)
  b = SignedWord(31)
  c = SignedWord(reg = code.gp_return)

  byte_mask = Bits(0xFF)

  c.v = a + (b & byte_mask)

  ptx.set_active_code(None)
  r = proc.execute(code)
  # code.print_code()
  assert(r == 42)
  return


def TestBits():
  from corepy.arch.ptx.platform import Processor, InstructionStream

  code = InstructionStream()
  proc = Processor()

  ptx.set_active_code(code)
  
  b = Bits(0xB0)
  e = Bits(0xE0000)
  a = Bits(0xCA)
  f = Bits(0x5)
  x = Bits(0, reg = code.gp_return)
  
  mask = Bits(0xF)
  byte = Bits(8) # 8 bits
  halfbyte = Bits(4) 

  f.v = (a & mask) ^ f
  x.v = (b << byte) | (e >> byte) | ((a & mask) << halfbyte) | (f | mask)

  r = proc.execute(code)
  assert(r == 0xBEAF)
  return
  
def TestFloatingPoint(float_type):
  from corepy.arch.ptx.platform import Processor, InstructionStream
  
  code = InstructionStream()
  proc = Processor()

  ptx.set_active_code(code)

  x = float_type(1.0)
  y = float_type(2.0)
  z = float_type(3.0)

  a = float_type()
  b = float_type()
  c = float_type()
  d = float_type()

  # Set the size of the float based on whether its double or single
  # Initialize a data array based on float type as well.
  if float_type == SingleFloat:
    float_size = 4
    data = array.array('f', (1.0, 2.0, 3.0, 4.0))
  else:
    float_size = 8
    data = array.array('d', (1.0, 2.0, 3.0, 4.0))

  # Create some data
  addr = data.buffer_info()[0]

  # Load from addr
  a.load(addr) 

  # Load from addr with idx in register
  offset = Bits(float_size)
  b.load(data.buffer_info()[0], offset)

  # Load from addr with constant idx 
  c.load(data.buffer_info()[0], float_size * 2)
  
  # Load from addr with addr as a register
  reg_addr = Bits(addr)
  d.load(reg_addr)
  
  r = float_type(reg = code.fp_return)

  r.v = (x + y) / y

  r.v = fmadd(a, y, z + z) + fnmadd(a, y, z + z) + fmsub(x, y, z) + fnmsub(x, y, z) 
  x.v = -x
  r.v = r + x - x + a + b - c + d - d

  # Store from addr
  a.v = 11.0
  a.store(addr) 

  # Store from addr with idx in register
  offset = Bits(float_size)
  b.v = 12.0
  b.store(data.buffer_info()[0], offset)

  # Store from addr with constant idx
  c.v = 13.0
  c.store(data.buffer_info()[0], float_size * 2)
  
  # Store from addr with addr as a register
  d.v = 14.0
  reg_addr = UnsignedWord(addr)
  reg_addr.v = reg_addr + float_size * 3
  d.store(reg_addr)

  
  r = proc.execute(code, mode='fp')
  assert(r == 0.0)
  assert(data[0] == 11.0)
  assert(data[1] == 12.0)
  assert(data[2] == 13.0)
  assert(data[3] == 14.0)
  
  return

if __name__=='__main__':
  from corepy.arch.ptx.lib.util import RunTest
  RunTest(SimpleTest)
  RunTest(TestFloatingPoint, SingleFloat)
  RunTest(TestFloatingPoint, DoubleFloat)
  RunTest(TestBits)
