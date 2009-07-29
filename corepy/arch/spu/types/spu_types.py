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

import corepy.lib.extarray as extarray
import corepy.arch.spu.isa as spu
import corepy.arch.spu.lib.spu_extended as spuex
import corepy.arch.spu.lib.util as util
import corepy.spre.spe as spe

from corepy.spre.syn_util import most_specific, make_user_type

_array_type   = type(array.array('I', [1]))
_extarray_type   = type(extarray.extarray('I', [1]))
INT_ARRAY_TYPES = ('b', 'h', 'i', 'B', 'H', 'I')
INT_ARRAY_SIZES = {'b':16, 'h':8, 'i':4, 'B':16, 'H':8, 'I':4}
INT_SIZES       = {'b':1,  'c':1, 'h':2, 'i':4, 'B':1,  'H':2, 'I':4}

FLOAT_ARRAY_TYPES = ('f', 'd')
FLOAT_ARRAY_SIZES = {'f':4, 'd':2}
FLOAT_SIZES       = {'f':4,  'd':8}


def _upcast(a, b, inst):
  return inst.ex(a, b, type_cls = most_specific(a, b))

def _firstcast(a, b, inst):
  return inst.ex(a, b, type_cls = a.var_cls)

def _reversecast(a, b, inst):
  return inst.ex(b, a, type_cls = a.var_cls)

class operator(object):
  """
  Dispatch to an instruction based on the type of other.

  Type map is a list of pairs containing a type or tuples of types and
  the instruction to call when type(other) is a subclass of a type in
  the list.

  This class is used to form the operators for the Type classes.

  See the following link for a description of how __get__ works:

  http://users.rcn.com/python/download/Descriptor.htm#functions-and-methods
  """

  def __init__(self, default, type_map = [], cast = _firstcast):
    object.__init__(self)
    self.default = default
    self.type_map = type_map
    self.cast = cast
    return

  def __get__(self, obj, objtype = None):
    """
    Get is called with the bound object as obj.
    """
    def invoke(other):
      # Operate on type classes?
      t_other = type(other)
      if hasattr(other, "type_cls"):
        t_other = other.type_cls

      t_obj   = type(obj)
      if hasattr(obj, "type_cls"):
        t_obj = obj.type_cls
      
      # If obj and other are the same type, use the default instruction
      if t_obj == t_other: return self.cast(obj, other, self.default)
      # if isinstance(obj, t_other): return self.cast(obj, other, self.default)
                            
      # Otherwise, search the type map for a match
      for types, inst in self.type_map:
        if issubclass(t_other, types):
          return self.cast(obj, other, inst)

      # If there is no match, see if other is a subclass of obj
      # (do this after the search in case the subclass overrides the behavior)
      if issubclass(t_other, t_obj):
        return self.cast(obj, other, self.default)

      raise Exception('Unable to determine proper type cast from ' + str(t_other) + ' to ' + str(type(obj)))
      return

    return invoke


class SPUType(spe.Type):
  def __init__(self, *args, **kargs):
    super(SPUType, self).__init__(*args, **kargs)
    self.storage = None
    return
  
  def _get_active_code(self):
    return spu.get_active_code()

  def _set_active_code(self, code):
    return spu.set_active_code(code)
  active_code = property(_get_active_code, _set_active_code)


class BitType(SPUType):
  register_type_id = 'gp'
  array_typecodes = ('c', 'b', 'B', 'h', 'H', 'i', 'I', 'f') # all valid typecodes
  array_typecode  = None # typecode for this class
  literal_types = (int,long, list, tuple, _array_type, _extarray_type)

  # Operators
  __or__ = operator(spu.or_, cast = _upcast)
  or_ = staticmethod(__or__)

  __and__ = operator(spu.and_, cast = _upcast)
  and_ = staticmethod(__and__)

  __xor__ = operator(spu.xor, cast = _upcast)
  xor = staticmethod(__xor__)


  def copy_register(self, other):
    return self.code.add(spu.ai(self, other, 0))

  def _set_literal_value(self, value):
    if type(value) in (_array_type, _extarray_type):

      if self.array_typecode != value.typecode:
        print "Warning: array typecode does not match variable type - I hope you know what you're doing!"
        
      util.vector_from_array(self.code, self, value)

      self.code.prgm.add_storage(value)
      self.storage = self.value
      
      # elif type(self.value) is _numeric_type:
      #   raise Exception('Numeric types not yet supported')

    elif type(value) in (int, long):

      if self.array_typecode not in INT_ARRAY_TYPES:
        print "Warning: int does not match variable type - I hope you know what you're doing!"

      util.load_word(self.code, self, value)
    else:
      # print "Warning: unknown type for %s -> %s, defaulting to 'I'" % (str(self.value), str(type(self.value)))
      # self.typecode = 'I'
      raise Exception("Warning: unknown type for %s -> %s, defaulting to 'I'" % (str(self.value), str(type(self.value))))
    
    if self.array_typecode is not None and INT_ARRAY_SIZES[self.array_typecode] != 4:
      print "Warning: Only 4-byte integers are supported for spu variables from arrays"

    self.code.prgm.add_storage(self.storage)
    return


class HalfwordType(BitType):
  array_typecode  = 'H'

class SignedHalfwordType(HalfwordType):
  array_typecode  = 'h' 


class WordType(BitType):
  array_typecode  = 'I' 

  __add__ = operator(spu.a, (
    (int, spuex.a_immediate),
    ))
  add = staticmethod(__add__)

  __lshift__ = operator(spu.shl, (
    (HalfwordType, spu.shl),
    ((int, long),  spu.shli)
    ))
  lshift = staticmethod(__lshift__)

  __rshift__ = operator(spuex.shr, (
    (HalfwordType, spuex.shr),
    ))
  rshift = staticmethod(__rshift__)

  __gt__ = operator(spu.cgt, (
    ((int, long), spuex.cgt_immediate),
    ))
  gt = staticmethod(__gt__)

  __lt__ = operator(spuex.lt, (
    ((int, long), spuex.lti),
    ))
  lt = staticmethod(__lt__)

  __eq__ = operator(spu.ceq, (
    ((int, long), spuex.ceq_immediate),
    ))
  eq = staticmethod(__eq__)


class SignedWordType(WordType):
  array_typecode  = 'i'   

  __add__ = operator(spu.a, (
    ((int, long), spuex.a_immediate),
    ))
  add = staticmethod(__add__)


  __radd__ = operator(spu.a, (
    ((int, long), spuex.a_immediate),
    ), cast = _reversecast)
  radd = staticmethod(__radd__)

  __sub__ = operator(spuex.sub, (
    ((int, long), spuex.subi),
    ))
  sub = staticmethod(__sub__)

  __rsub__ = operator(spu.sf, (
    ((int, long), spu.sfi),
    ), cast = _reversecast)
  rsub = staticmethod(__rsub__)


# Extra Operators
HalfwordType.__lshift__ = operator(spu.shlh, (
  (WordType,    spu.shlh),
  ((int, long), spu.shlhi)
  ))
HalfwordType.lshift = staticmethod(HalfwordType.__lshift__)


class SingleFloatType(SPUType):
  register_type_id = 'gp'
  array_typecodes = ('f')
  array_typecode  = 'f' # typecode for this class
  literal_types = (float, list, tuple, array)

  def _set_literal_value(self, value):

    # Convert lists and tuples to 'f' arrays
    if isinstance(value, (list, tuple)):
      value = array.array(self.array_typecode, value)
    
    if type(value) in (_array_type, _extarray_type):

      if self.array_typecode != value.typecode:
        print "Warning: array typecode does not match variable type - I hope you know what you're doing!"

      # Convert the float array to an integer array to prevent Python from
      # improperly casting floats to ints
      int_value = array.array('I')
      int_value.fromstring(value.tostring())
      
      util.vector_from_array(self.code, self, int_value)

      self.code.prgm.add_storage(value)
      self.code.prgm.add_storage(int_value)
      self.storage = self.value
      
      # elif type(self.value) is _numeric_type:
      #   raise Exception('Numeric types not yet supported')

    elif type(value) in (float,):

      if self.array_typecode not in FLOAT_ARRAY_TYPES:
        print "Warning: int does not match variable type - I hope you know what you're doing!"

      # Convert to bits
      af = array.array('f', (value,))
      int_value = array.array('I')
      int_value.fromstring(af.tostring())
      
      util.load_word(self.code, self, int_value[0])
    else:
      # print "Warning: unknown type for %s -> %s, defaulting to 'I'" % (str(self.value), str(type(self.value)))
      # self.typecode = 'I'
      raise Exception("Warning: unknown type for %s -> %s, defaulting to 'I'" % (str(self.value), str(type(self.value))))

    return


class DoubleFloatType(SPUType):
  register_type_id = 'gp'
  array_typecodes = ('d')
  array_typecode  = 'd' # typecode for this class
  literal_types = (float, list, tuple, array)

  def _set_literal_value(self, value):
    print 'TODO: DoubleFloatType: set_literal_value'


_user_types = ( # name, type class
  ('Bits', BitType),
#  ('Byte', ByteType),
#  ('SignedByte', SignedByteType),
  ('Halfword', HalfwordType),    
  ('SignedHalfword', SignedHalfwordType),
  ('Word', WordType),  
  ('SignedWord', SignedWordType),
  ('SingleFloat', SingleFloatType),
  ('DoubleFloat', DoubleFloatType),  
  )

array_spu_lu = {} # array_typecode: type

for name, type_cls in _user_types:
  # Expanded make_user_type to get namespaces correct
  # make_user_type(*(t + (globals(),)))
  
  var_cls = type(name, (spe.Variable, type_cls), {'type_cls': type_cls})
  expr_cls = type(name + 'Ex', (spe.Expression, type_cls), {'type_cls': type_cls})

  type_cls.var_cls = var_cls
  type_cls.expr_cls = expr_cls
  
  globals()[name] = var_cls
  array_spu_lu[globals()[name].array_typecode] = locals()[name]


# ------------------------------------------------------------
# Unit Tests
# ------------------------------------------------------------

def TestBits():
  x = Bits(0)
  y = Bits(0)
  z = Bits(0)

  z.v = (x | y) & (x ^ y)

  return

def TestHalfword():
  x = Halfword(0)
  y = Halfword(0)
  z = Halfword(0)

  z.v = x << y
  z.v = x << 3  
  
  return


def TestWord():
  x = Word(0)
  y = Word(0)
  z = Word(0)

  z.v = x << y
  z.v = x << 3
  z.v = x >> y    

  z.v = x
  return

def _bits_to_float(bits):
  ab = array.array('I', (bits,))
  af = array.array('f')
  af.fromstring(ab.tostring())
  return  af[0]

def TestFloatScalar():
  from corepy.arch.spu.platform import InstructionStream, Processor
  import corepy.arch.spu.lib.dma as dma
  import corepy.arch.spu.platform as env

  prgm = env.Program()
  code = prgm.get_stream()
  spu.set_active_code(code)

  x = SingleFloat(1.0)
  y = SingleFloat(2.0)
  r = SingleFloat(0.0, reg = code.fp_return)

  r.v = spu.fa.ex(x, y)

  prgm.add(code)
  proc = env.Processor()
  result = proc.execute(prgm, mode='fp')
  assert(result == (1.0 + 2.0))
  
  return


def TestFloatArray():
  from corepy.arch.spu.platform import InstructionStream, Processor
  import corepy.arch.spu.lib.dma as dma
  import corepy.arch.spu.platform as env

  prgm = env.Program()
  code = prgm.get_stream()
  spu.set_active_code(code)

  x = SingleFloat([1.0, 2.0, 3.0, 4.0])
  y = SingleFloat([0.5, 1.5, 2.5, 3.5])
  sum = SingleFloat(0.0)

  sum.v = spu.fa.ex(x, y)

  r = SingleFloat([0.0, 0.0, 0.0, 0.0], reg = code.fp_return)

  for i in range(4):
    r.v = spu.fa.ex(sum, r)
    spu.rotqbyi(sum, sum, 4)
 
  prgm.add(code) 
  proc = env.Processor()
  result = proc.execute(prgm, mode='fp')

  x_test = array.array('f', [1.0, 2.0, 3.0, 4.0])
  y_test = array.array('f', [0.5, 1.5, 2.5, 3.5])
  r_test = 0.0
  for i in range(4):
    r_test += x_test[i] + y_test[i]

  assert(result == r_test)
  
  return

def RunTest(test):
  import corepy.arch.spu.platform as env
  #from corepy.arch.spu.platform import InstructionStream, Processor

  prgm = env.Program()
  code = prgm.get_stream()
  spu.set_active_code(code)

  test()

  prgm.add(code)
  prgm.print_code()
  proc = env.Processor()
  proc.execute(prgm)
  return


if __name__=='__main__':
  RunTest(TestBits)
  RunTest(TestHalfword)
  RunTest(TestWord)  
  TestFloatScalar()
  TestFloatArray()
