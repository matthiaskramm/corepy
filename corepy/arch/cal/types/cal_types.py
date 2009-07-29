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
import corepy.arch.cal.isa as cal
import corepy.arch.cal.lib.cal_extended as calex
import corepy.arch.cal.lib.util as util

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


class CALType(spe.Type):
  def __init__(self, *args, **kargs):
    super(CALType, self).__init__(*args, **kargs)
    self.storage = None
    return
  
  def _get_active_code(self):
    return cal.get_active_code()

  def _set_active_code(self, code):
    return cal.set_active_code(code)
  active_code = property(_get_active_code, _set_active_code)


class BitType(CALType):
  #register_type_id = 'gp'
  register_type_id = 'r'
  array_typecodes = ('c', 'b', 'B', 'h', 'H', 'i', 'I', 'f') # all valid typecodes
  array_typecode  = None # typecode for this class
  literal_types = (int, long, list, tuple, _array_type, _extarray_type)

  # Operators
  __or__ = operator(cal.ior, cast = _upcast)
  or_ = staticmethod(__or__)

  __and__ = operator(cal.iand, cast = _upcast)
  and_ = staticmethod(__and__)

  __xor__ = operator(cal.ixor, cast = _upcast)
  xor = staticmethod(__xor__)

  def copy_register(self, other):
    return self.code.add(cal.mov(self, other))

  def _set_literal_value(self, value):
    if type(value) in (_array_type, _extarray_type):

      if self.array_typecode != value.typecode:
        print "Warning: array typecode does not match variable type - I hope you know what you're doing!"
        
      util.vector_from_array(self.code, self, value)

      self.code.add_storage(value)
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

    self.code.add_storage(self.storage)
    return

class WordType(BitType):
  array_typecode  = 'I' 

  __add__ = operator(cal.iadd, (
    ((int,long), calex.iaddi),
    ))
  add = staticmethod(__add__)

  __mod__ = operator(cal.umod, (
    ((int,long), calex.umodi),
    ))
  mod = staticmethod(__mod__)
  
  __mul__ = operator(cal.umul, (
    ((int,long), calex.umuli),
    ))
  mul = staticmethod(__mul__)
  
  __lshift__ = operator(cal.ishl, (
    ((int, long),  calex.ishli)
    ))
  lshift = staticmethod(__lshift__)

  __rshift__ = operator(cal.ushr, (
    ((int, long), calex.ushri),
    ))
  rshift = staticmethod(__rshift__)

class SignedWordType(WordType):
  array_typecode  = 'i'   

  __add__ = operator(cal.iadd, (
    ((int, long), calex.iaddi),
    ))
  add = staticmethod(__add__)

  __radd__ = operator(cal.iadd, (
    ((int, long), calex.iaddi),
    ), cast = _reversecast)
  radd = staticmethod(__radd__)

  __mul__ = operator(cal.imul, (
    ((int,long), calex.imuli),
    ))
  mul = staticmethod(__mul__)

  __neg__ = operator(cal.inegate)
  neg = staticmethod(__neg__)

  __rshift__ = operator(cal.ishr, (
    ((int, long), calex.ishri),
    ))
  rshift = staticmethod(__rshift__)

  __sub__ = operator(calex.isub, (
    ((int, long), calex.isubi),
    ))
  sub = staticmethod(__sub__)

  __rsub__ = operator(calex.isubf, (
    ((int, long), calex.isubfi),
    ), cast = _reversecast)
  rsub = staticmethod(__rsub__)

class SingleFloatType(CALType):
  #register_type_id = 'gp'
  register_type_id = 'r'
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
      

      self.code.add_storage(value)
      self.code.add_storage(int_value)
      self.storage = self.value
      
      # elif type(self.value) is _numeric_type:
      #   raise Exception('Numeric types not yet supported')

    elif type(value) in (float,):

      if self.array_typecode not in FLOAT_ARRAY_TYPES:
        print "Warning: int does not match variable type - I hope you know what you're doing!"

      ## Convert to bits
      #af = array.array('f', (value,))
      #int_value = array.array('I')
      #int_value.fromstring(af.tostring())
      
      #util.load_word(self.code, self, int_value[0])
      util.load_word(self.code, self, value)
    else:
      # print "Warning: unknown type for %s -> %s, defaulting to 'I'" % (str(self.value), str(type(self.value)))
      # self.typecode = 'I'
      raise Exception("Warning: unknown type for %s -> %s, defaulting to 'I'" % (str(self.value), str(type(self.value))))

    return

  __add__ = operator(cal.add, (
      ((float, int, long), calex.addi),
      ))
  add = staticmethod(__add__)

  __radd__ = operator(calex.radd, (
      ((float, int, long), calex.raddi),
      ))
  radd = staticmethod(__radd__)

  __sub__ = operator(cal.sub, (
      ((float, int, long), calex.subi),
      ))
  sub = staticmethod(__sub__)

  __rsub__ = operator(calex.subf, (
      ((float, int, long), calex.subfi),
      ))
  sub = staticmethod(__sub__)

  __mul__ = operator(cal.mul, (
      ((float, int, long), calex.muli),
      ))
  mul = staticmethod(__mul__)

  __div__ = operator(cal.div, (
      ((float, int, long), calex.divi),
      ))
  div = staticmethod(__div__)

  __rdiv__ = operator(calex.rdiv, (
      ((float, int, long), calex.rdivi),
      ))
  div = staticmethod(__div__)
  
class DoubleFloatType(CALType):
  #register_type_id = 'gp'
  register_type_id = 'r'
  array_typecodes = ('d')
  array_typecode  = 'd' # typecode for this class
  literal_types = (float, list, tuple, array)

  def _set_literal_value(self, value):
    print 'TODO: DoubleFloatType: set_literal_value'

  __add__ = operator(cal.dadd, (
      ((float,int,long), calex.daddi),
      ))
  add = staticmethod(__add__)

  __sub__ = operator(calex.dsub, (
      ((float,int,long), calex.dsubi),
      ))
  sub = staticmethod(__sub__)

  __mul__ = operator(cal.dmul, (
      ((float,int,long), calex.dmuli),
      ))
  mul = staticmethod(__mul__)

  __div__ = operator(cal.ddiv, (
      ((float,int,long), calex.ddivi),
      ))
  mul = staticmethod(__mul__)

  __rdiv__ = operator(calex.drdiv, (
      ((float,int,long), calex.drdivi),
      ))
  mul = staticmethod(__mul__)

_user_types = ( # name, type class
  ('Bits', BitType),
  ('Word', WordType),  
  ('SignedWord', SignedWordType),
  ('SingleFloat', SingleFloatType),
  ('DoubleFloat', DoubleFloatType),  
  )

array_cal_lu = {} # array_typecode: type

for name, type_cls in _user_types:
  # Expanded make_user_type to get namespaces correct
  # make_user_type(*(t + (globals(),)))
  
  var_cls = type(name, (spe.Variable, type_cls), {'type_cls': type_cls})
  expr_cls = type(name + 'Ex', (spe.Expression, type_cls), {'type_cls': type_cls})

  type_cls.var_cls = var_cls
  type_cls.expr_cls = expr_cls
  
  globals()[name] = var_cls
  array_cal_lu[globals()[name].array_typecode] = locals()[name]


# ------------------------------------------------------------
# Unit Tests
# ------------------------------------------------------------

def TestBits():
  x = Bits(0)
  y = Bits(0)
  z = Bits(0)

  z.v = (x | y) & (x ^ y)

  return

def TestSignedWord():
  x = SignedWord(0)
  y = SignedWord(0)
  z = SignedWord(0)

  z.v = x + y
  z.v = x + 3
  z.v = x + y

  z.v = x
  z.v = x - 12
  z.v = z - x
  z.v = 12 - x
  return

def _bits_to_float(bits):
  ab = array.array('I', (bits,))
  af = array.array('f')
  af.fromstring(ab.tostring())
  return  af[0]

def TestFloatScalar():

  x = SingleFloat(1.0)
  y = SingleFloat(2.0)
  z = SingleFloat()

  z.v = x + y
  z.v = x - y
  z.v = x + 5.0
  z.v = x - 5.0
  z.v = 500.0 - x
  z.v = x/y
  z.v = x/5
  z.v = 5/x

  return


def RunTest(test):
  import corepy.arch.cal.platform as env

  prgm = env.Program()
  code = prgm.get_stream()
  cal.set_active_code(code)

  test()

  prgm.add(code)
  prgm.print_code()
  proc = env.Processor(0)
  proc.execute(prgm, (0, 0, 128, 128))
  return


if __name__=='__main__':
  #RunTest(TestBits)
  #RunTest(TestSignedWord)  
  RunTest(TestFloatScalar)
  #TestFloatArray()
