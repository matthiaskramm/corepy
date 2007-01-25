
import array

import corepy.arch.spu.isa as spu
import corepy.arch.spu.lib.spu_extended as spuex
import corepy.arch.spu.lib.util as util
import corepy.spre.spe as spe

from corepy.spre.syn_util import most_specific, make_user_type

_array_type   = type(array.array('I', [1]))
INT_ARRAY_TYPES = ('b', 'h', 'i', 'B', 'H', 'I')
INT_ARRAY_SIZES = {'b':16, 'h':8, 'i':4, 'B':16, 'H':8, 'I':4}
INT_SIZES       = {'b':1,  'c':1, 'h':2, 'i':4, 'B':1,  'H':2, 'I':4}

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


def _upcast(a, b, inst):
    return inst.ex(a, b, type_cls = most_specific(a, b))


class BitType(SPUType):
  register_type_id = 'gp'
  array_typecodes = ('c', 'b', 'B', 'h', 'H', 'i', 'I', 'f') # all valid typecodes
  array_typecode  = None # typecode for this class
  literal_types = (int,long, list, tuple, _array_type)

  def __or__(self, other):
    if isinstance(other, BitType):
      return _upcast(self, other, spu.or_)
  or_ = staticmethod(__or__)

  def __and__(self, other):
    if isinstance(other, BitType):
      return _upcast(self, other, spu.and_)
  and_ = staticmethod(__and__)

  def __xor__(self, other):
    if isinstance(other, BitType):
      return _upcast(self, other, spu.xor)
  xor = staticmethod(__xor__)

  def copy_register(self, other):
    return self.code.add(spu.ai(self, other, 0))

  def _set_literal_value(self, value):
    if type(value) is _array_type:

      if self.array_typecode != value.typecode:
        print "Warning: array typecode does not match variable type - I hope you know what you're doing!"
        
      util.vector_from_array(self.code, self, value)

      self.code.add_storage(value)
      self.storage = self.value
      
      # elif type(self.value) is _numeric_type:
      #   raise Exception('Numeric types not yet supported')

    elif type(self.value) is int:

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


class HalfwordType(BitType):
  array_typecode  = 'H'
  
  def __lshift__(self, amount):
    if isinstance(amount, (Halfword, Word)):
      return spu.shlh.ex(self, amount, type_cls = self.var_cls)
    elif isinstance(amount, self.literal_types):
      return spu.shlhi.ex(self, amount, type_cls = self.var_cls)
  lshift = staticmethod(__lshift__)


class SignedHalfwordType(HalfwordType):
  array_typecode  = 'h' 


class WordType(BitType):
  array_typecode  = 'I' 

  def __lshift__(self, amount):
    if issubclass(type(amount), (HalfwordType, WordType)):
      return spu.shl.ex(self, amount, type_cls = self.var_cls)
    elif isinstance(amount, self.literal_types):
      return spu.shli.ex(self, amount, type_cls = self.var_cls)
  lshift = staticmethod(__lshift__)

  def __rshift__(self, amount):
    if isinstance(amount, (Halfword, Word)):
      return spuex.shr.ex(self, amount, type_cls = self.var_cls)
    # elif isinstance(amount, int):
    #  return spu.shli.ex(self, amount, type_cls = self.var_cls)
  lshift = staticmethod(__lshift__)


class SignedWordType(WordType):
  array_typecode  = 'i'   

  def __add__(self, other):
    if isinstance(other, SignedWord):
      return spu.a.ex(self, other, type_cls = self.var_cls)
    elif isinstance(other, int) and (-512 < other < 512):
      return spu.ai.ex(self, other, type_cls = self.var_cls)
  add = staticmethod(__add__)

  def __sub__(self, other): 
    # RD = RB - RA
    if isinstance(other, SignedWord):
      return spu.sf.ex(self, other, type_cls = self.var_cls)
    elif isinstance(other, int):
      return spu.ai.ex(self, -other, type_cls = self.var_cls)
  sub = staticmethod(__sub__)


class SingleFloatType(SPUType):
  register_type_id = 'gp'
  array_typecodes = ('f')
  array_typecode  = 'f' # typecode for this class
  literal_types = (float, list, tuple, array)

  def _set_literal_value(self, value):
    print 'TODO: SingleFloatType: set_literal_value'

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

for t in _user_types:
  make_user_type(*(t + (globals(),)))
  array_spu_lu[globals()[t[0]].array_typecode] = globals()[t[0]]


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

def RunTest(test):
  from corepy.arch.spu.platform import InstructionStream, Processor

  code = InstructionStream()
  spu.set_active_code(code)

  test()
  
  code.print_code()
  proc = Processor()
  proc.execute(code)
  return


if __name__=='__main__':
  RunTest(TestBits)
  RunTest(TestHalfword)
  RunTest(TestWord)  
  
