
import array

import corepy.arch.spu.isa as spu
import corepy.arch.spu.lib.spu_extended as spuex
import corepy.spre.spe as spe

from corepy.spre.syn_util import most_specific, make_user_type

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
  literal_types = (int,long, list, tuple, array)

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

  def _set_literal_value(self, value):
    print 'TODO: BitType: set_literal_value'

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
    if isinstance(amount, (Halfword, Word)):
      return spu.shl.ex(self, amount, type_cls = self.var_cls)
    elif isinstance(amount, self.literal_types):
      return spu.shli.ex(self, amount, type_cls = self.var_cls)
  lshift = staticmethod(__lshift__)

  def __rshift__(self, amount):
    if isinstance(amount, (Halfword, Word)):
      return spuex.shr.ex(self, amount, type_cls = self.var_cls)
    # elif isinstance(amount, self.literal_types):
    #    return spu.shli.ex(self, amount, type_cls = self.var_cls)
  lshift = staticmethod(__lshift__)


class SignedWordType(WordType):
  array_typecode  = 'i'   

  def __add__(self, other):
    if isinstance(other, SignedWord):
      return spu.a.ex(self, other, type_cls = self.var_cls)
  add = staticmethod(__add__)

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
  
