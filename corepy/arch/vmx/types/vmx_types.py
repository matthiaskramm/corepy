import array

import corepy.arch.ppc.isa as ppc
import corepy.arch.vmx.isa as vmx
import corepy.spre.spe as spe
import corepy.arch.ppc.lib.util as util

_array_type   = type(array.array('I', [1]))
INT_ARRAY_SIZES = {'b':16, 'h':8, 'i':4, 'B':16, 'H':8, 'I':4}

def _most_specific(a, b, default = None):
  """
  If a and b are from the same hierarcy, return the more specific of
  [type(a), type(b)], or the default type if they are from different
  hierarchies. If default is None, return type(a), or type(b) if a
  does not have a type_cls
  """
  if (hasattr(a, 'type_cls') and hasattr(a, 'type_cls')):
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


class VMXType(spe.Type):
  def __init__(self, *args, **kargs):
    super(VMXType, self).__init__(*args, **kargs)
    self.storage = None
    return
  
  def _get_active_code(self):
    return ppc.get_active_code()

  def _set_active_code(self, code):
    return ppc.set_active_code(code)
  active_code = property(_get_active_code, _set_active_code)


class BitType(VMXType):
  register_type_id = 'vector'
  array_typecodes = ('c', 'b', 'B', 'h', 'H', 'i', 'I', 'f') # all valid typecodes
  array_typecode  = None # typecode for this class
  literal_types = (int,long, list, tuple, array)

  def _upcast(self, other, inst):
    return inst.ex(self, other, type_cls = _most_specific(self, other))

#   def __and__(self, other):
#     if isinstance(other, BitType):
#       return self._upcast(other, ppc.andx)
#     elif isinstance(other, _int_literals):
#       return ppc.andi.ex(self, other, type_cls = self.var_cls)
#     raise Exception('__and__ not implemented for %s and %s' % (type(self), type(other)))    

#   and_ = staticmethod(__and__)

#   def __lshift__(self, other):
#     if isinstance(other, BitType):
#       return ppc.slwx.ex(self, other, type_cls = self.var_cls)
#     raise Exception('__lshift__ not implemented for %s and %s' % (type(self), type(other)))    
#   lshift = staticmethod(__lshift__)

#   def __rshift__(self, other):
#     if isinstance(other, BitType):
#       return ppc.srwx.ex(self, other, type_cls = self.var_cls)
#     raise Exception('__rshift__ not implemented for %s and %s' % (type(self), type(other)))    
#   rshift = staticmethod(__rshift__)

#   def __or__(self, other):
#     if isinstance(other, BitType):
#       return self._upcast(other, ppc.orx)
#     elif isinstance(other, _int_literals):
#       return ppc.ori.ex(self, other, type_cls = self.var_cls)
#     raise Exception('__or__ not implemented for %s and %s' % (type(self), type(other)))    
#   or_ = staticmethod(__or__)

#   def __xor__(self, other):
#     if isinstance(other, BitType):
#       return self._upcast(other, ppc.xorx)
#     elif isinstance(other, _int_literals):
#       return ppc.xori.ex(self, other, type_cls = self.var_cls)
#     raise Exception('__xor__ not implemented for %s and %s' % (type(self), type(other)))    
#  xor = staticmethod(__xor__)

  def _set_literal_value(self, value):
    if type(value) is _array_type:
      if value.typecode not in self.array_typecodes:
        raise Exception("Array typecode '%s' is not supported" % (value.typecode,))

      if len(value) < INT_ARRAY_SIZES[self.array_typecode]:
        print 'Warning: Variable array initializer has fewer elements than the corresponding vector: %d < %d' % (
          len(value), INT_ARRAY_SIZES[self.array_typecode])
      util.load_vector(self.code, self.reg, value.buffer_info()[0])
      self.storage = value

    elif type(value) in (list, tuple):
      if len(value) < INT_ARRAY_SIZES[self.array_typecode]:
        print 'Warning: Variable %s initializer has fewer elements than the corresponding vector: %d < %d' % (
          type(value), len(value), INT_ARRAY_SIZES[self.array_typecode])
      
      storage = array.array(self.array_typecode, value)
      util.load_vector(self.code, self.reg, storage.buffer_info()[0])
      self.storage = storage
      
    elif type(value) in self.literal_types:
      if (value & 0x1F) == value and isinstance(self, (SignedByteType, SignedHalfwordType, SignedWordType)):
        # Use the splat instructions
        if isinstance(self, SignedByteType):
          self.code.add(vmx.vspltisb(self.reg, value))
        elif isinstance(self, SignedHalfwordType):
          self.code.add(vmx.vspltish(self.reg, value))
        elif isinstance(self, SignedWordType):
          self.code.add(vmx.vspltisw(self.reg, value))
        else:
          raise Exception('Unsupported typecode for vector literal splat: ' + str(type(self)))
      else:
        splat = [self.value for i in range(INT_ARRAY_SIZES[self.array_typecode])]
        vsplat = array.array(self.array_typecode, splat)

        util.load_vector(self.code, self.reg, vsplat.buffer_info()[0])
        self.code.add_storage(vsplat)
        self.storage = vsplat
        
    self.value = value

    if self.storage is not None:
      self.code.add_storage(self.storage)

    return


class ByteType(BitType):
  array_typecode  = 'B'

class SignedByteType(BitType):
  array_typecode  = 'b'

class HalfwordType(BitType):
  array_typecode  = 'H'

class SignedHalfwordType(BitType):
  array_typecode  = 'h'

class WordType(BitType):
  array_typecode  = 'I'

class SignedWordType(BitType):
  array_typecode  = 'i'

  
class SingleFloatType(VMXType):
  register_type_id = 'fp'
  array_typecode  = 'f'
  literal_types = (float,)

#   def __abs__(self):
#     return ppc.fabsx.ex(self, type_cls = self.var_cls)
#   abs = staticmethod(__abs__)
  
#   def __add__(self, other):
#     if isinstance(other, SingleFloatType):
#       return ppc.faddsx.ex(self, other, type_cls = self.var_cls)
#     raise Exception('__add__ not implemented for %s and %s' % (type(self), type(other)))        
#   add = staticmethod(__add__)
  
#   def __div__(self, other):
#     if isinstance(other, SingleFloatType):
#       return ppc.fdivsx.ex(self, other, type_cls = self.var_cls)
#     raise Exception('__div__ not implemented for %s and %s' % (type(self), type(other)))    
#   div = staticmethod(__div__)

#   def __mul__(self, other):
#     if isinstance(other, SingleFloatType):
#       return ppc.fmulsx.ex(self, other, type_cls = self.var_cls)
#     raise Exception('__mul__ not implemented for %s and %s' % (type(self), type(other)))
#   mul = staticmethod(__mul__)

#   def __neg__(self):
#     return ppc.fnegx.ex(self, type_cls = self.var_cls)
#   neg = staticmethod(__neg__)

#   def __sub__(self, other):
#     if isinstance(other, SingleFloatType):
#       return ppc.fsubsx.ex(self, other, type_cls = self.var_cls)
#     raise Exception('__sub__ not implemented for %s and %s' % (type(self), type(other)))
#   sub = staticmethod(__sub__)

  def _set_literal_value(self, value):
    BitType._set_literal_value(self, value)
    return

_user_types = ( # name, type class
  ('Bits', BitType),
  ('Byte', ByteType),
  ('SignedByte', SignedByteType),
  ('Halfword', HalfwordType),    
  ('SignedHalfword', SignedHalfwordType),
  ('Word', WordType),  
  ('SignedWord', SignedWordType),
  ('SingleFloat', SingleFloatType),
  )

for t in _user_types:
  util.make_user_type(*(t + (globals(),)))


# ------------------------------------------------------------
# Unit Tests
# ------------------------------------------------------------

def TestLiterals():
  from corepy.arch.ppc.platform import Processor, InstructionStream
  code = InstructionStream()
  proc = Processor()

  ppc.set_active_code(code)
  vmx.set_active_code(code)

  zero = Bits.cast(SignedByte(0))

  target = Bits()

  # Signed versions use splat, unsigned arrays
  b  = Byte(2)
  sb = SignedByte(-2)
  vmx.vaddsbs(b, b, sb)

  h  = Halfword(9999)
  sh = SignedHalfword(-9999)
  vmx.vaddshs(h, h, sh)

  w  = Word(99999)
  sw = SignedWord(-99999)
  vmx.vaddsws(w, w, sw)

  # Combine the results (should be [0,0,0,0])
  vmx.vor(target, b, h)
  vmx.vor(target, target, w)

  # Array initializers
  b  = Byte(range(16))
  sb = SignedByte(range(16))
  vmx.vsubsbs(b, b, sb)
  vmx.vor(target, target, b)
  
  h  = Halfword([9999,9998,9997,9996,9995,9994,9993,9992])
  sh = SignedHalfword([9999,9998,9997,9996,9995,9994,9993,9992])
  vmx.vsubshs(h, h, sh)
  vmx.vor(target, target, h)
  
  w  = Word([99999,99998,99997,99996])
  sw = SignedWord([99999,99998,99997,99996])
  vmx.vsubsws(w, w, sw)

  target.v = vmx.vor.ex(target, w)
  
  result = array.array('I', [42,42,42,42])
  r_addr = code.acquire_register()
  util.load_word(code, r_addr, result.buffer_info()[0])

  vmx.stvx(target, 0, r_addr)

  ppc.set_active_code(None)
  vmx.set_active_code(None)
  r = proc.execute(code)
  for i in result:
    assert(i == 0)
  # for i in result: print '%08X' % i,
  # print
  
  return

if __name__=='__main__':
  from corepy.arch.ppc.lib.util import RunTest
  RunTest(TestLiterals)