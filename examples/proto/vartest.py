# Copyright 2006-2007 The Trustees of Indiana University.

# This software is available for evaluation purposes only.  It may not be
# redistirubted or used for any other purposes without express written
# permission from the authors.

# Authors:
#   Christopher Mueller (chemuell@cs.indiana.edu)
#   Andrew Lumsdaine    (lums@cs.indiana.edu)

import arch.spu as spu
import spre.spe as spe

class Byte_t: pass
class Halfword_t: pass
class Word_t: pass

class Literal(object):
  def __init__(self, value):
    self.value = self.validate(value)
    return

  def _cast(cls, value):
    if isinstance(value, cls):
      return value
    else:
      return cls(value)
  cast = classmethod(_cast)
    
class I10(Literal):
  def validate(self, value):
    if not (issubclass(type(value), Literal) or type(value) is int):
      raise TypeError('Cannot convert %s to %s' % (type(value), type(self)))

    if issubclass(type(value), Literal):
      value = value.value

    if (value & 0x3FF) == value:
      return value
    else:
      print 'Warning: %d does not fit in 10 bits' 
      return value
    return
    
class I7(Literal):
  def validate(self, value):
    if not (issubclass(type(value), Literal) or type(value) is int):
      raise TypeError('Cannot convert %s to %s' % (type(value), type(self)))

    if issubclass(type(value), Literal):
      value = value.value

    if (value & 0x7F) == value:
      return value
    else:
      print 'Warning: %d does not fit in 10 bits' 
      return value
    return
    

class Variable(object):
  def __init__(self, reg, value = None, code = None):
    self.value = value
    self.reg = reg
    self.code = code
    if value is not None:
      self.v = self.value
    return

  def get_value(self): return self.value
  def _set_value(self, v): self.set_value(v)
  v = property(get_value, _set_value)

  def set_value(self, value):
    value.eval(reg = self.reg)
    return

  def _cast(cls, other):
    newinst = None

    if isinstance(other, spe.Expression):
      print 'Casting to:', cls.expr_cls
      newinst = cls.expr_cls(other._inst, *other._operands, **other._koperands)
      
    elif isinstance(other, Variable):
      newinst = cls(other.reg, code = self.code)
      newinst.value = value

    return newinst

  cast = classmethod(_cast)
    
class BitOps(object):
  def __init__(self, im_type = Byte_t):
    self.im_type = im_type
    return
  
  def __or__(self, other):
    if isinstance(other, Bits):
      return spu._or.ex(a, other)
    elif issubclass(other, Literal) or type(other) is int:
      if self.im_type == Byte_t:
        return spu.orbi.ex(a, I10.cast(other))
      elif self.im_type == Halfword_t:
        return spu.orhi.ex(a, I10.cast(other))
      elif self.im_type == Word_t:
        return spu.ori.ex(a, I10.cast(other))

    return TypeError('Unsupported type for bitwise or: %s' % (str(type(other))))

  or_ = staticmethod(__or__)

class SignedHalfwordOps(BitOps):
  def __add__(self, other):
    if type(other) is SignedWord:
      return spu.ah.ex(self, other)
    elif issubclass(other, Literal) is Literal or type(other) is int:
      return spu.ahi.ex(self, I10.cast(other))

    raise TypeError('Unsupported type for SingedWord add: %s' % (str(type(other))))

  add = staticmethod(__add__)
  
class SignedWordOps(BitOps):
  def __add__(self, other):
    retVal = None
    if isinstance(other, SignedWord):
      retVal = spu.a.ex(self, other)
    elif isinstance(other, (Literal, int)):
      retVal = spu.ai.ex(self, I10.cast(other))

    if retVal is not None:
      retVal.code = self.code

    if retVal is None:
      raise TypeError('Unsupported type for SingedWord add: %s' % (str(type(other))))

    return retVal

  add = staticmethod(__add__)

class Bits(Variable, BitOps): pass
class SignedHalfword(Variable, SignedHalfwordOps): pass
class SignedWord(Variable, SignedWordOps): pass
class Quadword(Variable, BitOps): pass

class SignedHalfwordEx(spe.Expression, SignedHalfword): pass
class SignedWordEx(spe.Expression, SignedWord): pass
class QuadwordEx(spe.Expression, Bits): pass

SignedWord.expr_cls = SignedWordEx

spu.a.expr_cls = SignedWordEx
spu.ai.expr_cls = SignedWordEx
  
def rotate(a, b, by_bytes = False):
  retVal = None
  if (isinstance(a, (Variable, spe.Expression)) and isinstance(b, (Variable, spe.Expression))):
    if isinstance(b, SignedWord):
      if isinstance(a, SignedWord):
        retVal = SignedWordEx(spu.rot, a, b)
      elif isinstance(a, SignedHalfword):
        retVal = SignedHalfwordEx(spu.roth, a, b)
      elif isinstance(a, Quadword and not by_bytes):
        retVal = QuadwordEx(spu.rotqbi, a, b)
      elif isinstance(a, Quadword):
        retVal = QuadwordEx(spu.rotqby.ex, a, b)
  elif ((isinstance(a, Variable) and isinstance(b, Literal)) or
        (isinstance(a, Variable) and isinstance(b, int))):
    if isinstance(b, int) or b.size_type is not I7: b = I7.cast(b)

    if isinstance(a, SignedWord):
      retVal = SignedWordEx(spu.rot, a, b)
    elif isinstance(a, SignedHalfword):
      retVal = SignedHalfwordEx(spu.roth, a, b)
    elif isinstance(a, Quadword and not by_bytes):
      retVal = QuadwordEx(spu.rotqbi, a, b)
    elif isinstance(a, Quadword):
      retVal = QuadwordEx(spu.rotqby, a, b)

  if retVal is not None:
    retVal.code = a.code
  else:
    raise TypeError("Incompatible types for rotate: %s %s" % (type(a), type(b)))
  return retVal

class InstructionStream:
  def __init__(self): self.reg = 10
  def acquire_register(self):
    self.reg += 1
    return self.reg
    
code = InstructionStream()

a = SignedWord(reg = 1, code = code)
b = SignedWord(reg = 2, code = code)
c = SignedWord(reg = 3, code = code)
d = SignedHalfword(reg = 4, code = code)

c.v = a + b + 1 + SignedWord.cast(rotate(d, 1)) + SignedWordOps.add(c, 1)
