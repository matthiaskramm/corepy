# Copyright (c) 2006-2008 The Trustees of Indiana University.                   
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


# Placeholders for unfinished portions...

import spre.spe
import arch.spu as spu

class SPURegister(object): pass
  
class SPUInstructionStream(spre.spe.InstructionStream):
  RegisterFiles = ((SPURegister, [spre.spe.Register(SPURegister, i) for i in range(0,128)]),)

  exec_module   = None
  align         = 16 # 128 is max efficiency, 16 is what array currently does
  instruction_type  = 'I'
  default_register_type = SPURegister

code = SPUInstructionStream()

class I10(object):
  def _cast(value, type_type = None):
    # TODO: This is not quite right...make sure type_type checks value, too.
    if type(value) is int:
      return spre.spe.Immediate(value, I10, type_type)
  cast = staticmethod(_cast)

class Byte_t(object): pass
class Halfword_t(object): pass
class Word_t(object): pass
class Quadword_t(object): pass


_integer_add ={
  Word_t:     {Word_t: spu.a, I10: spu.ai, int: spu.ai},
  Halfword_t: {Halfword_t: spu.ah, I10: spu.ahi, int: spu.ahi}
  }

class SignedInteger(object):
  instructions = {
    'add': _integer_add
    }

class UnsignedInteger(object):
  instructions = {
    'add': _integer_add
    }


# UGH!!!
# testing for register instance isn't quite right, since these may be chained:
#   Bits.xor(a, Bits.or_(b, c))
# The _type_ of the operand is all that can matter.
# Given this, I'm leaning towards a large, but flat type hierarchy.  Maybe use multiple inheritance:

class SignedWord(SPURegister, SignedInteger, Word): pass
class SignedHalfword(SPURegister, SignedInteger, Halfword): pass

class SignedImmediate(spe.Immediate, SignedInteger): pass

class Bits(object):
  def or_(a, b, im_type = Byte_t):
    if isinstance(b, spe.Register):
      return spu._or.ex(a, b)
    elif isinstnace(b, spe.Immediate):
      if im_type = Byte_t:
        return spu.orbi.ex(a, I10.cast(b))
      elif im_type = Halfword_t:
        return spu.orhi.ex(a, I10.cast(b))
      elif im_type = Word_t:
        return spu.ori.ex(a, I10.cast(b))

    return NotImplemented
  
  def xor(a, b, im_type = Byte_t):
    if isinstance(b, spe.Register):
      return spu.xor.ex(a, b)
    elif isinstnace(b, spe.Immediate):
      if im_type = Byte_t:
        return spu.xorbi.ex(a, I10.cast(b))
      elif im_type = Halfword_t:
        return spu.xorhi.ex(a, I10.cast(b))
      elif im_type = Word_t:
        return spu.xori.ex(a, I10.cast(b))
      
    return NotImplemented

  def or_across(a, b):
    if isinstance(a, spe.Register) and isinstance(b, spe.Register):
      return spu.orx.ex(a, b)

    return NotImplemented
  
  def eq(a, b):
    if isinstance(a, spe.Register) and isinstance(b, spe.Register):
      return spu.eqv.ex(a, b)
    
    return NotImplemented

  def select(a, b, c):
    if isinstance(a, spe.Register) and isinstance(b, spe.Register) and isinstance(b, spe.Register) :
      return spu.selb.ex(a, b, c)
    
    return NotImplemented


class Literal(object):
  def __init__(self, value):
    self.value = self.validate(value):
    return

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
    

class Variable(object):
  def __init__(self, reg, value = None):
    self.value = value
    self.reg = reg

    if value is not None:
      self.v = self.value
    return

  def get_value(self): return self.value
  def _set_value(self, v): self.set_value(v)
  v = property(get_value, _set_value)

  def _cast(cls, other):
    if issubclass(type(other), Variable):
      return cls(other.reg)

class Bits(Variable):
  def __init__(self, reg, value = None):
    self.value = None
    self.reg = reg
    self.im_type = Byte_t
    
    if self.value is not None:
      self.v = self.value
    return
  
  def __or__(self, other):
    if isinstance(other, Bits):
      return spu._or.ex(a, other)
    elif issubclass(other, Literal) or type(other) is int:
      if self.im_type = Byte_t:
        return spu.orbi.ex(a, I10.cast(other))
      elif self.im_type = Halfword_t:
        return spu.orhi.ex(a, I10.cast(other))
      elif self.im_type = Word_t:
        return spu.ori.ex(a, I10.cast(other))

    return TypeError('Unsupported type for bitwise or: %s' % (str(type(other))))
  

class SignedHalfword(Bits):
  def __add__(self, other):
    if type(other) is SignedWord:
      return spu.ah.ex(self, other)
    elif issubclass(other, Literal) is Literal or type(other) is int:
      return spu.ahi.ex(self, I10.cast(other))

    raise TypeError('Unsupported type for SingedWord add: %s' % (str(type(other))))
  
class SignedWord(Bits):
  def __add__(self, other):
    if type(other) is SignedWord:
      return spu.a.ex(self, other)
    elif type(other) is Literal or type(other) is int:
      return spu.ai.ex(self, I10.cast(other))

    raise TypeError('Unsupported type for SingedWord add: %s' % (str(type(other))))

                    
  

operators = {
  'add': (spu.a, spu.ai, spu.ah, spu.ahi)
  }



def add(a, b):
  if (isinstance(a, spe.Register) and isinstance(b, spe.Register)):
    if a.semantic_type == b.semantic_type:
      if (a.semantic_type == SignedInteger) or (a.semantic_type == SignedInteger):
        if a.size_t == b.size_type:
          if a.size_t == Word_t:
            return spu.a.ex(a, b)
          elif a.size_t == Halfword_t:
            return spu.ah.ex(a, b)
      elif (a.semantic_type == SPFloat):
        return spu.fa.ex(a, b)
      elif (a.semantic_type == DPFloat):
        return spu.dfa.ex(a, b)
  elif ((isinstance(a, spe.Register) and isinstance(b, spe.Immediate))
        (isinstance(a, spe.Register) and isinstance(b, int))):
    if isinstance(b, int) or b.size_type is not I10: b = I10.cast(b)    
    if a.size_t == Word_t:
      return spu.ai.ex(a, b)
    elif a.size_t == Halfword_t:
      return spu.ahi.ex(a, b)

  return NotImplemented

def rotate(a, b, by_bytes = False):
  if (isinstance(a, spe.Register) and isinstance(b, spe.Register)):
    if (b.semantic_type == UnsignedInteger) or (b.semantic_type == SignedInteger):
      if a.size_type == Word_t:
        return spu.rot.ex(a, b)
      elif a.size_type == Halfword_t:
        return spu.roth.ex(a, b)
      elif a.size_type == Quadword_t and not by_bytes:
        return spu.rotqbi.ex(a, b)
      elif a.size_type == Quadword_t:
        return spu.rotqby.ex(a, b)
  elif ((isinstance(a, spe.Register) and isinstance(b, spe.Immediate))
        (isinstance(a, spe.Register) and isinstance(b, int))):
    if isinstance(b, int) or b.size_type is not I7: b = I7.cast(b)

    if a.size_type == Word_t:
      return spu.roti.ex(a, b)
    elif a.size_type == Halfword_t:
      return spu.rothi.ex(a, b)
    elif a.size_type == Quadword_t and not by_bytes:
      return spu.rotqbii.ex(a, b)
    elif a.size_type == Quadword_t:
      return spu.rotqbyi.ex(a, b)

  return NotImplemented




SignedWord = spre.spe.CoreType(SPURegister, Word_t, SignedInteger)
SignedHalfword = spre.spe.CoreType(SPURegister, Halfword_t, SignedInteger)
SignedI10 = spre.spe.CoreType(spre.spe.Immediate, I10, SignedInteger)

spu.a.type_signature = (SignedWord, SignedWord, SignedWord)
spu.ai.type_signature = (SignedWord, SignedWord, I10)

spu.ah.type_signature = (SignedHalfword, SignedHalfword, SignedHalfword)
spu.ahi.type_signature = (SignedHalfword, SignedHalfword, I10)

a = spre.spe.Variable(SPURegister, Word_t, SignedInteger, code.acquire_register())
b = spre.spe.Variable(SPURegister, Word_t, SignedInteger, code.acquire_register())
c = spre.spe.Variable(SPURegister, Word_t, SignedInteger, code.acquire_register())

print spu.a(a, b, c)

spu.a.ex(a, spu.ai.ex(b, 1)).eval()

(a + I10.cast(1)).eval()

