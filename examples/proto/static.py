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



class Foo(object):
  ValidTypes = []
  def _a(a):
    print 'a:', a
  a = staticmethod(_a)

  def _cast(cls, value):
    if issubclass(type(value), Foo) or type(value) in cls.ValidTypes:
      print 'Casting %s to %s' % (str(value), str(cls))
    else:
      print 'Cannot cast'
  cast = classmethod(_cast)

  def assign(value): raise Exception('Please supply an assign method')
  def _assign_proxy(self, value):
    self.assign(value)
  def get_value(self): return None
  v = property(get_value, _assign_proxy)

class Bar(Foo):
  ValidTypes = [int]
  def _a(a):
    print 'b:', a
  a = staticmethod(_a)

  def assign(self, value):
    print 'hello', value
    
# class Baz(Foo, Bar): 
  

Foo.a(1)
Bar.a(2)
# Baz.a(3)

f = Foo()
b = Bar()

Foo.cast(f)
Bar.cast(f)
Foo.cast(b)
Bar.cast(b)
Foo.cast(1)
Bar.cast(2)

b.v = 12

class Baz(object):
  def __add__(self, other):
    if type(other) is not Baz:
      print 'No!'
    else:
      print 'badd!'

class Ping(object):
  def __add__(self, other):
    print 'padd!'


b = Baz()
p = Ping()

b + b
p + p
b + p


class Pong(Baz): pass
class Bong(Ping, Pong): pass

b = Bong()
print isinstance(b, Pong)
print isinstance(b, Ping)
print isinstance(b, Baz)
print isinstance(b, Bar)
print isinstance(b, Bong)

