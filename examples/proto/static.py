# Copyright 2006-2007 The Trustees of Indiana University.

# This software is available for evaluation purposes only.  It may not be
# redistirubted or used for any other purposes without express written
# permission from the authors.

# Authors:
#   Christopher Mueller (chemuell@cs.indiana.edu)
#   Andrew Lumsdaine    (lums@cs.indiana.edu)


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

