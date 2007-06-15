# Copyright 2006-2007 The Trustees of Indiana University.

# This software is available for evaluation purposes only.  It may not be
# redistirubted or used for any other purposes without express written
# permission from the authors.

# Authors:
#   Christopher Mueller (chemuell@cs.indiana.edu)
#   Andrew Lumsdaine    (lums@cs.indiana.edu)

# ------------------------------
# Helpers
# ------------------------------

import spe

# Dec->Binary format converter from:
#  http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/219300
bstr_pos = lambda n: n>0 and bstr_pos(n>>1)+str(n&1) or ''
def DecToBin(n):
  """
  Convert an integer into binary coded string.
  """
  s = bstr_pos(n)
  if len(s) < 32:
    s = '0' * (32 - len(s)) + s
  return s

def BinToDec(b):
  """
  Convert a binary coded string to a decimal integer
  """

  l = list(b)
  l.reverse()
  p = 1
  d = 0
  for bit in l:
    d += p * int(bit)
    p = p << 1
  return d

def make_user_type(name, type_cls, g = None):
  """
  Create a Variable class and an Expression class for a type class.

  This is equivalent to creating two classes and updating the type
  class (except that the Expression class is not added to the global 
  namespace):

    class [name](spe.Variable, type_cls):
      type_cls = type_cls
    class [name]Ex(spe.Exression, type_cls):
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
  # if g is None: g = globals()
  # g[name] = var_cls

  return var_cls



def most_specific(a, b, default = None):
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

