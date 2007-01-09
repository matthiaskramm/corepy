
import corepy.arch.ppc.isa as ppc
import corepy.arch.vmx.isa as vmx
import corepy.spre.spe as spe

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

# Note: load_* will be replaced by the more complete memory classes at some point.

def load_word(code, r_target, word):
  """
  Generate the instruction sequence to load a word into r-target.
  
  This can be used for any value greater than 2^16.
  """
  # Put the lower 16 bits into r-temp
  start = code.add(ppc.addi(r_target, 0, word))
  
  # Addis r-temp with the upper 16 bits (shifted add immediate) and
  # put the result in r-target
  if (word & 0xFFFF) != word:
    code.add(ppc.addis(r_target, r_target, ((word + 32768) >> 16)))
  return start


def load_vector(code, v_target, addr):
  """
  Generate the code to load a vector into a vector register.
  """
  r_temp = code.acquire_register()

  load_word(code, r_temp, addr)
  code.add(vmx.lvx(v_target, 0, r_temp))
  code.release_register(r_temp)
  
  return

def RunTest(test, *ops):
  import sys, traceback
  try:
    test(*ops)
  except:
    info = sys.exc_info()
    file, line, func, text = traceback.extract_tb(info[2], 2)[1]
    print test.func_name, 'failed at line %d [%s]: \n  %s' % (line, info[0], info[1])
    traceback.print_tb(info[2])
    
  else:
    print test.func_name, ops, 'passed'

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
  if g is None: g = globals()
  g[name] = var_cls

  return
