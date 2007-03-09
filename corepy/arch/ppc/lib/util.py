# Copyright 2006-2007 The Trustees of Indiana University.

# This software is available for evaluation purposes only.  It may not be
# redistirubted or used for any other purposes without express written
# permission from the authors.

# Authors:
#   Christopher Mueller (chemuell@cs.indiana.edu)
#   Andrew Lumsdaine    (lums@cs.indiana.edu)

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
    if len(ops) > 0:
      print test.func_name, ops, 'passed'
    else:
      print test.func_name, 'passed'      


def return_var(var):
  if isinstance(var.reg, type(var.code.gp_return)):
    var.code.add(ppc.addi(var.code.gp_return, var, 0))
  elif isinstance(var.reg, type(var.code.fp_return)):
    var.code.add(ppc.fmrx(var.code.fp_return, var))
  else:
    raise Exception('Return not supported for %s registers' % (str(type(var.reg))))
  return
  
