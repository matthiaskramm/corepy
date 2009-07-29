# Copyright (c) 2006-2009 The Trustees of Indiana University.                   
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

  # Put the lower 16 bits into r_target
  start = code.add(ppc.addi(r_target, 0, word & 0xFFFF))

  uw = (word >> 16) & 0xFFFF
  msb = word & 0x8000

  if msb != 0:
    # lower 16-bit MSB is set, upper 16 bits are 1, adjust uw
    # If all upper 16 bits are 1, that is the value -1, so add 1 back in.
    uw = (uw + 1) & 0xFFFF

  # Only issue addis if the value added (uw) is not zero.
  if uw != 0:
    code.add(ppc.addis(r_target, r_target, uw))

  return start


def load_vector(code, v_target, addr):
  """
  Generate the code to load a vector into a vector register.
  """
  r_temp = code.prgm.acquire_register()

  load_word(code, r_temp, addr)
  code.add(vmx.lvx(v_target, 0, r_temp))
  code.prgm.release_register(r_temp)
  
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
  if isinstance(var.reg, type(var.code.prgm.gp_return)):
    var.code.add(ppc.addi(var.code.prgm.gp_return, var, 0))
  elif isinstance(var.reg, type(var.code.prgm.fp_return)):
    var.code.add(ppc.fmrx(var.code.prgm.fp_return, var))
  else:
    raise Exception('Return not supported for %s registers' % (str(type(var.reg))))
  return
  
