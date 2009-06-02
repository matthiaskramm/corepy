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

import corepy.arch.x86.isa as x86
import corepy.spre.spe as spe

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

# Note: load_* will be replaced by the more complete memory classes at some point.

def load_word(code, r_target, word):
  """
  Generate the instruction sequence to load a word into r-target.
  
  This can be used for any value greater than 2^16.
  TODO - afriedle - what ranges does this work for on x86?
  """
  return code.add(x86.mov(r_target, word))


def load_float(code, reg, val, clear = False):
  data = extarray.extarray('f', (val,))
  data.change_type('I')

  # reg better be an mmx or xmm, should we check?
  code.add(x86.push(data[0]))
  code.add(x86.pshufd(reg, mem.MemRef(regs.rsp, data_size = 128), 0))
  return code.add(x86.add(regs.rsp, 8))


def load_double(code, reg, val):
  data = extarray.extarray('d', (val,))
  data.change_type('I')

  # reg better be an mmx or xmm, should we check?
  code.add(x86.push(data[0]))
  code.add(x86.push(data[1]))
  code.add(x86.pshufd(reg, mem.MemRef(regs.rsp, data_size = 128), 0x44))
  return code.add(x86.add(regs.rsp, 8))


def load_vector(code, v_target, addr):
  """
  Generate the code to load a vector into a vector register.
  TODO - afriedle - not implemented yet for x86
  """
  raise NameError('Not Implemented')

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
    var.code.add(x86.mov(var.code.gp_return, var))
 #elif isinstance(var.reg, type(var.code.fp_return)):
 #    var.code.add(x86.fmrx(var.code.fp_return, var))
  else:
    raise Exception('Return not supported for %s registers' % (str(type(var.reg))))
  return
  
