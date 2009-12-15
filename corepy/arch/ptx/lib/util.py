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

import corepy.spre.spe as spe
import corepy.arch.cal.isa as cal
import corepy.lib.extarray as extarray

def load_word(code, r_target, word):
  l = code.prgm.acquire_register((word, word, word, word))
  code.add(cal.mov(r_target, l.x))
  code.prgm.release_register(l)

  return


def load_float(code, reg, val):
  data = extarray.extarray('f', (val,))
  data.change_type('I')

  return load_word(code, reg, data[0])


def vector_from_array(code, r_target, a):
  """
  Generate the instructions to fill a vector register with the values
  from an array.
  """

  l = code.prgm.acquire_register((a[0], a[1], a[2], a[3]))
  code.add(cal.mov(r_target, l))
  code.prgm.release_register(l)

  return

def get_param_reg(code, param, dict, copy = True):
  """ Take a parameter given to a function, which may be a value or a
      register containing that value, and return a register containing the
      value.

      If copy is True, a new register is always returned.  Otherwise if a
      register was passed in, that register is returned unchanged. 

      dict is a dictionary used internally between get_param_reg() and
      put_param_reg() to keep track of whether registers have been allocated for
      parameters.  A function should use one (initially empty) dictionary for
      all of its parameters.
  """

  reg = None

  if isinstance(param, (spe.Register, spe.Variable)):
    if copy == True:
      # TODO - behave differently if at an even/odd spot
      reg = code.prgm.acquire_register()
      code.add(spu.ori(reg, param, 0))
      dict[reg] = True
    else:
      reg = param
      dict[reg] = False
  else: # TODO - check types?
    reg = code.prgm.acquire_register()
    load_word(code, reg, param)
    dict[reg] = True

  return reg


def put_param_reg(code, reg, dict):
  """Check a register containing a parameter, release the register if the
     provided dictionary indicates it was acquired by get_param_reg()/
  """
  if dict[reg] == True:
    code.prgm.release_register(reg)


# ------------------------------------------------------------
# Unit Test Code
# ------------------------------------------------------------

if __name__=='__main__':
  pass
