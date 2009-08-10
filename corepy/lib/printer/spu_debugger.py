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
import corepy.spre.syn_util as syn_util

class SPU_Debugger(object):
  """
  InstructionStream printer for the Interactive SPU debugger.

  Output syntax from this printer is designed to be easily used by the SPU
  debugger:

    ilhu(3, 0xDEAD)
    iohl(3, 0xBEEF)
    stqd(3, 0, 1)

  """

  def __init__(self):
    return

  def __del__(self):
    return

  def header(self, fd):
    return

  def footer(self, fd):
    return

  def prologue(self, fd):
    """ Allow the module to print a prologue header if desired.
        The return value should be a boolean indicating whether prologue
        instructions should be printed. """
    return False

  def epilogue(self, fd):
    """ Allow the module to print a prologue header if desired.
        The return value should be a boolean indicating whether epilogue
        instructions should be printed. """
    return False

  def stream(self, fd, stream):
    return

  def string(self, fd, str):
    """Print a string (assumedly representing an instruction)."""
    print >>fd, "\t%s" % (str)
    return

  def instruction(self, fd, inst):
    op_str = ', '.join([self.str_op(op) for op in inst._supplied_operands])
    for k, v in inst._supplied_koperands.items():
      op_str += ", %s = %s" % (str(k), str(v))

    print >>fd, "%s(%s)" % (inst.__class__.__name__, op_str)
    return

  def label(self, fd, lbl):
    print >>fd, "\n%s:" % lbl.name
    return

  def str_op(self, op):
    if isinstance(op, spe.Register):
      return str(op.reg)
    elif isinstance(op, spe.Variable):
      return str(op.reg.reg)
    return str(op)

