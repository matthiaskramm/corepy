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
import corepy.arch.spu.isa as spu

class SPU_Asm(object):
  """
  SPU GAS-compatible assembly syntax printer.

  Output syntax from this printer is designed to look like GAS (AT&T) syntax
  assembly code.

    ilhu(3, 0xDEAD)
    iohl(3, 0xBEEF)
    stqd(3, 0, 1)

  Several options are available for modifying the output (with defaults):
    show_prologue = True    Whether the prologue code should be printed
    show_epilogue = False   Whether the epilogue code should be printed
    comment_chan = False    Whether wrch/rdch instructions should be commented
    verbose = False         Print extra comment information
  """

  def __init__(self, show_prologue = True, show_epilogue = False,
                     comment_chan = False, verbose = False):
    self.show_prologue = show_prologue
    self.show_epilogue = show_epilogue
    self.comment_chan = comment_chan
    self.verbose = verbose

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
    return self.show_prologue

  def epilogue(self, fd):
    """ Allow the module to print a prologue header if desired.
        The return value should be a boolean indicating whether epilogue
        instructions should be printed. """
    return self.show_epilogue

  def stream(self, fd, stream):
    if self.verbose:
      print >>fd
      print >>fd, "# InstructionStream %x\n" % id(stream)
    return

  def label(self, fd, lbl):
    print >>fd, "\n%s:" % lbl.name
    return

  def str_op(self, op):
    t = type(op)
    if isinstance(op, spe.Register):
      return "$%d" % op.reg
    elif isinstance(op, spe.Variable):
      return "$%d" % op.reg.reg
    return str(op)

  def instruction(self, fd, inst):
    # TODO - any other instructions to be handled specially?
    prefix = ""
    if isinstance(inst, (spu.stqd, spu.lqd)):
      ops = inst._supplied_operands
      name = inst.__class__.__name__
      print >>fd, "\t%s %s, %s(%s)" % (name, self.str_op(ops[0]), self.str_op(ops[2]), self.str_op(ops[1]))
      return
    elif self.comment_chan == True and isinstance(inst, (spu.rdch, spu.wrch)):
      # Comment rdch/wrch instructions, spu_timing doesn't like them
      prefix = "# "

    op_str = ', '.join([self.str_op(op) for op in inst._supplied_operands])
    # TODO - what to do about keywords?
    #for k, v in inst._supplied_koperands.items():
    #  op_str += ", %s = %s" % (str(k), str(v))

    name = inst.__class__.__name__.strip("_")
    print >>fd, "\t%s%s %s" % (prefix, name, op_str)
    return

