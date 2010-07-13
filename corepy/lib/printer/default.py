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

class Default(object):
  """
  Default, or CorePy-style InstructionStream printer.

  Output syntax from this printer is designed to look like the original
  Python-language CorePy code:

    ilhu(3, 0xDEAD)
    iohl(3, 0xBEEF)
    stqd(3, 0, 1)

  Several options are available for modifying the output (with defaults):
    show_prologue = False   Whether the prologue code should be printed
    show_epilogue = False   Whether the epilogue code should be printed
    show_hex = False        Show instruction encoding in hexadecimal
    show_binary = False     Show instruction encoding in binary
    line_numbers = False    Print a line number on each instruction if True
    inst_prefix = ""        Prefix each instruction with this string
    verbose = False         Print extra comment information
  """

  def __init__(self, show_prologue = False, show_epilogue = False,
                     show_hex = False, show_binary = False,
                     line_numbers = False, inst_prefix = "",
                     verbose = False):
    self.show_prologue = show_prologue
    self.show_epilogue = show_epilogue
    self.show_hex = show_hex
    self.show_binary = show_binary
    self.line_numbers = line_numbers
    self.inst_prefix = inst_prefix
    self.verbose = verbose

    self._line_num = 0
    self._hex_len = 0
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
    if self.show_prologue:
      print >>fd
      if self.verbose:
        print >>fd, "# Prologue\n"

      self._line_num = 0
    return self.show_prologue

  def epilogue(self, fd):
    """ Allow the module to print a prologue header if desired.
        The return value should be a boolean indicating whether epilogue
        instructions should be printed. """
    if self.show_epilogue:
      print >>fd
      if self.verbose:
        print >>fd, "# Epilogue\n"

      self._line_num = 0
    return self.show_epilogue

  #def body(self, fd):
  #  print >>fd
  #  if self.verbose:
  #    print >>fd, "# Body \n"

  #  self._line_num = 0
  #  return

  def stream(self, fd, stream):
    print >>fd
    if self.verbose:
      print >>fd, "# InstructionStream %x\n" % id(stream)

    self._line_num = 0
    return

  def string(self, fd, str):
    """Print a string (assumedly representing an instruction)."""
    print >>fd, "\t%s" % (str)
    return

  def instruction(self, fd, inst):
    if self.line_numbers:
      print >>fd, "%d\t" % (self._line_num),
      self._line_num += 1

    op_str = ', '.join([self.str_op(op) for op in inst._supplied_operands])
    for k, v in inst._supplied_koperands.items():
      op_str += ", %s = %s" % (str(k), str(v))

    print >>fd, "%s%s(%s)" % (self.inst_prefix, inst.__class__.__name__, op_str)
    if self.show_hex == True:
      print >>fd, "%x %s" % (self._hex_len, self.hex_inst(inst))
    if self.show_binary == True:
      print >>fd, self.binary_inst(inst)
    return

  def label(self, fd, lbl):
    if self.line_numbers:
      print >>fd, "\n\tLabel(%s)" % (lbl.name)
    else:
      print >>fd, "\nLabel(%s)" % lbl.name
    return

  def str_op(self, op):
    if isinstance(op, spe.Register):
      return str(op)
    elif isinstance(op, spe.Variable):
      return str(op.reg)
    return str(op)

  def hex_inst(self, inst):
    render = inst.render()
    if isinstance(render, (int, long)):
      hex = '%08x' % (render)
      self._hex_len += 4
    else:
      hex = ''
      for byte in render:
        hex += '%02x' % (byte)
        self._hex_len += 1
    return hex

  def binary_inst(self, inst):
    render = inst.render()
    if isinstance(render, (int, long)):
      bin = syn_util.DecToBin(render)
    else:
      bin = ''
      for byte in render:
        bin += syn_util.DecToBin(byte)[24:32]
    return bin


