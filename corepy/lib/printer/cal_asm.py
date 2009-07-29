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

class CAL_Asm(object):
  """
  CAL IL-compatible assembly syntax printer.

  Output syntax from this printer is designed to look like CAL IL syntax
  assembly code.

  Several options are available for modifying the output (with defaults):
    show_prologue = True    Whether the prologue code should be printed
    show_epilogue = False   Whether the epilogue code should be printed
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
    # CAL has a trivial/nonstandard prologue, just print it here.
    print >>fd, "\til_ps_3_0"
    return False

  def epilogue(self, fd):
    """ Allow the module to print a prologue header if desired.
        The return value should be a boolean indicating whether epilogue
        instructions should be printed. """
    # CAL has no epilogue, so always return false.
    return False

  def stream(self, fd, stream):
    if self.verbose:
      print >>fd
      print >>fd, "# InstructionStream %x\n" % id(stream)
    return

  def label(self, fd, lbl):
    print >>fd, "\n%s:" % lbl.name
    return

  def instruction(self, fd, inst):
    # On CAL, instructions are rendered to IL-compatible strings.
    # So just render the instructions and print the strings.
    print >>fd, "\t%s" % (inst.render())
    return

  def string(self, fd, str):
    """Print a string (assumedly representing an instruction)."""
    print >>fd, "\t%s" % (str)
    return

