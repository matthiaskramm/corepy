# Copyright (c) 2006-2008 The Trustees of Indiana University.                   
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

# Testing the bug in bi

import array
import time

import corepy.arch.spu.isa as spu
from corepy.arch.spu.types.spu_types import SignedWord, SingleFloat
from corepy.arch.spu.lib.iterators import memory_desc, spu_vec_iter, \
     stream_buffer, syn_iter, parallel
import corepy.arch.spu.lib.dma as dma
from corepy.arch.spu.platform import InstructionStream, ParallelInstructionStream, \
     NativeInstructionStream, Processor, spu_exec

  
def bi_bug():
  """
  A very simple SPU that computes 11 + 31 and returns 0xA on success.
  """
  code = InstructionStream()
  proc = Processor()

  spu.set_active_code(code)

  # Acquire two registers
  stop_inst = SignedWord(0x200D)
  stop_addr = SignedWord(0x0)

  spu.stqa(stop_inst, 0x0)
  spu.bi(stop_addr)
  spu.stop(0x200A)
  
  r = proc.execute(code) 
  assert(r == 0xD)

  return

if __name__=='__main__':
  bi_bug()
