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

import array
import time

import corepy.arch.spu.isa as spu
from corepy.arch.spu.types.spu_types import SignedWord, SingleFloat
#from corepy.arch.spu.lib.iterators import memory_desc, spu_vec_iter, \
#     stream_buffer, syn_iter, parallel
import corepy.arch.spu.lib.dma as dma
import corepy.arch.spu.lib.util as util
from corepy.arch.spu.platform import InstructionStream, ParallelInstructionStream, \
     Processor, aligned_memory, spu_exec
     #NativeInstructionStream, Processor, aligned_memory, spu_exec

  
def SimpleSPU():
  """
  A very simple SPU that computes 11 + 31 and returns 0xA on success.
  """
  code = InstructionStream()
  proc = Processor()

  spu.set_active_code(code)
  

  # Acquire two registers
  #x    = code.acquire_register()
  x = code.gp_return
  test = code.acquire_register()

  lbl_brz = code.get_label("BRZ")
  lbl_skip = code.get_label("SKIP")

  spu.hbrr(lbl_brz, lbl_skip)
  spu.xor(x, x, x) # zero x
  spu.ai(x, x, 11) # x = x + 11
  spu.ai(x, x, 31) # x = x + 31

  spu.ceqi(test, x, 42) # test = (x == 42)

  # If test is false (all 0s), skip the stop(0x100A) instruction
  code.add(lbl_brz)
  spu.brz(test, lbl_skip)
  spu.stop(0x100A)
  code.add(lbl_skip)
  spu.stop(0x100B)
 
  code.print_code(hex = True, pro = True, epi = True) 
  r = proc.execute(code, mode = 'int', stop = True) 
  print "ret", r
  assert(r[0] == 42)
  assert(r[1] == 0x100A)


  code = InstructionStream()
  spu.set_active_code(code)

  lbl_loop = code.get_label("LOOP")
  lbl_break = code.get_label("BREAK")

  r_cnt = code.acquire_register()
  r_stop = code.acquire_register()
  r_cmp = code.acquire_register()
  r_foo = code.gp_return

  spu.ori(r_foo, code.r_zero, 0)
  spu.ori(r_cnt, code.r_zero, 0)
  util.load_word(code, r_stop, 10)

  code.add(lbl_loop)

  spu.ceq(r_cmp, r_cnt, r_stop)
  spu.brnz(r_cmp, lbl_break)
  spu.ai(r_cnt, r_cnt, 1)

  spu.a(r_foo, r_foo, r_cnt)

  spu.br(lbl_loop)
  code.add(lbl_break)

  code.print_code()
  r = proc.execute(code, mode = 'int', stop = True)
  print "ret", r
  assert(r[0] == 55)

  return


if __name__=='__main__':
  SimpleSPU()
  
  #for i in [4100, 10000, 20000, 30000]:
  #  MemoryDescExample(i)

  #DoubleBufferExample()
  #SpeedTest()

