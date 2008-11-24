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

import corepy.lib.extarray as extarray
import corepy.arch.spu.isa as spu
import corepy.arch.spu.platform as env
import corepy.arch.spu.lib.dma as dma
from corepy.arch.spu.lib.util import load_word

import time

if __name__ == '__main__':
  ITERS = 500000
  #ITERS = 15

  code = env.InstructionStream()
  proc = env.Processor()
  spu.set_active_code(code)
  psmap = extarray.extarray('I', 131072 / 4)
  data = extarray.extarray('I', range(0, 16))

  r_sum = code.gp_return
  r_cnt = code.acquire_register()

  spu.xor(r_sum, r_sum, r_sum)
  load_word(code, r_cnt, ITERS)

  lbl_loop = code.get_label("loop")
  code.add(lbl_loop)

  reg = dma.spu_read_in_mbox(code)

  spu.ai(r_sum, r_sum, 1)
  dma.spu_write_out_intr_mbox(code, r_sum)
  #dma.spu_write_out_mbox(code, reg)

  code.release_register(reg)

  spu.ai(r_cnt, r_cnt, -1)
  spu.brnz(r_cnt, lbl_loop)
 
  reg = dma.spu_read_signal1(code)
  spu.ori(code.gp_return, reg, 0)


  spu.il(r_cnt, 0)
  spu.il(r_sum, 16 * 4)

  r_data = code.acquire_register()
  r_cmp = code.acquire_register()
  r_lsa = code.acquire_register()

  spu.il(r_lsa, 0x1000)

  lbl_incloop = code.get_label("incloop")
  code.add(lbl_incloop)

  spu.lqx(r_data, r_cnt, r_lsa)
  spu.ai(r_data, r_data, 2)
  spu.stqx(r_data, r_cnt, r_lsa)

  spu.ai(r_cnt, r_cnt, 16)
  spu.ceq(r_cmp, r_cnt, r_sum)
  spu.brz(r_cmp, lbl_incloop)

  dma.spu_write_out_mbox(code, code.r_zero)

  t3 = time.time()
  id = proc.execute(code, async = True, mode = 'int')


  t1 = time.time()
  for i in xrange(0, ITERS):
    #env.spu_exec.write_in_mbox(id, 1)
    #env.spu_exec.write_in_mbox(id, 1)
    env.spu_exec.write_in_mbox(id, i)
    #cnt = env.spu_exec.stat_in_mbox(id)
    #print "cnt %x" % cnt

    #cnt = env.spu_exec.stat_out_mbox(id)
    #while cnt == 0:
    #  cnt = env.spu_exec.stat_out_mbox(id)
    #env.spu_exec.read_out_mbox(id)

    cnt = env.spu_exec.stat_out_ibox(id)
    while cnt == 0:
      cnt = env.spu_exec.stat_out_ibox(id)
    val = env.spu_exec.read_out_ibox(id)
  t2 = time.time()

  # Use the PPU to DMA data to the SPU ls
  # The SPU will wait for the DMA to complete
  env.spu_exec.spu_getb(id, 0x1000, data.buffer_info()[0], 16 * 4, 2, 0, 0)
  env.spu_exec.read_tag_status_all(id, 1 << 2)

  env.spu_exec.write_signal(id, 1, 0x1234)
  env.spu_exec.read_out_mbox(id)

  env.spu_exec.spu_putb(id, 0x1000, data.buffer_info()[0], 16 * 4, 2, 0, 0)
  env.spu_exec.read_tag_status_all(id, 1 << 2)



  ret = proc.join(id)
  t4 = time.time()

  print "data", data
  print "inner time %0.5f" % (t2 - t1)
  print "outer time %0.5f" % (t4 - t3)
  print "ret %x" % ret
 
