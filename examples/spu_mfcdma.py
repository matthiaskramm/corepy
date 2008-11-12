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

if __name__ == '__main__':
  a = extarray.extarray('i', range(0, 32))
  b = extarray.extarray('i', [0 for i in range(0, 32)])
  code = env.InstructionStream()
  proc = env.Processor()

  spu.set_active_code(code)

  r_lsa = code.acquire_register()   # Local Store address
  r_mma = code.acquire_register()   # Main Memory address
  r_size = code.acquire_register()  # Size in bytes
  r_tag = code.acquire_register()   # DMA Tag

  # Set the parameters for a GET command
  abi = a.buffer_info()

  spu.il(r_lsa, 0x1000)               # Local Store address 0x1000
  load_word(code, r_mma, abi[0])      # Main Memory address of array a
  spu.il(r_size, a.itemsize * abi[1]) # Size of array a in bytes
  spu.il(r_tag, 12)                   # DMA tag 12

  # Issue a DMA GET command
  dma.mfc_get(code, r_lsa, r_mma, r_size, r_tag)

  # Wait for completion
  # Set the completion mask; here we complete tag 12
  spu.il(r_tag, 1 << 12)
  dma.mfc_write_tag_mask(code, r_tag)
  dma.mfc_read_tag_status_all(code)


  # Set the parameters for a PUT command
  bbi = b.buffer_info()

  spu.il(r_lsa, 0x1000)               # Local Store address 0x1000
  load_word(code, r_mma, bbi[0])      # Main Memory address of array b
  spu.il(r_size, b.itemsize * bbi[1]) # Size of array b in bytes
  spu.il(r_tag, 12)                   # DMA tag 12

  # Issue a DMA PUT command
  dma.mfc_put(code, r_lsa, r_mma, r_size, r_tag)
  
  # Wait for completion
  # Set the completion mask; here we complete tag 12
  spu.il(r_tag, 1 << 12)
  dma.mfc_write_tag_mask(code, r_tag)
  dma.mfc_read_tag_status_all(code)

  code.release_register(r_lsa)
  code.release_register(r_mma)
  code.release_register(r_size)
  code.release_register(r_tag)

  # Execute the code
  proc.execute(code)

  # Check the results 
  for i in range(0, 32):
    if b[i] != i:
      print "ERROR %d %d %d" % (i, b[i], a[i])

 

