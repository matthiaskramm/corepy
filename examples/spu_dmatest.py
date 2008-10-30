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

  abi = a.buffer_info()
  print "abi", abi, a.itemsize
  dma.mem_get(code, 0x1000, abi[0], abi[1] * a.itemsize, 2)
  dma.mem_complete(code, 2)

  bbi = b.buffer_info()
  print "bbi", bbi, b.itemsize
  dma.mem_put(code, 0x1000, bbi[0], bbi[1] * b.itemsize, 2)
  dma.mem_complete(code, 2)

  code.print_code()
  proc.execute(code)
 
  for i in range(0, 32):
    if b[i] != i:
      print "ERROR %d %d %d" % (i, b[i], a[i])

 
