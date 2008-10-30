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

def foo():
  return

import sys
sys.path.append('../..')


from ispu import ISPU
import corepy.arch.spu.isa as spu

cli = ISPU()

cli.start()

ia = 127
ib = 126

fa = 125
fb = 124

y0 = 120
y1 = 121
y2 = 122

t1 = 119

result  = 118

ione = 110
fone = 111

insts = [
  # Create fone = 1.0, fa = 2.0 and fb = 4.0
  spu.ai(ione, 0, 1),  
  spu.ai(ia, 0, 2),
  spu.ai(ib, 0, 4),
  spu.cuflt(fone, ione, 155),  
  spu.cuflt(fa, ia, 155),
  spu.cuflt(fb, ib, 155),

  # Compute 1/fb
  spu.frest(y0, fb),
  spu.fi(y1, fb, y0),
  spu.fnms(t1, fb, y1, fone),
  spu.fma(y2, t1, y1, y1),

  spu.fm(result, fa, y2)
  ]

for inst in insts:
  cli.execute(inst)

regs = cli.get_regs()

for reg in (ione, fone, ia, ib, fa, fb, y0, y1, y2, t1, result):
  print reg, '0x%08X 0x%08X 0x%08X 0x%08X' % regs[reg]

cli.stop()
