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

import inspect

def reg_block():
  mr = 4
  nr = 4
  
  def load_a(i): print 'load a[%d]' % i
  def load_b(i): print 'load b[%d]' % i
  
  def compute(i,j):
    print 'compute c[%d][%d]' % (i,j)
    
  # load_a(0)
  # load_b(0)
  
  # compute(0, 0)
  
  load_a(0)
  load_b(0)
  
  # Register block
  for ri in range(0, mr):
    
    compute(ri, ri)
    
    if ri < (mr - 1):
      load_a(ri+1)
      load_b(ri+1)
      
    for ci in range(0, ri):
      compute(ci, ri)
      
    for cj in range(0, ri):
      compute(ri, cj)     
      
    compute(mr-1, mr-1)

    #   if ri == 0:
    #     compute(0, 0)
    #   elif (ri - 1 != 0):
    #     compute(ri-1, ri-1)

    return

from Numeric import *

def pack_b():
  B = zeros((10, 10))
  a = arange(10)
  for i in range(10):
    B[i,:] = a + i * 10

  B.shape = (10,10)

  tB = arange(25) * 0
  tB.shape = (5,5)

  K, N = B.shape
  nc, kc = tB.shape

  B.shape  = (K * N,)
  tB.shape = (nc * kc,)

  B_offset = 3 * 10 + 0

  for i in range(kc):
    B_row = B_offset + i * N
    for j in range(nc):
      b  = B_row + j
      tb = j * kc + i
      tB[tb] = B[b]
        
  tB.shape = (nc, kc)
  B.shape  = (K, N)
  
  print tB
  return

pack_b()
