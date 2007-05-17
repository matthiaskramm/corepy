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
