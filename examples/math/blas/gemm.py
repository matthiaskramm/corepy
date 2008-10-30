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

# Matrix-Matrix Multiplication Examples
# NOTE: This is a work in progress

import Numeric
import time

# ------------------------------------------------------------
# Python
# ------------------------------------------------------------

def invalid(A, B, C): pass

def python_mm(A, B, C):
  """
  Naive pure Python implementation.
  """

  C[:,:] = C * 0
  
  for i in range(A.shape[0]):
    for j in range(A.shape[1]):
      for k in range(B.shape[1]):
        C[i][k] += A[i][j] * B[j][k]

  return

def numeric_mm(A, B, C, mc, kc, nc, mr=1, nr=1):
  C[:,:] = Numeric.matrixmultiply(A, B)
  return


def _gebp_opt1(A, B, C, nc):
  n = C.shape[1]

  # Assume B is packed
  # Pack A into tA
  
  for j in range(0, n, nc):
    # Load Bj into cache
    # Load Cj into cache

    # Cj += ABj + Cj
    ABj = Numeric.matrixmultiply(A, B[:,j:j+nc])
    Numeric.add(C[:,j:j+nc], ABj, C[:,j:j+nc])

    # Store Caux into memory
  return

def _gepp_blk_var1(A, B, C, mc, nc):
  m = C.shape[1]
  
  # Pack B into tB

  for i in range(0, m, mc):
    _gebp_opt1(A[i:i+mc,:], B, C[i:i+mc, :], nc)
  
  return

_MC, _KC, _NC = 64, 64, 128
# _MC, _KC, _NC = 1024, 1024, 1024

def numeric_gemm_var1(A, B, C, mc, kc, nc, mr=1, nr=1):
  """
  GEMM_VAR1, top branch 
  """
  
  m, n = C.shape
  k = A.shape[0]

  mc = min(mc, m)
  kc = min(kc, k)
  nc = min(nc, n)  
  
  for k in range(0, k, kc):
    _gepp_blk_var1(A[:,k:k+kc], B[k:k+kc,:], C, mc, nc)
  return


def numeric_gemm_var1_flat(A, B, C, mc, kc, nc, mr=1, nr=1):
  M, N = C.shape
  K = A.shape[0]

  mc = min(mc, M)
  kc = min(kc, K)
  nc = min(nc, N)  

  tA = Numeric.zeros((mc, kc), typecode = Numeric.Float)
  tB = Numeric.zeros((kc, N), typecode = Numeric.Float)  

  for k in range(0, K, kc):
    # Pack B into tB
    tB[:,:] = B[k:k+kc:,:]
    
    for i in range(0, M, mc):
      imc = i+mc
      # Pack A into tA
      tA[:,:] = A[i:imc,k:k+kc]
      
      for j in range(0, N): # , nc):
        # Cj += ABj + Cj
        # jnc = j+nc
        ABj = Numeric.matrixmultiply(tA, tB[:,j])
        Numeric.add(C[i:imc:,j], ABj, C[i:imc:,j])
        
        # Store Caux into memory
  return

def numeric_gemm_var1_row(A, B, C, mc, kc, nc, mr=1, nr=1):
  """
  Observations on flat/row:
  - Using the same basic approach and keeping the parameters
    equal,  flat is faster by about 100 MFlops.  This is
    counterintuitive, since row is optimized for the row major layout
    of the arrays and flat is optimized for column major layout.
  - The key to getting row to be faster lies in making sure the kernel
    operations really take advantage of the row layout by adjusting
    nc, which impacts row operations in the inner loop:
    - In Python, the final add has the largest effect on performance 
      - for flat, nc controls how much of each row is copied
      - for row,  nc controls how much of each row is copied
      - How much of each row is copied appears to have the biggest
        impact
  - In the end, turns out that doing vector/matrix multiply in the
    innermost loop is most fair.  Cache effects aren't really
    observable since matrixmultiply is already pretty fast.  Making
    the block sizes bigger just gives more to an already fast kernel.
  """
  M, N = C.shape
  K = A.shape[0]

  nc = min(nc, N)
  kc = min(kc, K)
  mc = min(mc, M)
  
  tA = Numeric.zeros((M, kc), typecode = Numeric.Float)
  tB = Numeric.zeros((kc, nc), typecode = Numeric.Float)  

  for k in range(0, K, kc):
    # Pack A into tA
    tA[:,:] = A[:,k:k+kc]

    for j in range(0, N, nc):
      jnc = j+nc
      # Pack B into tB
      tB[:,:] = B[k:k+kc, j:jnc]

      for i in range(0, M): # , mc):
        # Ci = AiB + Ci
        # imc = i+mc
        ABi = Numeric.matrixmultiply(tA[i,:], tB)
        Numeric.add(C[i,j:jnc], ABi, C[i,j:jnc])
        
        # Store Caux into memory
  return


# ------------------------------------------------------------
# Synthetic
# ------------------------------------------------------------

import corepy.arch.ppc.isa as ppc
import corepy.arch.ppc.platform as synppc
import corepy.arch.ppc.types.ppc_types as ppcvar

from corepy.arch.ppc.lib.iterators import syn_range, syn_iter, CTR

class SynPackB:

  def _init_constants(self, code, tB, N):
    code.add_storage(tB)
    
    nc, kc = tB.shape

    self.block = _block()
    self.block.nc = nc
    self.block.kc = kc    

    self.dim = _dim(-1, -1, N)
    self.tB = tB
    
    return

  def _init_vars(self, vb = None, vtB = None):
    
    self.vN = ppcvar.UnsignedWord(self.dim.N)
    
    self.vB = ppcvar.UnsignedWord()
    self.vBi = ppcvar.UnsignedWord()

    self.bij = ppcvar.UnsignedWord()
    self.tbji = ppcvar.UnsignedWord()

    self.vb_local = True
    if vb is None:
      self.vb = ppcvar.DoubleFloat()
    else:
      self.vb = vb; self.vb_local = False

    self.vtB_local = True
    if vtB is None:
      self.vtB = ppcvar.UnsignedWord(synppc.array_address(self.tB))
    else:
      self.vtB = vtB; self.vtB_local = False

    return

  def _init_vars2(self, vars, fvar, vtB):
    """
    Use variables in vars instead of allocating new ones.
    """

    # vB cannot change between runs, allocate locally.
    self.vB = ppcvar.UnsignedWord()
    
    self.vN = vars[0]
    self.vN.v = self.dim.N

    self.vBi = vars[1]
    self.bij = vars[2]
    self.tbji = vars[3]

    self.vtB = vtB

    self.vb = fvar
    return
  

  def _load_params(self, pvB = 3):
    self.vB.copy_register(pvB) # First parameter    
    return

  def _pack_b(self, code):
    kc, nc, = self.block.kc, self.block.nc
    
    vb, vB, vtB, vBi, bij, tbji, vN = (
      self.vb, self.vB, self.vtB, self.vBi, self.bij, self.tbji, self.vN)

    
    for i in syn_iter(code, kc):
      vBi.v = i * vN
      
      for j in syn_iter(code, nc):
        bij.v  = (vBi + j) * 8
        tbji.v = (j * kc + i) * 8
        vb.load(vB, bij)
        vb.store(vtB, tbji)        

    return

  def synthesize(self, code, tB, N):
    """
    Extract a block from B and pack it for fast access.

    tB is transposed.
    """
    old_code = ppc.get_active_code()
    ppc.set_active_code(code)

    code.add_storage(tB)

    self._init_constants(code, tB, N)
    self._init_vars()
    self._load_params()
    self._pack_b(code)

    ppc.set_active_code(old_code)        
    return

class _dim:
  def __init__(self, M, K, N):
    self.M = M
    self.K = K
    self.N = N
    return

class _stride:
  def __init__(self, row, col):
    self.row = row
    self.col = col
    return

class _block:
  def __init__(self):
    # empty -- use appropriate property names (e.g, mr/nr
    return

gepb_simple, gepb_prefetch, gepb_prefetch_hand = range(3)

class SynGEPB:

  def __init__(self, mode = gepb_simple):
    self.mode = mode
    return
  
  def reg_loop_4x4(self, a, b, c, mr, nr, p_tA, p_tB, A_row_stride, B_col_stride):
    a[0].load(p_tA, 0 * A_row_stride)
    b[0].load(p_tB, 0 * B_col_stride)

    c[0][0].v = ppcvar.fmadd(a[0], b[0], c[0][0]); 
    
    b[1].load(p_tB, 1 * B_col_stride)          
    a[1].load(p_tA, 1 * A_row_stride)
    
    c[0][1].v = ppcvar.fmadd(a[0], b[1], c[0][1]); 
    c[1][0].v = ppcvar.fmadd(a[1], b[0], c[1][0])
    
    a[2].load(p_tA, 2 * A_row_stride)
    b[2].load(p_tB, 2 * B_col_stride)          
    
    c[1][1].v = ppcvar.fmadd(a[1], b[1], c[1][1])
    c[2][0].v = ppcvar.fmadd(a[2], b[0], c[2][0])          
    
    c[0][2].v = ppcvar.fmadd(a[0], b[2], c[0][2]); 
    c[1][2].v = ppcvar.fmadd(a[1], b[2], c[1][2])
    
    c[2][1].v = ppcvar.fmadd(a[2], b[1], c[2][1])
    
    a[3].load(p_tA, 3 * A_row_stride)
    b[3].load(p_tB, 3 * B_col_stride)
    
    
    c[2][2].v = ppcvar.fmadd(a[2], b[2], c[2][2])
    
    c[3][0].v = ppcvar.fmadd(a[3], b[0], c[3][0])
    c[3][1].v = ppcvar.fmadd(a[3], b[1], c[3][1])
    c[3][2].v = ppcvar.fmadd(a[3], b[2], c[3][2])
    c[3][3].v = ppcvar.fmadd(a[3], b[3], c[3][3])
    
    c[0][3].v = ppcvar.fmadd(a[0], b[3], c[0][3]); 
    c[1][3].v = ppcvar.fmadd(a[1], b[3], c[1][3])
    c[2][3].v = ppcvar.fmadd(a[2], b[3], c[2][3])

    return

  def reg_loop_simple(self, a, b, c, mr, nr, p_tA, p_tB, A_row_stride, B_col_stride):
    # Load the next values from tA and tB -- generating loops
    for ai in range(mr):
      a[ai].load(p_tA, ai * A_row_stride)
    
    for bj in range(nr):
      b[bj].load(p_tB, bj * B_col_stride)
    
    # Update c -- generating loop
    for ci in range(mr):
      for cj in range(nr):
        c[ci][cj].v = ppcvar.fmadd(a[ci], b[cj], c[ci][cj])
          
    return

  def reg_loop_interleaved(self, a, b, c, mr, nr, p_tA, p_tB, A_row_stride, B_col_stride):
    # Generate the operations so that there is some overlap between
    # loads and computations.
    
    def load_a(i): a[i].load(p_tA, i * A_row_stride)
    def load_b(i): b[i].load(p_tB, i * B_col_stride)
    def compute(i,j): c[i][j].v = ppcvar.fmadd(a[i], b[j], c[i][j]); 
    
    load_a(0)
    load_b(0)          
    
    for ri in range(0, mr):
      compute(ri, ri)
      
      if ri < (mr - 1):
        load_a(ri+1)
        load_b(ri+1)
        
      for ci in range(0, ri):
        compute(ci, ri)
        
      for cj in range(0, ri):
        compute(ri, cj)     
    # /end for ri

    if mr > 1:
      compute(mr-1, mr-1)
    
    return

  def c_aux_save_simple(self, code):
    # Copy C_aux to C
    # ~620 MFlops
    nc, mr, a, b, p_C, p_C_aux, C_col_stride = (
      self.block.nc, self.block.mr, self.a, self.b, self.p_C, self.p_C_aux, self.C_strides.col)

    for jj in syn_range(code, 0, nc * C_col_stride, C_col_stride):
      for ci in range(mr):
    
        a[ci].load(p_C[ci], jj)
        b[ci].load(p_C_aux[ci], jj)
    
        a[ci].v = a[ci] + b[ci]
        a[ci].store(p_C[ci], jj)
    
      # /end for ci
    # /end for jj
    return

  def c_aux_save_hand(self, code):
    mr = self.block.mr
    nc = self.block.nc
    p_C = self.p_C
    p_C_aux = self.p_C_aux
    
    c_mem = self.a + self.b
    if mr > 1:  c_aux = self.c[0] + self.c[1]
    else:       c_aux = self.c[0]
    
    for jj in syn_range(code, 0, nc * self.C_strides.col, self.C_strides.col):
      JJ = jj
      
      # ~ 650 Mflops (one extra unroll does not affect result)
      if mr >= 1:
        c_mem[0].load(p_C[0], JJ)
        c_aux[0].load(p_C_aux[0], JJ)
        
      if mr >= 2:
        c_mem[1].load(p_C[1], JJ)
        c_aux[1].load(p_C_aux[1], JJ)
        
      if mr >= 1:
        c_mem[0].v = c_mem[0] + c_aux[0]
        
      if mr >= 2:
        c_mem[1].v = c_mem[1] + c_aux[1]
        
      if mr >= 3:
        c_mem[2].load(p_C[2], JJ)
        c_aux[2].load(p_C_aux[2], JJ)
        
      if mr >= 4:
        c_mem[3].load(p_C[3], JJ)
        c_aux[3].load(p_C_aux[3], JJ)
        
      if mr >= 3:
        c_mem[2].v = c_mem[2] + c_aux[2]
        
      if mr >= 4:
        c_mem[3].v = c_mem[3] + c_aux[3]
        
      if mr >= 1: c_mem[0].store(p_C[0], JJ)
      if mr >= 2: c_mem[1].store(p_C[1], JJ)
      if mr >= 3: c_mem[2].store(p_C[2], JJ)
      if mr >= 4: c_mem[3].store(p_C[3], JJ)

    # /end for jj
    return
    
  def prefetch_load(self, A, B):  return ppc.dcbt(A, B)
  def prefetch_store(self, A, B): return ppc.dcbtst(A, B)

  def _init_constants(self, M, K, N, kc, nc, mr, nr, _transpose):
    # Constants
    C_col_stride = 8    
    C_row_stride = N * C_col_stride

    # (Assume A and B are packed)
    A_col_stride = 8
    A_row_stride = kc * A_col_stride

    # B_col_stride = 8
    # B_row_stride = nc * B_col_stride
    if _transpose:
      B_row_stride = 8
      B_col_stride = kc * B_row_stride
    else:
      B_col_stride = 8      
      B_row_stride = nc * B_col_stride

    self.A_strides = A_row_stride
    self.A_col_stride = A_col_stride

    self.B_row_stride = B_row_stride
    self.B_col_stride = B_col_stride

    self.C_row_stride = C_row_stride
    self.C_col_stride = C_col_stride

    self.A_strides = _stride(A_row_stride, A_col_stride)
    self.B_strides = _stride(B_row_stride, B_col_stride)
    self.C_strides = _stride(C_row_stride, C_col_stride)
    
    self.dims = _dim(M, K, N)

    self.block = _block()
    self.block.kc = kc
    self.block.nc = nc
    self.block.mr = mr
    self.block.nr = nr
    
    return

  def _init_vars(self):
    # Address variables
    self.r_tA_addr = ppcvar.UnsignedWord()
    self.r_tB_addr = ppcvar.UnsignedWord()
    self.r_C_addr  = ppcvar.UnsignedWord()
    self.r_C_aux_addr  = ppcvar.UnsignedWord()    

    self.p_tA = ppcvar.UnsignedWord()
    self.p_tB = ppcvar.UnsignedWord()
    self.p_C  = [ppcvar.UnsignedWord() for i in range(self.block.mr)]
    self.p_C_aux = [ppcvar.UnsignedWord() for i in range(self.block.mr)]

    self.vC_row_stride = ppcvar.UnsignedWord(self.C_strides.row)

    mr, nr = self.block.mr, self.block.nr
    
    # Inner loop variables
    if mr != nr:
      raise Exception('mr (%d) should equal nr (%d)' % (mr, nr))

    self.a = [ppcvar.DoubleFloat() for i in range(mr)]
    self.b = [ppcvar.DoubleFloat() for j in range(nr)]

    self.a_pre = [ppcvar.DoubleFloat() for i in range(mr)]
    self.b_pre = [ppcvar.DoubleFloat() for j in range(nr)]

    self.c = [[ppcvar.DoubleFloat() for j in range(nr)] for i in range(mr)]

    return

  def _load_params(self, ptA_addr = 3, ptB_addr = 4, pC_addr = 5, pC_aux_addr = 6):

    # Load addresses from function parameters
    ppc.addi(self.r_tA_addr, ptA_addr, 0)
    ppc.addi(self.r_tB_addr, ptB_addr, 0)
    ppc.addi(self.r_C_addr,  pC_addr, 0)
    ppc.addi(self.r_C_aux_addr,  pC_aux_addr, 0)    

    return

  def _init_pointers(self):

    # Set the array pointers 
    self.p_tA.v = self.r_tA_addr
    self.p_tB.v = self.r_tB_addr

    for ci in range(self.block.mr):
      self.p_C[ci].v     = self.r_C_addr + ci * self.vC_row_stride # self.C_strides.row
      self.p_C_aux[ci].v = self.r_C_aux_addr + ci * (self.block.nc * self.C_strides.col)

    return

  def k_loop_prefetch(self, code):
    A_row_stride = self.A_strides.row
    A_col_stride = self.A_strides.col

    B_row_stride = self.B_strides.row
    B_col_stride = self.B_strides.col

    kc, mr, nr = self.block.kc, self.block.mr, self.block.nr
    a, b, c = self.a, self.b, self.c
    a_pre, b_pre = self.a_pre, self.b_pre
    p_tA, p_tB = self.p_tA, self.p_tB

    # Increment p_tA, p_tB
    a[0].load(p_tA, 0 * A_row_stride)
    b[0].load(p_tB, 0 * B_col_stride)
    b[1].load(p_tB, 1 * B_col_stride)                    
    a[1].load(p_tA, 1 * A_row_stride)
    
    b[2].load(p_tB, 2 * B_col_stride)          
    b[3].load(p_tB, 3 * B_col_stride)
    
    a[2].load(p_tA, 2 * A_row_stride)
    a[3].load(p_tA, 3 * A_row_stride)
    
    p_tA.v = p_tA + A_col_stride
    p_tB.v = p_tB + B_row_stride
    
    for k in syn_iter(code, kc / 2 , mode = CTR): # syn_range(code, 0, kc * 8, 8):
      # self.reg_loop_simple(a, b, c, mr, nr, p_tA, p_tB, A_row_stride, B_row_stride)
      
      a_pre[0].load(p_tA, 0 * A_row_stride)
      b_pre[1].load(p_tB, 1 * B_col_stride)
      b_pre[0].load(p_tB, 0 * B_col_stride)          
      a_pre[1].load(p_tA, 1 * A_row_stride)
      
      b_pre[2].load(p_tB, 2 * B_col_stride)
      a_pre[2].load(p_tA, 2 * A_row_stride)

      b_pre[3].load(p_tB, 3 * B_col_stride)          
      a_pre[3].load(p_tA, 3 * A_row_stride)
      
      p_tA.v = p_tA + A_col_stride
      p_tB.v = p_tB + B_row_stride
      
      c[0][0].v = ppcvar.fmadd(a[0], b[0], c[0][0]); 
      c[0][1].v = ppcvar.fmadd(a[0], b[1], c[0][1]);
      c[1][0].v = ppcvar.fmadd(a[1], b[0], c[1][0])
      c[1][1].v = ppcvar.fmadd(a[1], b[1], c[1][1])
      
      c[1][2].v = ppcvar.fmadd(a[1], b[2], c[1][2])
      c[0][2].v = ppcvar.fmadd(a[0], b[2], c[0][2]); 
      c[2][0].v = ppcvar.fmadd(a[2], b[0], c[2][0])
      c[2][1].v = ppcvar.fmadd(a[2], b[1], c[2][1])
      
      c[2][2].v = ppcvar.fmadd(a[2], b[2], c[2][2])
      c[2][3].v = ppcvar.fmadd(a[2], b[3], c[2][3])
      c[0][3].v = ppcvar.fmadd(a[0], b[3], c[0][3]);
      c[1][3].v = ppcvar.fmadd(a[1], b[3], c[1][3])
      
      c[3][0].v = ppcvar.fmadd(a[3], b[0], c[3][0])
      c[3][1].v = ppcvar.fmadd(a[3], b[1], c[3][1])
      c[3][2].v = ppcvar.fmadd(a[3], b[2], c[3][2])
      c[3][3].v = ppcvar.fmadd(a[3], b[3], c[3][3])
      
      a[0].load(p_tA, 0 * A_row_stride)
      b[1].load(p_tB, 1 * B_col_stride)
      b[0].load(p_tB, 0 * B_col_stride)
      a[1].load(p_tA, 1 * A_row_stride)
      
      b[2].load(p_tB, 2 * B_col_stride)
      b[3].load(p_tB, 3 * B_col_stride)
      
      a[2].load(p_tA, 2 * A_row_stride)
      a[3].load(p_tA, 3 * A_row_stride)
      
      p_tA.v = p_tA + A_col_stride
      p_tB.v = p_tB + B_row_stride
      
      c[0][0].v = ppcvar.fmadd(a_pre[0], b_pre[0], c[0][0]); 
      c[0][1].v = ppcvar.fmadd(a_pre[0], b_pre[1], c[0][1]);
      c[1][0].v = ppcvar.fmadd(a_pre[1], b_pre[0], c[1][0])
      c[1][1].v = ppcvar.fmadd(a_pre[1], b_pre[1], c[1][1])
      
      c[1][2].v = ppcvar.fmadd(a_pre[1], b_pre[2], c[1][2])
      c[0][2].v = ppcvar.fmadd(a_pre[0], b_pre[2], c[0][2]); 
      c[2][0].v = ppcvar.fmadd(a_pre[2], b_pre[0], c[2][0])
      c[2][1].v = ppcvar.fmadd(a_pre[2], b_pre[1], c[2][1])
      
      c[2][2].v = ppcvar.fmadd(a_pre[2], b_pre[2], c[2][2])
      c[2][3].v = ppcvar.fmadd(a_pre[2], b_pre[3], c[2][3])
      c[0][3].v = ppcvar.fmadd(a_pre[0], b_pre[3], c[0][3]);
      c[1][3].v = ppcvar.fmadd(a_pre[1], b_pre[3], c[1][3])
      
      c[3][0].v = ppcvar.fmadd(a_pre[3], b_pre[0], c[3][0])
      c[3][1].v = ppcvar.fmadd(a_pre[3], b_pre[1], c[3][1])
      c[3][2].v = ppcvar.fmadd(a_pre[3], b_pre[2], c[3][2])
      c[3][3].v = ppcvar.fmadd(a_pre[3], b_pre[3], c[3][3])
      
      # /end for k
    return

  def k_loop_prefetch_simple(self, code):
    kc, mr, nr = self.block.kc, self.block.mr, self.block.nr
    a, b, c = self.a, self.b, self.c
    a_pre, b_pre = self.a_pre, self.b_pre
    p_tA, p_tB = self.p_tA, self.p_tB

    # Load the next values from tA and tB 
    for ai in range(mr):
      a[ai].load(p_tA, ai * self.A_strides.row)
      
    for bj in range(nr):
      b[bj].load(p_tB, bj * self.B_strides.col)

    p_tA.v = p_tA + self.A_strides.col
    p_tB.v = p_tB + self.B_strides.row
        

    # Inner loop over k
    for k in syn_iter(code, kc / 2, mode = CTR): # syn_range(code, 0, kc * 8, 8):

      # Iteration 1 -- load [a,b]_pre, compute [a,b]
      # Load the prefetch values from tA and tB 
      for ai in range(mr):
        a_pre[ai].load(p_tA, ai * self.A_strides.row)
    
      for bj in range(nr):
        b_pre[bj].load(p_tB, bj * self.B_strides.col)

      p_tA.v = p_tA + self.A_strides.col
      p_tB.v = p_tB + self.B_strides.row

      # Update c
      for ci in range(mr):
        for cj in range(nr):
          c[ci][cj].v = ppcvar.fmadd(a[ci], b[cj], c[ci][cj])

      # Iteration 2l -- oad [a,b], compute [a,b]_pre
      # Load the prefetch values from tA and tB 
      for ai in range(mr):
        a_pre[ai].load(p_tA, ai * self.A_strides.row)
    
      for bj in range(nr):
        b_pre[bj].load(p_tB, bj * self.B_strides.col)

      p_tA.v = p_tA + self.A_strides.col
      p_tB.v = p_tB + self.B_strides.row

      # Update c
      for ci in range(mr):
        for cj in range(nr):
          c[ci][cj].v = ppcvar.fmadd(a_pre[ci], b_pre[cj], c[ci][cj])
          
    # /end for k

    return


  def k_loop_simple(self, code):
    kc, mr, nr = self.block.kc, self.block.mr, self.block.nr
    a, b, c = self.a, self.b, self.c
    p_tA, p_tB = self.p_tA, self.p_tB
    
    # Inner loop over k
    for k in syn_iter(code, kc, mode = CTR): # syn_range(code, 0, kc * 8, 8):
      # Load the next values from tA and tB -- generating loops
      for ai in range(mr):
        a[ai].load(p_tA, ai * self.A_strides.row)
    
      for bj in range(nr):
        b[bj].load(p_tB, bj * self.B_strides.col)
    
      # Update c -- generating loop
      for ci in range(mr):
        for cj in range(nr):
          c[ci][cj].v = ppcvar.fmadd(a[ci], b[cj], c[ci][cj])
          
      p_tA.v = p_tA + self.A_strides.col
      p_tB.v = p_tB + self.B_strides.row
    # /end for k

    return

  def _gepb(self, code):
    a, b, c = self.a, self.b, self.c
    p_tA, p_tB = self.p_tA, self.p_tB
    M = self.dims.M
    mr, nr, nc = self.block.mr, self.block.nr, self.block.nc

    # For each row in C
    for ii in syn_range(code, 0, M, mr):

      # Reset p_tB
      p_tB.v = self.r_tB_addr

      # For each column in C
      for jj in syn_range(code, 0, nc, nr):

        # Set p_tA to the current row in tA
        p_tA.v = self.r_tA_addr + ii * self.A_strides.row # * mr
            
        # Set p_tB to the current col in tB
        p_tB.v = self.r_tB_addr + jj * self.B_strides.col
        
        # Zero the c register block
        ppc.fsubx(c[0][0], c[0][0], c[0][0])
        for ci in range(mr):
          for cj in range(nr):
            c[ci][cj].copy_register(c[0][0])

        # self.k_loop_simple(code)
        if self.mode == gepb_simple:
          self.k_loop_simple(code)                        
        elif self.mode == gepb_prefetch:
          self.k_loop_prefetch_simple(code)              
        elif self.mode == gepb_prefetch_hand:
          self.k_loop_prefetch(code)              
        else:
          raise Exception("Unknown inner loop mode: %s" % (str(self.mode)))
        
        # Save c_current to c_aux 
        # (this is OK performance-wise)
        for ci in range(mr):
          for cj in range(nr):
            c[ci][cj].store(self.p_C_aux[ci], cj * self.C_strides.col)
            
        # Increment the sub-matrix in C_aux
        for ci in range(mr):
          self.p_C_aux[ci].v = self.p_C_aux[ci] + self.C_strides.col * nr

      # /end for jj

      # Reset p_C_aux
      for ci in range(mr):
        self.p_C_aux[ci].v = self.r_C_aux_addr + ci * (nc * self.C_strides.col)

      # Copy C_aux to C
      # self.c_aux_save_simple(code)
      self.c_aux_save_hand(code)

      # Increment p_C
      for ci in range(mr):
        self.p_C[ci].v = self.p_C[ci] + self.vC_row_stride * mr

    # /end for ii

    return
  
  def synthesize(self, code, M, K, N, kc, nc, mr = 1, nr = 1, _transpose = False): 
    """
    tA is M  x nc
    tB is nc x kc
    C  is M  x nc
    I  is the current block column in C
    """


    old_code = ppc.get_active_code()
    ppc.set_active_code(code)

    self._init_constants(M, K, N, kc, nc, mr, nr, _transpose)

    self._init_vars()
    self._load_params()
    self._init_pointers()

    self._gepb(code)
    
    ppc.set_active_code(old_code)
    return


class SynGEPP:
  def __init__(self, gepb_mode = gepb_simple):
    self.gepb_mode = gepb_mode
    return

  def synthesize(self, code, tB, M, K, N, kc, nc, mr = 1, nr = 1):
    old_code = ppc.get_active_code()
    ppc.set_active_code(code)

    gepb  = SynGEPB(self.gepb_mode)
    packb = SynPackB()

    gepb._init_constants(M, K, N, kc, nc, mr, nr, True)
    packb._init_constants(code, tB, N)

    gepb._init_vars()

    # Reuse the C/C_aux registers for B.  They are set in init pointers.
    packb._init_vars2(gepb.p_C, gepb.c[0][0], gepb.r_tB_addr)
    
    gepb._load_params()
    packb._load_params(pvB = 7)

    # kN = k * N * 8
    # for j in range(0, N * 8, nc * 8):
    for j in syn_iter(code, N, nc):
      # # Pack B into tB -- tB1.transpose(B[k:k+kc, j:j+nc])
      # pack_params.p1 = B_addr + kN + j # (k * N + j) * 8      

      packb.vN.v = N
      packb._pack_b(code)

      # proc.execute(cgepb, params = pm)
      gepb._init_pointers()
      gepb._gepb(code)

      # pm.p3 += nc8      
      gepb.r_C_addr.v = gepb.r_C_addr + nc * 8

      packb.vB.v = packb.vB + nc * 8      

    # /end for j

    ppc.set_active_code(old_code)
    return 


def syn_gemm(A, B, C, mc, kc, nc, mr=1, nr=1, gepb_mode = gepb_simple):
  """
  """
  cgepb = synppc.InstructionStream()
  cpackb = synppc.InstructionStream()  
  proc = synppc.Processor()

  gepb = SynGEPB(gepb_mode)  

  packb = SynPackB()
  
  M, N = C.shape
  K = A.shape[0]

  nc = min(nc, N)
  kc = min(kc, K)
  mc = min(mc, M)

  tA = Numeric.zeros((M, kc), typecode = Numeric.Float)
  tB = Numeric.zeros((nc, kc), typecode = Numeric.Float) + 14.0
  C_aux = Numeric.zeros((mr, nc), typecode=Numeric.Float)

  cgepb.set_debug(True)
  gepb.synthesize(cgepb, M, K, N, kc, nc, mr, nr, _transpose = True)
  cgepb.cache_code()
  # cgepb.print_code()

  cpackb.set_debug(True)
  packb.synthesize(cpackb, tB, N)
  cpackb.cache_code()
  # cpackb.print_code()  

  B_addr = synppc.array_address(B)
  C_addr = synppc.array_address(C)

  pack_params = synppc.ExecParams()
  pm = synppc.ExecParams()

  pm.p1 = synppc.array_address(tA)
  pm.p2 = synppc.array_address(tB)
  pm.p3 = C_addr
  pm.p4 = synppc.array_address(C_aux)  

  nc8 = nc * 8
  total = 0.0

  start = time.time()
  
  # print hex(pm.p3), hex(pm.p4)
  k = 0
  for k in range(0, K, kc):
    # Pack A into tA
    tA[:,:] = A[:,k:k+kc]

    pm.p3 = C_addr

    # kN = B_addr + k * N * 8
    pack_params.p1 =  B_addr + k * N * 8
    for j in range(0, N, nc):
      # print k, j, M, K, N, kc, nc, mr, nr
      # Pack B into tB --
      # tB[:,:] = Numeric.transpose(B[k:k+kc, j:j+nc])
      proc.execute(cpackb, params = pack_params)

      # start1 = time.time()
      proc.execute(cgepb, params = pm)
      # stop1  = time.time()
      # total += stop1 - start1
      # print 'ping'
      pack_params.p1 +=  nc8
      pm.p3 += nc8 

  end = time.time()

  return end - start

def syn_gemm_prefetch(A, B, C, mc, kc, nc, mr=1, nr=1):
  return syn_gemm(A, B, C, mc, kc, nc, mr, nr, gepb_mode = gepb_prefetch)

def syn_gemm_hand(A, B, C, mc, kc, nc, mr=1, nr=1):
  return syn_gemm(A, B, C, mc, kc, nc, mr, nr, gepb_mode = gepb_prefetch_hand)
    
   
def syn_gemm_pp(A, B, C, mc, kc, nc, mr=1, nr=1, gepb_mode = gepb_simple):
  """
  """
  cgepp = synppc.InstructionStream()
  proc = synppc.Processor()

  gepp = SynGEPP(gepb_mode)  
  
  M, N = C.shape
  K = A.shape[0]

  nc = min(nc, N)
  kc = min(kc, K)
  mc = min(mc, M)

  tA = Numeric.zeros((M, kc), typecode = Numeric.Float)
  tB = Numeric.zeros((nc, kc), typecode = Numeric.Float) + 14.0
  C_aux = Numeric.zeros((mr, nc), typecode=Numeric.Float) 

  cgepp.set_debug(True)
  gepp.synthesize(cgepp, tB, M, K, N, kc, nc, mr, nr)
  cgepp.cache_code()
  # cgepp.print_code()

  B_addr = synppc.array_address(B)
  C_addr = synppc.array_address(C)

  pack_params = synppc.ExecParams()
  pm = synppc.ExecParams()

  pm.p1 = synppc.array_address(tA)
  pm.p2 = synppc.array_address(tB)
  pm.p3 = C_addr
  pm.p4 = synppc.array_address(C_aux)  

  nc8 = nc * 8
  total = 0.0

  start = time.time()

  k = 0
  for k in range(0, K, kc):
    # Pack A into tA
    tA[:,:] = A[:,k:k+kc]

    pm.p3 = C_addr
    pm.p5 = B_addr + k * N * 8
    proc.execute(cgepp, params = pm)

  end = time.time()

  return end - start

def syn_gemm_pp_prefetch(A, B, C, mc, kc, nc, mr=1, nr=1):
  return syn_gemm_pp(A, B, C, mc, kc, nc, mr, nr, gepb_mode = gepb_prefetch)

def syn_gemm_pp_hand(A, B, C, mc, kc, nc, mr=1, nr=1):
  return syn_gemm_pp(A, B, C, mc, kc, nc, mr, nr, gepb_mode = gepb_prefetch_hand)


# ------------------------------------------------------------
# Test Helpers
# ------------------------------------------------------------

def _mflops(m, k, n, elapsed):
  return 2 * m * k * n / (elapsed * 1.0e6)

class _result:
  """
  Container for holding and printing reuslts.
  """

  def __init__(self, name, m, k, n, mc, kc, nc, times):
    self.name = name
    self.short_name = name[:3] + '...' + name[-10:]
    if len(self.short_name) < 20:
      self.short_name += ' ' * (20 - len(self.short_name))
    self.size = (m, k, n)
    self.blocks = (mc, kc, nc)
    self.m = m
    self.avg = reduce(lambda a,b: a + b, times) / float(len(times))
    self.min = min(times)
    self.max = max(times)

    self.avg_mflops = _mflops(m, k, n, self.avg)
    self.min_mflops = _mflops(m, k, n, self.max)
    self.max_mflops = _mflops(m, k, n, self.min)
    return

  def __str__(self, sep = '\t'):
    self.sep = sep
    return '%(short_name)s%(sep)s%(m)s%(sep)s%(blocks)s%(sep)s%(avg_mflops).4f%(sep)s%(min_mflops).4f%(sep)s%(max_mflops).4f%(sep)s' % \
           self.__dict__
  

def _validate(name, m, n, k, C, C_valid):
  """
  Validate by comparing the contents of C to the contents of C_valid
  The eq matrix should contain all 1s if the matrices are the same.
  """

  # Compare the matrices and flatten the results
  eq = C == C_valid
  eq.shape = (eq.shape[0] * eq.shape[1],)

  # Reduce the results: 1 is valid, 0 is not
  if Numeric.product(eq) == 0:
    eq.shape = C.shape
    for i, row in enumerate(eq[:]):
      # or i == 3 or i == 4
      # (i == 0 or i == 1 or i == 2 ) and 
      if Numeric.product(row) == 0:
        if True or (0 <= i <= 3): # (i == 99999):
          print 'row', i, 'failed'
          print C[i,:4]
          print C_valid[i,:4]
          print row#[:4]
      else:
        print 'row', i, 'succeeded'
    raise Exception("Algorithm '%s' failed validation for %d x %d x %d matrices" %
                    (name, m, k, n))
  print name, 'passed'
  return

def run_alg(alg, A, B, C, mc, kc, nc, mr, nr, niters):
  """
  Time an experiment niters times.
  """
  times = []
  for i in range(niters):
    Numeric.multiply(C, 0.0, C)
    start = time.time()
    t = alg(A, B, C, mc, kc, nc, mr, nr)
    end   = time.time()

    if t is not None:
      times.append(t)
    else:
      times.append(end - start)

  return times

def create_matrices(m, k, n):
  # A = Numeric.arange(m*k, typecode = Numeric.Float)
  A = Numeric.zeros((m,k), typecode = Numeric.Float) + 1.0
  # B = Numeric.arange(k*n, typecode = Numeric.Float)
  B = Numeric.zeros((k,n), typecode = Numeric.Float) + 1.0
  C = Numeric.zeros((m,n), typecode = Numeric.Float)

  A.shape = (m, k)
  B.shape = (k, n)
  C.shape = (m, n)

  return A, B, C

# ------------------------------------------------------------
# Unit Tests
# ------------------------------------------------------------

def test_gebp_opt1():
  m, k, n = (32, 32, 128)
  A, B, C = create_matrices(m, k, n)
  nc = 32

  _gebp_opt1(A, B, C, nc)

  C_valid = Numeric.matrixmultiply(A, B)

  _validate('test_gebp_opt1', m,n,k, C, C_valid)
  return


def test_syn_gepb():
  gepb = SynGEPB()
  code = synppc.InstructionStream()
  code.set_debug(True)
  
  m, k, n = (128, 32, 32)
  A, B, C = create_matrices(m, k, n)
  kc = k
  nc = 32

  mr = 4
  nr = 4
  
  C_aux = Numeric.zeros((mr, nc), typecode=Numeric.Float) # + 13.0
  
  A_addr = synppc.array_address(A)
  B_addr = synppc.array_address(B)
  C_addr = synppc.array_address(C)
  C_aux_addr = synppc.array_address(C_aux)

  gepb.synthesize(code, m, k, n, kc, nc, mr, nr) # , A_addr, B_addr, C_addr)

  # code.print_code()
  
  params = synppc.ExecParams()
  params.p1 = A_addr
  params.p2 = B_addr
  params.p3 = C_addr
  params.p4 = C_aux_addr  
  
  # code.print_code()
  proc = synppc.Processor()
  proc.execute(code, params = params)

  C_valid = Numeric.matrixmultiply(A, B)

  _validate('syn_gepb', m,n,k, C, C_valid)
  return

def test_gepp_blk_var1():
  m, k, n = (128, 128, 128)
  A, B, C = create_matrices(m, k, n)

  mc = 32
  nc = 32

  _gepp_blk_var1(A[:,0:mc], B[0:mc,:], C, mc, nc)

  C_valid = Numeric.zeros((m, n), typecode=Numeric.Float)
  for i in range(0, m, mc):
    C_valid[i:i+mc,:] = Numeric.matrixmultiply(A[i:i+mc,0:mc], B[0:mc,:])

  _validate('test_gepp_blk_var1', m,n,k, C, C_valid)
  return

def test_numeric_gemm_var1():
  # m, k, n = (128, 128, 128)
  m, k, n = (8, 8, 8)
  A, B, C = create_matrices(m, k, n)

  mc = 32
  kc = 32
  nc = 32

  numeric_gemm_var1(A, B, C, mc, kc, nc)

  C_valid = Numeric.matrixmultiply(A, B)
  
  _validate('test_gepp_blk_var1', m,n,k, C, C_valid)
  return

def test_numeric_gemm_var1_flat():
  m, k, n = (256, 256, 256)
  # m, k, n = (8, 8, 8)
  A, B, C = create_matrices(m, k, n)

  mc = 32
  kc = 32
  nc = 32

  numeric_gemm_var1_flat(A, B, C, mc, kc, nc)

  C_valid = Numeric.matrixmultiply(A, B)
  
  _validate('test_gepp_blk_var1', m,n,k, C, C_valid)
  return

def test_numeric_gemm_var1_row():
  m, k, n = (256, 256, 256)
  # m, k, n = (8, 8, 8)
  A, B, C = create_matrices(m, k, n)

  nc = 32
  kc = 32
  mr = 1
  nr = 1
  numeric_gemm_var1_row(A, B, C, nc, kc, mr, nr)

  C_valid = Numeric.matrixmultiply(A, B)
  
  _validate('test_gepp_blk_var1', m,n,k, C, C_valid)
  return

def test_syn_gemm():
  # m, k, n = (512, 512, 512)
  m, k, n = (256, 256, 256)
  # m, k, n = (128, 128, 128)
  # m, k, n = (64, 64, 64)
  
  # m, k, n = (8, 8, 8)
  A, B, C = create_matrices(m, k, n)

  mc = 32
  kc = 64
  nc = 32
  
  mr = 4
  nr = 4

  syn_gemm(A, B, C, mc, kc, nc, mr = mr, nr = nr)
  C[:,:] = C * 0
  syn_gemm(A, B, C, mc, kc, nc, mr = mr, nr = nr)  

  C_valid = Numeric.matrixmultiply(A, B)
  
  _validate('syn_gemm', m,n,k, C, C_valid)
  return

def test_syn_gemm_pp():
  # m, k, n = (512, 512, 512)
  m, k, n = (256, 256, 256)
  # m, k, n = (128, 128, 128)
  # m, k, n = (64, 64, 64)
  
  # m, k, n = (8, 8, 8)
  A, B, C = create_matrices(m, k, n)

  mc = 32
  kc = 32
  nc = 64
  
  mr = 4
  nr = 4

  syn_gemm_pp(A, B, C, mc, kc, nc, mr = mr, nr = nr)

  print A[0, :4]
  print B[0, :4]  
  C_valid = Numeric.matrixmultiply(A, B)

  _validate('syn_gemm_pp', m,n,k, C, C_valid)
  return


def test_syn_pack_b():

  # Create a 10x10 B array of indices
  B = Numeric.zeros((10, 10), typecode = Numeric.Float)
  a = Numeric.arange(10)

  for i in range(10):
    B[i,:] = a + i * 10

  B.shape = (10,10)

  # Create the packed array
  tB = Numeric.arange(25, typecode = Numeric.Float) * 0.0
  tB.shape = (5,5)

  B_offset = 3 * 10 + 0

  K, N = B.shape
  nc, kc = tB.shape

  pack_b = SynPackB()
  
  code = synppc.InstructionStream()
  proc = synppc.Processor()
  params = synppc.ExecParams()

  pack_b.synthesize(code, tB, N)
  
  params.p1 = synppc.array_address(B) + B_offset * 8

  proc.execute(code, params = params)
  
  
  # Validate
  B.shape  = (K * N,)

  tB_valid = Numeric.arange(nc*kc, typecode = Numeric.Float) * 0.0

  for i in range(kc):
    B_row = B_offset + i * N
    for j in range(nc):
      b  = B_row + j
      tb = j * kc + i
      tB_valid[tb] = B[b]
        
  tB_valid.shape = (nc,kc)
  B.shape  = (K, N)

  _validate('syn_pack_b', nc, nc, N, tB, tB_valid)

  return

def test(algs, niters = 5, validate = False):
  """
  Test a numcer of algorithms and return the results.
  """

  if type(algs) is not list:
    algs = [algs]

  results = []
  m,n,k = (64,64,64)

  # Cache effects show up between 2048 and 4096 on a G4
  tests = [8, 16, 32, 64] # , 128, 256, 512, 768, 1024, 2048, 4096]: [2048, 4096]: #
  tests = [128, 256, 512, 1024, 2048, 4096]
  # tests = [4096]
  # tests = [2048]
  # tests = [512] 
  for size in tests: 
    m,n,k = (size, size, size)
    A, B, C = create_matrices(m, k, n)

    if validate:
      C_valid = Numeric.matrixmultiply(A, B)

    for alg in algs:
      print alg.func_name, size
      if validate:
        alg(A, B, C)
        _validate(alg.func_name, m, n, k, C, C_valid)

      # KC = [32, 64, 128, 256] # , 512]
      KC = [32, 64, 128, 256] # , 256] # , 512]      
      # KC = [64]
      KC = [128, 256]
      KC = [256]
      NC = [32, 64, 128, 256] # , 512]
      # NC = [32, 32, 32] 
      # NC = [256, 256, 256]
      # NC = [128, 128, 128]
      NC = [128, 256]
      NC = [128]

      for mc in [32]: # , 64, 128, 256]:
        for kc in KC:
          for nc in NC:
            # print '  ', mc, kc, nc
            # try:
            if True:
              times = run_alg(alg, A, B, C, mc, kc, nc, 4, 4, niters)
              result = _result(alg.func_name, m, k, n, mc, kc, nc, times)
              results.append(result)
              print result
            # except:
            #   print 'Failed: ', alg.func_name, m, k, n, mc, kc, nc, times

  return results

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
  import sys
  colors = 'rgbko'

  # algs = [numeric_mm, numeric_gemm_var1_flat, numeric_gemm_var1_row, syn_gemm]
  algs = [# numeric_gemm_var1_row,
          syn_gemm, syn_gemm_prefetch, syn_gemm_hand,
          syn_gemm_pp, syn_gemm_pp_prefetch, syn_gemm_pp_hand]
  # algs = [syn_gemm_hand, syn_gemm_pp_hand]
  
  # algs = [numeric_gemm_var1_flat, numeric_gemm_var1_row]
  results = test(algs)

  if 'plot' in sys.argv:
    import pylab
    algs = {} # alg: [max...]
    size = [] # m

    for result in results:
      if not algs.has_key(result.name):
        algs[result.name] = []

      algs[result.name].append(result.max_mflops)

      if result.size[0] not in size:
        size.append(result.size[0])

    for i, alg in enumerate(algs.keys()):
      times = algs[alg]
      lines = pylab.plot(size, times)
      pylab.setp(lines, color=colors[i])
      print alg, colors[i]
    pylab.show()
  else:
    print 'Algorithm           \tSize\tBlocks        \tAvg      \tMin      \tMax      \t'

    for result in results:
      print str(result)

  return

if __name__=='__main__':
  # test_gebp_opt1()
  # test_gepp_blk_var1()
  # test_numeric_gemm_var1()
  # test_numeric_gemm_var1_flat()
  # test_numeric_gemm_var1_row()
  # test_syn_gepb()
  # test_syn_gemm()
  # test_syn_gemm_pp()
  # test_syn_pack_b()
  main()
  # proto_reg_mm_4x4()
  
  
