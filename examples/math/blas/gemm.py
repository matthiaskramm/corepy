# Matrix-Matrix Multiplication Examples

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

def numeric_mm(A, B, C, mc, kc, nc, mr=1, kr=1, nr=1):
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

def numeric_gemm_var1(A, B, C, mc, kc, nc, mr=1, kr=1, nr=1):
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


def numeric_gemm_var1_flat(A, B, C, mc, kc, nc, mr=1, kr=1, nr=1):
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

def numeric_gemm_var1_row(A, B, C, mc, kc, nc, mr=1, kr=1, nr=1):
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

from corepy.arch.ppc.lib.iterators import syn_range

class SynGEPB:

  def synthesize(self, code, M, K, N, kc, nc): 
    """
    tA is M  x nc
    tB is nc x kc
    C  is M  x nc
    I  is the current block column in C
    """

    old_code = ppc.get_active_code()
    ppc.set_active_code(code)

    # Address variables
    r_tA_addr = ppcvar.UnsignedWord()
    r_tB_addr = ppcvar.UnsignedWord()
    r_C_addr  = ppcvar.UnsignedWord()
    r_C_aux_addr  = ppcvar.UnsignedWord()    

    ppc.addi(r_tA_addr, 3, 0)
    ppc.addi(r_tB_addr, 4, 0)
    ppc.addi(r_C_addr,  5, 0)
    ppc.addi(r_C_aux_addr,  6, 0)    
    
    p_tA = ppcvar.UnsignedWord()
    p_tB = ppcvar.UnsignedWord()
    p_C  = ppcvar.UnsignedWord()
    p_C_aux = ppcvar.UnsignedWord()    

    p_tA.v = r_tA_addr
    p_tB.v = r_tB_addr    
    p_C.v  = r_C_addr
    p_C_aux.v  = r_C_aux_addr    
    
    tB_offset = ppcvar.UnsignedWord()

    # Inner loop variables
    a = ppcvar.DoubleFloat()
    b = ppcvar.DoubleFloat()
    c = ppcvar.DoubleFloat()            
    c_current = ppcvar.DoubleFloat()

    # Constants
    C_col_stride = 8    
    C_row_stride = N * C_col_stride

    # Assume A and B are packed
    A_col_stride = 8
    A_row_stride = kc * A_col_stride

    B_col_stride = 8
    B_row_stride = nc * B_col_stride
    
    # For each row in C
    for ii in syn_range(code, 0, M):
      # Set p_C to the current row in C
      p_C.v = r_C_addr + ii * C_row_stride
      p_C_aux.v = r_C_aux_addr

      # Reset p_tB
      p_tB.v = r_tB_addr

      # For each column in C
      for jj in syn_range(code, 0, nc * C_col_stride, C_col_stride):
        
        # Set p_tA to the current row in tA
        p_tA.v = r_tA_addr + ii * A_row_stride

        # Register-scale BP

        # Initialize c_current 
        c_current.v = 0.0

        # for j in syn_range(code, 0, nc * 8, 8):
        tB_offset.v = 0

        for k in syn_range(code, 0, kc * 8, 8):
            
            # Load the next values from tA and tB
            a.load(p_tA, k)
            b.load(p_tB, tB_offset)
            
            # Update c
            c_current.v = ppcvar.fmadd(a, b, c_current)
            
            # Increment tB's offset by a full row
            tB_offset.v = tB_offset + B_row_stride
          # /end for j

        # /end for k

        # Set p_tB to the current col in tB
        p_tB.v = p_tB + B_col_stride

        # Save c_current to c_aux
        c_current.store(p_C_aux, jj)

      # /end for jj

      # Copy C_aux to C
      for jj in syn_range(code, 0, nc * C_col_stride, C_col_stride):
        c.load(p_C, jj)
        c_current.load(r_C_aux_addr, jj)
        c.v = c + c_current
        c.store(p_C, jj)
        
    # /end for ii

    ppc.set_active_code(old_code)
    return

# class SynGEPP:

#   def synthesize(self, code, M, K, N, kc, nc):

#     r_B_addr = ppcvar.UnsignedWord()
#     r_B_addr
#     for j in range(0, N, nc):
#       jnc = j+nc
#       # Pack B into tB
#       tB[:,:] = B[k:k+kc, j:jnc]

#       proc.execute(code, params = params)
#       params.p3 += nc8

#     return
  
def syn_gemm(A, B, C, mc, kc, nc, mr=1, kr=1, nr=1):
  """
  """
  code = synppc.InstructionStream()
  proc = synppc.Processor()

  gepb = SynGEPB()
  params = synppc.ExecParams()

  C_aux = Numeric.zeros(nc, typecode=Numeric.Float)

  M, N = C.shape
  K = A.shape[0]

  nc = min(nc, N)
  kc = min(kc, K)
  mc = min(mc, M)
  # code.set_debug(True)
  gepb.synthesize(code, M, N, K, kc, nc)
  code.cache_code()
  # code.print_code()
  
  start = time.time()

  tA = Numeric.zeros((M, kc), typecode = Numeric.Float)
  tB = Numeric.zeros((kc, nc), typecode = Numeric.Float)  

  params.p1 = synppc.array_address(tA)
  params.p2 = synppc.array_address(tB)

  params.p4 = synppc.array_address(C_aux)  

  C_addr = synppc.array_address(C)
  nc8 = nc * 8

  k = 0
  for k in range(0, K, kc):
    # Pack A into tA
    tA[:,:] = A[:,k:k+kc]

    params.p3 = C_addr

    for j in range(0, N, nc):
      jnc = j+nc
      # Pack B into tB
      tB[:,:] = B[k:k+kc, j:jnc]

      proc.execute(code, params = params)
      params.p3 += nc8

      #  for i in range(0, M): # , mc):
      #    # Ci = AiB + Ci
      #    # imc = i+mc
      #    ABi = Numeric.matrixmultiply(tA[i,:], tB)
      #    Numeric.add(C[i,j:jnc], ABi, C[i,j:jnc])

  end = time.time()
  return end - start


# ------------------------------------------------------------
# Test Harness
# ------------------------------------------------------------

def _mflops(m, k, n, elapsed):
  return 2 * m*n*k / (elapsed * 1.0e6)

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
      if Numeric.product(row) == 0:
        print 'row', i, 'failed'
        # print C[i,:]
        # print C_valid[i,:]
        
    raise Exception("Algorithm '%s' failed validation for %d x %d x %d matrices" %
                    (name, m, k, n))
  return

def run_alg(alg, A, B, C, mc, kc, nc, niters):
  """
  Time an experiment niters times.
  """
  times = []
  for i in range(niters):
    start = time.time()
    t = alg(A, B, C, mc, kc, nc)
    end   = time.time()

    if t is not None:
      times.append(t)
    else:
      times.append(end - start)

  return times

def create_matrices(m, k, n):
  A = Numeric.arange(m*k, typecode = Numeric.Float)
  B = Numeric.arange(k*n, typecode = Numeric.Float)
  C = Numeric.zeros(m*n, typecode = Numeric.Float)  

  A.shape = (m, k)
  B.shape = (k, n)
  C.shape = (m, n)
  return A, B, C

def test(algs, niters = 2, validate = False):
  """
  Test a numcer of algorithms and return the results.
  """

  if type(algs) is not list:
    algs = [algs]

  results = []
  m,n,k = (64,64,64)

  # Cache effects show up between 2048 and 4096 on a G4
  tests = [8, 16, 32, 64] # , 128, 256, 512, 768, 1024, 2048, 4096]: [2048, 4096]: #
  tests = [128, 256, 512] # , 1024, 2048] #, 4096]
  tests = [512]
  for size in tests: 
    m,n,k = (size, size, size)
    A, B, C = create_matrices(m, k, n)

    if validate:
      C_valid = Numeric.matrixmultiply(A, B)

    for alg in algs:
      Numeric.multiply(C, 0, C)
      print alg.func_name, size
      if validate:
        alg(A, B, C)
        _validate(alg.func_name, m, n, k, C, C_valid)

      for mc in [32, 64, 128, 256]:
        for kc in [32, 64, 128, 256]:
          for nc in [32, 64, 128, 256]:
            print '  ', mc, kc, nc
            times = run_alg(alg, A, B, C, mc, kc, nc, niters)
            result = _result(alg.func_name, m, k, n, mc, kc, nc, times)
            results.append(result)

  return results

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

  C_aux = Numeric.zeros(kc, typecode=Numeric.Float)
  
  A_addr = synppc.array_address(A)
  B_addr = synppc.array_address(B)
  C_addr = synppc.array_address(C)
  C_aux_addr = synppc.array_address(C_aux)
  
  gepb.synthesize(code, m, k, n, kc, nc) # , A_addr, B_addr, C_addr)

  params = synppc.ExecParams()
  params.p1 = A_addr
  params.p2 = B_addr
  params.p3 = C_addr
  params.p4 = C_aux_addr  
  
  # code.print_code()
  proc = synppc.Processor()
  proc.execute(code, params = params)

  C_valid = Numeric.matrixmultiply(A, B)

  _validate('test_gebp_opt1', m,n,k, C, C_valid)
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
  kr = 1

  numeric_gemm_var1_row(A, B, C, nc, kc, mr, kr)

  C_valid = Numeric.matrixmultiply(A, B)
  
  _validate('test_gepp_blk_var1', m,n,k, C, C_valid)
  return

def test_syn_gemm():
  m, k, n = (512, 512, 512)
  # m, k, n = (8, 8, 8)
  A, B, C = create_matrices(m, k, n)

  mc = 32
  kc = 32
  nc = 64
  
  mr = 1
  kr = 1

  syn_gemm(A, B, C, mc, kc, nc)

  C_valid = Numeric.matrixmultiply(A, B)
  
  _validate('test_gepp_blk_var1', m,n,k, C, C_valid)
  return

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
  import sys
  colors = 'rgbko'

  # algs = [numeric_mm, numeric_gemm_var1_flat, numeric_gemm_var1_row, syn_gemm]
  algs = [syn_gemm]  
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
    print 'Algorithm           \tSize        \tBlocks        \tAvg      \tMin      \tMax      \t'

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
  main()
  
  
