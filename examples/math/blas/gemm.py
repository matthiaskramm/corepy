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

def numeric_mm(A, B, C):
  C[:,:] = Numeric.matrixmultiply(A, B)
  return


def _gebp_opt1(A, B, C, nb):
  n = C.shape[1]

  # Assume B is packed
  # Pack A into tA
  
  for j in range(0, n, nb):
    # Load Bj into cache
    # Load Cj into cache

    # Cj += ABj + Cj
    ABj = Numeric.matrixmultiply(A, B[:,j:j+nb])
    Numeric.add(C[:,j:j+nb], ABj, C[:,j:j+nb])

    # Store Caux into memory
  return

def _gepp_blk_var1(A, B, C, mb, nb):
  m = C.shape[1]
  
  # Pack B into tB

  for i in range(0, m, mb):
    _gebp_opt1(A[i:i+mb,:], B, C[i:i+mb, :], nb)
  
  return

_MB, _KB, _NB = 128, 128, 256

def numeric_gemm_var1(A, B, C, mb = _MB, kb = _KB, nb = _NB):
  """
  GEMM_VAR1, top branch 
  """
  
  m, n = C.shape
  k = A.shape[0]

  mb = min(mb, m)
  kb = min(kb, k)
  nb = min(nb, n)  
  
  for k in range(0, k, kb):
    _gepp_blk_var1(A[:,k:k+kb], B[k:k+kb,:], C, mb, nb)
  return


def numeric_gemm_var1_flat(A, B, C, mb = _MB, kb = _KB, nb = _NB):
  M, N = C.shape
  K = A.shape[0]

  mb = min(mb, M)
  kb = min(kb, K)
  nb = min(nb, N)  

  tA = Numeric.zeros((mb, kb), typecode = Numeric.Float)
  tB = Numeric.zeros((kb, N), typecode = Numeric.Float)  

  for k in range(0, K, kb):
    # Pack B into tB
    tB[:,:] = B[k:k+kb:,:]
    
    for i in range(0, M, mb):

      # Pack A into tA
      tA[:,:] = A[i:i+mb,k:k+kb]
      
      for j in range(0, N, nb):
        # Cj += ABj + Cj
        ABj = Numeric.matrixmultiply(tA, tB[:,j:j+nb])
        Numeric.add(C[i:i+mb:,j:j+nb], ABj, C[i:i+mb:,j:j+nb])
        
        # Store Caux into memory
  return




# ------------------------------------------------------------
# Test Harness
# ------------------------------------------------------------

def _mflops(m, n, k, elapsed):
  return 2 * m*n*k / (elapsed * 1.0e6)

class _result:
  """
  Container for holding and printing reuslts.
  """

  def __init__(self, name, m, n, k, times):
    self.name = name

    self.size = (m, n, k)
    
    self.avg = reduce(lambda a,b: a + b, times) / float(len(times))
    self.min = min(times)
    self.max = max(times)

    self.avg_mflops = _mflops(m, n, k, self.avg)
    self.min_mflops = _mflops(m, n, k, self.max)
    self.max_mflops = _mflops(m, n, k, self.min)
    return

  def __str__(self, sep = '\t'):
    self.sep = sep
    return '%(name)s%(sep)s%(size)s%(sep)s%(avg_mflops).4f%(sep)s%(min_mflops).4f%(sep)s%(max_mflops).4f%(sep)s' % \
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
        print C[i,:]
        print C_valid[i,:]        
    raise Exception("Algorithm '%s' failed validation for %d x %d x %d matrices" %
                    (name, m, k, n))
  return

def run_alg(alg, A, B, C, niters):
  """
  Time an experiment niters times.
  """
  times = []
  for i in range(niters):
    start = time.time()
    alg(A, B, C)
    end   = time.time()
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
  Test a number of algorithms and return the results.
  """

  if type(algs) is not list:
    algs = [algs]

  results = []
  m,n,k = (64,64,64)

  # Cache effects show up between 2048 and 4096 on a G4
  tests = [8, 16, 32, 64] # , 128, 256, 512, 768, 1024, 2048, 4096]: [2048, 4096]: #
  tests = [128, 256, 512, 1024, 2048, 4096]
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
        
      times = run_alg(alg, A, B, C, niters)
      result = _result(alg.func_name, m, n, k, times)
      results.append(result)

  return results

# ------------------------------------------------------------
# Unit Tests
# ------------------------------------------------------------

def test_gebp_opt1():
  m, k, n = (32, 32, 128)
  A, B, C = create_matrices(m, k, n)
  nb = 32

  _gebp_opt1(A, B, C, nb)

  C_valid = Numeric.matrixmultiply(A, B)

  _validate('test_gebp_opt1', m,n,k, C, C_valid)
  return

def test_gepp_blk_var1():
  m, k, n = (128, 128, 128)
  A, B, C = create_matrices(m, k, n)

  mb = 32
  nb = 32

  _gepp_blk_var1(A[:,0:mb], B[0:mb,:], C, mb, nb)

  C_valid = Numeric.zeros((m, n), typecode=Numeric.Float)
  for i in range(0, m, mb):
    C_valid[i:i+mb,:] = Numeric.matrixmultiply(A[i:i+mb,0:mb], B[0:mb,:])

  _validate('test_gepp_blk_var1', m,n,k, C, C_valid)
  return

def test_numeric_gemm_var1():
  # m, k, n = (128, 128, 128)
  m, k, n = (8, 8, 8)
  A, B, C = create_matrices(m, k, n)

  mb = 32
  kb = 32
  nb = 32

  numeric_gemm_var1(A, B, C, mb, kb, nb)

  C_valid = Numeric.matrixmultiply(A, B)
  
  _validate('test_gepp_blk_var1', m,n,k, C, C_valid)
  return

def test_numeric_gemm_var1_flat():
  m, k, n = (256, 256, 256)
  # m, k, n = (8, 8, 8)
  A, B, C = create_matrices(m, k, n)

  mb = 32
  kb = 32
  nb = 32

  numeric_gemm_var1_flat(A, B, C, mb, kb, nb)

  C_valid = Numeric.matrixmultiply(A, B)
  
  _validate('test_gepp_blk_var1', m,n,k, C, C_valid)
  return

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
  import sys
  colors = 'rgbko'

  algs = [numeric_mm, numeric_gemm_var1, numeric_gemm_var1_flat]
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
    
  # print 'Algorithm  \tSize        \tAvg      \tMin      \tMax      \t'

  # for result in results:
  #   print str(result)

  return

if __name__=='__main__':
  # test_gebp_opt1()
  # test_gepp_blk_var1()
  # test_numeric_gemm_var1()
  # test_numeric_gemm_var1_flat()
  main()
  
