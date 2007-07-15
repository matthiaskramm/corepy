# PLaying around with FFTs in Python

# Based on:
#   http://en.literateprograms.org/Cooley-Tukey_FFT_algorithm_(C)
#   http://cnx.org/content/m12016/latest/
import math
import Numeric
import FFT
import time

def complex_from_polar(r, theta):
  return complex(r * math.cos(theta), r * math.sin(theta))

def DFT_naive(x):
  N = len(x)
  X = [0+0j for i in range(N)]

  for k in range(N):
    for n in range(N):
      X[k] = X[k] + x[n] * complex_from_polar(1, -2.0 * math.pi * n * k / N)
  
  return X


def DFT_naive_roots(x):
  N = len(x)
  X = [0+0j for i in range(N)]

  Nth_root = [complex_from_polar(1, -2.0*math.pi*k/N) for k in range(N)]
    
  for k in range(N):
    for n in range(N):
      X[k] = X[k] + x[n] * Nth_root[(n*k) % N]
  
  return X


def FFT_simple(x):
  if math.log(len(x), 2) % 1 != 0:
    raise Expcetion('len(x) [%d] must be a power of 2' % len(x))
    
  N = len(x)
  X = [0+0j for k in range(N)]

  if N == 1:
    X[0] = x[0]
    return X

  if N == 16:
    print 'PING!'
    return DFT_naive_roots(x)
  
  e = [0+0j for k in range(N/2)]
  d = [0+0j for k in range(N/2)]
  
  for k in range(N/2):
    e[k] = x[2*k]
    d[k] = x[2*k + 1]

  E = FFT_simple(e)
  D = FFT_simple(d)

  for k in range(N/2):
    D[k] = complex_from_polar(1, -2.0 * math.pi * k / N) * D[k]

  for k in range(N/2):
    X[k]       = E[k] + D[k]
    X[k + N/2] = E[k] - D[k]

  return X

# ------------------------------------------------------------
# Tests
# ------------------------------------------------------------

def test_DFT_native():
  x = Numeric.arange(256, typecode=Numeric.Complex)

  start = time.time()
  for i in range(10):
    X = DFT_naive(x)
  stop = time.time()
  print '%.6f' % ((stop - start) / 10.0)
  
  XX = FFT.fft(x)

  # for x1, x2 in zip(X, XX):
  #    print x1, x2
  return

def test_DFT_native_roots():
  x = Numeric.arange(256, typecode=Numeric.Complex)

  start = time.time()
  for i in range(10):
    X = DFT_naive_roots(x)
  stop = time.time()
  print '%.6f' % ((stop - start) / 10.0)

  XX = FFT.fft(x)

  # for x1, x2 in zip(X, XX):
  #   print x1, x2
  return

def test_FFT_simple():
  x = Numeric.arange(256, typecode=Numeric.Complex)

  start = time.time()
  for i in range(1):
    X = FFT_simple(x)
  stop = time.time()
  print '%.6f' % ((stop - start) / 1.0)

  XX = FFT.fft(x)

  for x1, x2 in zip(X, XX):
    print x1, x2
  return

if __name__=='__main__':
  # test_DFT_native()
  # test_DFT_native_roots()
  test_FFT_simple()
