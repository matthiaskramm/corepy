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

# PLaying around with FFTs in Python

# Based on:
#   http://en.literateprograms.org/Cooley-Tukey_FFT_algorithm_(C)
#   http://cnx.org/content/m12016/latest/
import math
import Numeric
import FFT
import time


def _rev_bits(bits, l=32):
  """
  Reverse bits
  """
  rev = 0
  
  for i in range(l):
    if bits & (1 << i) != 0:
      rev |= 1 << (l-i - 1)

  return rev


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

def FFT_bit_reversal(x):
  """
  Based on table 12-4 in DSP guide.
  """
  
  N = len(x)
  log_N = math.log(N, 2)
  X = [0+0j for k in range(N)]
  
  for i in range(0, N):
    j = _rev_bits(i, int(log_N))
    X[j] = x[i]

  for l in range(int(log_N)):
    le = 2**(l+1)
    le2 = le / 2

    ur = 1.0
    ui = 0.0

    sr = math.cos(math.pi / le2)
    si = -math.sin(math.pi / le2)

    for j in range(0, le2):
      for i in range(j, N, le):
        ip = i + le2
        tr = X[ip].real * ur - X[ip].imag * ui
        ti = X[ip].real * ui + X[ip].imag * ur
        X[ip] = complex(X[i].real - tr, X[i].imag - ti)
        X[i]  = complex(X[i].real + tr, X[i].imag + ti)
      tr = ur
      ur = tr * sr - ui * si
      ui = tr * si + ui * sr
      
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

  i = 0
  for x1, x2 in zip(X, XX):
    # Complex eq is too sensitive, compare string representations
    s1, s2 = str(x1), str(x2)
    assert(s1 == s2)
  return

def test_FFT_bit_reversal():
  x = Numeric.arange(16, typecode=Numeric.Complex)

  start = time.time()
  for i in range(10):
    X = FFT_bit_reversal(x)
  stop = time.time()
  print '%.6f' % ((stop - start) / 10.0)

  # start = time.time()
  XX = FFT.fft(x)
  # stop = time.time()
  # print '%.6f' % ((stop - start) / 1.0)

  i = 0
  for x1, x2 in zip(X, XX):
    # Complex eq is too sensitive, compare string representations
    s1, s2 = str(x1), str(x2)
    assert(s1 == s2)
  return

if __name__=='__main__':
  # test_DFT_native()
  # test_DFT_native_roots()
  test_FFT_simple()
  test_FFT_bit_reversal()
