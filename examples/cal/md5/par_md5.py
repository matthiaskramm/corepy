import corepy.lib.extarray as extarray

import corepy.arch.cal.isa as cal
import corepy.arch.cal.lib.cal_extended as calex
import corepy.arch.cal.platform as env
import corepy.arch.cal.types as types
import corepy.arch.cal.types.registers as reg

import math
import time

class Context(object):
  def __init__(self):
    self.state = extarray.extarray('I', 4)
    self.count = extarray.extarray('I', 2)
    self.buffer = extarray.extarray('B', 64)

class ParContext(object):
  def __init__(self, number):
    self.number = number
    self.statea = extarray.extarray('I', number)
    self.stateb = extarray.extarray('I', number)
    self.statec = extarray.extarray('I', number)
    self.stated = extarray.extarray('I', number)
    self.count = extarray.extarray('I', 2*number)
    self.buffer = extarray.extarray('B', 64*number)
    #self.buffer = [extarray.extarray('B', 4*number) for i in range(16)]

TIME = 0.0

xcode = None
xproc = None
input_buf = None
output_buf = None

def F(x, y, z, r):
  """
  md5 F 'macro'
  x, y, z, r should be registers that are scalars
  puts result in r
  """
  # return (x & y) | (~x & z)
  global xcode
  temp = xcode.acquire_register()
  cal.iand(r, x, y)       #  x & y
  cal.inot(temp, x)       # temp = ~x
  cal.iand(temp, temp, z) # temp = (~x) & z
  cal.ior(r, r, temp)
  xcode.release_register(temp)
  
def G(x, y, z, r):
  """
  md5 G 'macro'
  x, y, z, r should be registers that are scalars
  puts result in r
  """
  # return (x & z) | (y & ~z)
  global xcode
  temp = xcode.acquire_register()
  cal.iand(r, x, z)       # x & z
  cal.inot(temp, z)       # temp = ~z
  cal.iand(temp, y, temp) # temp = y & ~z
  cal.ior(r, r, temp)
  xcode.release_register(temp)

def H(x, y, z, r):
  """
  md5 H 'macro'
  x, y, z, r should be registers that are scalars
  puts result in r
  """
  # return x ^ y ^ z
  global xcode
  cal.ixor(r, x, y)
  cal.ixor(r, r, z)

def I(x, y, z, r):
  """
  md5 I 'macro'
  x, y, z, r should be registers that are scalars
  puts result in r
  """
  # return y ^ (x | ~z)
  global xcode
  cal.inot(r, z)    # ~z
  cal.ior(r, x, r)  # x | ~z
  cal.ixor(r, y, r) # y ^ (x | ~z)

def FF(a1, b1, c1, d1, x1, s1, ac1):
  global xcode
  l = xcode.acquire_register((ac1, ac1, ac1, ac1))
  temp1 = xcode.acquire_register()
  temp2 = xcode.acquire_register()

  F(b1, c1, d1, temp1)
  cal.iadd(a1, a1, temp1)
  cal.iadd(a1, a1, x1)
  cal.iadd(a1, a1, l)
  
  cal.ishl(temp1, a1, s1)
  cal.ushr(temp2, a1, s1(neg=('x', 'y', 'z', 'w')))
  cal.ior(a1, temp1, temp2)
  cal.iadd(a1, a1, b1)

  xcode.release_register(l)
  xcode.release_register(temp1)
  xcode.release_register(temp2)

def GG(a1, b1, c1, d1, x1, s1, ac1):
  global xcode

  l = xcode.acquire_register((ac1, ac1, ac1, ac1))
  temp1 = xcode.acquire_register()
  temp2 = xcode.acquire_register()

  G(b1, c1, d1, temp1)
  cal.iadd(a1, a1, temp1)
  cal.iadd(a1, a1, x1)
  cal.iadd(a1, a1, l)
  
  cal.ishl(temp1, a1, s1)
  cal.ushr(temp2, a1, s1(neg=('x', 'y', 'z', 'w')))
  cal.ior(a1, temp1, temp2)
  cal.iadd(a1, a1, b1)

  xcode.release_register(l)
  xcode.release_register(temp1)
  xcode.release_register(temp2)

def HH(a1, b1, c1, d1, x1, s1, ac1):
  global xcode

  l = xcode.acquire_register((ac1, ac1, ac1, ac1))
  temp1 = xcode.acquire_register()
  temp2 = xcode.acquire_register()
  H(b1, c1, d1, temp1)
  cal.iadd(a1, a1, temp1)
  cal.iadd(a1, a1, x1)
  cal.iadd(a1, a1, l)
  
  cal.ishl(temp1, a1, s1)
  cal.ushr(temp2, a1, s1(neg=('x', 'y', 'z', 'w')))
  cal.ior(a1, temp1, temp2)
  cal.iadd(a1, a1, b1)

  xcode.release_register(l)
  xcode.release_register(temp1)
  xcode.release_register(temp2)

def II(a1, b1, c1, d1, x1, s1, ac1):
  global xcode

  l = xcode.acquire_register((ac1, ac1, ac1, ac1))
  temp1 = xcode.acquire_register()
  temp2 = xcode.acquire_register()

  I(b1, c1, d1, temp1)
  cal.iadd(a1, a1, temp1)
  cal.iadd(a1, a1, x1)
  cal.iadd(a1, a1, l)
  
  cal.ishl(temp1, a1, s1)
  cal.ushr(temp2, a1, s1(neg=('x', 'y', 'z', 'w')))
  cal.ior(a1, temp1, temp2)
  cal.iadd(a1, a1, b1)

  xcode.release_register(l)
  xcode.release_register(temp1)
  xcode.release_register(temp2)

def ParEncode(num, output, input, length):
  for k in range(num):
    i = 0
    for j in range(0, length, 4):
      output[k*length + j] = input[k*length/4 + i] & 0xff
      output[k*length + j+1] = (input[k*length/4 + i] >> 8) & 0xff
      output[k*length + j+2] = (input[k*length/4 + i] >> 16) & 0xff
      output[k*length + j+3] = (input[k*length/4 + i] >> 24) & 0xff
      i += 1

def ParDecode (num, output, input, inputi, length):
  for k in range(num):
    i = 0
    for j in range(0, length, 4):
      output[k*length/4 + i] = input[k*length + inputi+j]
      output[k*length/4 + i] |= input[k*length + inputi+j+1]<<8
      output[k*length/4 + i] |= input[k*length + inputi+j+2]<<16
      output[k*length/4 + i] |= input[k*length + inputi+j+3]<<24
      i += 1

def ParMD5Transform(parcontext, parblock, blocki):
  num = parcontext.number

  temp_block = extarray.extarray('I', 16*num)
  ParDecode(num, temp_block, parblock, blocki, 64)

  proc = env.Processor(0)

  N = int(math.sqrt(num/4))
  #print "N = ", N
  def address_4_1d(i, pitch=64):
    x = i % N
    y = i // 64*4
    #return x*4 + y*pitch*4*4
    return i
  def address_4_2d(x, y, pitch=64):
    return x*4 + y*pitch*4

  input_statea = proc.alloc_remote('I', 4, N, N)
  input_stateb = proc.alloc_remote('I', 4, N, N)
  input_statec = proc.alloc_remote('I', 4, N, N)
  input_stated = proc.alloc_remote('I', 4, N, N)
  input_block = [proc.alloc_remote('I', 4, N, N) for i in range(16)]
  outputa = proc.alloc_remote('I', 4, N, N)
  outputb = proc.alloc_remote('I', 4, N, N)
  outputc = proc.alloc_remote('I', 4, N, N)
  outputd = proc.alloc_remote('I', 4, N, N)

  for j in range(N):
    for i in range(N):
      for k in range(4):
        input_statea[address_4_2d(i, j) + k] = parcontext.statea[k + (i + j*N)*4]
        input_stateb[address_4_2d(i, j) + k] = parcontext.stateb[k + (i + j*N)*4]
        input_statec[address_4_2d(i, j) + k] = parcontext.statec[k + (i + j*N)*4]
        input_stated[address_4_2d(i, j) + k] = parcontext.stated[k + (i + j*N)*4]
  for k in range(N):
    for j in range(N):
      for l in range(4):
        for i in range(16):
          input_block[i][address_4_2d(j, k) + l] = temp_block[i + (j + k*N)*4*16 + l*16]

  global xcode
  if xcode == None:
    xcode = env.InstructionStream()
    cal.set_active_code(xcode)

    S11 = xcode.acquire_register((7, 7, 7, 7))
    S12 = xcode.acquire_register((12, 12, 12, 12))
    S13 = xcode.acquire_register((17, 17, 17, 17))
    S14 = xcode.acquire_register((22, 22, 22, 22))
    S21 = xcode.acquire_register((5, 5, 5, 5))
    S22 = xcode.acquire_register((9, 9, 9, 9))
    S23 = xcode.acquire_register((14, 14, 14, 14))
    S24 = xcode.acquire_register((20, 20, 20, 20))
    S31 = xcode.acquire_register((4, 4, 4, 4))
    S32 = xcode.acquire_register((11, 11, 11, 11))
    S33 = xcode.acquire_register((16, 16, 16, 16))
    S34 = xcode.acquire_register((23, 23, 23, 23))
    S41 = xcode.acquire_register((6, 6, 6, 6))
    S42 = xcode.acquire_register((10, 10, 10, 10))
    S43 = xcode.acquire_register((15, 15, 15, 15))
    S44 = xcode.acquire_register((21, 21, 21, 21))

    a = xcode.acquire_register()
    b = xcode.acquire_register()
    c = xcode.acquire_register()
    d = xcode.acquire_register()
    x = [xcode.acquire_register() for i in range(16)]
    r = xcode.acquire_register()
    
    cal.dcl_resource(0, cal.pixtex_type.twod, cal.fmt.uint, UNNORM=True) # statea
    cal.dcl_resource(1, cal.pixtex_type.twod, cal.fmt.uint, UNNORM=True) # stateb
    cal.dcl_resource(2, cal.pixtex_type.twod, cal.fmt.uint, UNNORM=True) # statec
    cal.dcl_resource(3, cal.pixtex_type.twod, cal.fmt.uint, UNNORM=True) # stated
    for i in range(16):
      cal.dcl_resource(i + 4, cal.pixtex_type.twod, cal.fmt.uint, UNNORM=True)
    cal.dcl_output(reg.o0, USAGE=cal.usage.generic)
    cal.dcl_output(reg.o1, USAGE=cal.usage.generic)
    cal.dcl_output(reg.o2, USAGE=cal.usage.generic)
    cal.dcl_output(reg.o3, USAGE=cal.usage.generic)

    cal.sample(0, 0, a, reg.v0.xy)
    cal.sample(1, 0, b, reg.v0.xy)
    cal.sample(2, 0, c, reg.v0.xy)
    cal.sample(3, 0, d, reg.v0.xy)

    for i in range(16):
      cal.sample(i+4, 0, x[i], reg.v0.xy)

      # Round 1
    FF (a, b, c, d, x[ 0], S11, 0xd76aa478); # 1 
    FF (d, a, b, c, x[ 1], S12, 0xe8c7b756); # 2 
    FF (c, d, a, b, x[ 2], S13, 0x242070db); # 3 
    FF (b, c, d, a, x[ 3], S14, 0xc1bdceee); # 4 
    FF (a, b, c, d, x[ 4], S11, 0xf57c0faf); # 5 
    FF (d, a, b, c, x[ 5], S12, 0x4787c62a); # 6 
    FF (c, d, a, b, x[ 6], S13, 0xa8304613); # 7 
    FF (b, c, d, a, x[ 7], S14, 0xfd469501); # 8 
    FF (a, b, c, d, x[ 8], S11, 0x698098d8); # 9 
    FF (d, a, b, c, x[ 9], S12, 0x8b44f7af); # 10 
    FF (c, d, a, b, x[10], S13, 0xffff5bb1); # 11 
    FF (b, c, d, a, x[11], S14, 0x895cd7be); # 12 
    FF (a, b, c, d, x[12], S11, 0x6b901122); # 13 
    FF (d, a, b, c, x[13], S12, 0xfd987193); # 14 
    FF (c, d, a, b, x[14], S13, 0xa679438e); # 15 
    FF (b, c, d, a, x[15], S14, 0x49b40821); # 16 

    # Round 2 
    GG (a, b, c, d, x[ 1], S21, 0xf61e2562); # 17 
    GG (d, a, b, c, x[ 6], S22, 0xc040b340); # 18 
    GG (c, d, a, b, x[11], S23, 0x265e5a51); # 19 
    GG (b, c, d, a, x[ 0], S24, 0xe9b6c7aa); # 20 
    GG (a, b, c, d, x[ 5], S21, 0xd62f105d); # 21 
    GG (d, a, b, c, x[10], S22,  0x2441453); # 22 
    GG (c, d, a, b, x[15], S23, 0xd8a1e681); # 23 
    GG (b, c, d, a, x[ 4], S24, 0xe7d3fbc8); # 24 
    GG (a, b, c, d, x[ 9], S21, 0x21e1cde6); # 25 
    GG (d, a, b, c, x[14], S22, 0xc33707d6); # 26 
    GG (c, d, a, b, x[ 3], S23, 0xf4d50d87); # 27 
    GG (b, c, d, a, x[ 8], S24, 0x455a14ed); # 28 
    GG (a, b, c, d, x[13], S21, 0xa9e3e905); # 29 
    GG (d, a, b, c, x[ 2], S22, 0xfcefa3f8); # 30 
    GG (c, d, a, b, x[ 7], S23, 0x676f02d9); # 31 
    GG (b, c, d, a, x[12], S24, 0x8d2a4c8a); # 32 

    # Round 3 
    HH (a, b, c, d, x[ 5], S31, 0xfffa3942); # 33 
    HH (d, a, b, c, x[ 8], S32, 0x8771f681); # 34 
    HH (c, d, a, b, x[11], S33, 0x6d9d6122); # 35 
    HH (b, c, d, a, x[14], S34, 0xfde5380c); # 36 
    HH (a, b, c, d, x[ 1], S31, 0xa4beea44); # 37 
    HH (d, a, b, c, x[ 4], S32, 0x4bdecfa9); # 38 
    HH (c, d, a, b, x[ 7], S33, 0xf6bb4b60); # 39 
    HH (b, c, d, a, x[10], S34, 0xbebfbc70); # 40 
    HH (a, b, c, d, x[13], S31, 0x289b7ec6); # 41 
    HH (d, a, b, c, x[ 0], S32, 0xeaa127fa); # 42 
    HH (c, d, a, b, x[ 3], S33, 0xd4ef3085); # 43 
    HH (b, c, d, a, x[ 6], S34,  0x4881d05); # 44 
    HH (a, b, c, d, x[ 9], S31, 0xd9d4d039); # 45 
    HH (d, a, b, c, x[12], S32, 0xe6db99e5); # 46 
    HH (c, d, a, b, x[15], S33, 0x1fa27cf8); # 47 
    HH (b, c, d, a, x[ 2], S34, 0xc4ac5665); # 48 
  
    # Round 4 
    II (a, b, c, d, x[ 0], S41, 0xf4292244); # 49 
    II (d, a, b, c, x[ 7], S42, 0x432aff97); # 50 
    II (c, d, a, b, x[14], S43, 0xab9423a7); # 51 
    II (b, c, d, a, x[ 5], S44, 0xfc93a039); # 52 
    II (a, b, c, d, x[12], S41, 0x655b59c3); # 53 
    II (d, a, b, c, x[ 3], S42, 0x8f0ccc92); # 54 
    II (c, d, a, b, x[10], S43, 0xffeff47d); # 55 
    II (b, c, d, a, x[ 1], S44, 0x85845dd1); # 56 
    II (a, b, c, d, x[ 8], S41, 0x6fa87e4f); # 57 
    II (d, a, b, c, x[15], S42, 0xfe2ce6e0); # 58 
    II (c, d, a, b, x[ 6], S43, 0xa3014314); # 59 
    II (b, c, d, a, x[13], S44, 0x4e0811a1); # 60 
    II (a, b, c, d, x[ 4], S41, 0xf7537e82); # 61 
    II (d, a, b, c, x[11], S42, 0xbd3af235); # 62 
    II (c, d, a, b, x[ 2], S43, 0x2ad7d2bb); # 63 
    II (b, c, d, a, x[ 9], S44, 0xeb86d391); # 64


    cal.mov('o0', a)
    cal.mov('o1', b)
    cal.mov('o2', c)
    cal.mov('o3', d)

    xcode.release_register(a)
    xcode.release_register(b)
    xcode.release_register(c)
    xcode.release_register(d)
    for xi in x:
      xcode.release_register(xi)

  xcode.set_remote_binding('i0', input_statea)
  xcode.set_remote_binding('i1', input_stateb)
  xcode.set_remote_binding('i2', input_statec)
  xcode.set_remote_binding('i3', input_stated)
  for i in range(16): #range(len(input_block)):
    xcode.set_remote_binding('i' + str(i+4), input_block[i])
  xcode.set_remote_binding('o0', outputa)
  xcode.set_remote_binding('o1', outputb)
  xcode.set_remote_binding('o2', outputc)
  xcode.set_remote_binding('o3', outputd)

  domain = (0, 0, N, N)
  global TIME
  start_time = time.time()
  proc.execute(xcode, domain)
  end_time = time.time()
  TIME += (end_time - start_time)
  for j in range(N):
    for i in range(N):
      for k in range(4):
        parcontext.statea[k + (i + j*N)*4] += outputa[address_4_2d(i, j) + k]
        parcontext.stateb[k + (i + j*N)*4] += outputb[address_4_2d(i, j) + k]
        parcontext.statec[k + (i + j*N)*4] += outputc[address_4_2d(i, j) + k]
        parcontext.stated[k + (i + j*N)*4] += outputd[address_4_2d(i, j) + k]

  proc.free_remote(input_statea)
  proc.free_remote(input_stateb)
  proc.free_remote(input_statec)
  proc.free_remote(input_stated)
  for block in input_block:
    proc.free_remote(block)
  proc.free_remote(outputa)
  proc.free_remote(outputb)
  proc.free_remote(outputc)
  proc.free_remote(outputd)

def ParMD5Init(parcontext):
  for i in range(parcontext.number):
    parcontext.count[i*2 + 0] = 0
    parcontext.count[i*2 + 1] = 0
    parcontext.statea[i] = 0x67452301
    parcontext.stateb[i] = 0xefcdab89
    parcontext.statec[i] = 0x98badcfe
    parcontext.stated[i] = 0x10325476

def ParMD5Update(parcontext, parinput, inputLen):
  # we're assuming all lengths are the same - this is very important
  num = parcontext.number

  index = (parcontext.count[0] // 8) % 64
  for k in range(num):
    parcontext.count[k*2] = ((inputLen * 8) + parcontext.count[k*2]) % 2**32
    if parcontext.count[k*2] < inputLen * 8: # deal with overflow
      parcontext.count[k*2 + 1] += 1
    parcontext.count[k*2 + 1] += inputLen >> 29 # ?

  partLen = 64 - index
  if inputLen >= partLen:
    for k in range(num):
      for i in range(partLen):
        parcontext.buffer[k*64 + index + i] = parinput[k*inputLen + i]
    ParMD5Transform (parcontext, parcontext.buffer, 0)
    #print map(hex, (parcontext.statea[1], parcontext.stateb[1], parcontext.statec[1], parcontext.stated[1]))
    #print map(hex, (parcontext.buffer[64 + z] for z in range(64)))
    i = partLen
    while i + 63 < inputLen:
      ParMD5Transform (parcontext, parinput, i);
      #print map(hex, (parcontext.statea[1], parcontext.stateb[1], parcontext.statec[1], parcontext.stated[1]))
      #print map(hex, (parcontext.buffer[64 + z] for z in range(64)))
      i += 64
    index = 0
    _i = i
  else:
    _i = 0

  if type(parinput) == str:
    for k in range(num):
      for i in range(inputLen-_i):
        parcontext.buffer[k*64 + index+i] = ord(parinput[k*inputLen + _i+i])
  else:
    for k in range(num):
      for i in range(inputLen-_i):
        parcontext.buffer[k*64 + index+i] = parinput[k*inputLen + _i+i]

def ParMD5Final(pardigest, parcontext):
  num = parcontext.number

  #print map(hex, [parPADDING[k*64] for k in range(num)])
  parbits = extarray.extarray('B', 8*num)
  ParEncode(num, parbits, parcontext.count, 8)
  index = (parcontext.count[0] // 8) % 64
  #import pdb
  #pdb.set_trace()
  if index < 56:
    padLen = 56 - index
  else:
    padLen = 120 - index
  parPADDING = extarray.extarray('B', padLen*num)
  for k in range(num):
    for i in range(padLen):
      parPADDING[k*padLen + i] = 0
    parPADDING[k*padLen] = 128
  ParMD5Update(parcontext, parPADDING, padLen)
  ParMD5Update(parcontext, parbits, 8)
  state = extarray.extarray('I', 4*num)
  for k in range(num):
    state[k*4 + 0] = parcontext.statea[k]
    state[k*4 + 1] = parcontext.stateb[k]
    state[k*4 + 2] = parcontext.statec[k]
    state[k*4 + 3] = parcontext.stated[k]
  #print map(hex, state)
  ParEncode(num, pardigest, state, 16)

def ParMD5(num, ss, length):
  # len(ss) = num * length

  pardigest = extarray.extarray('B', 16*num)
  parcontext = ParContext(num)

  ParMD5Init(parcontext)
  #print map(hex, (parcontext.statea[1], parcontext.stateb[1], parcontext.statec[1], parcontext.stated[1]))
  #print map(hex, (parcontext.buffer[64 + z] for z in range(64)))
  ParMD5Update(parcontext, ss, length)
  ParMD5Final(pardigest, parcontext)

  #for k in range(num):
  for k in range(1):
    print map(hex, [pardigest[k*16 + i] for i in range(16)])
  #print map(hex, pardigest)
  check = True
  for k in range(num):
    for i in range(16):
      if pardigest[k*16 + i] != pardigest[i]:
        if check == True:
          print k, i
        check = False
  print "Results consistent = ", check


  #for k in range(num):
  #  ls = [pardigest[i] for i in range(k*16, (k+1)*16)]
  #  #print map(hex, map(int, ls))
  #print map(hex, map(int, ls))

#MD5('')
#print '========'
##for i in range(100):
#MD5('a')
#print '========'
#MD5('abc')
#print '========'
#MD5("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
#print '========'

#num = 1048576
#num = 512*4*512


# num should be x*x*4 for some integer x for MD5Transform to be able to figure it out
num = 64*64*4
#num = 4*4


#num = 4
s = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
#s = 'a'
ss = ''
for i in range(num):
  ss += s
TIME = 0
ParMD5(num, ss, len(s))
print "TIME = ", TIME
