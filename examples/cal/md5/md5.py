import corepy.lib.extarray as extarray

import corepy.arch.cal.isa as cal
import corepy.arch.cal.lib.cal_extended as calex
import corepy.arch.cal.platform as env
import corepy.arch.cal.types as types
import corepy.arch.cal.types.registers as reg

class Context(object):
  def __init__(self):
    self.state = extarray.extarray('I', 4)
    self.count = extarray.extarray('I', 2)
    self.buffer = extarray.extarray('B', 64)

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

def Encode(output, input, length):
  i = 0
  for j in range(0, length, 4):
    output[j] = input[i] & 0xff
    output[j+1] = (input[i] >> 8) & 0xff
    output[j+2] = (input[i] >> 16) & 0xff
    output[j+3] = (input[i] >> 24) & 0xff
    i += 1

def Decode (output, input, inputi, len):
  i = 0
  for j in range(0, len, 4):
    output[i] = input[inputi+j]
    output[i] |= input[inputi+j+1]<<8
    output[i] |= input[inputi+j+2]<<16
    output[i] |= input[inputi+j+3]<<24
    i += 1

def MD5Transform(state, block, blocki):
  proc = env.Processor(0)
  input_state = proc.alloc_remote('I', 4, 1, 1)
  input_block = proc.alloc_remote('I', 4, 4, 1)
  output = proc.alloc_remote('I', 4, 1, 1)


  for i in range(4):
    input_state[i] = state[i]
  Decode(input_block, block, blocki, 64)
  #print map(hex, input_block)

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
    
    cal.dcl_cb('cb0[1]')
    cal.dcl_cb('cb1[4]')
    cal.dcl_output('o0', USAGE=cal.usage.generic)

    cal.mov(a, 'cb0[0].x')
    cal.mov(b, 'cb0[0].y')
    cal.mov(c, 'cb0[0].z')
    cal.mov(d, 'cb0[0].w')
    for i in range(4):
      cal.mov(x[i*4], 'cb1[' + str(i) + '].x')
      cal.mov(x[i*4+1], 'cb1[' + str(i) + '].y')
      cal.mov(x[i*4+2], 'cb1[' + str(i) + '].z')
      cal.mov(x[i*4+3], 'cb1[' + str(i) + '].w')

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

    temp = xcode.acquire_register()
    #cal.mov(temp.x___, a.x)
    #cal.mov(temp._y__, b.xx)
    #cal.mov(temp.__z_, c.xxx)
    #cal.mov(temp.___w, d.xxxx)
    cal.iadd(temp, a.x000, b('0x00'))
    cal.iadd(temp, temp, c('00x0'))
    cal.iadd(temp, temp, d('000x'))
    cal.mov('o0', temp)

    xcode.release_register(a)
    xcode.release_register(b)
    xcode.release_register(c)
    xcode.release_register(d)
    for xi in x:
      xcode.release_register(xi)

    #for i, inst in enumerate(xcode._instructions):
    #  print inst.render()

  xcode.set_remote_binding('cb0', input_state)
  xcode.set_remote_binding('cb1', input_block)
  xcode.set_remote_binding('o0', output)

  domain = (0, 0, 1, 1)
  proc.execute(xcode, domain)

  state[0] += output[0]
  state[1] += output[1]
  state[2] += output[2]
  state[3] += output[3]

  print 'input  = ', map(hex, input_state)
  print 'output = ', map(hex, output)

  proc.free_remote(input_state)
  proc.free_remote(input_block)
  proc.free_remote(output)

  #xcode = None

def MD5Init(context):
  context.count[0] = 0
  context.count[1] = 0
  context.state[0] = 0x67452301
  context.state[1] = 0xefcdab89
  context.state[2] = 0x98badcfe
  context.state[3] = 0x10325476

def MD5Update(context, input, inputLen):
  index = (context.count[0] // 8) % 64
  context.count[0] = ((inputLen * 8) + context.count[0]) % 2**32
  if context.count[0] < inputLen * 8: # deal with overflow
    context.count[1] += 1
  context.count[1] += inputLen >> 29 # ?
  partLen = 64 - index
  if inputLen >= partLen:
    for i in range(partLen):
      context.buffer[index + i] = input[i]
    print map(hex, context.state)
    MD5Transform (context.state, context.buffer, 0)
    print map(hex, context.state)
    i = partLen
    while i + 63 < inputLen:
      MD5Transform (context.state, input, i);
      print map(hex, context.state)
      i += 64
    index = 0
    _i = i
  else:
    _i = 0

  if type(input) == str:
    for i in range(int(inputLen)-int(_i)):
      context.buffer[index+i] = ord(input[_i+i])
  else:
    for i in range(int(inputLen)-int(_i)):
      context.buffer[index+i] = input[_i+i]

def MD5Final(digest, context):
  PADDING = extarray.extarray('B', 64)
  for i in range(64):
    PADDING[i] = 0
  PADDING[0] = 128
  bits = extarray.extarray('B', 8)
  Encode(bits, context.count, 8)
  index = (context.count[0] // 8) % 64
  if index < 56:
    padLen = 56 - index
  else:
    padLen = 120 - index
  MD5Update(context, PADDING, padLen)
  MD5Update(context, bits, 8)
  print map(hex, context.state)
  Encode(digest, context.state, 16)

def MD5(s):
  digest = extarray.extarray('B', 16)
  length = len(s)
  context = Context()

  MD5Init(context)
  MD5Update(context, s, length)
  MD5Final(digest, context)

  print map(hex, map(int, digest))

MD5('')
print '========'
#for i in range(100):
MD5('a')
print '========'
MD5('abc')
print '========'
MD5("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
print '========'

