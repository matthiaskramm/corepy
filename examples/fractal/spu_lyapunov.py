import corepy.arch.spu.platform as synspu
import corepy.arch.spu.isa as spu
import corepy.arch.spu.types.spu_types as var
import corepy.arch.spu.lib.dma as dma
import corepy.arch.spu.lib.iterators as spuiter

import spu_log

import array

constants = {
  'BIT_0':   0x80000000,
  'TWO':     0x40000000
  }

def fdiv(code, d, x, y, one = None):
  """
  Single-precision floating point division for x / y
  """
  Y = code.acquire_registers(3)
  t = code.acquire_register()
  regs = Y[:]
  regs.append(t)

  if one is None:
    one = code.acquire_register()
    spu.xor(one, one, one)
    spu.ai(one, one, 1)
    spu.cuflt(one, one, 155)
    regs.append(one)

  # Compute 1/y (from SPU ISA 1.1, p208, Normal case)
  spu.frest(Y[0], y)
  spu.fi(Y[1], y, Y[0])
  spu.fnms(t, y, Y[1], one)
  spu.fma(Y[2], t, Y[1], Y[1])

  # Compute x * (1/y)
  spu.fm(d, x, Y[2])

  code.release_registers(regs)

  return

class LyapunovPoint:

  def __init__(self):
    self.pattern = None
    self.r1 = None
    self.r2 = None
    self.result   = None    
    self.max_init = None
    self.max_n    = None
    self.x0       = 0x3F000000 # 0.5

    self.log = None
    self.consts = {}
    return

  def set_pattern_reg(self, p): self.pattern = p
  def set_result_reg(self, r):  self.result = r

  def set_max_init(self, n): self.max_init = n
  def set_max_n(self, n):    self.max_n = n
  def set_x0(self, x):       self.x0 = x

  def set_r_regs(self, r1, r2):
    self.r1 = r1
    self.r2 = r2
    return

  def set_log(self, l): self.log = l

  def setup(self, code):
    for const in constants.keys():
      self.consts[const] = var.Word(constants[const])
    return

  def _next_r(self, r):
    """
    Set r with r1 if the next bit is 0 and r2 if it is 1.
    """

    bit_0 = self.consts['BIT_0']

    # Get the first bit in the pattern
    r.v = spu.and_.ex(self.pattern, bit_0)

    # See if it is '1'
    r.v = spu.ceq.ex(r, bit_0)

    # Copy the test result from byte 0 into all bytes to
    # fill the vector with 1's or 0's
    r.v = spu.shufb.ex(r, 0, 0)

    # Select r1 or r2
    r.v = spu.selb.ex(self.r1, self.r2, r)

    # Rotate the pattern by one
    self.pattern.v = spu.rotqbii.ex(self.pattern, 1)

    return


  def _check_inputs(self):
    if self.pattern is None: raise Exception('Please set pattern')
    if self.r1 is None: raise Exception('Please set r1')
    if self.r2 is None: raise Exception('Please set r2')
    if self.result is None: raise Exception('Please set result')
    if self.max_init is None: raise Exception('Please set max_init')
    if self.max_n is None: raise Exception('Please set max_n')
    if self.log is None: raise Exception('Please set log')
    return

  def synthesize(self, code):
    self._check_inputs()

    old_code = spu.get_active_code()
    spu.set_active_code(code)

    zero = var.Word(reg = code.r_zero)
    one = self.log.consts['ONE']
    two = self.consts['TWO']

    x   = var.Word(self.x0)
    r   = var.Word(0)
    cmp = var.Word(0)
    x_neg = var.Word(0)
    fmax  = var.Word(self.max_init)
    temp = var.SingleFloat()

    fmax.v = spu.cuflt.ex(fmax, 155)

    # Init
    for i in spuiter.syn_iter(code, self.max_init):
      # x = r[i % r_max] * x * (1.0 - x)      
      self._next_r(r)
      temp.v = spu.fs.ex(one, x)
      x.v = spu.fm.ex(x, temp)
      x.v = spu.fm.ex(r, x)

    #  if x == float('-infinity'):
    #    return -10.0

    # Derive Exponent
    total = var.Word(0)
    logx  = var.SingleFloat()

    for i in spuiter.syn_iter(code, self.max_n):    
      # x = ri * x * (1.0 - x)
      self._next_r(r)
      temp.v = spu.fs.ex(one, x)
      x.v = spu.fm.ex(x, temp)
      x.v = spu.fm.ex(r, x)

      # logx = ri - 2.0 * ri * x
      logx.v = spu.fm.ex(two, x)
      logx.v = spu.fm.ex(r, logx)
      logx.v = spu.fs.ex(r, logx)

      # abs(logx)
      x_neg.v = spu.fs.ex(zero, logx)
      cmp.v = spu.fcgt.ex(logx, zero)
      logx.v = spu.selb.ex(x_neg, logx, cmp)
      # logx.v = spu.selb.ex(logx, x_neg, cmp)

      # log(logx)
      self.log.set_result(logx)
      self.log.set_x(logx)
      self.log.synthesize(code)

      # total = total + x
      total.v = spu.fa.ex(total, logx)

    # return total / float(max_n)    
    fdiv(code, self.result, total, fmax, one)

    spu.set_active_code(code)
    return


class LyapunovBlock:
  def __init__(self):
    self.range = None   # array('f', [r1,r1,r1,r1,r2,r2,r2,r2,r1inc,...,r2inc])
    self.pattern = None # array('I', [p0_31, p32_63, p64_95, p95_128])
    self.w = None
    self.h = None

    self.renderer = None
    self.ly_point = LyapunovPoint()

    return

  def set_size(self, w, h): self.w, self.h = (w, h)
  def set_range(self, r):   self.range = r
  def set_pattern(self, p): self.pattern = p

  def set_max_init(self, n): self.ly_point.set_max_init(n)
  def set_max_n(self, n):    self.ly_point.set_max_n(n)  

  def set_renderer(self, r): self.renderer = r

  def _load_parameters(self, code):
    range_md = spuiter.memory_desc('I') # use 'I' for sizeof(float)
    range_md.from_array(self.range)

    pattern_md = spuiter.memory_desc('I')
    pattern_md.from_array(self.pattern)

    # Range is in address 0-63
    range_md.get(code, 0)

    # Pattern is at address 64
    pattern_md.get(code, 64)

    return

  def synthesize(self, code):
    old_code = spu.get_active_code()
    spu.set_active_code(code)

    self._load_parameters(code)

    log = spu_log.SPULog()
    log.setup(code)

    if self.renderer is not None:
      self.renderer.setup(code)
      self.renderer.set_one(log.consts['ONE'])

    r1_inc = var.SingleFloat()
    r2_inc = var.SingleFloat()
    r1 = var.SingleFloat()
    r2 = var.SingleFloat()
    result = var.SingleFloat()
    pattern = var.Word(0)

    self.ly_point.set_pattern_reg(pattern)
    self.ly_point.set_result_reg(result)
    self.ly_point.set_r_regs(r1, r2)
    self.ly_point.set_log(log)
    self.ly_point.setup(code)

    spu.lqa(r1, 0)
    spu.lqa(r2, 4)    
    spu.lqa(r1_inc, 8)
    spu.lqa(r2_inc, 12)
    spu.lqa(pattern, 16)

    for y in spuiter.syn_iter(code, self.h):
      spu.lqa(r1, 0)

      for x in spuiter.syn_iter(code, self.w):
        self.ly_point.synthesize(code)
        r1.v = spu.fa.ex(r1, r1_inc)

        if self.renderer is not None:
          # result.v = spu.fm.ex(r1, r2)
          self.renderer.set_result_reg(result)
          self.renderer.synthesize(code)

      if self.renderer is not None:
        self.renderer.row_complete(code)
      r2.v = spu.fa.ex(r2, r2_inc)

    # return Numeric.where(Numeric.less(results, 0), results, 0)

    spu.set_active_code(old_code)
    return 


class MailboxRenderer:
  """
  Send the 32-bit result to the outbox.
  """
  def __init__(self):
    self.result = None
    return

  def set_result_reg(self, r): self.result = r

  def synthesize(self, code):
    old_code = spu.get_active_code()
    spu.set_active_code(code)

    if self.result is None: raise Exception('Please set result')

    spu.wrch(self.result, dma.SPU_WrOutMbox)

    spu.set_active_code(old_code)
    return

  def row_complete(self, code): pass


class FBRenderer:
  """
  Render to a framebuffer
  """
  def __init__(self):
    self.result = None
    self.lsa  = None
    self.addr = None
    self.w = None
    self._stride = None

    self.one = None

    self.uint2rgb = None
    self.ff = None    

    self.x_offset = None
    self.y_offset = None    
    self.stride = None
    self.offset = None
    return

  def set_lsa(self, lsa): self.lsa = lsa
  def set_addr(self, addr): self.addr = addr  
  def set_width(self, w): self.w = w
  def set_stride(self, s): self._stride = s
  def set_result_reg(self, r): self.result = r

  def set_one(self, one): self.one = one

  def setup(self, code):
    if self.addr is None: raise Exception('Please set addr')
    if self._stride is None: raise Exception('Please set stride')        

    self.x_offset = var.Word(0)
    self.y_offset = var.Word(self.addr)
    self.stride = var.Word(self._stride * 4)

    # Mask to extract the lowest 2 bytes from each word in the first vector
    # into RGB and the first byte from the second vector into A
    self.uint2rgba = var.Word(array.array('I', [0x03030310, 0x07070710, 0x0B0B0B10, 0x0F0F0F10]))
    self.ff = var.Word(0xFF000000)

    return

  def synthesize(self, code):
    """
    Render a vector with 4 pixels.
    """
    old_code = spu.get_active_code()
    spu.set_active_code(code)

    if self.x_offset is None: raise Exception('Please call setup')
    if self.result is None: raise Exception('Please set result')
    if self.one is None: raise Exception('Please set one')

    # Make the part of the result positive and subtract 1
    # to transform (-1,-oo) into (0,oo)
    self.result.v = spu.fs.ex(0, self.result)
    self.result.v = spu.fs.ex(self.result, self.one)

    # Convert the result to an unsigned int, scaling by 2^4 to put 
    # values between 0 and 16 in the gradient.  Values outside [0,16] 
    # are 0 or FF
    self.result.v = spu.cfltu.ex(self.result, 170) # 173 - 169 == 4

    # Extract the first two bytes from the result into the RGB positions
    # and set alpha to 0xFF
    self.result.v = spu.shufb.ex(self.result, self.ff, self.uint2rgba)

    # Save the result and increment the offset
    spu.stqd(self.result, self.x_offset, self.lsa >> 4)
    spu.ai(self.x_offset, self.x_offset, 16)

    spu.set_active_code(old_code)
    return

  def row_complete(self, code):
    """
    Save the current row to the framebuffer.
    """

    if self.w is None: raise Exception('Please set width')
    if self.lsa is None: raise Exception('Please set lsa')    
    if self.y_offset is None: raise Exception('Please call setup')

    md = spuiter.memory_desc('I', size = self.w)

    md.set_addr_reg(self.y_offset)
    md.put(code, self.lsa)

    self.y_offset.v = self.y_offset + self.stride
    spu.xor(self.x_offset, self.x_offset, self.x_offset)
    return

def _pattern2vector(pattern):
  """
  Encode a string of 1's and 0's into a 128-bit bit vector.
  """

  if 128 % len(pattern) != 0: raise Exception('Pattern length must be a factor of 128')
  pattern = pattern * (128 / len(pattern))

  bv = array.array('I', [0,0,0,0])

  size = 128 / 4
  for i in range(size):
    for j in range(4):
      b = int(pattern[size * j + i])
      if b == 1:
        bv[j] = bv[j] | (1 << (size - i - 1))

  return bv


class MailboxLyapunov:

  def __init__(self):
    return

  def generate(self, results, pattern, r1_range, r2_range, max_init, max_n, size):

    # Setup the range parameter array
    r1_inc = (r1_range[1] - r1_range[0]) / size[0]
    r2_inc = (r2_range[1] - r2_range[0]) / size[1]

    ranges = array.array('f', [0.0] * 16)
    for i in range(4):
      ranges[i]      = r1_range[0]
      ranges[4 + i]  = r2_range[0]
      ranges[8 + i]  = r1_inc
      ranges[12 + i] = r2_inc

    # Setup the pattern vector
    bits = _pattern2vector(pattern)

    # Copy the paramters to aligned buffers
    a_ranges = synspu.aligned_memory(len(ranges), typecode='I')
    a_ranges.copy_to(ranges.buffer_info()[0], len(ranges))

    a_pattern = synspu.aligned_memory(len(bits), typecode='I')
    a_pattern.copy_to(bits.buffer_info()[0], len(bits))

    renderer = MailboxRenderer()
    ly_block = LyapunovBlock()

    ly_block.set_size(size[0], size[1])
    ly_block.set_range(a_ranges)
    ly_block.set_pattern(a_pattern)
    ly_block.set_max_init(max_init)
    ly_block.set_max_n(max_n)
    ly_block.set_renderer(renderer)

    code = synspu.InstructionStream()
    ly_block.synthesize(code)

    proc = synspu.Processor()

    spe_id = proc.execute(code, mode = 'async')

    for i in range(size[0] * size[1]):
      while synspu.spu_exec.stat_out_mbox(spe_id) == 0: pass
      print 'ly said: 0x%X' % (synspu.spu_exec.read_out_mbox(spe_id))

    proc.join(spe_id)

    #   for x in range(size[0]):
    #     r2 = r2_range[0] + r2_inc
    #     print 'col:', x, r1, r2

    #     for y in range(size[1]):
    #       results[y, x] = lyapunov_point(pattern, r1, r2, max_init, max_n)
    #       r2 += r2_inc      
    #     r1 += r1_inc      

    return

cell_fb = synspu.cell_fb

class FramebufferLyapunov:

  def __init__(self):
    return

  def generate(self, results, pattern, r1_range, r2_range, max_init, max_n, size):
    # Connect to the framebuffer
    fb = cell_fb.framebuffer()
    cell_fb.fb_open(fb)

    # Setup the range parameter array
    r1_inc = (r1_range[1] - r1_range[0]) / size[0]
    r2_inc = (r2_range[1] - r2_range[0]) / size[1]

    ranges = array.array('f', [0.0] * 16)
    for i in range(4):
      ranges[i]      = r1_range[0] + float(i) * r1_inc # horizontal is simd
      ranges[4 + i]  = r2_range[0]
      ranges[8 + i]  = r1_inc * 4.0
      ranges[12 + i] = r2_inc
    print ranges

    # Setup the pattern vector
    bits = _pattern2vector(pattern)

    # Copy the paramters to aligned buffers
    a_ranges = synspu.aligned_memory(len(ranges), typecode='I')
    a_ranges.copy_to(ranges.buffer_info()[0], len(ranges))

    a_pattern = synspu.aligned_memory(len(bits), typecode='I')
    a_pattern.copy_to(bits.buffer_info()[0], len(bits))

    renderer = FBRenderer()
    renderer.set_lsa(0x100)
    renderer.set_addr(cell_fb.fb_addr(fb, 0))
    renderer.set_width(size[0])
    renderer.set_stride(fb.stride)

    ly_block = LyapunovBlock()

    ly_block.set_size(size[0], size[1])
    ly_block.set_range(a_ranges)
    ly_block.set_pattern(a_pattern)
    ly_block.set_max_init(max_init)
    ly_block.set_max_n(max_n)
    ly_block.set_renderer(renderer)

    code = synspu.InstructionStream()
    ly_block.synthesize(code)

    proc = synspu.Processor()

    cell_fb.fb_clear(fb, 0)      
    while True:
      spe_id = proc.execute(code)
      cell_fb.fb_wait_vsync(fb)
      cell_fb.fb_flip(fb, 0)

    cell_fb.fb_close(fb)

    return


def TestPattern2Vector():
  p1 = ['10', 0xAAAAAAAA]
  p2 = ['1100', 0xCCCCCCCC]
  p3 = ['11100110', 0xE6E6E6E6]

  for p in (p1, p2, p3):
    v = _pattern2vector(p[0])
    for i in range(4):
      assert(v[i] == p[1])

  return



def TestMailboxLyapunov():

  ml = MailboxLyapunov()
  ml.generate(None, '01', [2.0, 4.0], [2.0, 4.0], 200, 400, [200, 200])

  return

def TestFramebufferLyapunov():

  ml = FramebufferLyapunov()
  ml.generate(None, '01', [1.0, 256.0], [1.0, 256.0], 200, 400, [256, 256])

  return

if __name__=='__main__':
  # TestPattern2Vector()
  # TestMailboxLyapunov()
  TestFramebufferLyapunov()
