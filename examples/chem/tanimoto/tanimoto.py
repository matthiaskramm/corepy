# Jaccard/Tanimoto SPU Bit Vector Comparison Kernel

import array

import corepy.arch.spu.platform as synspu
import corepy.arch.spu.isa as spu
import corepy.arch.spu.types.spu_types as var
import corepy.arch.spu.lib.iterators as spuiter
import corepy.arch.spu.lib.dma as dma


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

class Tanimoto:
  def __init__(self):
    self._one = None
    self._n_bits = 128 # must be multiples of 128

    self._x_regs = None
    self._y_regs = None
    self._result = None

    return

  def set_n_bits(self, n):
    if n % 128 != 0:
      raise Exception("n_bits must be a multiple of 128")

    self._n_bits = n    
    return

  def set_x_regs(self, regs):
    if len(regs) != self._n_bits / 128:
      raise Exception("Too many registers supplied for x_regs")

    self._x_regs = regs
    return

  def set_y_regs(self, regs):
    if len(regs) != self._n_bits / 128:
      raise Exception("Too many registers supplied for x_regs")

    self._y_regs = regs
    return

  def set_result(self, reg):
    self._result = reg
    return
  
  def _ab(self, x, y, ab, temp):

    spu.xor(temp, x, y)
    spu.cntb(temp, temp)
    spu.sumb(temp, temp, 0)
    spu.a(ab, ab, temp)

    return

  def _c(self, x, y, c, temp):

    spu.and_(temp, x, y)
    spu.cntb(temp, temp)
    spu.sumb(temp, temp, 0)
    spu.a(c, c, temp)
    
    return

  def _reduce_word(self, words, result):
    """
    Add-reduce a vector of words into the preferred
    slot of result.
    """

    for i in range(4):
      spu.a(result, words, result)
      spu.rotqbyi(words, words, 4)

    return
  
  def _compute_ratio(self, ab, c, result):

    # Convert ab and c to float
    spu.cuflt(ab, ab, 155)
    spu.cuflt(c,   c, 155)

    # Compute ab = ab + c
    spu.fa(ab, ab, c)

    # Compute c / (ab + c)

    fdiv(spu.get_active_code(), result, c, ab, self._one)
    
    return

  def synthesize_constants(self, code):
    old_code = spu.get_active_code()
    spu.set_active_code(code)
    
    self._one = code.acquire_register()
    spu.xor(self._one, self._one, self._one)
    spu.ai(self._one, self._one, 1)
    spu.cuflt(self._one, self._one, 155)
    
    if old_code is not None:
      spu.set_active_code(old_code)

    return

  def release_constants(self, code):
    code.release_release_registers((self._one,))
    return

  def synthesize(self, code):
    if self._x_regs is None:  raise Exception("Please set x_regs")        
    if self._y_regs is None:  raise Exception("Please set y_regs")
    if self._result is None:  raise Exception("Please set result register")    

    old_code = spu.get_active_code()
    spu.set_active_code(code)    

    regs = []

    if self._one is None:
      self.synthesize_constants(code)
      regs.append(self._one)      


    ab = code.acquire_register()
    c  = code.acquire_register()
    ab_temp = code.acquire_register()
    c_temp  = code.acquire_register()
    result  = code.acquire_register()
    regs = regs + [ab, c, ab_temp, c_temp]

    nregs = self._n_bits / 128

    for i in range(nregs):
      self._ab(self._x_regs[i], self._y_regs[i], ab, ab_temp)
      self._c( self._x_regs[i], self._y_regs[i],  c,  c_temp)
    
    self._reduce_word(ab, ab_temp)
    self._reduce_word( c,  c_temp)

    self._compute_ratio(ab_temp, c_temp, result)

    code.release_registers(regs)
    if old_code is not None:
      spu.set_active_code(old_code)

    return


class TanimotoBlock:
  """
  Compute the Tanimoto coefficient for all pairs in the m x n block of
  bit vectors.
  """

  def __init__(self):
    self._x_addr = None
    self._y_addr = None
    self._n_bits = None
    self._m = None
    self._n = None

    self._save_op = None
    return

  def set_x_addr(self, addr): self._x_addr = addr
  def set_y_addr(self, addr): self._y_addr = addr

  def set_n_bits(self, n): self._n_bits = n
  def set_block_size(self, m, n):
    self._m = m
    self._n = n
    return

  def set_save_op(self, op): self._save_op = op

  def _load_bit_vector(self, addr, regs):

    for i in range(len(regs)):
      spu.lqd(regs[i], addr, i)
    
    return

  
  def synthesize(self, code):
    old_code = spu.get_active_code()
    spu.set_active_code(code)

    # Sanity checks
    if self._x_addr is None: raise Exception("Please set x_addr")
    if self._y_addr is None: raise Exception("Please set y_addr")
    if self._n_bits is None: raise Exception("Please set n_bits")
    if self._m is None: raise Exception("Please set m")
    if self._n is None: raise Exception("Please set n")    
    
    # Acquire a registers for the bit vectors and result
    n_vecs = self._n_bits / 128
    x_regs = [code.acquire_register() for i in range(n_vecs)]
    y_regs = [code.acquire_register() for i in range(n_vecs)]
    result = code.acquire_register()

    x_addr = var.Word()
    y_addr = var.Word()
    
    # Setup the Tanimito kernel
    tan = Tanimoto()

    tan.set_n_bits(self._n_bits)
    tan.set_x_regs(x_regs)
    tan.set_y_regs(y_regs)
    tan.set_result(result)

    tan.synthesize_constants(code)

    # Setup the save op
    save_op = self._save_op
    if save_op is not None:
      save_op.setup(code)
      
    # Create the iterators
    xiter = spuiter.syn_iter(code, self._m)
    yiter = spuiter.syn_iter(code, self._n)

    # Synthesize the block comparison loops
    x_addr.v = self._x_addr

    for x_off in xiter:
      x_addr.v = x_addr + 16 * n_vecs
      y_addr.v = self._y_addr

      self._load_bit_vector(x_addr, x_regs)

      for y_off in yiter:
        y_addr.v = y_addr + 16 * n_vecs

        self._load_bit_vector(y_addr, y_regs)
        tan.synthesize(code)

        if save_op is not None:
          save_op.synthesize(code, x_off, y_off, result)
    # /x_off

    if save_op is not None:
      save_op.cleanup(code)

    if old_code is not None:
      spu.set_active_code(old_code)
    
    return
  

class SaveOp:
  def __init__(self, md_results, total_size, threshold):
    self._md = md_results
    self._threshold = threshold
    self._count = None
    return

  def setup(self, code):
    self._count = SignedWord(0)
    self._threshold = SingleFloat([threshold] * 4)
    return

  def synthesize(self, code, x_off, y_off, result):
    # test result against the threshold
    # if result > threshold:
    #   save [x_off, y_off, result] to ls
    #   inc count
    #   if count > md_size:
    #     flush md
    #     resent ls_count
    #     if total_count > available memory:
    #       signal controller
    #       stop
    return

  def cleanup(self, code):
    # save any remaining results to main memory
    return


class CountOp:
  def __init__(self):
    self._count = None
    return

  def setup(self, code):
    self._count = var.Word(0)
    return

  def synthesize(self, code, x_off, y_off, result):
    self._count.v = self._count + 1
    return

  def cleanup(self, code):
    spu.wrch(self._count, dma.SPU_WrOutMbox)
    return



def TestTanimoto():
  code = synspu.InstructionStream()
  proc = synspu.Processor()

  code.set_debug(True)
  
  x_regs = code.acquire_registers(2)
  y_regs = code.acquire_registers(2)
  result = code.acquire_register()

  tan = Tanimoto()

  tan.set_n_bits(256)
  tan.set_x_regs(x_regs)
  tan.set_y_regs(y_regs)
  tan.set_result_reg(result)
  
  tan.synthesize(code)

  code.print_code()

  proc.execute(code)

  # TODO: Do a real test, not just a synthesis test
  return

def TestTanimotoBlock():
  code = synspu.InstructionStream()
  proc = synspu.Processor()

  tb = TanimotoBlock()
  op = CountOp()

  code.set_debug(True)
  
  m = 128
  n = 128
  n_vecs = 4
  n_bits = 128 * n_vecs

  tb.set_n_bits(n_bits)
  tb.set_block_size(m, n)

  tb.set_x_addr(0)
  tb.set_y_addr(m * n_vecs * 16)

  tb.set_save_op(op)

  tb.synthesize(code)

  code.print_code()
  return

  spe_id = proc.execute(code, mode = 'async')

  while synspu.spu_exec.stat_out_mbox(spe_id) == 0: pass
  print 'tb said: 0x%X' % (synspu.spu_exec.read_out_mbox(spe_id))

  proc.join(spe_id)

  return

if __name__=='__main__':
  # Test()
  TestTanimotoBlock()
