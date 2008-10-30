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

# Jaccard/Tanimoto SPU Bit Vector Comparison Kernel

import array
import time

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


  def _ab_c(self, x, y, ab, c, ab_temp, c_temp):
    """
    Interleave ab and c computations
    """
    spu.xor(ab_temp, x, y)
    spu.and_(c_temp, x, y)
    
    spu.cntb(ab_temp, ab_temp)
    spu.cntb(c_temp, c_temp)
    
    spu.sumb(ab_temp, ab_temp, 0)
    spu.sumb(c_temp, c_temp, 0)
    
    spu.a(ab, ab, ab_temp)
    spu.a(c, c, c_temp)
    
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
      # self._ab(self._x_regs[i], self._y_regs[i], ab, ab_temp)
      # self._c( self._x_regs[i], self._y_regs[i],  c,  c_temp)
      self._ab_c(self._x_regs[i], self._y_regs[i], ab, c, ab_temp, c_temp)
      
    self._reduce_word(ab, ab_temp)
    self._reduce_word( c,  c_temp)

    self._compute_ratio(ab_temp, c_temp, result)

    print '%d registers,' % (len(regs) + len(self._x_regs) + len(self._y_regs)),
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
    self._threshold = None

    self._save_op = None
    return

  def set_x_addr(self, addr): self._x_addr = addr
  def set_y_addr(self, addr): self._y_addr = addr

  def set_n_bits(self, n): self._n_bits = n
  def set_block_size(self, m, n):
    self._m = m
    self._n = n
    return

  def set_threshold(self, threshold): self._threshold = threshold
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

    if self._save_op is not None:
      if self._threshold is not None:
        threshold = var.SingleFloat(self._threshold)
      else:
        threshold = var.SingleFloat(0.0)
      bcmp = var.Word(0)
    
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
      save_op.setup()
      
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
          spu.fcgt(bcmp, result, threshold)
          save_op.test(bcmp, result, x_off, y_off)

    # /x_off

    if old_code is not None:
      spu.set_active_code(old_code)
    
    return


# ------------------------------------------------------------
# Buffer save operations
# ------------------------------------------------------------


class LocalSave:
  def __init__(self):
    self._md_results = None

    self._branch_idx = None
    self._block_idx  = None
    self._count = None
    self._save_value = None
    self._word_mask = None
    
    # Save parameters
    self._cmp = None
    self._score = None
    self._x_off = None
    self._y_off = None

    # Main memory save
    self._save_op = None
    
    return

  def set_md_results(self, md): self._md_results = md
  def set_mm_save_op(self, op): self._save_op = op
  
  def setup(self):
    self._count = var.SignedWord(0)
    self._save_value = var.SignedWord(0)
    self._word_mask = var.SignedWord(array.array('I', [0xFFFFFFFF, 0, 0, 0]))


    if self._save_op is not None:
      self._save_op.setup()
    
    return
  
  def test(self, cmp, score, x_off, y_off):
    code = spu.get_active_code()
    self._branch_idx = len(code)
    spu.stop(0xB)
    # spu.nop(0)
    self._cmp = cmp
    self._score = score
    self._x_off = x_off
    self._y_off = y_off
    return

  def block(self):
    code = spu.get_active_code()
    self._block_idx = len(code)

    # --> add the branch instruction (use brz (?) to always branch, nop to never branch)
    code[self._branch_idx] = spu.nop(0, ignore_active = True)
    # code[self._branch_idx] = spu.brnz(self._cmp, self._block_idx - self._branch_idx, ignore_active = True)
    # code[self._branch_idx] = spu.brz(self._cmp, self._block_idx - self._branch_idx, ignore_active = True)

    # Pack result into vector
    #   [x][y][score][--]

    # Zero the save value
    spu.xor(self._save_value, self._save_value, self._save_value)

    # Copy the score
    spu.selb(self._save_value, self._save_value, self._score, self._word_mask)    
    spu.rotqbyi(self._save_value, self._save_value, 12)

    # Copy the y value
    spu.selb(self._save_value, self._save_value, self._y_off, self._word_mask)
    spu.rotqbyi(self._save_value, self._save_value, 12)        

    # Copy the x value
    spu.selb(self._save_value, self._save_value, self._x_off, self._word_mask)
    
    # Save value to local store
    spu.stqx(self._save_value, self._count, self._md_results.r_addr)
    
    self._count.v = self._count.v + 16

    # --> MemorySave test
    cmp = self._save_value # reuse the save register
    spu.ceq.ex(cmp, self._count, self._md_results.r_size)

    if self._save_op is not None:
      self._save_op.test(cmp, self._count)
      
    # Just reset for now
    spu.selb(self._count, self._count, 0, cmp)

    # Return to the loop
    idx = len(code)
    spu.br(- (idx - self._branch_idx - 1))
    
    return

class MemorySave:
  def __init__(self):
    self._md_save = None # Main memory buffer

    self._branch_idx = None
    self._block_idx  = None
    self._count = None
    self._cmp = None
    
    return

  def set_md_save_buffer(self, md): self._md_save = md
  
  def setup(self):
    return
  
  def test(self, cmp, count_var):
    code = spu.get_active_code()
    self._branch_idx = len(code)
    spu.stop(0xB)
    # spu.nop(0)
    self._cmp = cmp
    self._count = count_var
    return

  def block(self):
    code = spu.get_active_code()
    self._block_idx = len(code)

    # --> add the branch instruction
    code[self._branch_idx] = spu.nop(0, ignore_active = True)
    code[self._branch_idx] = spu.brnz(self._cmp, self._block_idx - self._branch_idx,
                                      ignore_active = True)
    
    # FILL IN HERE
    
    # Return to the loop
    idx = len(code)
    spu.br(- (idx - self._branch_idx - 1))
    
    return


class MemoryFull:
  def __init__(self):
    return

  def save_test(self):
    # test
    # branch
    return

  def save_block(self):
    # signal PPU that the main memory buffer is full
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

class CountOp2:
  def __init__(self):
    self._count = None
    return

  def pre_setup(self, code):
    self._count = var.Word(0)
    return

  def setup(self, code):
    return

  def synthesize(self, code, x_off, y_off, result):
    self._count.v = self._count + 1
    return

  def cleanup(self, code):
    return

  def post_cleanup(self, code):
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

def TestTanimotoBlock(n_vecs = 4):
  code = synspu.InstructionStream()
  proc = synspu.Processor()

  code.set_debug(True)
  spu.set_active_code(code)
  
  tb = TanimotoBlock()
  ls_save = LocalSave()
  mm_save = MemorySave()

  code.set_debug(True)

  # Input block parameters
  m = 128
  n = 64
  # n_vecs = 9
  n_bits = 128 * n_vecs

  # Main memory results buffer
  # max_results = 2**16
  max_results = 16384
  words_per_result = 4

  mm_results_data = array.array('I', [12 for i in range(max_results * words_per_result)])
  #mm_results_buffer = synspu.aligned_memory(max_results * words_per_result, typecode = 'I')
  # mm_results_buffer.copy_to(mm_results_data.buffer_info()[0], len(mm_results_data))

  mm_results = spuiter.memory_desc('I')
  #mm_results.from_array(mm_results_buffer)
  mm_results.from_array(mm_results_data)

  mm_save.set_md_save_buffer(mm_results)
    
  # Local Results buffer
  buffer_size = var.SignedWord(16384)
  buffer_addr = var.SignedWord(m * n * n_vecs * 4)
  ls_results = spuiter.memory_desc('B')
  ls_results.set_size_reg(buffer_size)
  ls_results.set_addr_reg(buffer_addr)

  ls_save.set_md_results(ls_results)
  ls_save.set_mm_save_op(mm_save)

  # Setup the TanimotoBlock class
  tb.set_n_bits(n_bits)
  tb.set_block_size(m, n)

  tb.set_x_addr(0)
  tb.set_y_addr(m * n_vecs * 16)
  tb.set_save_op(ls_save)

  # Main test loop
  n_samples = 10000
  for samples in spuiter.syn_iter(code, n_samples):
    tb.synthesize(code)

  spu.wrch(buffer_size, dma.SPU_WrOutMbox)
  
  spu.stop(0x2000) 

  # "Function" Calls
  ls_save.block()
  mm_save.block()

  # code.print_code()
  start = time.time()
  spe_id = proc.execute(code, async=True)
  
  while synspu.spu_exec.stat_out_mbox(spe_id) == 0: pass
  # print 'tb said: 0x%X' % (synspu.spu_exec.read_out_mbox(spe_id))
  stop = time.time()

  # mm_results_buffer.copy_from(mm_results_data.buffer_info()[0], len(mm_results_data))
  
  proc.join(spe_id)
  total = stop - start
  bits_sec = (m * n * n_bits * n_samples) / total / 1e9
  ops_per_compare = 48 * 4 + 8  # 48 SIMD instructions, 8 scalar
  insts_per_compare = 56
  gops = (m * n * n_vecs * n_samples * ops_per_compare ) / total / 1e9
  ginsts = (m * n * n_vecs * n_samples * insts_per_compare ) / total / 1e9  
  print '%.6f sec, %.2f Gbits/sec, %.2f GOps, %.2f GInsts, %d insts' % (
    total, bits_sec, gops, ginsts, code.size())
  return

if __name__=='__main__':
  # Test()

  for n_vecs in range(24):
    print '%d bits/vec,' %  (n_vecs * 128),
    TestTanimotoBlock(n_vecs)
