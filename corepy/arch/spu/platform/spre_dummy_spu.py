# Copyright (c) 2006-2009 The Trustees of Indiana University.                   
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

__doc__="""
Dummy versions of the SPRE classes for the SPU.  These can be used to
develop SPU code on platforms without direct SPU support.
"""

import array
import math

import corepy.spre.spe as spe

class ExecParams(object):
  def __init__(self):
    # $r3
    self.addr = None  # address of syn code
    self.p1   = None 
    self.p2   = None 
    self.p3   = None 

    # $r4
    self.size = None       # size of syn code
    self.p4   = None 
    self.p5   = None 
    self.p6   = None 

    # $r5
    self.p7   = None 
    self.p8   = None 
    self.p9   = None 
    self.p10  = None 
    return 

import corepy.arch.spu.isa as spu
import corepy.arch.spu.lib.util as util

# ------------------------------
# Registers
# ------------------------------

class SPURegister(spe.Register): pass

# ------------------------------
# Constants
# ------------------------------

WORD_TYPE = 'I'           # array type that corresponds to 1 word
WORD_SIZE = 4             # size in bytes of one word
WORD_BITS = WORD_SIZE * 8 # number of bits in a word

INT_SIZES = {'b':1,  'c':1, 'h':2, 'i':4, 'B':1,  'H':2, 'I':4}

# ------------------------------
# Constants
# ------------------------------

# Parameters - (register, slot)
REG, SLOT = (0, 1)

spu_param_1 = (3, 1)
spu_param_2 = (3, 2)
spu_param_3 = (3, 3)

spu_param_4 = (4, 1)
spu_param_5 = (4, 2)
spu_param_6 = (4, 3)

spu_param_7 = (5, 0)
spu_param_8 = (5, 1)
spu_param_9 = (5, 2)
spu_param_10 = (5, 3)

N_SPUS = 6

# ------------------------------------------------------------
# Aligned Memory
# ------------------------------------------------------------

class aligned_memory(object):
  def __init__(self, size, alignment = 128, typecode = 'B'):
    print 'Using dummy aligned memory'
    self.data = array.array(typecode, range(size))
    self.typecode = typecode
    return

  def __str__(self): return '<aligned_memory typecode = %s addr = 0x%X size = %d ' % (
    self.data.typecode, self.get_addr(), self.get_size())

  def get_addr(self): return self.data.buffer_info()[0]
  def get_size(self): return len(self.data) * INT_SIZES[self.typecode]
  
  def __len__(self):
    return self.get_size() / INT_SIZES[self.typecode]
  
  def buffer_info(self):
    return (self.get_addr(), self.get_size())

  def copy_to(self, source, size):
    return 

  def copy_from(self, dest, size):
    return 

  def word_at(self, index, signed = False):
    """
    Minor hack to give fast access to data...
    TODO: full array-type interface?
    """
    return 0

# ------------------------------------------------------------
# Dummy spe_exec
# ------------------------------------------------------------

class DummyExec(object):
  ExecParams = ExecParams
  
  def _make_executable(addr, size): return 0
  make_executable = staticmethod(_make_executable)
  
  def _cancel_async(spe_id): return 0
  cancel_async = staticmethod(_cancel_async)
  
  def _suspend_async(spe_id): return 0
  suspend_async = staticmethod(_suspend_async)
  def _resume_async(spe_id): return 0
  resume_async = staticmethod(_resume_async)
  def _wait_async(spe_id, result): return 0
  wait_async = staticmethod(_wait_async)
  def _join_async(spe_id): return 0
  join_async = staticmethod(_join_async)
  def _execute_param_async(addr,params): return 0
  execute_param_async = staticmethod(_execute_param_async)
  def _execute_async(addr): return 0
  execute_async = staticmethod(_execute_async)
  def _execute_int(addr): return 0
  execute_int = staticmethod(_execute_int)
  def _execute_param_int(addr, params): return 0
  execute_param_int = staticmethod(_execute_param_int)
  def _execute_void(addr): return
  execute_void = staticmethod(_execute_void)
  def _execute_void(addr,  params): return
  execute_void = staticmethod(_execute_void)
  def _execute_fp(addr): return 0.0
  execute_fp = staticmethod(_execute_fp)

  def _read_out_mbox(spe_id): return 0
  read_out_mbox = staticmethod(_read_out_mbox)
  def _stat_out_mbox(spe_id): return 0
  stat_out_mbox = staticmethod(_stat_out_mbox)
  def _write_in_mbox(spe_id, data): return 0
  write_in_mbox = staticmethod(_write_in_mbox)
  def _stat_in_mbox(spe_id): return 0
  stat_in_mbox = staticmethod(_stat_in_mbox)
  def _write_signal(spe_id, signal_reg, data): return 0
  write_signal = staticmethod(_write_signal)
  def _wait_stop_event(spe_id): return 0
  wait_stop_event = staticmethod(_wait_stop_event)
  def _spu_putb(speid, ls, ea, size, tag, tid, rid): return 0
  spu_putb = staticmethod(_spu_putb)
  def _spu_getb(speid, ls, ea, size, tag, tid, rid): return 0
  spu_getb = staticmethod(_spu_getb)
  def _read_tag_status_all(speid, mask): return 0
  read_tag_status_all = staticmethod(_read_tag_status_all)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def copy_param(code, target, source):
  """
  Copy a parameter from source reg to preferred slot in the target reg.
  For params in slot 0, this is just and add immediate.
  For params in other slots, the source is rotated.
  Note that other values in the source are copied, too.
  """
  if source[SLOT] != 0:
    code.add(spu.rotqbyi(target, source[REG], source[SLOT] * 4))
  else:
    code.add(spu.ai(target, source[REG], 0))
  return

ALIGN_UP = 0
ALIGN_DOWN = 1

def align_addr(addr, align = 16, dir = ALIGN_DOWN):
  """
  Round an address to the nearest aligned address based on align.
  Round up or down based on dir.
  """

  if dir == ALIGN_DOWN:
    return addr - (addr % align)
  else:
    return addr + (align - addr % align)
  
# ------------------------------------------------------------
# InstructionStream
# ------------------------------------------------------------

class InstructionStream(spe.InstructionStream):
  """
  SPU Instruction Stream.  
  Two assumptions:
    o We have the processor untill we're done
    o If we're prempted, the whole state is saved automagically

  Based on these and the fact that we are a leaf node, no register
  saves are attempted and only the raw instructions stream (no
  prologue/epilogue) is used.
  """

  # Class attributes
  RegisterFiles = (('gp', SPURegister, range(0,128)),)
  default_register_type = SPURegister
  
  exec_module   = DummyExec
  align         = 16 # 128 is max efficiency, 16 is what array currently does
  instruction_type  = WORD_TYPE
  
  def __init__(self, optimize=False):
    spe.InstructionStream.__init__(self)

    self._optimize = optimize

    return

  # ------------------------------
  # Execute/ABI support
  # ------------------------------

  def _synthesize_prologue(self):
    """
    Setup register 0.
    """

    self._prologue = InstructionStream()
    
    # Reserve register r0 for the value zero
    self.acquire_register(reg = 0)
    util.load_word(self._prologue, 0, 0, zero = False)

    return

  def _synthesize_epilogue(self):
    """
    Do nothing.
    """

    return

  def cache_code(self):
    """
    Add a stop signal with return type 0x2000 (EXIT_SUCCESS) to the
    end if the instruction stream. (BE Handbook, p. 422).
    """

    # Generate the prologue
    self._synthesize_prologue()

    # Don't have a real epilogue.
    self.add(spu.stop(0x2000))
    # self._check_alignment(self._code, 'spu code')

    # self.exec_module.make_executable(self._code.buffer_info()[0], len(self._code))

    # Append our instructions to the prologue's, first making sure the alignment is correct.
    if len(self._prologue._code) % 2 == 1: # Odd number of instructions
      self._prologue.add(spu.lnop(0))

    self._prologue._code.extend(self._code)
    self._prologue._check_alignment(self._prologue._code, 'spu prologue')
    
    self._epilogue = self    
    self._cached = True
    return


  def add_return(self):
    """
    Do nothing.
    """
    return

  def add_jump(self, addr):
    """
    No nothing.
    """
    return

  def align_code(self, boundary):
    """
    Insert the appropraite nop/lnops to align the next instruction
    on the byte boudary.  boundary must be a multiple of four.
    """
    word_align = boundary / 4

    while len(self._code) % word_align:
      if len(self._code) % 2 == 0:
        self.add(spu.nop(0), True)
      else:
        self.add(spu.lnop(0), True)

    return

  def add(self, inst, optimize_override = False):

    if not optimize_override and self._optimize:
      # binary_string_inst = spu.DecToBin(inst)
      op = 'nop'
      # if binary_string_inst[0:3] in spu.inst_opcodes:
      #   op = spu.inst_opcodes[binary_string_inst[0:3]]
      # elif binary_string_inst[0:6] in spu.inst_opcodes:
      #   op = spu.inst_opcodes[binary_string_inst[0:6]]
      # elif binary_string_inst[0:7] in spu.inst_opcodes:
      #   op = spu.inst_opcodes[binary_string_inst[0:7]]
      # elif binary_string_inst[0:8] in spu.inst_opcodes:
      #   op = spu.inst_opcodes[binary_string_inst[0:8]]
      # elif binary_string_inst[0:9] in spu.inst_opcodes:
      #   op = spu.inst_opcodes[binary_string_inst[0:9]]
      # elif binary_string_inst[0:10] in spu.inst_opcodes:
      #   op = spu.inst_opcodes[binary_string_inst[0:10]]
        
      pipeline = inst.cycles[0]
        
      if (len(self._code) % 2 == 0) and pipeline == 0:   
        InstructionStream.add(self, inst)

      elif (len(self._code) % 2 == 1) and pipeline == 1:
        InstructionStream.add(self, inst)
      elif (len(self._code) % 2 == 0) and pipeline == 1:
        InstructionStream.add(self, spu.nop(0))
        InstructionStream.add(self, inst)
      elif (len(self._code) % 2 == 1) and pipeline == 0:
        InstructionStream.add(self, spu.lnop(0))
        InstructionStream.add(self, inst)

    else:
      spe.InstructionStream.add(self, inst)

    # Invalidate the cache
    self._cached = False
    return len(self._code)


class ParallelInstructionStream(InstructionStream):

  def __init__(self, optimize=False):
    InstructionStream.__init__(self, optimize)

    self.r_rank = self.acquire_register()
    self.r_size = self.acquire_register()

    self.r_block_size = None
    self.r_offset     = None

    # All the params are stored in r_rank
    self.r_params = self.r_rank

    # User/library supplied data size, used by processor to determine
    # block and offset for an execution run.  This value is in bytes.
    self.raw_data_size = None
    
    return

  def _synthesize_prologue(self):
    """
    Add raw_data_size/offest support code.
    """

    InstructionStream._synthesize_prologue(self)

    # Parallel parameters are passed in the prefered slot and the next
    # slot of the user arugment.
    self._prologue.add(spu.shlqbyi(self.r_rank, SPURegister(3, None), 4)) 
    self._prologue.add(spu.shlqbyi(self.r_size, SPURegister(3, None), 8)) 

    if self.raw_data_size is not None:
      self.acquire_block_registers()

      self._prologue.add(spu.shlqbyi(self.r_block_size, SPURegister(4, None), 4)) 
      self._prologue.add(spu.shlqbyi(self.r_offset, SPURegister(4, None), 8)) 
    else:
      print 'no raw data'
    return

  def acquire_block_registers(self):
    if self.r_block_size is None:
      self.r_block_size = self.acquire_register()
    if self.r_offset is None:
      self.r_offset     = self.acquire_register()

    # print 'offset/block_size', self.r_offset, self.r_block_size
    return
  
    
  def release_parallel_registers(self):
    self.release_register(self.r_rank)
    self.release_register(self.r_size)

    if self.r_block_size is not None:
      self.release_register(self.r_block_size)
    if self.r_offset is not None:
      self.release_register(self.r_offset)
      
    return



def _copy_params(params, rank, size):
  """
  Copy params.
  """
  ret = ExecParams()

  ret.addr = params.addr
  ret.p1 = rank
  ret.p2 = size
  ret.p3 = params.p3

  ret.size = params.size        

  ret.p4 = params.p4
  ret.p5 = params.p5
  ret.p6 = params.p6
  ret.p7 = params.p7
  ret.p8 = params.p8
  ret.p9 = params.p9
  ret.p10 = params.p10
  
  return ret


class Processor(spe.Processor):
  exec_module = DummyExec
  ExecParams = ExecParams
  
  def execute(self, code, mode = 'int', debug = False, params = None, n_spus = 1):
    """
    Execute the instruction stream in the code object.

    Execution modes are:

      'int'  - return the intetger value in register gp_return when
               execution is complete
      'fp'   - return the floating point value in register fp_return
               when execution is complete
      'void' - return None
      'async'- execute the code in a new thread and return the thread
               id immediately

    If debug is True, the buffer address and code length are printed
    to stdout before execution.

    ParallelExecutionStream execution:
    
    If code is a ParallelInstructionStream code.n_spus threads are
    created and the parameter structure is set up with world_size=n_spus
    and rank values for each thread. A list containing the speids is
    returned.

    If raw_data_size is present and set on the code object, set the
    block_size and offset parameters.

    The parameters for parallel execution are:

      p1 = rank ($r3.2)
      p2 = size ($r3.3)

      p4 = block_size ($r4.2)
      p5 = offset     ($r4.3)
    
    """

    if len(code._code) == 0:
      return None

    # Cache the code here
    if not code._cached:
      code.cache_code()

    # Setup the parameter structure
    if params is None:
      params = ExecParams()

    addr = code._prologue.inst_addr()
    params.addr = addr
    params.size = len(code._prologue._code) * 4 # size in bytes

    retval = None

    if type(code) is ParallelInstructionStream:
      # Parallel SPU execution
      speids = []
      if n_spus > 8:
        raise Exception("Too many SPUs requests (%d > 8)" % n_spus)

      # print 'Regs:', code.r_rank, code.r_size, code.r_block_size, code.r_offset

      # Set up the parameters and execute each spu thread
      for i in range(n_spus):
        pi = _copy_params(params, i, n_spus)

        if hasattr(code, "raw_data_size") and code.raw_data_size is not None:
          pi.p4 = int(code.raw_data_size / n_spus)  # block_size
          pi.p5 = pi.p4 * i                         # offset

          # print 'Executing: 0x%x %d %d %d %d' % (pi.addr, pi.p1, pi.p2, pi.p4, pi.p5)
        speids.append(spe.Processor.execute(self, code, debug=debug, params=pi, mode='async'))

      # Handle blocking execution modes
      if mode != 'async':
        reterrs = [self.join(speid) for speid in speids]
        retval = reterrs
      else:
        retval = speids
    else:
      # Single SPU execution
      retval = spe.Processor.execute(self, code, mode, debug, params)

    return retval

spu_exec = DummyExec

# ------------------------------------------------------------
# Unit tests
# ------------------------------------------------------------

def TestInt():
  code = InstructionStream()
  proc = Processor()

  spu.set_active_code(code)
  
  r13 = code.acquire_register(reg = 13)
  r20 = code.acquire_register(reg = 20)
  spu.ai(r20, r20, 13)
  spu.ai(r13, r13, 13)
  spu.ai(r13, r13, 13)
  spu.ai(r13, r13, 13)
  spu.ai(r13, r13, 13)
  spu.ai(r13, r13, 13)
  
  spu.stop(0x200D)

  code.print_code()
  r = proc.execute(code) # , debug = True)
  print 'int result:', r
  # while True:
  #   pass
  return


def TestParams():
  # Run this with a stop instruction and examine the registers
  code = InstructionStream()
  proc = Processor()

  # code.add(spu.stop(0xA))
  code.add(spu.stop(0x200D))
  
  params = ExecParams()

  params.p1  = 1 
  params.p2  = 2 
  params.p3  = 3 

  params.p4  = 4 
  params.p5  = 5 
  params.p6  = 6 

  params.p7  = 7 
  params.p8  = 8 
  params.p9  = 9 
  params.p10 = 10


  r = proc.execute(code, params = params)
  # print 'int result:', r
  # while True:
  #   pass
  return


def TestAlignedMemory():
  import spuiter
  n = 10000
  a = array.array('I', range(n))
  aa = aligned_memory(len(a), typecode='I')
  aa.copy_to(a.buffer_info()[0], len(a))

  # aa.print_memory()
  print str(aa), '0x%X, %d' % a.buffer_info()
  
  code = InstructionStream()
  proc = Processor()
  
  md = spuiter.memory_desc('I')
  md.from_array(aa)
  print str(md)
  md.get(code, 0)
  
  ls = spuiter.memory_desc('I', 0, n)
  seq_iter = spuiter.spu_vec_iter(code, ls)

  for i in seq_iter:
    i.v = i + i

  print str(md)
  md.put(code, 0)

  r = proc.execute(code, mode = 'int')
  # print a
  aa.copy_from(a.buffer_info()[0], len(a))
  # aa.print_memory()  
  print a[:20]
  print a[4090:4105]
  print a[8188:8200]    
  print a[-20:]  
  return

def TestParallel():
  # Run this with a stop instruction and examine the registers and memory
  code = ParallelInstructionStream()
  proc = Processor()

  code.raw_data_size = 128*8

  r = code.acquire_register()
  code.add(spu.ai(r, r, 0xCAFE))
  code.add(spu.ai(r, r, 0xBABE))    
  code.add(spu.stop(0x2000))

  r = proc.execute(code, mode='async', n_spus = 6)

  for speid in r:
    proc.join(speid)

  assert(True)
  return


def TestOptimization():
  import time
  import spuiter
  import spuvar
  code1 = InstructionStream(optimize=False)
  code2 = InstructionStream(optimize=True)
  proc = Processor()
  for code in [code1, code2]:
    x = spuvar.spu_int_var(code, 0)
    y = spuvar.spu_int_var(code, 0)
    for i in spuiter.syn_iter(code, pow(2, 14)):
      x.v = x + x
      y.v = y + y
    s = time.time()
    proc.execute(code)
    e = time.time()
    print "Total time: ", e - s
  print "(First time is withOUT optimization.)"

def TestInt2(i0 = 0, i1 = 1):
  i2 = i0 + i1
  i3 = i1 + i2
  
  code = InstructionStream()
  proc = Processor()

  r_loop = 4
  r_address = 5
  r0 = 6
  r1 = 7
  r2 = 8
  r3 = 9
  
  # Load arguments into a quadword
  
  #################
  # Pack quadword #
  #################

  def load_value_int32(code, reg, value, clear = False):
    # obviously, value should be 32 bit integer
    code.add(spu.ilhu(reg, value / pow(2, 16)))      # immediate load halfword upper
    code.add(spu.iohl(reg, value % pow(2, 16))) # immediate or halfword lower
    if clear:
      code.add(spu.shlqbyi(reg, reg, 12)) # shift left qw by bytes, clears right bytes
    return

  load_value_int32(code, r0, i0, True)
  load_value_int32(code, r1, i1, True)
  code.add(spu.rotqbyi(r1, r1, 12)) # rotate qw by bytes
  load_value_int32(code, r2, i2, True)
  code.add(spu.rotqbyi(r2, r2, 8))
  load_value_int32(code, r3, i3, True)
  code.add(spu.rotqbyi(r3, r3, 4))
  code.add(spu.a(r0, r0, r1))
  code.add(spu.a(r0, r0, r2))
  code.add(spu.a(r0, r0, r3)) 

  ##########

  # Main loop to calculate Fibnoccai sequence

  load_value_int32(code, r_address, pow(2, 16), clear_bits = False) # start at 64K

  load_value_int32(code, r_loop, 0, clear_bits = False)
  start_label = code.size() + 1




  code.add(spu.sfi(r_loop, r_loop, 1))
  code.add(spu.brnz(r_loop, (-(next - start_label) * spu.WORD_SIZE)))

  #

  code.add(spu.stop(0x2005))

  r = proc.execute(code)
  # assert(r == 12)
  # print 'int result:', r

  return

if __name__ == '__main__':
  TestInt()
  TestParams()
  TestParallel()
  # TestOptimization()
  # TestAlignedMemory()
