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

# SPU Iterator Hierarchy

import array

import corepy.spre.spe as spe
import corepy.lib.extarray as extarray
import corepy.arch.spu.platform as env
import corepy.arch.spu.isa as spu
import corepy.arch.spu.types.spu_types as var
import corepy.arch.spu.lib.dma as dma
import corepy.arch.spu.lib.util as util

def _mi(cls):
  """
  Return the machine order for an instruction.
  """
  return cls.machine_inst._machine_order
  

CTR = 0
DEC = 1
INC = 2

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

_array_type   = type(array.array('I', [1]))
_extarray_type = type(extarray.extarray('I', [1]))

def _typecode(a):
  if type(a) in (_array_type, _extarray_type):
    return a.typecode
  elif type(a) is memory_desc:
    return a.typecode
  else:
    raise Exception('Unknown array type ' + type(a))

def _array_address(a):
  if type(a) in (_array_type, _extarray_type):
    return a.buffer_info()[0]
  elif type(a) is memory_desc:
    return a.addr
  else:
    raise Exception('Unknown array type ' + type(a))

# ------------------------------
# PPU Memory
# ------------------------------

# TODO - AWF - this isn't use by anything except for spu_vec_iter.  Why?
#  Even then most of this class doesn't appear to be used.
class memory_desc(object):
  def __init__(self, typecode, addr = None, size = None):
    self.typecode = typecode
    self.addr = addr
    self.size = size

    self.r_addr = None
    self.r_size = None
    return

  def __str__(self): return '<memory_desc typcode = %s addr = 0x%X size = %d r_addr = %s r_size = %s>' % (
    self.typecode, self.addr, self.size, str(self.r_addr), str(self.r_size))

  def set_addr_reg(self, reg): self.r_addr = reg
  def set_size_reg(self, reg): self.r_size = reg
  
  def __len__(self): return self.size

  def nbytes(self): return self.size * var.INT_SIZES[self.typecode]
  
  def from_buffer(self, b):
    """
    Extract the address and size from a buffer object.

    Note: this doesn't very well with buffer objects.
    """
    l = repr(b).split(' ')
    self.size = int(l[l.index('size') + 1])
    self.addr = int(l[l.index('ptr') + 1][:-1], 0)
    # print l, self.size, self.addr
    return

  def from_ibuffer(self, m):
    """
    Extract the address and size from an object that supports
    the buffer interface.

    This should be more flexible than the buffer object.
    """
    self.addr, self.size = synbuffer.buffer_info(m)
    self.size = self.size / var.INT_SIZES[self.typecode]    
    return

  def from_array(self, a):
    self.addr, self.size = a.buffer_info()
    return

  def get(self, code, lsa, tag = 1):
    return self._transfer_data(code, dma.mfc_get, lsa, tag)

  def put(self, code, lsa, tag = 2):
    return self._transfer_data(code, dma.mfc_put, lsa, tag)
  
  def _transfer_data(self, code, kernel, lsa, tag):
    """
    Load the data into the SPU memory
    """

    # Check the types
    if not isinstance(code, spe.InstructionStream):
      raise Exception('Code must be an InstructionStream')
    if not (isinstance(lsa, int) or issubclass(type(lsa), (spe.Register, spe.Variable))):
      raise Exception('lsa must be an integer, Register, or Variable')
    
    old_code = spu.get_active_code()
    spu.set_active_code(code)

    # Acquire registers for address and size, if they were not supplied by the user
    if self.r_addr is None: r_ea_data = code.prgm.acquire_register()
    else:                   r_ea_data = self.r_addr
      
    if self.r_size is None: r_size = code.prgm.acquire_register()
    else:                   r_size = self.r_size

    # Create variables 
    ea_addr      = var.SignedWord(reg = r_ea_data)
    aligned_size = var.SignedWord(0)
    mod_16       = var.SignedWord(0xF)

    # Initialize the lsa_addr variable. 
    if isinstance(lsa, int):
      # From a constant
      ls_addr   = var.SignedWord(lsa)
    elif issubclass(type(lsa), (spe.Register, spe.Variable)):
      # From a variable
      ls_addr   = var.SignedWord()      
      ls_addr.v = lsa
      
      
    tag_var = var.SignedWord(tag)
    cmp = var.SignedWord(0)

    # Load the effective address
    if self.r_addr is None:
      if self.addr % 16 != 0:
        print '[get_memory] Misaligned data'

      util.load_word(code, ea_addr, self.addr)

    # Load the size, rounding up as required to be 16-byte aligned
    if self.r_size is None:
      rnd_size = self.size * var.INT_SIZES[self.typecode]
      if rnd_size < 16:
        rnd_size = 16
      elif (rnd_size % 16) != 0:
        rnd_size += (16 - (rnd_size % 16))
      util.load_word(code, aligned_size, rnd_size)
    else:
      # TODO: !!! UNIT TEST THIS !!!
      # Same as above, but using SPU arithemtic to round
      size  = var.SignedWord(reg = r_size)
      sixteen  = var.SignedWord(16)
      cmp.v = ((size & mod_16) == size)
      aligned_size.v = size + (sixteen - (size & mod_16))
      spu.selb(aligned_size.reg, size.reg, aligned_size.reg, cmp.reg, order = _mi(spu.selb))
      code.release_register(sixteen.reg)

    # Use an auxillary register for the moving ea value if the
    # caller supplied the address register
    if self.r_addr is not None:
      ea_load   = var.SignedWord(0)
      ea_load.v = ea_addr
    else:
      ea_load = ea_addr # note that this is reference, not .v assignment

    # Transfer parameters
    buffer_size   = var.SignedWord(16384)
    remaining     = var.SignedWord(0)
    transfer_size = var.SignedWord(0)
    remaining.v   = aligned_size

    # Set up the iterators to transfer at most 16k at a time
    xfer_iter = syn_iter(code, 0, 16384)
    xfer_iter.set_stop_reg(aligned_size.reg)

    for offset in xfer_iter:
      cmp.v = buffer_size > remaining
      spu.selb(transfer_size, buffer_size, remaining, cmp)

      # Transfer the data
      kernel(code, ls_addr, ea_load, transfer_size, tag_var)
      ls_addr.v = ls_addr + buffer_size
      ea_load.v = ea_load + buffer_size

      remaining.v = remaining - buffer_size

    # Set the tag bit to tag
    dma.mfc_write_tag_mask(code, 1<<tag);

    # Wait for the transfer to complete
    dma.mfc_read_tag_status_all(code);

    # Release the registers
    code.release_register(buffer_size.reg)
    code.release_register(remaining.reg)
    code.release_register(aligned_size.reg)    
    code.release_register(transfer_size.reg)
    code.release_register(cmp.reg)
    code.release_register(ls_addr.reg)
    code.release_register(tag_var.reg)
    code.release_register(ea_load.reg)

    if old_code is not None:
      spu.set_active_code(old_code)
    return 


# ------------------------------------------------------------
# Iterators
# ------------------------------------------------------------  

class syn_iter(object):
  
  def __init__(self, code, count, step = 1, mode = INC, hint = True):
    object.__init__(self)
    
    self.code = code
    self.mode = mode
    self.hint = hint
    self.state = 0
    
    self.n = count
    self.step = step
    
    self.r_count = None
    self.r_stop  = None
    self.r_step  = None
    
    self.current_count = None

    self.start_label = None
    self.continue_label = None

    self.r_start = None
    self._external_start = False
    self._external_stop = False    
    
    return


  def get_acquired_registers(self):
    """
    This is a minor hack that returns a list of the acquired registers.
    It is intended to allow the caller to re-acquire the registers
    after the loop completes in cases where 'subroutines' that are called
    from the loop have not yet been synthesized.  By re-requiring the
    registers, the caller can ensure that the subroutines do not corrupt
    data in them.

    TODO: This is a temporary fix until a better resource management
          scheme is implemented.
    """

    regs = [self.r_count]

    if self.r_step is not None:
      regs.append(self.r_step)

    if not self._external_stop:
      regs.append(self.r_stop)
        
    return regs
    
  def set_start_reg(self, reg):
    self._external_start = True
    self.r_start = reg
    return
    
  def set_stop_reg(self, reg):
    self._external_stop = True
    self.r_stop = reg
    return
  
    
  def get_start(self):
    """
    Used in INC mode to start the count from somewhere other than
    zero.  Has no effect on CTR or DEC modes.
    """
    return 0

  def get_count(self):
    return self.n
  
  def n_steps(self):
    return self.n / self.step

  def step_size(self):
    return self.step
  
  def start(self, align = True, branch = True):
    """Do pre-loop iteration initialization"""
    if self.r_count is None:
      self.r_count = self.code.prgm.acquire_register()
      
    if self.mode == DEC:
      if self._external_start:
        self.code.add(spu.ai(self.r_count, self.r_start, 0))
      else:
        util.load_word(self.code, self.r_count, self.get_count())

    elif self.mode == INC:
      if self.r_stop is None and branch:
        self.r_stop = self.code.prgm.acquire_register()

      if self._external_start:
        self.code.add(spu.ai(self.r_count, self.r_start, 0))
      else:
        util.load_word(self.code, self.r_count, self.get_start())

      if branch and not self._external_stop:
        util.load_word(self.code, self.r_stop, self.get_count())

    # /end mode if
    
    if self.r_count is not None:
      self.current_count = var.SignedWord(code = self.code, reg = self.r_count)

    # If the step size doesn't fit in an immediate value, store it in a register
    # (-512 < word < 511):
    if not (-512 < self.step_size() < 511):
      self.r_step = self.code.prgm.acquire_register()
      util.load_word(self.code, self.r_step, self.step_size())

    # Label
    self.start_label = self.code.prgm.get_unique_label("SYN_ITER_START")
    self.code.add(self.start_label)

    # Create continue/branch labels so they can be referenced; they will be
    # added to the code in their appropriate locations.
    self.branch_label = self.code.prgm.get_unique_label("SYN_ITER_BRANCH")
    self.continue_label = self.code.prgm.get_unique_label("SYN_ITER_CONTINUE")
    return

  def setup(self):
    """Do beginning-of-loop iterator setup/initialization"""
    return

  def get_current(self):
    return self.current_count

  def cleanup(self):
    """Do end-of-loop iterator code"""
    # Update the current count
    if self.mode == DEC:
      if self.r_step is not None:
        self.code.add(spu.sf(self.r_count, self.r_step, self.r_count))
      else:
        self.code.add(spu.ai( self.r_count, self.r_count, -self.step_size()))
    elif self.mode == INC:
      if self.r_step is not None:
        self.code.add(spu.a(self.r_count, self.r_count, self.r_step))
      else:
        self.code.add(spu.ai(self.r_count, self.r_count, self.step_size()))
      
    return

  def end(self, branch = True):
    """Do post-loop iterator code"""
    if self.hint == True:
      self.code.add(spu.hbrr(self.branch_label, self.start_label))

    if self.mode == DEC:
      # branch if r_count is not zero (CR)
      #   Note that this relies on someone (e.g. cleanup()) setting the
      #   condition register properly.
      if branch:
        self.code.add(self.branch_label)
        self.code.add(spu.brnz(self.r_count, self.start_label))

      # Reset the counter in case this is a nested loop
      util.load_word(self.code, self.r_count, self.get_count())

    elif self.mode == INC:
      # branch if r_current < r_stop
      if branch:
        r_cmp_gt = self.code.prgm.acquire_register()

        self.code.add(spu.cgt(r_cmp_gt, self.r_stop, self.r_count))
        self.code.add(self.branch_label)
        self.code.add(spu.brnz(r_cmp_gt, self.start_label))

        self.code.prgm.release_register(r_cmp_gt)        

      # Reset the the current value in case this is a nested loop
      if self._external_start:
        self.code.add(spu.ai(self.r_count, self.r_start, 0))
      else:
        util.load_word(self.code, self.r_count, self.get_start())

    if self.r_count is not None:
      self.code.prgm.release_register(self.r_count)
    if self.r_stop is not None and not self._external_stop:
      self.code.prgm.release_register(self.r_stop)      

    return


  def add_continue(self, code, idx, branch_inst = spu.br):
    """
    Insert a branch instruction to branch to the end of the loop.
    """
    #if self.continue_label is None:
    #  raise Exception('Continue point not set.  Has the loop been synthesized yet?')

    #next = (self.continue_label - idx)
    # print 'Continue:', next, idx, self.continue_label
    #code[idx] = branch_inst(next)
    #code[idx] = branch_inst(self.continue_label)
    code.add(branch_inst(self.continue_label))
    return
  
  def __iter__(self):
    self.start()
    return self

  def next(self):

    if self.state == 0:
      self.state = 1
      self.setup()
      return self.get_current()
    else:
      self.code.add(self.continue_label)
      self.cleanup()
      self.end()
      raise StopIteration

    return


class syn_range(syn_iter):
  """
  Purpose: Iterate a set number of times and make the current
           iteration count available as a variable.
  """

  def __init__(self, code, start, stop = None, step = 1):
    if stop is None:
      stop = start
      start = 0

    syn_iter.__init__(self, code, stop, step = step, mode = INC)

    self.istart = start
    
    return

  def get_start(self):
    return self.istart

def _overlap(lsa, lsb, size):
  lsa_in_lsb = (lsa < lsb) and (lsa + size) > lsb
  lsb_in_lsa = (lsb < lsa) and (lsb + size) > lsa  
  return lsa_in_lsb or lsb_in_lsa



_strides = {'b':1, 'h':2, 'i':4, 'B':1, 'H':2, 'I':4, 'f':4, 'd':8}
_vector_sizes = {'b':16, 'h':8, 'i':4, 'B':16, 'H':8, 'I':4, 'f':4}

class spu_vec_iter(syn_iter):
  """
  Purpose: Iterate over the values as vectors.
  """

  def __init__(self, code, data, step = 1, length = None, store_only = False,
               addr_reg = None, save = True, type_cls = None):
    self.var_type = type_cls or var.array_spu_lu[data.typecode]

    if type(data) not in (_array_type, _extarray_type, memory_desc):
      raise Exception('Unsupported array type')

    if _typecode(data) not in _vector_sizes.keys():
      raise Exception('Unsupported array data type for vector operations: ' + data.typecode)

    stop = 0
    self.data = data
    self.addr_reg = addr_reg
    self.store_only = store_only
    self.save = save
    if length is None:
      length = len(data)

    t = _typecode(data)
    step = (step * _vector_sizes[_typecode(data)]) * _strides[t]
    stop = _strides[t] * length # len(data)
    self.typecode = t

    syn_iter.__init__(self, code, stop, step, mode = INC)

    self.r_current = None
    self.r_addr = None
    self.current_var = None
  
    return

  def get_acquired_registers(self):
    """
    See comment in syn_iter.
    """
    regs = syn_iter.get_acquired_registers(self)

    regs.append(self.r_current)
  
    if self.addr_reg is None:
      regs.append(self.r_addr)
    
    return regs

  def get_current(self): return self.current_var

  def load_current(self):
    return self.code.add(spu.lqx(self.r_current, self.r_addr, self.r_count))

  def store_current(self):
    return self.code.add(spu.stqx(self.r_current, self.r_addr, self.r_count))    

  def make_current(self):
    return self.var_type(code = self.code, reg = self.r_current)

  def init_address(self):
    if self.addr_reg is None:
      return util.load_word(self.code, self.r_addr, _array_address(self.data))
  
  def start(self, align = True, branch = True):
    self.r_current = self.code.prgm.acquire_register()

    # addr_reg is the user supplied address for the data
    if self.addr_reg is None:
      self.r_addr = self.code.prgm.acquire_register()
    else:
      self.r_addr = self.addr_reg

    syn_iter.start(self, align, branch)
    self.current_var = self.make_current()
    self.init_address()

    # print self.r_count, self.r_stop, self.r_current, self.r_addr, self.data.buffer_info()[0]

    return

  def setup(self):
    if not self.store_only:
      self.load_current()
    syn_iter.setup(self)
    return

  def cleanup(self):
    if self.current_var.assigned and self.save:
      self.store_current()
    syn_iter.cleanup(self)
    return

  def end(self, branch = True):
    if self.r_current is not None:
      self.code.prgm.release_register(self.r_current)

    if self.r_addr is not None and self.addr_reg is None:
      self.code.prgm.release_register(self.r_addr)

    syn_iter.end(self, branch)
    return


class stream_buffer(syn_range):
  """
  Manage a buffered data stream from main memory.
  """
  def __init__(self, code, ea, data_size, buffer_size, ls, buffer_mode='single', save = False):
    """Stream buffer.  If save is True, buffers will be written back to main
       memory."""
    syn_range.__init__(self, code, ea, ea + data_size, buffer_size)

    # Buffer addresses
    if buffer_mode == 'single':
      self.lsa = ls
      self.lsb = ls
    elif buffer_mode == 'double':
      if type(ls) is list:
        if _overlap(ls[0], ls[1], buffer_size):
          raise Exception('Local store buffers overlap')
        self.lsa, self.lsb = ls
      else:
        # Assume contiguous buffers: lsa = ls, lsb = ls + buffer_size
        self.lsa = ls
        self.lsb = ls + buffer_size
      
    else:
      raise Exception('Unknown buffering mode: ' + buffer_mode)
  
    self.buffer_mode = buffer_mode
    self.save = save
  
    self.ls = None
    self.tag = None
    self.buffer_size = None
    self.ibuffer_size = buffer_size
    return

  def set_ea_addr_reg(self, reg):
    self.set_start_reg(reg)
    return

  def set_ea_size_reg(self, reg):
    self.set_stop_reg(reg)
    return

  
  # ------------------------------
  # Buffer management
  # ------------------------------

  def _toggle(self, var):
    """
    Use rotate to toggle between two preferred slot  values in a vector.
    """
    if self.buffer_mode == 'double':
      self.code.add(spu.rotqbyi(var.reg, var.reg, 4))
    return

  def _swap_buffers(self):
    return

  def _load_buffer(self):
    # TODO - AWF - some optimization is possible here.
    #  rather than skipping around the DMA get on the last iteration, short out
    #  of the loop completely.  Saves doing the check twice..
    #  Also as soon as we do this first check, we know we are going to go
    #  through the loop again.  Again, no need for a second conditional at the
    #  end, just increment counters and always branch.  A hint could be added
    #  right before the DMA get.

    # Don't perform the load the last time through the loop
    r_cmp = self.code.prgm.acquire_register()

    # Compare count == step
    self.code.add(spu.ceq(r_cmp, self.r_stop, self.r_count))

    # Create a skip label and add the branch
    skip_label = self.code.prgm.get_unique_label("STREAM_BUFFER_SKIP")
    self.code.add(spu.brnz(r_cmp, skip_label))

    # Start the DMA get
    dma.mfc_get(self.code, self.ls, syn_range.get_current(self), self.buffer_size, self.tag)

    # Add the branch label
    self.code.add(skip_label)

    self.code.prgm.release_register(r_cmp)
    return

  def _save_buffer(self):
    dma.mfc_put(self.code, self.ls, syn_range.get_current(self), self.buffer_size, self.tag)
    return

  def _wait_buffer(self):
    # TODO - BUG HERE!!
    # Here's what happens: a variable 'mask' is created, then used.  When this
    # code finishes with the variable, it calls mask.release_register() to
    # release the underlying register, which is no longer needed.  But,
    # release_register() sets mask.reg to None.  Although it appears mask would
    # go out of scope here and be garbage collected, it does not!  mask is
    # still referred to by self.code, since instructions have been added that
    # reference it.  The problem is that if these instructions ever need to be
    # rendered again -- like say, for print_code() -- mask.reg.reg is None,
    # which makes it impossible to render the instruction.
    mask = var.SignedWord(1, self.code)
    mask.v = mask << self.tag

    dma.mfc_write_tag_mask(self.code, mask)
    reg = dma.mfc_read_tag_status_all(self.code)
    self.code.prgm.release_register(reg)

    #mask.release_register()

    return

  # ------------------------------
  # Iterator methods
  # ------------------------------

  def get_current(self):
    """
    Overload current to return the local buffer address.
    Use syn_range.get_current(self) to get the ea/count variable.
    """
    return self.ls


  def _inc_ea(self):
    """
    Increment the ea/count register by step size.  This is used for double buffering.
    """
    if self.r_step is not None:
      vstep = var.SignedWord(code = self.code, reg = self.r_step)
      self.current_count.v = self.current_count + vstep 
    else:
      self.current_count.v = self.current_count + self.step_size()
    return

  def _dec_ea(self):
    """
    Decrement the ea/count register by step size.  This is used for double buffering.
    """
  
    if self.r_step is not None:
      vstep = var.SignedWord(code = self.code, reg = self.r_step)
      self.current_count.v = self.current_count - vstep 
    else:
      self.current_count.v = self.current_count - self.step_size()
    return


  def start(self, align = True, branch = True):    
    """Do pre-loop iteration initialization"""

    syn_range.start(self, align = align, branch = branch)
    if not hasattr(self, 'skip_start_post'):
      self._start_post()
    return

  def _start_post(self):
    # Initialize the buffer size
    self.buffer_size = var.SignedWord(self.ibuffer_size, self.code)
  
    # Initialize the ls and tag vectors with (optionally) alternating values
    if self.buffer_mode == 'single':
      self.ls  = var.SignedWord(self.lsa, self.code)
      self.tag = var.SignedWord(1, self.code)
    else:
      self.ls  = var.SignedWord(array.array('i', [self.lsa, self.lsb, self.lsa, self.lsb]),  self.code)
      self.tag = var.SignedWord(array.array('i', [1, 2, 1, 2]), self.code)

    # For double buffering, load the first buffer
      self._load_buffer()
  
    # Update the start label (make a new one and add it)
    self.start_label = self.code.prgm.get_unique_label("STREAM_BUFFER_START")
    self.code.add(self.start_label)
    return


  def setup(self):
    """Do beginning-of-loop iterator setup/initialization"""

    syn_range.setup(self)

    # Toggle the tag and set the ls to next
    if self.buffer_mode == 'double':
      self._toggle(self.tag)
      self._toggle(self.ls)
      self._inc_ea()

    # Start the transfer of next
    if self.save:
      self._wait_buffer()
    
    self._load_buffer()

    # Reset tag/ls
    if self.buffer_mode == 'double':    
      self._toggle(self.tag)
      self._toggle(self.ls)
      self._dec_ea()

    # Wait for current to complete
    self._wait_buffer()
  
    return


  def cleanup(self):
    """Do end-of-loop iterator code"""

    # Save current
    if self.save:
      self._save_buffer()
    
    # Swap buffers
    self._toggle(self.tag)
    self._toggle(self.ls)

    # Update the counter
    syn_range.cleanup(self)
  
    return

  def end(self, branch = True):
    """Do post-loop iterator code"""
    syn_range.end(self, branch = branch)
  
    self.code.prgm.release_register(self.ls.reg)
    self.code.prgm.release_register(self.tag.reg)
    self.code.prgm.release_register(self.buffer_size.reg)

    return
    
  
class zip_iter: pass

class parallel(object):
  def __init__(self, obj):
    object.__init__(self)
    self.obj = obj

    if type(obj.code.prgm) is not env.ParallelProgram:
      raise Exception("ParallelProgram required")

    if obj.code.prgm.raw_data_size is not None:
      print 'Warning (parallel): raw_data_size is already set'
  
    if type(self.obj) is zip_iter:
      self.obj.iters = [parallel(i) for i in self.obj.iters]
  
    self.state = 0
    return

  def get_start(self): return self.obj.get_start()
  def get_count(self): return self.obj.get_count()
  def n_steps(self):   return self.obj.n_steps()
  def step_size(self): return self.obj.step_size()
  def setup(self): return self.obj.setup()
  def get_current(self): return self.obj.get_current()
  def cleanup(self): return self.obj.cleanup()
  def end(self, branch = True): return self.obj.end(branch)


  def _update_inc_count(self):
    code = self.obj.code

    code.prgm.acquire_block_registers()
  
    r_block_size = code.prgm.r_block_size
    r_offset = code.prgm.r_offset
  
    # Determine the block size for each loop
    code.prgm.raw_data_size = self.get_count() - self.get_start()
    # synppc.load_word(code, r_block_size, self.get_count() - self.get_start())
    # code.add(synppc.ppc.divw(r_block_size, r_block_size, code.r_size))
  
    # Determine the offset for the current block and update the r_count
    # (this is primarily for range, which uses different values in r_count
    #  to initialize ranges that don't start at 0)
    # code.add(synppc.ppc.mullw(r_offset, code.r_rank, r_block_size))
    code.add(spu.a(self.obj.r_count, r_offset, self.obj.r_count))

    # Offset is rank * block_size
    # Count is count + offset
    # Stop is count + block_size
    if self.obj.r_stop is not None:
      code.add(spu.a(self.obj.r_stop, r_block_size, self.obj.r_count))

    # code.prgm.release_register(r_offset)
    # code.prgm.release_register(r_block_size)
    return
    
  def start(self, align = True, branch = True):
    # HACK to get double buffering and parallel working together
    if hasattr(self.obj, '_start_post'):
      self.obj.skip_start_post = True
  
    self.obj.start(align = False, branch = branch)

    code = self.obj.code
    # replace count with rank
    if self.obj.mode == CTR:
      raise Exception('Parallel CTR loops not supported')
    elif self.obj.mode == DEC:
      raise Exception('Parallel DEC loops not supported')
    elif self.obj.mode == INC:
      self._update_inc_count()
    
    if align and branch:
      self.obj.code.align(16)
      # Align the start of the loop on a 16 byte boundary
#      while (code.size()) % 4 != 0:
#        if code.size() % 2 == 0:
#          code.add(spu.nop(0))
#        else:
#          code.add(spu.lnop(0))
        
    # Update the real iterator's label
    self.obj.start_label = code.prgm.get_unique_label("PARALLEL_START")

    # HACK end
    if hasattr(self.obj, '_start_post'):
      self.obj._start_post()

    return 

  def end(self, branch = True):
    self.obj.end(branch)
  
    if self.obj.mode == CTR and branch:
      raise Exception('Parallel CTR loops not supported')
    elif self.obj.mode == DEC:
      raise Exception('Parallel DEC loops not supported')
    elif self.obj.mode == INC:
      self._update_inc_count()

    return

  def init_address(self):
    # Call syn_iters init self.code
    self.obj.init_address(self)

    # Update the address with the offset
    # For variable iterators, this is the value already computed for r_count
    self.obj.code.add(spu.a(self.r_addr, self.obj.r_count, self.r_addr))

    return

  def __iter__(self):
    self.start()
    return self

  def next(self):

    if self.state == 0:
      self.state = 1
      self.setup()
      return self.get_current()
    else:
      self.cleanup()
      self.end()
      raise StopIteration

    return



# ------------------------------------------------------------
# Tests
# ------------------------------------------------------------


def TestSPUIter():
  size = 32
  data = extarray.extarray('I', range(size))
  prgm = env.Program()
  code = prgm.get_stream()

  r_ea_data = prgm.acquire_register()
  r_ls_data = prgm.acquire_register()
  r_size    = prgm.acquire_register()
  r_tag     = prgm.acquire_register()  

  #print 'array ea: %X' % (data.buffer_info()[0])
  #print 'r_zero = %s, ea_data = %s, ls_data = %s, r_size = %s, r_tag = %s' % (
  #  str(code.r_zero), str(r_ea_data), str(r_ls_data), str(r_size), str(r_tag))
  
  # Load the effective address
  util.load_word(code, r_ea_data, data.buffer_info()[0])

  # Load the size
  util.load_word(code, r_size, size * 4)

  # Load the tag
  code.add(spu.ai(r_tag, code.r_zero, 12))

  # Load the lsa
  code.add(spu.ai(r_ls_data, code.r_zero, 0))

  # Load the data into address 0
  dma.mfc_get(code, r_ls_data, r_ea_data, r_size, r_tag)

  # Set the tag bit to 12
  dma.mfc_write_tag_mask(code, 1<<12);

  # Wait for the transfer to complete
  dma.mfc_read_tag_status_all(code);

  # Increment the data values by 1 using an unrolled loop (no branches)
  # r_current = code.acquire_register()
  current = var.SignedWord(0, code)
  
  # Use an SPU iter
  for lsa in syn_iter(code, size * 4, 16):
    code.add(spu.lqx(current, code.r_zero, lsa))
    # code.add(spu.ai(1, r_current, r_current))
    current.v = current + current
    code.add(spu.stqx(current, code.r_zero, lsa))    

  # code.prgm.release_register(r_current)
  #current.release_register(code)
  
  # Store the values back to main memory

  # Load the tag
  code.add(spu.ai(r_tag, code.r_zero, 13))

  # Load the data into address 0
  dma.mfc_put(code, r_ls_data, r_ea_data, r_size, r_tag)

  # Set the tag bit to 12
  dma.mfc_write_tag_mask(code, 1<<13);

  # Wait for the transfer to complete
  dma.mfc_read_tag_status_all(code);

  # Cleanup
  prgm.release_register(r_ea_data)
  prgm.release_register(r_ls_data)  
  prgm.release_register(r_size)
  prgm.release_register(r_tag)  

  # Stop for debugging
  # code.add(spu.stop(0xA))

  # Execute the code
  prgm.add(code)
  proc = env.Processor()
  r = proc.execute(prgm)

  for i in range(0, size):
    assert(data[i] == i + i)

  return



def TestSPUParallelIter(data, size, n_spus = 6, buffer_size = 16, run_code = True):
  import time
  # n_spus = 8
  # buffer_size = 16 # 16 ints/buffer
  # n_buffers   = 4  # 4 buffers/spu
  # n_buffers = size / buffer_size
  # size = buffer_size * n_buffers * n_spus
  # data = array.array('I', range(size + 2))

  #data = env.aligned_memory(n, typecode = 'I')
  #data.copy_to(data_array.buffer_info()[0], len(data_array))


  # print 'Data align: 0x%X, %d' % (data.buffer_info()[0], data.buffer_info()[0] % 16)

  code = env.ParallelInstructionStream()
  # code = env.InstructionStream()

  r_zero    = code.acquire_register()
  r_ea_data = code.acquire_register()
  r_ls_data = code.acquire_register()
  r_size    = code.acquire_register()
  r_tag     = code.acquire_register()  

  # Load zero
  util.load_word(code, r_zero, 0)

  # print 'array ea: 0x%X 0x%X' % (data.buffer_info()[0], long(data.buffer_info()[0]))
  # print 'r_zero = %d, ea_data = %d, ls_data = %d, r_size = %d, r_tag = %d' % (
  #   r_zero, r_ea_data, r_ls_data, r_size, r_tag)

  # Load the effective address
  if data.buffer_info()[0] % 16 == 0:
    util.load_word(code, r_ea_data, data.buffer_info()[0])
  else: 
    util.load_word(code, r_ea_data, data.buffer_info()[0] + 8)

  ea_start = data.buffer_info()[0]
  # Iterate over each buffer
  for ea in parallel(syn_range(code, ea_start, ea_start + size * 4 , buffer_size * 4)):
    # ea = var.SignedWord(code = code, reg = r_ea_data)
  
    # print 'n_iters:', size / buffer_size
    # for i in syn_range(code, size / buffer_size):

    # code.add(spu.stop(0xB))
  
    # Load the size
    util.load_word(code, r_size, buffer_size * 4)

    # Load the tag
    code.add(spu.ai(r_tag, r_zero, 12))

    # Load the lsa
    code.add(spu.ai(r_ls_data, r_zero, 0))

    # Load the data into address 0
    dma.mfc_get(code, r_ls_data, ea, r_size, r_tag)

    # Set the tag bit to 12
    dma.mfc_write_tag_mask(code, 1<<12);

    # Wait for the transfer to complete
    dma.mfc_read_tag_status_all(code);

    # Increment the data values by 1 using an unrolled loop (no branches)
    # r_current = code.acquire_register()
    current = var.SignedWord(0, code)

    count = var.SignedWord(0, code)
    # Use an SPU iter
    for lsa in syn_iter(code, buffer_size * 4, 16):
      code.add(spu.lqx(current, r_zero, lsa))
      # code.add(spu.ai(1, r_current, r_current))
      current.v = current + current
      code.add(spu.stqx(current, r_zero, lsa))    
      count.v = count + 1

    code.add(spu.stqx(count, r_zero, 0))
  
    # code.release_register(r_current)
    current.release_registers(code)

    # Store the values back to main memory

    # Load the tag
    code.add(spu.ai(r_tag, r_zero, 13))

    # Load the data into address 0
    dma.mfc_put(code, r_ls_data, ea.reg, r_size, r_tag)

    # Set the tag bit to 13
    dma.mfc_write_tag_mask(code, 1<<13);

    # Wait for the transfer to complete
    dma.mfc_read_tag_status_all(code);


    # code.add(spu.stop(0xB))

    # Update ea
    # ea.v = ea + (buffer_size * 4)
  # /for ea address 


  # Cleanup
  code.release_register(r_zero)
  code.release_register(r_ea_data)
  code.release_register(r_ls_data)  
  code.release_register(r_size)
  code.release_register(r_tag)  

  if not run_code:
    return code

  # Stop for debugging
  # code.add(spu.stop(0xA))

  # Execute the code
  proc = env.Processor()
  #data.copy_from(data_array.buffer_info()[0], len(data_array))  
  def print_blocks():
    for i in range(0, size, buffer_size):
      # print data[i:(i + buffer_size)]
      print data[i + buffer_size],
    print '' 
  
  # print_blocks()
  s = time.time()
  r = proc.execute(code, n_spus = n_spus)
  # r = proc.execute(code)
  t = time.time() - s
  # print_blocks()

  return t

# LOG = {1:0, 2:1, 4:2, 8:3}

def ParallelTests():
  max_exp = 16
  max_size = pow(2, max_exp)
  print 'Creating data...'
  data = extarray.extarray('I', range(max_size))
  
  print 'Executing Tests...'
  # t = TestSPUParallelIter(data, 8192, n_spus = 1, buffer_size = 128)
  # return 

  i = 0
  for exponent in range(13, max_exp + 1):
    size = pow(2, exponent)
    for n_spus in [1, 2, 4]:

      # Increase the buffer size until to the largest possible factor for the
      # number of SPUs or 4096 (*4=16k), whichever is smaller
      for buffer_exp in range(2, min(exponent - LOG[n_spus] - 2, 12)):
        buffer_size = pow(2, buffer_exp)
        # for buffer_size in [4]:
        t = 0.0
        print 'try\t%d\t%d\t%d\t-.-' % (size, n_spus, buffer_size)
        # for i in range(10):
        t += TestSPUParallelIter(data, size, n_spus = n_spus, buffer_size = buffer_size)
        
        print 'test\t%d\t%d\t%d\t%.8f' % (size, n_spus, buffer_size, t / 10.0)
        # print 'count:', i
        i += 1
  return


def TestStreamBufferSingle(n_spus = 1):
  n = 1024
  a = extarray.extarray('I', range(n))
  buffer_size = 128

  if n_spus > 1:  prgm = env.ParallelProgram()
  else:           prgm = env.Program()
  code = prgm.get_stream()
  
  current = var.SignedWord(0, code)

  addr = a.buffer_info()[0]
  stream = stream_buffer(code, addr, n * 4, buffer_size, 0, save = True)  
  if n_spus > 1:  stream = parallel(stream)

  #r_bufsize = code.acquire_register()
  #r_lsa = code.acquire_register()
  #r_current = code.acquire_register()
  
  for buffer in stream:
    #util.load_word(code, r_bufsize, buffer_size)
    #code.add(spu.il(r_lsa, 0))

    #loop = code.size()
    
    #code.add(spu.lqx(r_current, buffer, r_lsa))
    #code.add(spu.a(r_current, r_current, r_current))
    #code.add(spu.stqx(r_current, buffer, r_lsa))

    #code.add(spu.ai(r_bufsize, r_bufsize, -16))
    #code.add(spu.ai(r_lsa, r_lsa, 16))
    #code.add(spu.brnz(r_bufsize, loop - code.size()))

    for lsa in syn_iter(code, buffer_size, 16):
      code.add(spu.lqx(current, lsa, buffer))
      current.v = current + current
      #current.v = 5
      code.add(spu.stqx(current, lsa, buffer))
      

  prgm.add(code)
  proc = env.Processor()
  r = proc.execute(prgm, n_spus = n_spus)

  for i in range(0, n):
    assert(a[i] == i + i)
  
  return


def TestVecIter(n_spus = 1):
  n = 1024
  a = extarray.extarray('I', range(n))
  
  buffer_size = 16

  if n_spus > 1:  prgm = env.ParallelProgram()
  else:           prgm = env.Program()
  code = prgm.get_stream()

  current = var.SignedWord(0, code)

  stream = stream_buffer(code, a.buffer_info()[0], n * 4, buffer_size, 0, save = True)  
  if n_spus > 1:  stream = parallel(stream)

  md = memory_desc('i', 0, buffer_size)

  for buffer in stream:
    for current in spu_vec_iter(code, md):
      current.v = current + current

  prgm.add(code)
  proc = env.Processor()
  r = proc.execute(prgm, n_spus = n_spus)

  for i in range(0, n):
    assert(a[i] == i + i)

  return


def TestContinueLabel(n_spus = 1):
  n = 1024
  a = extarray.extarray('I', range(n))
  
  buffer_size = 16

  if n_spus > 1:  prgm = env.ParallelProgram()
  else:           prgm = env.Program()
  code = prgm.get_stream()
  
  current = var.SignedWord(0, code)
  test    = var.SignedWord(0, code)
  four    = var.SignedWord(4, code)    

  stream = stream_buffer(code, a.buffer_info()[0], n * 4, buffer_size, 0, save = True)  
  if n_spus > 1:  stream = parallel(stream)

  md = memory_desc('i', 0, buffer_size)
  lsa_iter = spu_vec_iter(code, md)

  for buffer in stream:
    for current in lsa_iter:
      current.v = current + current

      test.v = (current == four)
      code.add(spu.gbb(test, test))
      #lbl_continue = code.add(spu.stop(0xC)) - 1 # Place holder for the continue
      #lsa_iter.add_continue(code, 0, lambda lbl, reg = test.reg: spu.brz(reg, lbl))
      code.add(spu.brz(test.reg, lsa_iter.continue_label))
      current.v = current + current

    #lsa_iter.add_continue(code, lbl_continue, lambda next, reg = test.reg: spu.brz(reg, next))
 
  prgm.add(code) 
  proc = env.Processor()
  r = proc.execute(prgm, n_spus = n_spus)

  for i in range(0, n):
    if i >= 4:
      assert(a[i] == i + i)
    else:
      #print a[i]
      assert(a[i] == i * 4)
  return


def TestStreamBufferDouble(n_spus = 1):
  n = 2048
  a = extarray.extarray('I', range(n))
  
  buffer_size = 32

  if n_spus > 1:  prgm = env.ParallelProgram()
  else:           prgm = env.Program()
  code = prgm.get_stream()

  current = var.SignedWord(0, code)

  addr = a.buffer_info()[0]
  n_bytes = n * 4
  #print 'addr 0x%(addr)x %(addr)d' % {'addr':a.buffer_info()[0]}, n_bytes, buffer_size

  stream = stream_buffer(code, addr, n_bytes, buffer_size, 0, buffer_mode='double', save = True)
  if n_spus > 1:  stream = parallel(stream)

  for buffer in stream:
    for lsa in syn_iter(code, buffer_size, 16):
      code.add(spu.lqx(current, lsa, buffer))
      current.v = current + current
      code.add(spu.stqx(current, lsa, buffer))

  prgm.add(code)
  proc = env.Processor()
  r = proc.execute(prgm, n_spus = n_spus)

  for i in range(0, len(a)):
    assert(a[i] == i + i)
  
  return


# def TestMemoryMap(n_spus = 1):
#   import mmap
#   import os
#   filename = 'spuiter.TestMemoryMap.dat'
#   n = 8192
#   print 'hello'
#   # Create a file
#   fw = open(filename, 'w')
#   fw.write('-' * (8192 + 32))
#   fw.close()

#   # Open the file again for memory mapping
#   f = open(filename, 'r+')
#   size = os.path.getsize(filename)
#   m = mmap.mmap(f.fileno(), n)
#   print 'size:', size, n
#   # Create a memory descriptor
#   md = memory_desc('I', size = size)
#   md.from_ibuffer(m)

#   if n_spus > 1:  code = env.ParallelInstructionStream()
#   else:           code = env.InstructionStream()

#   current = var.SignedWord(0, code)
#   X = var.SignedWord(0x58585858, code)
#   buffer_size = 16
  
#   # code.add(spu.stop(0xB))
#   stream = stream_buffer(code, md.addr, md.size, buffer_size, 0, buffer_mode='double', save = True)
#   if n_spus > 1:  stream = parallel(stream)
  
#   for buff in stream:
#     for lsa in syn_iter(code, buffer_size, 16):
#       code.add(spu.lqx(buff.reg, lsa.reg, current.reg))
#       current.v = X
#       code.add(spu.stqx(buff.reg, lsa.reg, current.reg))
    
#   proc = env.Processor()
#   r = proc.execute(code, n_spus = n_spus)

#   for i in range(0, n): # , buffer_size / 4):
#     # print a[i:(i+buffer_size/4)]
#     # assert(a[i] == i + i)
#     pass
#   return


# def TestBranchHinting():
#   import time
#   code = env.InstructionStream()
#   a = var.SignedWord(0, code)
#   s = time.time()
#   for i in syn_iter(code, pow(2, 16), hint=False):
#     a.v = a + a
#   e = time.time() - s
#   print "Without hint: ", e
#   s = time.time()
#   for i in syn_iter(code, pow(2, 16), hint=True):
#     a.v = a + a
#   e = time.time() - s
#   print "With hint: ", e
#   return

if __name__=='__main__':
  TestSPUIter()
  TestVecIter()
  ## TestMemoryMap(1)
  TestContinueLabel()  
  TestStreamBufferSingle(1)
  TestStreamBufferDouble(4)
    
  # TestSPUParallelIter()
  # ParallelTests()

  # TestZipIter()
