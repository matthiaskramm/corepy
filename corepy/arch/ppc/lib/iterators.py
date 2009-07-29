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

# Iterator Hierarchy

import array
import random

# import Numeric

import corepy.lib.extarray as extarray
import corepy.arch.ppc.platform as synppc
import corepy.arch.ppc.isa as ppc
import corepy.arch.vmx.isa as vmx

import corepy.arch.ppc.types.ppc_types as vars
import corepy.arch.vmx.types.vmx_types as vmx_vars
import corepy.arch.ppc.lib.util as util


# import synnumeric
# import synbuffer
# import metavar
# import metavec

class _numeric_type: pass

def _typecode(a):
  if type(a) in (_array_type, _extarray_type):
    return a.typecode
  elif type(a) is _numeric_type:
    return a.typecode()
  elif type(a) is memory_desc:
    return a.typecode
  else:
    raise Exception('Unknown array type ' + type(a))

def _array_address(a):
  if type(a) in (_array_type, _extarray_type):
    return a.buffer_info()[0]
  elif type(a) is _numeric_type:
    return synnumeric.array_address(a)
  elif type(a) is memory_desc:
    return a.addr
  else:
    raise Exception('Unknown array type ' + type(a))

class ParallelInstructionStream(synppc.InstructionStream):

  def __init__(self):
    synppc.InstructionStream.__init__(self)

    self.r_rank = self.acquire_register()
    self.r_size = self.acquire_register()

    return

  def release_parallel_registers(self):
    self.release_register(self.r_rank)
    self.release_register(self.r_size)
    return
  
  def _save_registers(self):
    """
    Add the parameter loading instructions to the prologue.
    """
    synppc.InstructionStream._save_registers(self)
    
    # Rank and Size are the only two parameters to the function.  Note
    # that the ABI conventions appear to be off by one word.  r1
    # contains the stack pointer.
    # self._prologue.add(ppc.lwz(self.r_rank, 1, 24)) # param 1 should be + 24
    # self._prologue.add(ppc.lwz(self.r_size, 1, 28)) # param 2 should be + 28

    # Register parameter values
    raise Exception('Fix this')
    self._prologue.add(ppc.addi(self.r_rank, 3, 0)) 
    self._prologue.add(ppc.addi(self.r_size, 4, 0)) 

    return


# _numeric_type = type(Numeric.array(1))
_array_type   = type(array.array('I', [1]))
_extarray_type   = type(extarray.extarray('I', [1]))

CTR = 0
DEC = 1
INC = 2

class syn_iter(object):
  
  def __init__(self, code, count, step = 1, mode = INC):

    self.code = code
    self.mode = mode
    self.state = 0
    
    self.n = count
    self.step = step
    
    self.r_count = None
    self.r_stop  = None

    # Hack to allow the caller to supply the stop register
    self.external_stop = False
    
    self.current_count = None
    self.start_label = None

    return

  def set_external_stop(self, r):
    self.r_stop = r
    self.external_stop = True
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

    if self.r_count is None:
      self.r_count = self.code.prgm.acquire_register()
      
    if self.mode == CTR and branch:
      if self.step_size() != 1:
        raise Exception('CTR loops must have step_size of 1, you used ' + str(self.step_size()))

      if self.external_stop:
        self.code.add(ppc.mtctr(self.r_stop))
      else:
        util.load_word(self.code, self.r_count, self.n_steps())
        self.code.add(ppc.mtctr(self.r_count))

      self.code.prgm.release_register(self.r_count)
      self.r_count = None

    elif self.mode == DEC:
      util.load_word(self.code, self.r_count, self.get_count())

    elif self.mode == INC:
      if self.r_stop is None and branch:
        self.r_stop = self.code.prgm.acquire_register()

      util.load_word(self.code, self.r_count, self.get_start())

      if branch and not self.external_stop:
        util.load_word(self.code, self.r_stop, self.get_count())

    # /end mode if

    if self.r_count is not None:
      self.current_count = vars.UnsignedWord(code = self.code, reg = self.r_count)

    if align and branch:
      # Align the start of the loop on a 16 byte boundary
      while (self.code.size()) % 4 != 0:
        self.code.add(ppc.noop())

    # Label
    self.start_label = self.code.prgm.get_unique_label("SYN_ITER_START")
    self.code.add(self.start_label)

    return

  def setup(self):
    return

  def get_current(self):
    return self.current_count

  def cleanup(self):
    # Update the current count
    if self.mode == DEC:
      # Note: using addic here may cause problems with zip/nested loops...tread with caution! 
      self.code.add(ppc.addic_(self.r_count, self.r_count, -self.step_size()))
    elif self.mode == INC:
      self.code.add(ppc.addi(self.r_count, self.r_count, self.step_size()))
      
    return

  def end(self, branch = True):
    if self.mode == CTR and branch:
        self.code.add(ppc.bdnz(self.start_label))

    elif self.mode == DEC:
      # branch if r_count is not zero (CR)
      #   Note that this relies on someone (e.g. cleanup()) setting the
      #   condition register properly.
      if branch:
        self.code.add(ppc.bgt(self.start_label))

      # Reset the counter in case this is a nested loop
      util.load_word(self.code, self.r_count, self.get_count())

    elif self.mode == INC:
      # branch if r_current < r_stop
      if branch:
        self.code.add(ppc.cmpw(0, self.r_count, self.r_stop))
        #self.code.add(ppc.cmp_(0, 2, self.r_count, self.r_stop))
        self.code.add(ppc.blt(self.start_label))
      
      # Reset the the current value in case this is a nested loop
      util.load_word(self.code, self.r_count, self.get_start())

    if self.r_count is not None:
      self.code.prgm.release_register(self.r_count)
      self.r_count = None
      
    if self.r_stop is not None and not self.external_stop:
      self.code.prgm.release_register(self.r_stop)      
      self.r_count = None
    return

  def __iter__(self):
    self.state = 0
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


class parallel(object):
  def __init__(self, obj):
    object.__init__(self)
    self.obj = obj

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
    r_block_size = code.prgm.acquire_register()
    r_offset = code.prgm.acquire_register()
    
    # Determine the block size for each loop
    util.load_word(code, r_block_size, self.get_count() - self.get_start())
    code.add(ppc.divw(r_block_size, r_block_size, code.r_size))
    
    # Determine the offset for the current block and update the r_count
    # (this is primarily for range, which uses different values in r_count
    #  to initialize ranges that don't start at 0)
    code.add(ppc.mullw(r_offset, code.r_rank, r_block_size))
    code.add(ppc.add(self.obj.r_count, self.obj.r_count, r_offset))
    
    if self.obj.r_stop is not None:
      code.add(ppc.add(self.obj.r_stop, self.obj.r_count, r_block_size))

    code.prgm.release_register(r_offset)
    code.prgm.release_register(r_block_size)
    return
      
  def start(self, align = True, branch = True):
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
      # Align the start of the loop on a 16 byte boundary
      while (code.size()) % 4 != 0:
        code.add(ppc.noop())
      
    # Update the real iterator's label
    self.obj.start_label = code.prgm.get_unique_label("PARALLEL_START")
    code.add(self.obj.start_label)

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
    self.obj.code.add(ppc.add(self.r_addr, self.r_addr, self.obj.r_count))

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


_int_types = ('b', 'h', 'i', 'B', 'H', 'I')
_float_types   = ('f','d')

_strides = {'b':1, 'h':2, 'i':4, 'B':1, 'H':2, 'I':4, 'f':4, 'd':8}
_loads  = {'b':ppc.lbzx, 'h':ppc.lhax, 'i':ppc.lwzx,
           'B':ppc.lbzx, 'H':ppc.lhzx, 'I':ppc.lwzx,
           'f':ppc.lfsx, 'd':ppc.lfdx}
_stores = {'b':ppc.stbx, 'h':ppc.sthx, 'i':ppc.stwx,
           'B':ppc.stbx, 'H':ppc.sthx, 'I':ppc.stwx,
           'f':ppc.stfsx, 'd':ppc.stfdx}


class memory_desc(object):
  def __init__(self, typecode, addr = None, size = None):
    self.typecode = typecode
    self.addr = addr
    self.size = size
    return

  def __len__(self): return self.size

  def from_buffer(self, b):
    """
    Extract the address and size from a buffer object.

    Note: this doesn't very well with buffer objects.
    """
    l = repr(b).split(' ')
    self.size = int(l[l.index('size') + 1])
    self.addr = int(l[l.index('ptr') + 1][:-1], 0)
    print l, self.size, self.addr
    return

  def from_ibuffer(self, m):
    """
    Extract the address and size from an object that supports
    the buffer interface.

    This should be more flexible than the buffer object.
    """
    self.addr, self.size = synbuffer.buffer_info(m)
    return


_array_ppc_lu = { # array_typecode: ppc_type
  'I': vars.Bits,
  'I': vars.UnsignedWord,
  'i': vars.SignedWord,
  'f': vars.SingleFloat,
  'd': vars.DoubleFloat
  }


class var_iter(syn_iter):
  """
  Purpose: Iterate over the values in a scalar array.
  """
  # int_type = metavar.int_var
  # float_type = metavar.float_var
  type_lu = _array_ppc_lu
  
  def __init__(self, code, data, step = 1, length = None, store_only = False, addr_reg = None, save = True):
    self.var_type = None
    self.reg_type = None

    stop = 0
    self.data = data
    self.addr_reg = addr_reg
    self.store_only = store_only
    self.save = save
  
    if length is None:
      length = len(data)

    if type(data) in (_array_type, _extarray_type):
      if (data.typecode in self.type_lu.keys()):
        self.var_type = self.type_lu[data.typecode]
        if data.typecode in ('f', 'd'):
          self.reg_type = 'fp'
      else:
        raise Exception('Unsupported array type: ' + data.typecode)
    
    elif type(data) is _numeric_type:
      raise Exception('Unsupported array type: ' + data.typecode)

    elif type(data) is memory_desc:
      if (data.typecode in self.type_lu.keys()):
        self.var_type = self.type_lu[data.typecode]
        if data.typecode in ('f', 'd'):
          self.reg_type = 'fp'
      else:
        raise Exception('Unsupported memory type: ' + data.typecode)
    
    else:
      raise Exception('Unknown data type:' + str(type(data)))

    t = _typecode(data)
    step = _strides[t] * step
    stop = _strides[t] * length # len(data)
    self.typecode = t

    syn_iter.__init__(self, code, stop, step, mode = INC)

    self.r_current = None
    self.r_addr = None
    self.current_var = None
  
    return

  def get_current(self): return self.current_var

  def load_current(self):
    return self.code.add(_loads[self.typecode](self.r_current, self.r_addr, self.r_count))

  def store_current(self): 
    return self.code.add(_stores[self.typecode](self.r_current, self.r_addr, self.r_count))

  def make_current(self):
    return self.var_type(code = self.code, reg = self.r_current)

  def init_address(self):
    if self.addr_reg is None:
      return util.load_word(self.code, self.r_addr, _array_address(self.data))
  
  def start(self, align = True, branch = True):
    self.r_current = self.code.prgm.acquire_register(reg_type = self.reg_type)

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
      self.r_current = None
    
    if self.r_addr is not None and self.addr_reg is None:
      self.code.prgm.release_register(self.r_addr)
      self.r_addr = None
    
    syn_iter.end(self, branch)
    return


_vector_sizes = {'b':16, 'h':8, 'i':4, 'B':16, 'H':8, 'I':4, 'f':4}

class vector_iter(var_iter):
  """
  Purpose: Iterate over the values in a scalar array returning vectors
           instead of vars.
  """
  type_lu = vmx_vars.array_vmx_lu

  def __init__(self, code, data, step = 1, length = None, store_only = False, addr_reg = None):
    if type(data) not in (_array_type, _extarray_type, _numeric_type):
      raise Exception('Unsupported array type')

    if _typecode(data) not in _vector_sizes.keys():
      raise Exception('Unsupported array data type for vector operations: ' + data.typecode)

    var_iter.__init__(self, code, data,
                      step = (step * _vector_sizes[_typecode(data)]),
                      length = length,
                      store_only = store_only,
                      addr_reg = addr_reg)
    
    # TODO - AWF - better way to force the reg_type to vector?
    #self.reg_type = 'vector'
    return

  def load_current(self):
    return self.code.add(vmx.lvx(self.r_current, self.r_count, self.r_addr))

  def store_current(self):
    return self.code.add(vmx.stvx(self.r_current, self.r_count, self.r_addr))


class zip_iter(syn_iter):
  """
  Purpose: Manage a set of iterators.
  """

  def __init__(self, code, *iters):

    count = min([i.n_steps() for i in iters])

    syn_iter.__init__(self, code, count, mode = INC)
    self.iters = iters

    return

  def start(self, align = True, branch = True):
    for i in self.iters: i.start(branch = False)
    syn_iter.start(self, align, branch)
    return

  def setup(self):
    for i in self.iters: i.setup()
    syn_iter.setup(self)
    return
  
  def get_current(self):
    return [i.get_current() for i in self.iters]

  def cleanup(self):
    for i in self.iters: i.cleanup()
    syn_iter.cleanup(self)
    return

  def end(self, branch = True):
    syn_iter.end(self, branch)
    for i in self.iters: i.end(branch = False)
    return



# class unroll_iter(syn_iter):
#   """
#   Purpose: Repeat an iterator body a set number of times.  Optionally
#            clone variables and reduce at the end of each iteration.
#   """
#  pass



# ------------------------------------------------------------
# Tests
# ------------------------------------------------------------

def TestIter():

  prgm = synppc.Program()
  code = prgm.get_stream()
  prgm.add(code)

  a = vars.SignedWord(0, code = code)
  
  for i in syn_iter(code, 16, 4):
    a.v = a + 1
 
  for i in syn_iter(code, 16, 4, mode = DEC):
    a.v = a + 1
 
  for i in syn_iter(code, 16, 4, mode = INC):
    a.v = a + 1

  for i in syn_iter(code, 16, 4, mode = INC):
    a.v = a + vars.SignedWord.cast(i)
    
  util.return_var(a)
  #a.release_register(code)
  
  proc = synppc.Processor()
  r = proc.execute(prgm)
  
  # print 'should be 36:', r
  assert(r == 36)
  return

def TestExternalStop():

  prgm = synppc.Program()
  code = prgm.get_stream()
  prgm.add(code)
  ppc.set_active_code(code)
  
  # Data
  data = array.array('d', range(5*5))

  # Constants - read only
  n_rows = vars.SignedWord(5)
  n_cols = vars.SignedWord(5)
  addr   = vars.SignedWord(data.buffer_info()[0])  
  dbl_size  = vars.SignedWord(synppc.WORD_SIZE * 2)
  row_bytes = vars.SignedWord(synppc.WORD_SIZE * 5 * 2)

  # Variables - read/write
  sum = vars.DoubleFloat(0.0)
  x = vars.DoubleFloat(0.0)

  offset = vars.SignedWord(0)

  # Iterators
  i_iter = syn_iter(code, 0, mode = INC)
  i_iter.set_external_stop(n_rows.reg)

  j_ctr = syn_iter(code, 0, mode = CTR)
  j_ctr.set_external_stop(n_cols.reg)

  for i in i_iter:
    offset.v = vars.SignedWord.cast(i) * row_bytes
    
    # Note that j_cnt is unreadable since it's in the ctr register
    for j_cnt in j_ctr:
      # Load the next vaule in the matrix
      ppc.lfdx(x, addr, offset)
      sum.v = vars.fmadd(x, x, sum) # sum += x*x
      offset.v = offset + dbl_size

  # code.add(ppc.Illegal())
  util.return_var(sum)

  proc = synppc.Processor()
  r = proc.execute(prgm, mode = 'fp')
  # print 'Test external stop: ', r
  assert(r == 4900.0)
    
  return


def TestNestedIter():

  prgm = synppc.Program()
  code = prgm.get_stream()
  prgm.add(code)
  ppc.set_active_code(code)

  a = vars.UnsignedWord(0)

  for i in syn_iter(code, 5):
    for j in syn_iter(code, 5):
      for k in syn_iter(code, 5):
        a.v = a + i + j + k
      
  util.return_var(a)
  #a.release_register()

  proc = synppc.Processor()
  r = proc.execute(prgm)

  # print 'should be 750:', r
  assert(r == 750)
  return

def TestRange():

  prgm = synppc.Program()
  code = prgm.get_stream()
  prgm.add(code)
  ppc.set_active_code(code)
  
  a = vars.UnsignedWord(0)

  for i in syn_range(code, 7):
    a.v = a + 1

  for i in syn_range(code, 20, 31):
    a.v = a + 1

  for i in syn_range(code, 20, 26, 2):
    a.v = a + 1
  
  util.return_var( a)
  #a.release_register(code)

  proc = synppc.Processor()
  r = proc.execute(prgm)

  # print 'should be 21:', r
  assert(r == 21)

  return


_expected = [10, 11, 12, 13]
def _array_check(result, expected = _expected):
  #if result.typecode == 'b':
  #  for x, y in zip(result, expected):
  #    print "types", type(x), type(y)
  #    assert(ord(x) == y)
  #else:
  for x, y in zip(result, expected):
    assert(x == y)


def TestVarIter():

  prgm = synppc.Program()
  code = prgm.get_stream()
  prgm.add(code)
  ppc.set_active_code(code)
  
  a = array.array('I', range(4))
  for i in var_iter(code, a):
    i.v = i + 10

  ai = array.array('i', range(4))
  for i in var_iter(code, ai):
    i.v = i + 10

    # b = array.array('H', range(4))
    # for i in var_iter(code, b):
    #   i.v = i + 10
    
    # bi = array.array('h', range(4))
    # for i in var_iter(code, bi):
    #   i.v = i + 10
    
    # c = array.array('B', range(4))
    # for i in var_iter(code, c):
    #   i.v = i + 10
    
    # ci = array.array('b', range(4))
    # for i in var_iter(code, ci):
    #   i.v = i + 10

  f = array.array('f', range(4))
  f10 = vars.SingleFloat(10.0)  
  for i in var_iter(code, f):
    i.v = i + f10

  d = array.array('d', range(4))
  d10 = vars.DoubleFloat(10.0)
  for i in var_iter(code, d):
    i.v = i + d10

  proc = synppc.Processor()
  r = proc.execute(prgm)

  _array_check(a)
  _array_check(ai)
  #  print b
  #  print bi
  #  print c
  #  print ci
  _array_check(f)
  _array_check(d)

  # print 'TODO: Implememnt the rest of the integer types (or have a clean way of upcasting to signed/unsigned int)'
  return


def TestMemoryDesc():

  prgm = synppc.Program()
  code = prgm.get_stream()
  prgm.add(code)
  ppc.set_active_code(code)

  a = array.array('I', range(4))
  m = memory_desc('I', a.buffer_info()[0], 4)

  for i in var_iter(code, m):
    i.v = i + 10

  proc = synppc.Processor()
  r = proc.execute(prgm)
  _array_check(a)
  return

# def TestMemoryMap():
#   """
#   Use mmap to map a file and a memory_desc to iterate over the contents.
#   """
#   import mmap
#   import os
#   filename = 'metaiter.TestMemoryMap.dat'

#   # Create a file
#   fw = open(filename, 'w')
#   fw.write('-' * 64)
#   fw.close()

#   # Open the file again for memory mapping
#   f = open(filename, 'r+')
#   size = os.path.getsize(filename)
#   m = mmap.mmap(f.fileno(), size)

#   # Create a descriptor
#   md = memory_desc('I')
#   md.from_ibuffer(m)

#   # Adjust the addr/size to iterate over the middle of the file
#   md.addr += 16
#   md.size = 8
  
#   code = synppc.InstructionStream()
  
#   # 32-bit word for the string 'XXXX'
#   X = metavar.var(code, 0x58585858)

#   # Replace the values in the file with X's
#   for i in var_iter(code, md):
#     i.v = X

#   proc = synppc.Processor()
#   r = proc.execute(code)
  
#   return

def TestVecIter():
  prgm = synppc.Program()
  code = prgm.get_stream()
  prgm.add(code)
  ppc.set_active_code(code)
  
  a = extarray.extarray('I', range(16))
  for i in vector_iter(code, a):
    i.v = vmx.vadduws.ex(i, i)

  ai = extarray.extarray('i', range(16))
  for i in vector_iter(code, ai):
    i.v = vmx.vaddsws.ex(i, i) 

  b = extarray.extarray('H', range(16))
  for i in vector_iter(code, b):
    i.v = vmx.vadduhs.ex(i, i) 

  bi = extarray.extarray('h', range(16))
  for i in vector_iter(code, bi):
    i.v = vmx.vaddshs.ex(i, i) 

  c = extarray.extarray('B', range(16))
  for i in vector_iter(code, c):
    i.v = vmx.vaddubs.ex(i, i) 

  ci = extarray.extarray('b', range(16))
  for i in vector_iter(code, ci):
    i.v = vmx.vaddsbs.ex(i, i) 

  ften = vmx_vars.BitType(10.0)
  f = extarray.extarray('f', range(16))
  for i in vector_iter(code, f):
    i.v = vmx.vaddfp.ex(i, i) 

  proc = synppc.Processor()
  r = proc.execute(prgm)

  expected = [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]

  _array_check(a, expected)
  _array_check(ai, expected)
  _array_check(b, expected)
  _array_check(bi, expected)
  _array_check(c, expected)
  _array_check(ci, expected)
  _array_check(f, expected)

  return

def TestZipIter():
  prgm = synppc.Program()
  code = prgm.get_stream()
  ppc.set_active_code(code)
  prgm.add(code)

  a = extarray.extarray('I', range(16, 32))
  b = extarray.extarray('I', range(32, 48))
  c = extarray.extarray('I', [0 for i in range(16)])
  
  sum = vars.UnsignedWord(0)

  for i, j, k in zip_iter(code, var_iter(code, a), var_iter(code, b),
                          var_iter(code, c, store_only = True)):
    k.v = i + j 
    sum.v = sum + 1
  
  av = vector_iter(code, extarray.extarray('I', range(16)))
  bv = vector_iter(code, extarray.extarray('I', range(16, 32)))
  cv = vector_iter(code, extarray.extarray('I', [0 for i in range(16)]), store_only = True)

  for i, j, k in zip_iter(code, av, bv, cv):
    k.v = vmx.vadduws.ex(i, j)  # i + j 

  util.return_var(sum)
  
  proc = synppc.Processor()
  r = proc.execute(prgm, mode = 'int')

  assert(r == 16)
  print a
  print b
  print c

  print av.data
  print bv.data
  print cv.data
  print 'TODO: Finish checking TestZipIter values'
  return


# def TestParallelIter():
#   code = ParallelInstructionStream()
#   proc = synppc.Processor()

#   result = array.array('I', [42,42,42,13,13,13])
#   data   = array.array('I', range(16))

#   # code.add(ppc.Illegal())    
  
#   a = metavar.var(code, 0)
#   rank = metavar.int_var(code, reg = code.r_rank)  
  
#   # for i, j in parallel(zip_iter(code, syn_iter(code, 16), syn_range(code, 16, 32))):
#   #   a.v = i
#   # for i in parallel(syn_range(code, 16, 32)):
#   #    a.v = i

#   for i in parallel(vector_iter(code, data)):
#     i.v = i + 1
                             
#   metavar.syn_return(code, a)
  
#   t1 = proc.execute(code, mode='async', params=(0,2,0))
#   t2 = proc.execute(code, mode='async', params=(1,2,0))

#   proc.join(t1)
#   proc.join(t2)

#   print data

#   return


if __name__=='__main__':
  # TestMemoryMap()
  util.RunTest(TestIter)
  util.RunTest(TestExternalStop)
  util.RunTest(TestNestedIter)
  util.RunTest(TestRange)
  util.RunTest(TestVarIter)
  util.RunTest(TestMemoryDesc)
  util.RunTest(TestVecIter)
  util.RunTest(TestZipIter)
  # TestParallelIter()
