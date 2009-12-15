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

import array

import corepy.spre.spe as spe
import corepy.lib.extarray as extarray
import corepy.arch.ptx.platform as env
import corepy.arch.ptx.isa as ptx
import corepy.arch.ptx.types.ptx_types as vars
import corepy.arch.ptx.types.registers as reg
import corepy.arch.ptx.lib.util as util

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

# ------------------------------------------------------------
# Iterators
# ------------------------------------------------------------  

class syn_iter(object):
  
  def __init__(self, code, count, step = 1, mode = INC, type='u64'):
    object.__init__(self)
    
    self.code = code
    self.mode = mode
    self.state = 0

    self.type = type
    
    self.n = count
    self.step = step
    if mode == INC:
      self.n_start = 0
      self.n_stop = count
    elif mode == DEC:
      self.n_start = count
      self.n_stop = 0

    self.r_count = None
    self.r_start = None
    self.r_stop  = None
    self.r_step  = None

    self._external_start = False
    self._external_stop = False    
    self._external_step = False
    if mode == INC:
      if isinstance(count, reg.ptxRegister):
        self.type = count.type
        self._external_stop = True
        self.r_stop = count
    elif mode == DEC:
      if isinstance(count, reg.ptxRegister) and mode == DEC:
        self.type = count.type
        self._external_start = True
        self.r_start = count
    if isinstance(step, reg.ptxRegister):
      self._external_step = True
      self.r_step = step

    self.current_count = None

    self.start_label = None

    return


  def get_acquired_registers(self):
    """
    This is a minor hack that returns a list of the acquired registers.
    It is intended to allow the ptxler to re-acquire the registers
    after the loop completes in cases where 'subroutines' that are ptxled
    from the loop have not yet been synthesized.  By re-requiring the
    registers, the ptxler can ensure that the subroutines do not corrupt
    data in them.

    TODO: This is a temporary fix until a better resource management
          scheme is implemented.
    """

    regs = [self.r_count]
    if not self._external_stop:
      regs.append(self.r_stop)
    if not self._external_start:
      regs.append(self.r_start)
    if not self._external_step:
      regs.append(self.r_step)

    return regs
    
  def set_start_reg(self, reg):
    self._external_start = True
    self.r_start = reg
    return
    
  def set_stop_reg(self, reg):
    self._external_stop = True
    self.r_stop = reg
    return
  
  def set_start(self, val):
    """
    Used in INC mode to start the count from somewhere other than
    zero.  Has no effect on CTR or DEC modes.
    """
    if isinstance(val, reg.ptxRegister):
      self.set_start_reg(val)
    else:
      self._external_start = False
      self.n_start = val
    return 0

  def step_size(self):
    return self.step
  
  def start(self, align = True, branch = True):
    """Do pre-loop iteration initialization"""

    if self.r_count is None:
      self.r_count = self.code.prgm.acquire_register(self.type)
      
    if self._external_start == False:
      self.code.add(ptx.mov(self.r_count, self.n_start))
    else:
      self.code.add(ptx.mov(self.r_count, self.r_start))

    self.start_label = self.code.prgm.get_unique_label("SYN_ITER_START")
    self.code.add(self.start_label)

    return

  def setup(self):
    """Do beginning-of-loop iterator setup/initialization"""
    return

  def get_current(self):
    ##return self.current_count
    #return self.r_count
    self.current_count = vars.UnsignedWord(code = self.code, reg = self.r_count)
    return self.current_count

  def cleanup(self):
    """Do end-of-loop iterator code"""
    # Update the current count
    if self.mode == DEC:
      if self._external_step:
        self.code.add(ptx.sub(self.r_count, self.r_count, self.r_step))
      else:
        self.code.add(ptx.sub(self.r_count, self.r_count, self.step))
    elif self.mode == INC:
      if self._external_step:
        self.code.add(ptx.add(self.r_count, self.r_count, self.r_step))
      else:
        self.code.add(ptx.add(self.r_count, self.r_count, self.step))
    return

  def end(self, branch = True):
    """Do post-loop iterator code"""

    p = self.code.prgm.acquire_register('pred')

    if self.mode == DEC:
      if self._external_stop:
        self.code.add(ptx.setp('gt', p, self.r_count, self.r_stop))
      else:
        self.code.add(ptx.setp('gt', p, self.r_count, self.n_stop))

    elif self.mode == INC:
      if self._external_stop:
        self.code.add(ptx.setp('lt', p, self.r_count, self.r_stop))
      else:
        self.code.add(ptx.setp('lt', p, self.r_count, self.n_stop))

    self.code.add(ptx.bra(self.start_label, pred=p))

    # Reset the the current value in case this is a nested loop
    if self._external_start:
      self.code.add(ptx.mov(self.r_count, self.r_start))
    else:
      self.code.add(ptx.mov(self.r_count, self.n_start))

    # TODO: erm put this back in
    #for reg in self.get_acquired_registers():
    #  self.code.prgm.release_register(reg)

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






def TestSynIterDec():
  import corepy.arch.ptx.isa as ptx
  import corepy.arch.ptx.types.registers as regs

  SIZE = 64

  proc = env.Processor(0)

  # build and run the kernel
  prgm = env.Program()
  code = prgm.get_stream()

  _mem = prgm.add_parameter('u64', name='_mem')

  memp = prgm.acquire_register('u64')
  counter = prgm.acquire_register('u32')
  code.add(ptx.ld('param', memp, regs.ptxAddress(_mem)))
  code.add(ptx.mov(counter, 0))
  for i in syn_iter(code, 5, step=1, mode=DEC):
    code.add(ptx.add(counter, counter, 1))
  code.add(ptx.st('global', regs.ptxAddress(memp), counter))
  prgm.add(code)

  ptx_mem_addr = proc.alloc_device('u32', 1)
  mem = extarray.extarray('I', 1)
  mem[0] = 5

  param_list = [ptx_mem_addr.address,]

  proc.copy(ptx_mem_addr, mem)
  prgm.cache_code()
  print prgm.render_string
  proc.execute(prgm, (1, 1, 1, 1, 1), param_list)
  proc.copy(mem, ptx_mem_addr)

  print mem

  #passed = True
  #for i in xrange(0, SIZE):
  #  if ext_output[i] != 5:
  #    passed = False
  #print "Passed == ", passed

  return


def TestSynIterInc():
  SIZE = 64

  # build and run the kernel
  prgm = env.Program()
  code = prgm.get_stream()

  code.add(ptx.dcl_output(reg.o0, USAGE=ptx.usage.pos))
  ones = prgm.acquire_register((1, 1, 1, 1))
  counter = prgm.acquire_register()
  code.add(ptx.mov(counter, ones))

  for i in syn_iter(code, 4, step=1, mode=INC):
    code.add(ptx.iadd(counter, counter, ones))

  code.add(ptx.mov(reg.o0, counter.x))

  domain = (0, 0, SIZE, SIZE)
  proc = env.Processor(0)

  ext_output=proc.alloc_remote('i', 1, SIZE)
  prgm.set_binding(reg.o0, ext_output)

  prgm.add(code)
  proc.execute(prgm, domain)

  passed = True
  for i in xrange(0, SIZE):
    if ext_output[i] != 5:
      passed = False
  print "Passed == ", passed

  proc.free(ext_output)

  return

#def TestSynIterInc():
#  SIZE = 128
#
#  # build and run the kernel
#  code = env.InstructionStream()
#
#  code.add(ptx.dcl_literal(reg.l0, 0, 4, 1, 0))
#  code.add(ptx.dcl_literal(reg.l1, 1, 1, 1, 1))
#  code.add(ptx.mov(reg.r1, reg.l1))
#  code.add(ptx.mov(reg.r0, reg.l0))
#  for i in syn_iter(code, 100, step=1, mode=INC):
#    code.add(ptx.iadd(reg.r1, reg.r1, reg.l1))
#
#  code.cache_code()
#  print code.render_string
#
#  domain = (0, 0, SIZE, SIZE)
#
#  proc = env.Processor()
#  proc.execute(code, 0, domain)
#
#  return


def TestSynIterDecFloat():
  SIZE = 64

  # build and run the kernel
  prgm = env.Program()
  code = prgm.get_stream()

  code.add(ptx.dcl_output(reg.o0, USAGE=ptx.usage.pos))

  ones = prgm.acquire_register((1, 1, 1, 1))
  counter = prgm.acquire_register()
  code.add(ptx.mov(counter, ones))

  for i in syn_iter_float(code, 4.0, step=1.0, mode=DEC):
    code.add(ptx.iadd(counter, counter, ones))

  code.add(ptx.mov(reg.o0, counter.x))

  domain = (0, 0, SIZE, SIZE)
  proc = env.Processor(0)

  ext_output=proc.alloc_remote('i', 1, SIZE, 1)
  prgm.set_binding(reg.o0, ext_output)

  prgm.add(code)
  proc.execute(prgm, domain)

  passed = True
  for i in xrange(0, SIZE):
    if ext_output[i] != 5:
      passed = False
  print "Passed == ", passed

  proc.free(ext_output)

  return


def TestSynIterIncFloat():
  SIZE = 64

  # build and run the kernel
  prgm = env.Program()
  code = prgm.get_stream()

  code.add(ptx.dcl_output(reg.o0, USAGE=ptx.usage.pos))

  ones = prgm.acquire_register((1, 1, 1, 1))
  counter = prgm.acquire_register()
  code.add(ptx.mov(counter, ones))

  for i in syn_iter_float(code, 4.0, step=1.0, mode=INC):
    code.add(ptx.iadd(counter, counter, ones))

  code.add(ptx.mov(reg.o0, counter.x))

  domain = (0, 0, SIZE, SIZE)
  proc = env.Processor(0)

  ext_output=proc.alloc_remote('i', 1, SIZE, 1)
  prgm.set_binding(reg.o0, ext_output)

  prgm.add(code)
  proc.execute(prgm, domain)

  passed = True
  for i in xrange(0, SIZE):
    if ext_output[i] != 5:
      passed = False
  print "Passed == ", passed

  proc.free(ext_output)

  return


def TestSynIterIncFloatExtStop():
  SIZE = 64

  # build and run the kernel
  prgm = env.Program()
  code = prgm.get_stream()

  code.add(ptx.dcl_output(reg.o0, USAGE=ptx.usage.pos))

  ones = prgm.acquire_register((1, 1, 1, 1))
  counter = prgm.acquire_register()
  code.add(ptx.mov(counter, ones))

  stop = prgm.acquire_register((0.0, 4.0, 1.0, 0.0))

  for i in syn_iter_float(code, stop.y, step=stop.z, mode=INC):
    code.add(ptx.iadd(counter, counter, ones))

  code.add(ptx.mov(reg.o0, counter.x))

  domain = (0, 0, SIZE, SIZE)
  proc = env.Processor(0)

  ext_output=proc.alloc_remote('i', 1, SIZE, 1)
  prgm.set_binding(reg.o0, ext_output)

  prgm.add(code)
  proc.execute(prgm, domain)

  passed = True
  for i in xrange(0, SIZE):
    if ext_output[i] != 5:
      passed = False
  print "Passed == ", passed

  proc.free(ext_output)

  return


def TestSynIterIncFloatExtStopExtStart():
  SIZE = 64

  # build and run the kernel
  prgm = env.Program()
  code = prgm.get_stream()

  code.add(ptx.dcl_output(reg.o0, USAGE=ptx.usage.pos))
  ones = prgm.acquire_register((1, 1, 1, 1))
  counter = prgm.acquire_register()
  code.add(ptx.mov(counter, ones))

  stop = prgm.acquire_register((4.0, 4.0, 4.0, 4.0))
  start = prgm.acquire_register((2.0, 2.0, 2.0, 2.0))
  step = prgm.acquire_register((1.0, 1.0, 1.0, 1.0))

  fiter = syn_iter_float(code, stop, step=step, mode=INC)
  fiter.set_start_reg(start)
  for i in fiter:
    code.add(ptx.iadd(counter, counter, ones))

  code.add(ptx.mov(reg.o0, counter.x))

  domain = (0, 0, SIZE, SIZE)
  proc = env.Processor(0)

  ext_output=proc.alloc_remote('i', 1, SIZE, 1)
  prgm.set_binding(reg.o0, ext_output)

  prgm.add(code)
  proc.execute(prgm, domain)

  passed = True
  for i in xrange(0, SIZE):
    if ext_output[i] != 3:
      passed = False
  print "Passed == ", passed

  proc.free(ext_output)

  return

if __name__=='__main__':
  TestSynIterDec()
  #TestSynIterInc()
  #TestSynIterDecFloat()
  #TestSynIterIncFloat()
  #TestSynIterIncFloatExtStop()
  #TestSynIterIncFloatExtStopExtStart()
