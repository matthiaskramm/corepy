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

import corepy.arch.x86.isa as x86
import corepy.arch.x86.types.registers as registers
import corepy.arch.x86.lib.memory as memory

CTR = 0
DEC = 1
INC = 2

_ws = 4

class syn_iter(object):

  def __init__(self, code, count, step = 1, mode = INC, count_reg = None, clobber_reg = None):

    if mode != CTR and count_reg == None:
      raise Exception('No count register was specified (a register must be specified for x86 unless mode CTR is used)')
    if mode == CTR:
      if count_reg != None and count_reg != registers.ecx:
        raise Exception('If mode is CTR, count_reg must be None or ecx')
      count_reg = registers.ecx
    
    self.code = code
    self.mode = mode
    self.state = 0

    self.external_start = False
    self.external_stop = False
    if self.mode == CTR:
      if step != 1:
        raise Exception('CTR loops must have step_size of 1, you used ' + str(self.step_size()))
    self.step = step
    if isinstance(step, registers.GPRegisterType) or isinstance(step, memory.MemRef):
      self.r_step = step
    else:
      self.r_step = None
      
    self.r_count = count_reg
    self.r_start  = None
    self.r_stop  = None
    if clobber_reg != None and not isinstance(clobber_reg, registers.GPRegisterType):
      raise Exception('clobber_reg must refer to a register.')
    else:
      self.r_clobber = clobber_reg

    if isinstance(count, registers.GPRegisterType) or isinstance(count, memory.MemRef):
      if mode == INC:
        self.external_stop = True
        self.r_stop  = count
      else:
        self.external_start = True
        self.r_start  = count
    else:
      self.n = count

    self.current_count = None

    self.start_label = None
    self.continue_label = None
    
    return

  def set_start(self, reg):
    self.external_start = True
    self.r_start = reg
    return

  def set_stop(self, reg):
    self.r_stop = reg
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
  
  def step_size(self):
    return self.step
  
  def start(self, align = True):

    if self.mode == CTR:
      if self.external_start:
        self.code.add(x86.mov(registers.ecx, self.r_start))
      else:
        self.code.add(x86.mov(registers.ecx, self.n))

    elif self.mode == DEC:
      if self.external_start:
        if self.r_start == None:
          raise Exception('No external start register was specified.')
        if isinstance(self.r_count, memory.MemRef) and isinstance(self.r_start, memory.MemRef):
          if self.r_clobber == None:
            raise Exception('Must specify clobber_reg if count_reg and start values are both stored in memory.')
          self.code.add(x86.mov(self.r_clobber, self.r_start))
          self.code.add(x86.mov(self.r_count, self.r_clobber))
        else:
          self.code.add(x86.mov(self.r_count, self.r_start))
      else:
        self.code.add(x86.mov(self.r_count, self.n))

    elif self.mode == INC:
      if self.external_stop:
        if self.r_stop == None:
          raise Exception('No external stop register was specified.')
      if self.external_start:
        if isinstance(self.r_count, memory.MemRef) and isinstance(self.r_start, memory.MemRef):
          self.code.add(x86_64.mov(self.r_clobber, self.r_start))
          self.code.add(x86_64.mov(self.r_count, self.r_clobber))
        else: 
          self.code.add(x86.mov(self.r_count, self.r_start))
      else:
        self.code.add(x86.mov(self.r_count, self.get_start()))

    # /end mode if

    if self.r_count is not None:
      #self.current_count = var.SignedWord(code = self.code, reg = self.r_count)
      self.current_count = self.r_count
      
    # Label
    self.start_label = self.code.prgm.get_unique_label("SYN_ITER_START")
    self.code.add(self.start_label)
    
    # Create continue/branch labels so they can be referenced; they will be
    # added to the code in their appropriate locations.
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
      if self.step_size() == 1:
        self.code.add(x86.dec(self.r_count))
      else:
        if self.r_step is not None:
          if isinstance(self.r_step, memory.MemRef):
            self.code.add(x86.mov(self.r_clobber, self.r_step))
            self.code.add(x86.add(self.r_count, self.r_clobber))
          else:
            self.code.add(x86.add(self.r_count, self.r_step))
        else:
          self.code.add(x86.add(self.r_count, self.step_size()))
    elif self.mode == INC:
      if self.step_size() == 1:
        self.code.add(x86.inc(self.r_count))
      else:
        if self.r_step is not None:
          if isinstance(self.r_step, memory.MemRef):
            self.code.add(x86.mov(self.r_clobber, self.r_step))
            self.code.add(x86.add(self.r_count, self.r_clobber))
          else:
            self.code.add(x86.add(self.r_count, self.r_step))
        else:
          self.code.add(x86.add(self.r_count, self.step_size()))
    return

  def end(self):
    """Do post-loop iterator code"""
    if self.mode == CTR:
      self.code.add(x86.loop(self.start_label))

    elif self.mode == DEC:
      # branch if r_count is not zero (CR)
      #   Note that this relies on someone (e.g. cleanup()) setting the
      #   condition register properly.
      if self.step_size() == 1:
        self.code.add(x86.jnz(self.start_label))
      else:
        self.code.add(x86.cmp(self.r_count, 0))
        self.code.add(x86.jg(self.start_label))

    elif self.mode == INC:
      if self.external_stop:
        if isinstance(self.r_count, memory.MemRef) and isinstance(self.r_stop, memory.MemRef):
          if self.r_clobber == None:
            raise Exception('Must specify clobber_reg if count and stop values are both stored in memory.')
            #self.code.add(x86.push(registers.eax))
            #if self.r_count.base != registers.esp:
            #  self.code.add(x86.mov(registers.eax, self.r_count))
            #else:
            #  oldm = self.r_count
            #  m = memory.MemRef(oldm.base, oldm.disp+8, oldm.index, oldm.scale, oldm.data_size)
            #  self.code.add(x86.mov(registers.eax, m))
            #if self.r_stop.base != registers.esp:
            #  self.code.add(x86.cmp(registers.eax, self.r_stop))
            #else:
            #  oldm = self.r_stop
            #  m = memory.MemRef(oldm.base, oldm.disp+8, oldm.index, oldm.scale, oldm.data_size)
            #  self.code.add(x86.cmp(registers.eax, m))
            #self.code.add(x86.pop(registers.eax))
          else:
            self.code.add(x86.mov(self.r_clobber, self.r_count))
            self.code.add(x86.cmp(self.r_clobber, self.r_stop))
        else:
          self.code.add(x86.cmp(self.r_count, self.r_stop))
      else:
        self.code.add(x86.cmp(self.r_count, self.n))
      self.code.add(x86.jnge(self.start_label))

      

    return

  def add_continue(self, code, idx, branch_inst = x86.jmp):
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
      return self.r_count
    else:
      self.code.add(self.continue_label)
      self.cleanup()
      self.end()
      raise StopIteration

    return

####################################################################################################

# Test with the stop value being an immediate value
def TestINCRegImm():
  A = extarray.extarray('l', 1000)
  B = extarray.extarray('l', 1000)

  for i in xrange(1000):
    A[i] = i

  prgm = env.Program()
  code = prgm.get_stream()
  a = registers.esi
  b = registers.edi
  code.add(x86.mov(a, memory.MemRef(registers.ebp, 2*_ws)))
  code.add(x86.mov(b, memory.MemRef(registers.ebp, 3*_ws)))
  i_iter = syn_iter(code, 1000, mode=INC, count_reg = registers.ecx)
  for i_ in i_iter:
    code.add(x86.mov(registers.eax, memory.MemRef(a, index=i_, scale=4)))
    code.add(x86.mov(memory.MemRef(b, index=i_, scale=4), registers.eax))

  params = env.ExecParams()
  params.p1 = A.buffer_info()[0]
  params.p2 = B.buffer_info()[0]
  prgm += code
  proc = env.Processor()
  proc.execute(prgm, mode='int', params=params)
  
  for i in range(len(B)):
    assert(B[i] == i)

# Test with the current count and the stop value being held in registers
def TestINCRegReg():
  A = extarray.extarray('l', 1000)
  B = extarray.extarray('l', 1000)

  for i in xrange(1000):
    A[i] = i

  prgm = env.Program()
  code = prgm.get_stream()
  a = registers.esi
  b = registers.edi
  code.add(x86.mov(a, memory.MemRef(registers.ebp, 2*_ws)))
  code.add(x86.mov(b, memory.MemRef(registers.ebp, 3*_ws)))
  n = registers.ebx
  code.add(x86.mov(n, 1000))
  i_iter = syn_iter(code, n, mode=INC, count_reg = registers.ecx)
  for i_ in i_iter:
    code.add(x86.mov(registers.eax, memory.MemRef(a, index=i_, scale=4)))
    code.add(x86.mov(memory.MemRef(b, index=i_, scale=4), registers.eax))

  params = env.ExecParams()
  params.p1 = A.buffer_info()[0]
  params.p2 = B.buffer_info()[0]
  prgm += code
  proc = env.Processor()
  proc.execute(prgm, mode='int', params=params)

  for i in range(len(B)):
    assert(B[i] == i)

# Test with the current count being held in a register and the stop value being in memory
def TestINCRegMem():
  A = extarray.extarray('l', 1000)
  B = extarray.extarray('l', 1000) 

  for i in xrange(1000):
    A[i] = i

  prgm = env.Program()
  code = prgm.get_stream()
  a = registers.esi
  b = registers.edi
  code.add(x86.mov(a, memory.MemRef(registers.ebp, 2*_ws)))
  code.add(x86.mov(b, memory.MemRef(registers.ebp, 3*_ws)))
  n = memory.MemRef(registers.ebp, 4*_ws)
  code.add(x86.mov(n, 1000))
  i_iter = syn_iter(code, n, mode=INC, count_reg = registers.ecx)
  for i_ in i_iter:
    code.add(x86.mov(registers.eax, memory.MemRef(a, index=i_, scale=4)))
    code.add(x86.mov(memory.MemRef(b, index=i_, scale=4), registers.eax))

  params = env.ExecParams()
  params.p1 = A.buffer_info()[0]
  params.p2 = B.buffer_info()[0]
  prgm += code
  proc = env.Processor()
  proc.execute(prgm, mode='int', params=params)

  for i in range(len(B)):
    assert(B[i] == i)

# Test with the stop value and the current count both being held in memory
def TestINCMemMem():
  A = extarray.extarray('i', 1000)
  B = extarray.extarray('i', 1000)

  for i in xrange(1000):
    A[i] = i

  prgm = env.Program()
  code = prgm.get_stream()
  a = registers.esi
  b = registers.edi
  code.add(x86.mov(a, memory.MemRef(registers.ebp, 2*_ws)))
  code.add(x86.mov(b, memory.MemRef(registers.ebp, 3*_ws)))
  n = memory.MemRef(registers.ebp, 4*_ws)
  code.add(x86.mov(n, 1000))
  i_iter = syn_iter(code, n, mode=INC, count_reg = memory.MemRef(registers.ebp, 5*_ws), clobber_reg = registers.eax)
  j = registers.ebx
  for i_ in i_iter:   
    code.add(x86.mov(j, i_))
    code.add(x86.mov(registers.eax, memory.MemRef(a, index=j, scale=4)))
    code.add(x86.mov(memory.MemRef(b, index=j, scale=4), registers.eax))

  params = env.ExecParams()
  params.p1 = A.buffer_info()[0]
  params.p2 = B.buffer_info()[0]
  prgm += code
  proc = env.Processor()
  proc.execute(prgm, mode='int', params=params)

  for i in range(len(B)):
    assert(B[i] == i)

# Test with the stop value and the current count both being held in memory with a non-unary step
def TestINCMemMem_ImmStep():
  A = extarray.extarray('l', 1000)
  B = extarray.extarray('l', 1000)

  for i in xrange(1000):
    A[i] = i

  prgm = env.Program()
  code = prgm.get_stream()
  a = registers.esi
  b = registers.edi
  code.add(x86.mov(a, memory.MemRef(registers.ebp, 2*_ws)))
  code.add(x86.mov(b, memory.MemRef(registers.ebp, 3*_ws)))
  n = memory.MemRef(registers.ebp, 4*_ws)
  code.add(x86.mov(n, 1000))
  i_iter = syn_iter(code, n, step=4, mode=INC, count_reg = memory.MemRef(registers.ebp, 5*_ws), clobber_reg = registers.eax)
  j = registers.ebx
  for i_ in i_iter:   
    code.add(x86.mov(j, i_))
    code.add(x86.mov(registers.eax, memory.MemRef(a, index=j, scale=4)))
    code.add(x86.mov(memory.MemRef(b, index=j, scale=4), registers.eax))

  params = env.ExecParams()
  params.p1 = A.buffer_info()[0]
  params.p2 = B.buffer_info()[0]
  prgm += code
  proc = env.Processor()
  proc.execute(prgm, mode='int', params=params)

  for i in range(len(B), 4):
    assert(B[i] == i)

# Test with the stop value and the current count both being held in memory with a non-unary step held in a register
def TestINCMemMem_RegStep():
  A = extarray.extarray('l', 1000)
  B = extarray.extarray('l', 1000)

  for i in xrange(1000):
    A[i] = i

  prgm = env.Program()
  code = prgm.get_stream()
  a = registers.esi
  b = registers.edi
  code.add(x86.mov(a, memory.MemRef(registers.ebp, 2*_ws)))
  code.add(x86.mov(b, memory.MemRef(registers.ebp, 3*_ws)))
  n = memory.MemRef(registers.ebp, 4*_ws)
  code.add(x86.mov(n, 1000))
  s = registers.edx
  code.add(x86.mov(s, 4))
  i_iter = syn_iter(code, n, step=s, mode=INC, count_reg = memory.MemRef(registers.ebp, 5*_ws), clobber_reg = registers.eax)
  j = registers.ebx
  for i_ in i_iter:   
    code.add(x86.mov(j, i_))
    code.add(x86.mov(registers.eax, memory.MemRef(a, index=j, scale=4)))
    code.add(x86.mov(memory.MemRef(b, index=j, scale=4), registers.eax))

  params = env.ExecParams()
  params.p1 = A.buffer_info()[0]
  params.p2 = B.buffer_info()[0]
  prgm += code
  proc = env.Processor()
  proc.execute(prgm, mode='int', params=params)

  for i in range(len(B), 4):
    assert(B[i] == i)

# Test with the stop value and the current count both being held in memory with a non-unary step held in memory
def TestINCMemMem_MemStep():
  A = extarray.extarray('l', 1000)
  B = extarray.extarray('l', 1000)

  for i in xrange(1000):
    A[i] = i

  prgm = env.Program()
  code = prgm.get_stream()
  a = registers.esi
  b = registers.edi
  code.add(x86.mov(a, memory.MemRef(registers.ebp, 2*_ws)))
  code.add(x86.mov(b, memory.MemRef(registers.ebp, 3*_ws)))
  code.add(x86.sub(registers.esp, 12))
  n = memory.MemRef(registers.esp)
  code.add(x86.mov(n, 1000))
  s = memory.MemRef(registers.esp, 2*_ws)
  code.add(x86.mov(s, 4))
  i_iter = syn_iter(code, n, step=s, mode=INC, count_reg = memory.MemRef(registers.esp, 1*_ws), clobber_reg = registers.eax)
  j = registers.ebx
  for i_ in i_iter:   
    code.add(x86.mov(j, i_))
    code.add(x86.mov(registers.eax, memory.MemRef(a, index=j, scale=4)))
    code.add(x86.mov(memory.MemRef(b, index=j, scale=4), registers.eax))

  code.add(x86.add(registers.esp, 12))

  params = env.ExecParams()
  params.p1 = A.buffer_info()[0]
  params.p2 = B.buffer_info()[0]
  prgm += code
  proc = env.Processor()
  proc.execute(prgm, mode='int', params=params)

  for i in range(len(B), 4):
    assert(B[i] == i)

###########################################################################################

# Test with initial count being an immediate value
def TestDECImm():
  A = extarray.extarray('l', 1000)
  B = extarray.extarray('l', 1000)

  for i in xrange(1000):
    A[i] = i

  prgm = env.Program()
  code = prgm.get_stream()
  a = registers.esi
  b = registers.edi
  code.add(x86.mov(a, memory.MemRef(registers.ebp, 2*_ws)))
  code.add(x86.mov(b, memory.MemRef(registers.ebp, 3*_ws)))
  i_iter = syn_iter(code, 1000, mode=DEC, count_reg = registers.ecx)
  for i_ in i_iter:
    code.add(x86.mov(registers.eax, memory.MemRef(a, index=i_, scale=4)))
    code.add(x86.mov(memory.MemRef(b, index=i_, scale=4), registers.eax))

  params = env.ExecParams()
  params.p1 = A.buffer_info()[0]-_ws
  params.p2 = B.buffer_info()[0]-_ws
  prgm += code
  proc = env.Processor()
  proc.execute(prgm, mode='int', params=params)
  
  for i in range(1, len(B)):
    assert(B[i] == i)

# Test with initial count in a register
def TestDECReg():
  A = extarray.extarray('l', 1000)
  B = extarray.extarray('l', 1000)

  for i in xrange(1000):
    A[i] = i

  prgm = env.Program()
  code = prgm.get_stream()
  a = registers.esi
  b = registers.edi
  code.add(x86.mov(a, memory.MemRef(registers.ebp, 2*_ws)))
  code.add(x86.mov(b, memory.MemRef(registers.ebp, 3*_ws)))
  n = registers.ebx
  code.add(x86.mov(n, 1000))
  i_iter = syn_iter(code, n, mode=DEC, count_reg = registers.ecx)
  for i_ in i_iter:
    code.add(x86.mov(registers.eax, memory.MemRef(a, index=i_, scale=4)))
    code.add(x86.mov(memory.MemRef(b, index=i_, scale=4), registers.eax))

  params = env.ExecParams()
  params.p1 = A.buffer_info()[0]-_ws
  params.p2 = B.buffer_info()[0]-_ws
  prgm += code
  proc = env.Processor()
  proc.execute(prgm, mode='int', params=params)

  for i in range(len(B)):
    assert(B[i] == i)

# Test with initial count in memory 
def TestDECMem():
  A = extarray.extarray('l', 1000)
  B = extarray.extarray('l', 1000)

  for i in xrange(1000):
    A[i] = i

  prgm = env.Program()
  code = prgm.get_stream()
  a = registers.esi
  b = registers.edi
  code.add(x86.mov(a, memory.MemRef(registers.ebp, 2*_ws)))
  code.add(x86.mov(b, memory.MemRef(registers.ebp, 3*_ws)))
  n = memory.MemRef(registers.esp)
  code.add(x86.push(1000))
  i_iter = syn_iter(code, n, mode=DEC, count_reg = registers.ecx)
  for i_ in i_iter:
    code.add(x86.mov(registers.eax, memory.MemRef(a, index=i_, scale=4)))
    code.add(x86.mov(memory.MemRef(b, index=i_, scale=4), registers.eax))

  code.add(x86.add(registers.esp, 4))

  params = env.ExecParams()
  params.p1 = A.buffer_info()[0]-_ws
  params.p2 = B.buffer_info()[0]-_ws
  prgm += code
  proc = env.Processor()
  proc.execute(prgm, mode='int', params=params)

  for i in range(1, len(B)):
    assert(B[i] == i)

# Test initial count stored in memory, and current count stored in memory
def TestDECMemMem():
  A = extarray.extarray('l', 1000)
  B = extarray.extarray('l', 1000)

  for i in xrange(1000):
    A[i] = i

  prgm = env.Program()
  code = prgm.get_stream()
  a = registers.esi
  b = registers.edi
  code.add(x86.mov(a, memory.MemRef(registers.ebp, 2*_ws)))
  code.add(x86.mov(b, memory.MemRef(registers.ebp, 3*_ws)))
  n = memory.MemRef(registers.esp)
  #code.add(x86.mov(n, 1000))
  code.add(x86.push(1000))
  i_iter = syn_iter(code, n, mode=DEC, count_reg = memory.MemRef(registers.esp, -1*_ws), clobber_reg=registers.eax)
  j = registers.ebx
  for i_ in i_iter:
    code.add(x86.mov(j, i_))
    code.add(x86.mov(registers.eax, memory.MemRef(a, index=j, scale=4)))
    code.add(x86.mov(memory.MemRef(b, index=j, scale=4), registers.eax))

  code.add(x86.add(registers.esp, 4))

  params = env.ExecParams()
  params.p1 = A.buffer_info()[0]-_ws
  params.p2 = B.buffer_info()[0]-_ws
  prgm += code
  proc = env.Processor()
  proc.execute(prgm, mode='int', params=params)

  for i in range(len(B)):
    assert(B[i] == i)

# Test initial count stored in memory, and current count stored in memory, with a non-unary step size
def TestDECMemMem_ImmStep():
  A = extarray.extarray('l', 1000)
  B = extarray.extarray('l', 1000)

  for i in xrange(1000):
    A[i] = i

  prgm = env.Program()
  code = prgm.get_stream()
  a = registers.esi
  b = registers.edi
  code.add(x86.mov(a, memory.MemRef(registers.ebp, 2*_ws)))
  code.add(x86.mov(b, memory.MemRef(registers.ebp, 3*_ws)))
  n = memory.MemRef(registers.esp)
  #code.add(x86.mov(n, 1000))
  code.add(x86.push(1000))
  i_iter = syn_iter(code, n, mode=DEC, step=-4, count_reg = memory.MemRef(registers.esp, -1*_ws), clobber_reg=registers.eax)
  j = registers.ebx
  for i_ in i_iter:
    code.add(x86.mov(j, i_))
    code.add(x86.mov(registers.eax, memory.MemRef(a, index=j, scale=4)))
    code.add(x86.mov(memory.MemRef(b, index=j, scale=4), registers.eax))

  code.add(x86.add(registers.esp, 4))

  params = env.ExecParams()
  params.p1 = A.buffer_info()[0]-_ws
  params.p2 = B.buffer_info()[0]-_ws
  prgm += code
  proc = env.Processor()
  proc.execute(prgm, mode='int', params=params)

  for i in range(len(B)-1, 0, -4):
    assert(B[i] == i)

# Test initial count stored in memory, and current count stored in memory with non-unary step stored in a register
def TestDECMemMem_RegStep():
  A = extarray.extarray('l', 1000)
  B = extarray.extarray('l', 1000)

  for i in xrange(1000):
    A[i] = i

  prgm = env.Program()
  code = prgm.get_stream()
  a = registers.esi
  b = registers.edi
  code.add(x86.mov(a, memory.MemRef(registers.ebp, 2*_ws)))
  code.add(x86.mov(b, memory.MemRef(registers.ebp, 3*_ws)))
  n = memory.MemRef(registers.esp, 0)
  #code.add(x86.mov(n, 1000))
  code.add(x86.push(1000))
  s = registers.edx
  code.add(x86.mov(s, -4))
  i_iter = syn_iter(code, n, mode=DEC, step=s, count_reg = memory.MemRef(registers.esp, -1*_ws), clobber_reg=registers.eax)
  j = registers.ebx
  for i_ in i_iter:
    code.add(x86.mov(j, i_))
    code.add(x86.mov(registers.eax, memory.MemRef(a, index=j, scale=4)))
    code.add(x86.mov(memory.MemRef(b, index=j, scale=4), registers.eax))

  code.add(x86.add(registers.esp, 4))

  params = env.ExecParams()
  params.p1 = A.buffer_info()[0]-_ws
  params.p2 = B.buffer_info()[0]-_ws
  prgm += code
  proc = env.Processor()
  proc.execute(prgm, mode='int', params=params)

  for i in range(len(B)-1, 0, -4):
    assert(B[i] == i)

# Test initial count stored in memory, and current count stored in memory with non-unary step stored in memory
def TestDECMemMem_MemStep():
  A = extarray.extarray('l', 1000)
  B = extarray.extarray('l', 1000)

  for i in xrange(1000):
    A[i] = i

  prgm = env.Program()
  code = prgm.get_stream()
  a = registers.esi
  b = registers.edi
  code.add(x86.mov(a, memory.MemRef(registers.ebp, 2*_ws)))
  code.add(x86.mov(b, memory.MemRef(registers.ebp, 3*_ws)))
  n = memory.MemRef(registers.esp, 0)
  code.add(x86.sub(registers.esp, 12))
  code.add(x86.mov(n, 1000))
  #code.add(x86.push(1000))
  s = memory.MemRef(registers.esp, 2*_ws)
  code.add(x86.mov(s, -4))
  i_iter = syn_iter(code, n, mode=DEC, step=s, count_reg = memory.MemRef(registers.esp, 1*_ws), clobber_reg=registers.eax)
  j = registers.ebx
  for i_ in i_iter:
    code.add(x86.mov(j, i_))
    code.add(x86.mov(registers.eax, memory.MemRef(a, index=j, scale=4)))
    code.add(x86.mov(memory.MemRef(b, index=j, scale=4), registers.eax))

  code.add(x86.add(registers.esp, 12))

  params = env.ExecParams()
  params.p1 = A.buffer_info()[0]-_ws
  params.p2 = B.buffer_info()[0]-_ws
  prgm += code
  proc = env.Processor()
  proc.execute(prgm, mode='int', params=params)

  for i in range(len(B)-1, 0, -4):
    assert(B[i] == i)


#########################################################################

# Test CTR mode with initial count being an immediate value
def TestCTRImm():
  A = extarray.extarray('l', 1000)
  B = extarray.extarray('l', 1000)

  for i in xrange(1000):
    A[i] = i

  prgm = env.Program()
  code = prgm.get_stream()
  a = registers.esi
  b = registers.edi
  code.add(x86.mov(a, memory.MemRef(registers.ebp, 2*_ws)))
  code.add(x86.mov(b, memory.MemRef(registers.ebp, 3*_ws)))
  i_iter = syn_iter(code, 1000, mode=CTR)
  for i_ in i_iter:
    code.add(x86.mov(registers.eax, memory.MemRef(a, index=i_, scale=4)))
    code.add(x86.mov(memory.MemRef(b, index=i_, scale=4), registers.eax))

  params = env.ExecParams()
  params.p1 = A.buffer_info()[0]-_ws
  params.p2 = B.buffer_info()[0]-_ws
  prgm += code
  proc = env.Processor()
  proc.execute(prgm, mode='int', params=params)

  for i in range(len(B)):
    assert(B[i] == i)

# Test CTR mode with initial count being stored in a register
def TestCTRReg():
  A = extarray.extarray('l', 1000)
  B = extarray.extarray('l', 1000)

  for i in xrange(1000):
    A[i] = i

  prgm = env.Program()
  code = prgm.get_stream()
  a = registers.esi
  b = registers.edi
  code.add(x86.mov(a, memory.MemRef(registers.ebp, 2*_ws)))
  code.add(x86.mov(b, memory.MemRef(registers.ebp, 3*_ws)))
  n = registers.ebx
  code.add(x86.mov(n, 1000))
  i_iter = syn_iter(code, n, mode=CTR)
  for i_ in i_iter:
    code.add(x86.mov(registers.eax, memory.MemRef(a, index=i_, scale=4)))
    code.add(x86.mov(memory.MemRef(b, index=i_, scale=4), registers.eax))

  params = env.ExecParams()
  params.p1 = A.buffer_info()[0]-_ws
  params.p2 = B.buffer_info()[0]-_ws

  prgm += code
  proc = env.Processor()
  proc.execute(prgm, mode='int', params=params)

  for i in range(len(B)):
    assert(B[i] == i)

############################################################################


if __name__=='__main__':
  import corepy.lib.extarray as extarray
  import corepy.arch.x86.platform as env
  TestINCRegImm()
  TestINCRegReg()
  TestINCRegMem()
  TestINCMemMem()
  TestINCMemMem_ImmStep()
  TestINCMemMem_RegStep()
  TestINCMemMem_MemStep()
  TestDECImm()
  TestDECReg()
  TestDECMem()
  TestDECMemMem()
  TestDECMemMem_ImmStep()
  TestDECMemMem_RegStep()
  TestDECMemMem_MemStep()
  TestCTRImm()
  TestCTRReg()
