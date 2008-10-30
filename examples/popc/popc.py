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

# Population count examples

import array

import corepy.arch.spu.platform as synspu
import corepy.arch.spu.isa as spu
import corepy.arch.spu.types.spu_types as var
import corepy.arch.spu.lib.iterators as spuiter
import corepy.arch.spu.lib.dma as dma



class syn_popc:
  def __init__(self):
    return

    
  def popc(self, count, x):
    """
    Add the number of 1 bits in each word in X to the value in count.
    """
    temp = spu.get_active_code().acquire_register()
    
    spu.cntb(temp, x)
    spu.sumb(temp, temp, 0)
    spu.a(count, count, temp)

    spu.get_active_code().release_register(temp)
    return

  def reduce_word(self, result, x):
    """
    Add-reduce a vector of words into the preferred
    slot of result.
    """

    for i in range(4):
      spu.a(result, x, result)
      spu.rotqbyi(x, x, 4)

    return

  def synthesize(self, code):
    old_code = spu.get_active_code()
    spu.set_active_code(code)

    # Reserve two variable registers
    count  = code.acquire_register()
    result = code.acquire_register()
    
    # 'Load' the input vector x from register 5
    x = code.acquire_register() 
    spu.ai(x, 5, 0)

    # Zero count and result
    spu.xor(count, count, count)
    spu.xor(result, result, result)
    
    # Inline the popc and reduce operations
    self.popc(count, x)
    self.reduce_word(result, count)

    # Send the result to the caller
    spu.wrch(result, dma.SPU_WrOutMbox)    

    code.release_register(x)
    spu.set_active_code(old_code)
    return


class syn_popc_var:
  def __init__(self):
    return

    
  def popc(self, count, x):
    """
    Add the number of 1 bits in each word in X to the value in count.
    """
    temp   = var.Word()
    temp.v = spu.cntb.ex(x)
    temp.v = spu.sumb.ex(temp, 0)
    count.v = count + temp

    return

  def reduce_word(self, result, x):
    """
    Add-reduce a vector of words into the preferred
    slot of result.
    """

    for i in range(4):
      result.v = result + x
      x.v = spu.rotqbyi.ex(x, 4)

    return

  def synthesize(self, code):
    old_code = spu.get_active_code()
    spu.set_active_code(code)

    # Create and initialize the variables
    count  = var.Word(0)
    result = var.Word(0)
    x = var.Word(0)    

    # 'Load' the input vector x from register 5
    x.v = spu.ai.ex(5, 0)
    
    # Inline the popc and reduce operations
    self.popc(count, x)
    self.reduce_word(result, count)

    # Send the result to the caller
    spu.wrch(result, dma.SPU_WrOutMbox)    

    spu.set_active_code(old_code)
    return


class syn_popc_stream:
  """
  Count the '1' bits in a data stream.
  """
  def __init__(self):
    self.stream_addr = None
    self.stream_size = None

    self.lsa = 0
    self.buffer_size = 256
    self.double_buffer = False
    return

  def set_stream_addr(self, addr): self.stream_addr = addr
  def set_stream_size(self, size): self.stream_size = size  

  def set_buffer_size(self, size): self.buffer_size = size
  def set_double_buffer(self, b):  self.double_buffer = b

  def synthesize(self, code):
    old_code = spu.get_active_code()
    spu.set_active_code(code)


    stream = spuiter.stream_buffer(code, self.stream_addr, self.stream_size * 4,
                                   self.buffer_size, self.lsa)
    ls_data = spuiter.memory_desc('I', self.lsa, self.buffer_size / 4)
    popc = syn_popc_var()

    x = var.Word(0)
    count = var.Word(0)
    total = var.Word(0)

    for buffer in stream:
      for x in spuiter.spu_vec_iter(code, ls_data, addr_reg = buffer):
        popc.popc(count, x)

    popc.reduce_word(total, count)

    # Send the result to the caller
    spu.wrch(total, dma.SPU_WrOutMbox)    

    spu.set_active_code(old_code)
    return

def test_c_popc():
  code = synspu.NativeInstructionStream("spu_popc")
  proc = synspu.Processor()

  params = synspu.spu_exec.ExecParams()
  params.p7  = 0x01010101 # 4 bits
  params.p8  = 0xFFFFFFFF # 32 bits
  params.p9  = 0x10101010 # 4 bits
  params.p10 = 0xFF0FF0F0 # 20 bits = 60 bits total
  
  count = proc.execute(code, mode='mbox', params = params)

  assert(count == 60)
  print 'test_syn_c passed'
  return


def test_syn(kernel):
  code = synspu.InstructionStream()
  proc = synspu.Processor()

  popc = kernel()
  popc.synthesize(code)

  params = synspu.spu_exec.ExecParams()
  params.p7  = 0x01010101 # 4 bits
  params.p8  = 0xFFFFFFFF # 32 bits
  params.p9  = 0x10101010 # 4 bits
  params.p10 = 0xFF0FF0F0 # 20 bits = 60 bits total
  
  count = proc.execute(code, mode='mbox', params = params)
  assert(count == 60)

  return

def test_syn_popc():
  test_syn(syn_popc)
  print 'test_syn_popc passed'
  return

def test_syn_popc_var():
  test_syn(syn_popc_var)
  print 'test_syn_popc_var passed'  
  return

def test_stream_popc():
  code = synspu.InstructionStream()
  proc = synspu.Processor()

  bits = array.array('I', range(1024))
  for i in range(0, 1024, 4):
    bits[i]   = 0x01010101 # 4 bits
    bits[i+1] = 0xFFFFFFFF # 32 bits
    bits[i+2] = 0x10101010 # 4 bits
    bits[i+3] = 0xFF0FF0F0 # 20 bits = 60 bits total
#    bits[i]   = 1
#    bits[i+1] = 2
#    bits[i+2] = 3
#    bits[i+3] = 4

    
  #abits = synspu.aligned_memory(len(bits), typecode = 'I')
  #abits.copy_to(bits.buffer_info()[0], len(bits))

  popc = syn_popc_stream()
  popc.set_stream_addr(bits.buffer_info()[0])
  popc.set_stream_size(len(bits))

  popc.synthesize(code)

  count = proc.execute(code, mode='mbox')
  print '-->', count
  assert(count == 60 * 1024 / 4)

  return


if __name__=='__main__':
  test_c_popc()  
  test_syn_popc()
  test_syn_popc_var()
  test_stream_popc()
