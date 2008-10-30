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

# Prototype of a save buffer

import corepy.arch.spu.platform as synspu
import corepy.arch.spu.isa as spu
import corepy.arch.spu.types.spu_types as var
import corepy.arch.spu.lib.iterators as spuiter
import corepy.arch.spu.lib.dma as dma
import corepy.arch.spu.lib.util as util


class SaveBuffer(object):
  """
  A save buffer consists of:
    Memory regions:
    - a local store buffer
      - a main memory backing buffer

    Code blocks:
      - save register to buffer
      - save buffer to main memory
      - infrom PPU that main memory buffer is full

  Code block ogranization experiments:

  Case 1:

  [save:
    save register
    if ls buffer full:
      [save to main memory]
      if main memory buffer full:
        [inform PPU]
  ]

  Case 2:
  
  [save:
    save register
    if ls buffer full:
      jmp to save to main memory
  


  
  [save to main memory:
     if main memory buffer full:
       [inform PPU]
  ]

  Case 3:
  
  [save:
    save register
    if ls buffer full:
      jmp to save to main memory
  


  
  [save to main memory:
     if main memory buffer full:
       jmp to inform PPU
  ]

  [inform PPU]
  """

  def __init__(self):
    object.__init__(self)

    # Store the addresses
    self.ls_buffer = None # ls_addr, ls_size, ls_offset, unused
    self.mm_buffer = None # mm_addr, mm_size, mm_offset, unused
    
    self.branch_ls_full = None
    self.jump_regs = None
    return

  def setup(self):
    self.ls_buffer = var.SignedWord(0)
    self.mm_buffer = var.SignedWord(0)        
    return
  
  def init_ls_buffer(self, addr, size, offset = 0):
    code = spu.get_active_code()

    util.set_slot_value(code, self.ls_buffer, 0, addr)
    util.set_slot_value(code, self.ls_buffer, 1, size)
    util.set_slot_value(code, self.ls_buffer, 2, offset)
    return

  def init_mm_buffer(self, addr, size, offset = 0):
    code = spu.get_active_code()

    util.set_slot_value(code, self.mm_buffer, 0, addr)
    util.set_slot_value(code, self.mm_buffer, 1, size)
    util.set_slot_value(code, self.mm_buffer, 2, offset)
    return

  def save_register(self, reg): # , branch_to_save = False):
    code = spu.get_active_code()

    offset = code.acquire_register()
    size = code.acquire_register()
    test = code.acquire_register()
    regs = [offset, size, test]
    
    spu.rotqbyi(offset, self.ls_buffer, 4)
    spu.rotqbyi(size,   self.ls_buffer, 8)

    spu.stqx(reg, self.ls_buffer, offset)
    
    spu.ai(offset, offset, 16)
    spu.ceq(test,  offset, size)

    spu.wrch(size, dma.SPU_WrOutMbox)
    spu.wrch(offset, dma.SPU_WrOutMbox)
    spu.wrch(test, dma.SPU_WrOutMbox)
    # !!! STOPPED HERE !!! THESE VALUES ARE WRONG !!!
    lbl_ls_full = code.size()
    spu.stop(0xB)
    self.save_ls_buffer(ls_size = size)

    spu.nop(0)
    code[lbl_ls_full] = spu.brz(test, (code.size() - lbl_ls_full), ignore_active = True)

    code.release_registers(regs)
    return

  def save_ls_buffer(self, ls_size = None, branch = False):
    code = spu.get_active_code()
    
    regs = []
    if ls_size is None:
      ls_size = code.acquire_register()
      regs.append(ls_size)

    # Set the main memory address
    mm_offset = code.acquire_register()
    regs.append(mm_offset)

    spu.rotqbyi(mm_offset, self.mm_buffer, 4)
    spu.a(mm_offset, mm_offset, self.mm_buffer)

    # Tranfer the buffer
    md = spuiter.memory_desc('b')
    md.set_size_reg(ls_size)
    md.set_addr_reg(mm_offset)

    md.put(code, self.ls_buffer)

    # Increment the main memory offset
    mm_size = code.acquire_register()
    regs.append(mm_size)

    spu.rotqbyi(mm_size, self.mm_buffer, 8)        
    spu.rotqbyi(mm_offset,  self.mm_buffer, 4)
    spu.a(mm_offset, mm_offset, mm_size)

    util.set_slot_value(code, self.mm_buffer, 2, mm_offset)
    
    # Reset the ls offset
    util.set_slot_value(code, self.ls_buffer, 2, 0)
    
    code.release_registers(regs)
    
    return



def TestSaveBuffer1():
  import array

  code = synspu.InstructionStream()
  proc = synspu.Processor()

  code.set_debug(True)
  spu.set_active_code(code)
  
  n = 2**14
  data = array.array('I', range(n))
  #data = synspu.aligned_memory(n, typecode = 'I')
  #data.copy_to(data_array.buffer_info()[0], len(data_array))


  save_buffer = SaveBuffer()
  
  save_buffer.setup()
  save_buffer.init_ls_buffer(0, 128)
  save_buffer.init_mm_buffer(data.buffer_info()[0], n)

  value = var.SignedWord(0xCAFEBABE)
  
  for i in spuiter.syn_iter(code, n / 4):
    save_buffer.save_register(value)

  code.print_code()
  spe_id = proc.execute(code, mode='async')

  for i in range(n/4):
    while synspu.spu_exec.stat_out_mbox(spe_id) == 0: pass
    print 'size: 0x%X' % (synspu.spu_exec.read_out_mbox(spe_id))

    while synspu.spu_exec.stat_out_mbox(spe_id) == 0: pass
    print 'offset: 0x%X' % (synspu.spu_exec.read_out_mbox(spe_id))

    while synspu.spu_exec.stat_out_mbox(spe_id) == 0: pass
    print 'test: 0x%X' % (synspu.spu_exec.read_out_mbox(spe_id))

  proc.join(spe_id)

  #data.copy_from(data_array.buffer_info()[0], len(data_array))  

  print data[:10]
  return


if __name__=='__main__':
  TestSaveBuffer1()
