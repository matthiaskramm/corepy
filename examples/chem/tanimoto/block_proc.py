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

# Synthetic component for processing blocks of data

import corepy.arch.spu.platform as synspu
import corepy.arch.spu.isa as spu
import corepy.arch.spu.types.spu_types as var
import corepy.arch.spu.lib.iterators as spuiter
import corepy.arch.spu.lib.dma as dma

__doc__ = """

How to handle blocked comparison matrices...

Option 1: SPU manages block loads/address increments/etc

Pros:

- No PPU overhead per block

Cons:

- Fixed set of blocks per SPU could lead to unbalanced loads


Option 2: PPU manages load balancing for blocked code

Pros:

- Easy(er) to manage load balancing

Cons:

- Potential to slow things down if block ops are fast -- becomes PPU bound


Test:

Implement both, measure performance.

"""

array_sizes = {'I':4}

class spu_bank:
  """
  Container for assigned spus.
  """
  def __init__(self):
    self._assigned = []
    self._running  = []
    self._waiting  = []
    return

  def idx(self, spuid): return self._assigned.index(spuid)
  def add(self, spuid): self._assigned.append(spuid)

  def remove(self, spuid):
    self._assigned.remove(spuid)

    if spuid in self._running:
      self._running.remove(spuid)
      log('Warning: removing running SPU from SPU bank')

    if spuid in self._waiting:
      self._waiting.remove(spuid)

    return

  def n_spus(self): return len(self._assigned)
  def n_waiting(self): return len(self._waiting)
  def n_running(self): return len(self._running)

  def get_all(self):return self._assigned[:]
  
  def set_all_waiting(self):
    self._waiting = self._assigned[:]
    self._running = []
    return

  def get_running(self): return self._running[:]
  def set_running(self, spuid):
    if spuid in self._waiting:
      self._waiting.remove(spuid)
    self._running.append(spuid)
    return

  def get_waiting(self): return self._waiting[:]
  def set_waiting(self, spuid):
    if spuid in self._running:
      self._running.remove(spuid)
    self._waiting.append(spuid)
    return
  
  def pop_waiting(self): return self._waiting.pop()

  def reset(self):
    self._assigned = []
    self._running  = []
    self._waiting  = []
    return

  def __str__(self):
    return "  %d spus\n  %d running\n  %d waiting" % (
      self.n_spus(), self.n_running(), self.n_waiting())


class PPU_BlockCompare(object):
  """
  Block management on the PPU
  """

  def __init__(self):
    self.spus = spu_bank()
    self.n_spus = 0

    self.proc = synspu.Processor()
    self.prog = None
    
    return


  def start_spus(self):
    if self.prog is None:
      return
    
    for i in range(self.n_spus):
      spu_id = proc.execute(self.prog, mode='async')

    return

  def wait_spus(self):

    return

  
  def run(self, spu_prog, data, n_spus = 6, block_size = 16384):
    if block_size % array_sizes[a.typecode] != 0:
      raise Exception('Block size must be a multiple of the array data type size')
    
    n = len(data)
    n_bytes = n * array_sizes[a.typecode]
    
    
    for row in range(0, n_bytes, block_size):
      for col in range(row, n_bytes, block_size):
        pass

    return



  

