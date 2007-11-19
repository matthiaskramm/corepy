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



  

