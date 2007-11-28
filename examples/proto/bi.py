# Testing the bug in bi

import array
import time

import corepy.arch.spu.isa as spu
from corepy.arch.spu.types.spu_types import SignedWord, SingleFloat
from corepy.arch.spu.lib.iterators import memory_desc, spu_vec_iter, \
     stream_buffer, syn_iter, parallel
import corepy.arch.spu.lib.dma as dma
from corepy.arch.spu.platform import InstructionStream, ParallelInstructionStream, \
     NativeInstructionStream, Processor, aligned_memory, spu_exec

  
def bi_bug():
  """
  A very simple SPU that computes 11 + 31 and returns 0xA on success.
  """
  code = InstructionStream()
  proc = Processor()

  spu.set_active_code(code)

  # Acquire two registers
  stop_inst = SignedWord(0x200D)
  stop_addr = SignedWord(0x0)

  spu.stqa(stop_inst, 0x0)
  spu.bi(stop_addr)
  spu.stop(0x200A)
  
  r = proc.execute(code) 
  assert(r == 0xD)

  return

if __name__=='__main__':
  bi_bug()
