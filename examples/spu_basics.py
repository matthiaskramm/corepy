# Copyright 2006-2007 The Trustees of Indiana University.

# This software is available for evaluation purposes only.  It may not be
# redistirubted or used for any other purposes without express written
# permission from the authors.

# Authors:
#   Christopher Mueller (chemuell@cs.indiana.edu)
#   Andrew Lumsdaine    (lums@cs.indiana.edu)

import array
import time

import corepy.arch.spu.isa as spu
from corepy.arch.spu.types.spu_types import SignedWord, SingleFloat
from corepy.arch.spu.lib.iterators import memory_desc, spu_vec_iter, \
     stream_buffer, syn_iter, parallel
import corepy.arch.spu.lib.dma as dma
from corepy.arch.spu.platform import InstructionStream, ParallelInstructionStream, \
     NativeInstructionStream, Processor, aligned_memory, spu_exec

  
def SimpleSPU():
  """
  A very simple SPU that computes 11 + 31 and returns 0xA on success.
  """
  code = InstructionStream()
  proc = Processor()

  spu.set_active_code(code)
  

  # Acquire two registers
  x    = code.acquire_register()
  test = code.acquire_register()

  spu.xor(x, x, x) # zero x
  spu.ai(x, x, 11) # x = x + 11
  spu.ai(x, x, 31) # x = x + 31

  spu.ceqi(test, x, 42) # test = (x == 42)

  # If test is false (all 0s), skip the stop(0x200A) instruction
  spu.brz(test, 2)
  spu.stop(0x200A)
  spu.stop(0x200B)
  
  r = proc.execute(code) 
  assert(r == 0xA)

  return


def MemoryDescExample(data_size = 20000):
  """
  This example uses a memory descriptor to move 20k integers back and 
  forth between main memory and the SPU local store. Each value is
  incremented by 1 while on the SPU.
  
  Memory descriptors are a general purpose method for describing a
  region of memory.  Memory is described by a typecode, address, and
  size.  Memory descriptors can be initialized by hand or from an
  array or buffer object.

  For main memory, memory descriptors are useful for transfering data
  between main memory and an SPU's local store.  The get/put methods
  on a memory descriptor generate the SPU code to move data of any
  size between main memory and local store.

  Memory descriptors can also be used with spu_vec_iters to describe
  the region of memory to iterate over.  The typecode in the memory
  descriptor is used to determine the type for the loop induction
  variable.

  Note that there is currently no difference between memory
  descriptors for main memory and local store.  It's up to the user to
  make sure the memory descriptor settings make sense in the current
  context.  (this will probably change in the near future)

  Note: get/put currently use loops rather than display lists for
        transferring data over 16k.
  """
  
  code = InstructionStream()
  proc = Processor()

  code.debug = True
  spu.set_active_code(code)

  # Create a python array
  data = array.array('I', range(data_size))

  # Align the data in the array
  a_data = aligned_memory(data_size, typecode = 'I')
  a_data.copy_to(data.buffer_info()[0], data_size)
  
  # Create memory descriptor for the data in main memory
  data_desc = memory_desc('I')
  data_desc.from_array(a_data)

  # Transfer the data to 0x0 in the local store
  data_desc.get(code, 0)

  # Create memory descriptor for the data in the local store for use
  # in the iterator  
  lsa_data = memory_desc('i', 0, data_size)

  # Add one to each value
  for x in spu_vec_iter(code, lsa_data):
    x.v = x + 1

  # Transfer the data back to main memory
  data_desc.put(code, 0)

  dma.spu_write_out_mbox(code, 0xCAFE)
  
  # Execute the synthetic program
  # code.print_code()
  
  spe_id = proc.execute(code, mode = 'async')
  proc.join(spe_id)

  # Copy it back to the Python array
  a_data.copy_from(data.buffer_info()[0], data_size)

  for i in range(data_size):
    assert(data[i] == i + 1)
  return


def DoubleBufferExample(n_spus = 6):
  """
  stream_buffer is an iterator that streams data from main memory to
  SPU local store in blocked buffers.  The buffers can be managed
  using single or double buffering semantics.  The induction variable
  returned by the buffer returns the address of the current buffer.

  Note: stream_buffer was designed before memory descriptors and has
        not been updated to support them yet.  The interface will
        change slightly when the memory classes are finalized.
  """
  n = 30000
  buffer_size = 16

  # Create an array and align the data
  a_array = array.array('I', range(n))
  a = aligned_memory(n, typecode = 'I')
  a.copy_to(a_array.buffer_info()[0], len(a_array))

  addr = a.buffer_info()[0]  
  n_bytes = n * 4

  if n_spus > 1:  code = ParallelInstructionStream()
  else:           code = InstructionStream()

  current = SignedWord(0, code)
  two = SignedWord(2, code)

  # Create the stream buffer, parallelizing it if using more than 1 SPU
  stream = stream_buffer(code, addr, n_bytes, buffer_size, 0, buffer_mode='double', save = True)
  if n_spus > 1:  stream = parallel(stream)

  # Loop over the buffers
  for buffer in stream:

    # Create an iterators that computes the address offsets within the
    # buffer.  Note: this will be supported by var/vec iters soon.
    for lsa in syn_iter(code, buffer_size, 16):
      code.add(spu.lqx(current, lsa, buffer))
      current.v = current - two
      code.add(spu.stqx(current, lsa, buffer))

  # Run the synthetic program and copy the results back to the array 
  proc = Processor()
  r = proc.execute(code, n_spus = n_spus)
  a.copy_from(a_array.buffer_info()[0], len(a_array))    

  for i in range(2, len(a_array)):
    try:
      assert(a_array[i] == i - 2)
    except:
      print 'DoubleBuffer error:', a_array[i], i - 2
  
  return

def SpeedTest(n_spus = 6, n_floats = 6):
  """
  Get a rough estimate of the maximum flop count.
  On a PS3 using all 6 spus, this is 152 GFlops.
  """

  if n_spus > 1:  code = ParallelInstructionStream()
  else:           code = InstructionStream()

  spu.set_active_code(code)
  
  f_range = range(n_floats)
  a = [SingleFloat(0.0) for i in f_range]
  b = [SingleFloat(0.0) for i in f_range]
  c = [SingleFloat(0.0) for i in f_range]  
  t = [SingleFloat(0.0) for i in f_range]

  outer = 2**12
  inner = 2**16
  unroll = 128
  fuse = 2
  simd = 4
  for x in syn_iter(code, outer):
    for y in syn_iter(code, inner):
      for u in range(unroll):
        for i in f_range:
          t[i].v = spu.fma.ex(a[i], b[i], c[i])
    

  # Run the synthetic program and copy the results back to the array 
  proc = Processor()
  start = time.time()
  r = proc.execute(code, n_spus = n_spus)
  stop = time.time()
  total = stop - start
  n_ops = long(outer) * inner * long(unroll) * long(n_floats) * long(fuse) * long(simd) * long(n_spus)
  print '%.6f sec, %.2f GFlops' % (total, n_ops / total / 1e9)

#   # Run the native program and copy the results back to the array
#   outer = 2**14
#   inner = 2**16
#   unroll = 1
#   fuse = 1
#   simd = 1

#   proc = Processor()
#   # ncode = NativeInstructionStream("a.out")
#   start = time.time()
#   r = proc.execute(ncode, n_spus = n_spus)
#   stop = time.time()
#   total = stop - start
#   n_ops = long(outer) * inner * long(unroll) * long(n_floats) * long(fuse) * long(simd) * long(n_spus)
#   print '%.6f sec, %.2f GFlops' % (total, n_ops / total / 1e9)

  results = """
  --> No optimizations
  Executing native code: a.out
  14.805322 sec, 20.89 GFlops

  --> Synthetic
  Platform: linux.spre_linux_spu
  no raw data
  65.023350 sec, 152.19 GFlops

  --> -O3 (fuse: 2, simd: 4)
  Executing native code: a.out
  7.407939 sec, 41.74 GFlops

  --> -O3 (fuse: 1, simd: 1)
  Executing native code: a.out
  7.403702 sec, 5.22 GFlops
  """
  return

if __name__=='__main__':
  SimpleSPU()
  
  for i in [4100, 10000, 20000, 30000]:
    MemoryDescExample(i)

  DoubleBufferExample()
  SpeedTest()

