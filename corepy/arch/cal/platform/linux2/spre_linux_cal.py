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

__doc__="""
SPE for the ATI CAL GPU environment
"""

import corepy.lib.extarray as extarray
import corepy.spre.spe as spe
import corepy.arch.cal.types.registers as reg
import corepy.arch.cal.isa as isa
import cal_exec

# ------------------------------
# Constants
# ------------------------------

WORD_TYPE = 'f'           # array type that corresponds to 1 word
WORD_SIZE = 4             # size in bytes of one word
WORD_BITS = WORD_SIZE * 8 # number of bits in a word

INT_SIZES = {'b':1,  'c':1, 'h':2, 'i':4, 'B':1,  'H':2, 'I':4}

N_GPUS = cal_exec.get_num_gpus()
HAS_NUMPY = cal_exec.HAS_NUMPY

if HAS_NUMPY:
  import numpy


# ------------------------------------------------------------
# InstructionStream
# ------------------------------------------------------------

class InstructionStream(spe.InstructionStream):
  """
  CAL Instruction Stream.  
  """

  def add(self, obj):
    if type(obj) == str:
      self._objects.append(obj)
    else:
      spe.InstructionStream.add(self, obj)


class Program(spe.Program):
  default_register_type = reg.TempRegister
  instruction_type  = WORD_TYPE

  stream_type = InstructionStream

  def __init__(self):
    spe.Program.__init__(self, None)

    # Array of literal register declarations
    # Added to by acquire_register(), rendered by synthesize_prologue()
    self._literals = []
    return


  def reset(self):
    spe.Program.reset(self)
    self._bindings = {}
    self._bindings_data = {}
    #self._declare_registers = {}

    if self._cached:
      cal_exec.free_image(self.render_code)
      self.render_string = None

    return


  def make_executable(self):
    return 


  def create_register_files(self):
    self._register_files[reg.TempRegister] = reg.r
    self._register_files[reg.LiteralRegister] = reg.l
    self._reg_type['r'] = reg.TempRegister
    self._reg_type['l'] = reg.LiteralRegister
    return

  
  def acquire_register(self, reg_type = None, reg_name = None):
    if isinstance(reg_type, (list, tuple)):
      # if this is a LiteralRegister, acquire and set the value
      l = spe.Program.acquire_register(self, reg_type='l', reg_name=reg_name)
      self._literals.append(isa.dcl_literal(l, reg_type[0], reg_type[1], reg_type[2], reg_type[3], ignore_active = True))
      return l
    else:
      return spe.Program.acquire_register(self,
          reg_type = reg_type, reg_name = reg_name)
      
  def release_register(self, register):
    if type(register) != reg.LiteralRegister:
      spe.Program.release_register(self, register)
    # print 'release', str(self._register_files[type])
    return 

  
  # ------------------------------
  # GPU memory binding management
  # ------------------------------

  def set_binding(self, regname, hdl):
    """Bind a register name (string or register object) to a local or remote
       memory allocation obtained from a Processor."""

    if isinstance(hdl, extarray.extarray):
      if not hasattr(hdl, "gpu_mem_handle"):
        raise TypeError("Not an extarray with a GPU memory handle")
      binding = hdl.gpu_mem_handle

    elif isinstance(hdl, numpy.ndarray):
      # NumPy array.. do we support it, and does it use a CAL buffer?
      if not HAS_NUMPY:
        raise ImportError("NumPy array support requires NumPy installation")
      if not isinstance(hdl.base, cal_exec.calmembuffer):
        raise TypeError("Not NumPy with a GPU memory buffer")

      # Build a binding from the underlying CAL buffer
      buf = hdl.base
      binding = [buf.res, buf.pointer]

    elif isinstance(hdl, LocalMemory):
      binding = hdl.binding

    if isinstance(regname, (reg.CALRegister, reg.CALBuffer)):
      regname = regname.name

    self._bindings_data[regname] = hdl
    self._bindings[regname] = binding
    #self._cached = False
    return

  def get_binding(self, regname):
    try:
      return self._bindings_data[regname]
    except KeyError:
      return None


  # ------------------------------
  # Execute/ABI support
  # ------------------------------

  def _synthesize_prologue(self):
    self._prologue = 'il_ps_3_0\n'

    self._prologue += '\n'.join([inst.render() for inst in self._literals]) + '\n'

    #print "PRLOG", self._prologue
    # Add declare instructions for any bindings
    #for (regname, (arr, kwargs)) in self._remote_bindings_data.items():
    #  print "synth prolog check binding", regname, kwargs
    #  # TODO - do this in set_bindings instead?
    #  if kwargs.has_key('decl') and kwargs['decl'] == False:
    #    continue;

    #  # Switch on the regname
    #  if regname[0] == 'o':
    #    inst = isa.dcl_output(regname, **kwargs)
    #    print "inserting inst", inst.render()
    #    self._prologue += inst.render() + '\n'
      #elif regname[0] == 'i':
      #  dim = isa.pixtex_type.oned
      #  #if self._remote_bindings[regname].
      #  inst = isa.dcl_resource(regname[1:], **kwargs)
      #  print "inserting inst", inst.render()
      #  self._prologue += inst.render() + '\n'

    return

  def _synthesize_epilogue(self):
    self._epilogue = 'end\n'
    return


  def _cache_code_S(self, render_string, stream):
    for obj in stream:
      if isinstance(obj, spe.Instruction):
        render_string += obj.render() + '\n'
      elif isinstance(obj, str):
        if obj[-1] == '\n':
          render_string += obj
        else:
          render_string += obj + '\n'
      elif isinstance(obj, Label):
        # TODO - AWF - make labels work
        raise Exception("Are labels supported? don't think so")
    return render_string


  def cache_code(self):
    if self._cached == True:
      return
    
    self._synthesize_prologue()
    self._synthesize_epilogue()

    render_string = ''
    for stream in self._objects:
      render_string = self._cache_code_S(render_string, stream._objects)

    self.render_string = self._prologue + render_string + self._epilogue

    #print self.render_string
    self.render_code = cal_exec.compile(self.render_string)
    self._cached = True
    return


# ------------------------------------------------------------
# Processor
# ------------------------------------------------------------

class LocalMemory(object):
  """
  Handle representing a local GPU memory allocation.
  """

  def __init__(self, fmt, width, height, globl, binding):
    self.fmt = fmt
    self.width = width
    self.height = height
    self.globl = globl
    self.binding = binding
    return


class Processor(spe.Processor):
  exec_module = cal_exec

  def __init__(self, device):
    """Create a new Processor representing a particular GPU in the system, 
       indexed by device."""
    spe.Processor.__init__(self)

    if device < 0 or device > N_GPUS:
      raise Exception("Invalid device number %d" % device)

    self.ctx = cal_exec.alloc_ctx(device)
    self.device = device
    return


  def __del__(self):
    cal_exec.free_ctx(self.ctx)
    return


  # ------------------------------
  # Memory Management
  # ------------------------------

  def _get_fmt(self, typecode, comps):
    # TODO - more format types
    if typecode == 'f':
      if comps == 1:
        fmt = cal_exec.FMT_FLOAT32_1
      elif comps == 2:
        fmt = cal_exec.FMT_FLOAT32_2
      elif comps == 4:
        fmt = cal_exec.FMT_FLOAT32_4
      else:
        raise Exception("Number of components must be 1, 2, or 4")
    elif typecode == 'i':
      if comps == 1:
        fmt = cal_exec.FMT_SIGNED_INT32_1
      elif comps == 2:
        fmt = cal_exec.FMT_SIGNED_INT32_2
      elif comps == 4:
        fmt = cal_exec.FMT_SIGNED_INT32_4
      else:
        raise Exception("Number of components must be 1, 2, or 4")
    elif typecode == 'I':
      if comps == 1:
        fmt = cal_exec.FMT_UNSIGNED_INT32_1
      elif comps == 2:
        fmt = cal_exec.FMT_UNSIGNED_INT32_2
      elif comps == 4:
        fmt = cal_exec.FMT_UNSIGNED_INT32_4
      else:
        raise Exception("Number of components must be 1, 2, or 4")
    else:
      raise Exception("Unsupported data type: " + str(typecode))
    return fmt


  def alloc_local(self, typecode, comps, width, height = 1, globl = False):
    """Allocate local GPU memory and return a handle for copying/binding."""
    fmt = self._get_fmt(typecode, comps)
    
    if globl:
      globl = cal_exec.GLOBAL_BUFFER

    # Allocate GPU memory and create a LocalMemory handle
    binding = cal_exec.alloc_local(self.device, fmt, width, height, globl)
    hdl = LocalMemory(fmt, width, height, globl, binding)
    return hdl


  def alloc_remote(self, typecode, comps, width, height = 1, globl = False):
    """Allocate an ExtArray backed by remote (main) memory."""
    fmt = self._get_fmt(typecode, comps)

    if globl:
      globl = cal_exec.GLOBAL_BUFFER

    # Allocate and initialize the memory
    # TODO - more operand error checking
    mem = cal_exec.alloc_remote(self.device, fmt, width, height, globl)
    arr = extarray.extarray(typecode, 0)

    arr.data_len = mem[2] * height * comps
    arr.set_memory(mem[1], arr.data_len * 4)
    arr.gpu_mem_handle = mem
    arr.gpu_device = self.device
    arr.gpu_width = width
    arr.gpu_pitch = mem[2]
    arr.gpu_height = height
    return arr


  def alloc_remote_npy(self, typecode, comps, width, height = 1, globl = False):
    """Allocate a NumPy ndarray backed by remote (main) memory."""
    if not HAS_NUMPY:
      raise ImportError("NumPy array support requires NumPy installation")

    fmt = self._get_fmt(typecode, comps)
    if typecode == 'f':
      dtype = numpy.float32
    elif typecode == 'i':
      dtype = numpy.int32
    elif typecode == 'I':
      dtype = numpy.uint32
    else:
      raise Exception("Unsupported data type: " + str(typecode))

    if globl:
      globl = cal_exec.GLOBAL_BUFFER

    buf = cal_exec.calmembuffer(self.device, fmt, width, height, globl)
    arr = numpy.frombuffer(buf, dtype=dtype)

    if height == 1:
      arr.shape = (width, comps)
    else:
      arr.shape = (buf.pitch, height, comps)

    return arr


  def free(self, hdl):
    #if not (isinstance(arr, extarray.extarray) and hasattr(arr, "gpu_mem_handle")):
    #  raise Exception("Not a register or extarray with a GPU memory handle")

    if isinstance(hdl, extarray.extarray):
      if not hasattr(hdl, "gpu_mem_handle"):
        raise TypeError("Not an extarray with a GPU memory handle")

      cal_exec.free_remote(hdl.gpu_mem_handle)

      del hdl.gpu_mem_handle
      del hdl.gpu_device
      del hdl.gpu_width
      del hdl.gpu_pitch

      hdl.set_memory(0, 0)
      hdl.data_len = 0
    elif isinstance(hdl, LocalMemory):
      cal_exec.free_local(hdl.binding)
      hdl.res = None
    else:
      raise TypeError("Unknown handle type %s" % (type(hdl)))
    return


  # ------------------------------
  # Kernel Execution
  # ------------------------------

  def copy(self, dst, src, async = False):
    """Copy memory from src to dst, using this GPU."""

    # Figure out what dst and src are and extract bindings
    if isinstance(dst, extarray.extarray):
      if not hasattr(dst, "gpu_mem_handle"):
        raise TypeError("dst is not an extarray with a GPU memory handle")

      dst_binding = dst.gpu_mem_handle
    elif isinstance(dst, numpy.ndarray):
      # NumPy array.. do we support it, and does it use a CAL buffer?
      if not HAS_NUMPY:
        raise ImportError("NumPy array support requires NumPy installation")
      if not isinstance(arr.base, cal_exec.calmembuffer):
        raise TypeError("Not NumPy with a GPU memory buffer")

      dst_binding = [dst.base.res, 0]
    elif isinstance(dst, LocalMemory):
      dst_binding = dst.binding

    if isinstance(src, extarray.extarray):
      if not hasattr(src, "gpu_mem_handle"):
        raise TypeError("src is not an extarray with a GPU memory handle")

      src_binding = src.gpu_mem_handle
    elif isinstance(src, numpy.ndarray):
      # NumPy array.. do we support it, and does it use a CAL buffer?
      if not HAS_NUMPY:
        raise ImportError("NumPy array support requires NumPy installation")
      if not isinstance(arr.base, cal_exec.calmembuffer):
        raise TypeError("Not NumPy with a GPU memory buffer")

      src_binding = [src.base.res, 0]
    elif isinstance(src, LocalMemory):
      src_binding = src.binding

    # Start the copy
    hdl = cal_exec.copy_async(self.ctx, dst_binding, src_binding)

    if async:
      return hdl

    # Not async, complete the copy here.
    cal_exec.join_copy(self.ctx, hdl)
    return


  def execute(self, prgm, domain = None, async = False):
    if not isinstance(prgm, Program):
      raise Exception("ERROR: Can only execute a Program, not %s" % type(prgm))

    prgm.cache_code() 

    if domain is None:
      try:
        arr = prgm.get_binding("o0")
      except KeyError:
        raise Exception("No domain specified and no o0 register bound")

      if isinstance(arr, extarray.extarray):
        domain = (0, 0, arr.gpu_width, arr.gpu_height)
      elif isinstance(arr, numpy.ndarray):
        domain = (0, 0, arr.base.width, arr.base.height)
      elif isinstance(arr, LocalMemory):
        domain = (0, 0, arr.width, arr.height)
      else:
        raise Exception("Invalid o0 binding!")

    if async:
      th = cal_exec.run_stream_async(prgm.render_code,
          self.ctx, domain, prgm._bindings)
      return (th, prgm)
    else:
      cal_exec.run_stream(prgm.render_code, self.ctx, domain, prgm._bindings)

      # Go through the bindings and re-set all the pointers
      #  When a kernel is executed, remote memory has to be unmapped and
      #  remapped, meaning the memory location can change.
      for (key, arr) in prgm._bindings_data.items():
        binding = prgm._bindings[key]
        if isinstance(arr, extarray.extarray):
          arr.set_memory(binding[1], arr.data_len * arr.itemsize)
        elif isinstance(arr, numpy.ndarray) and HAS_NUMPY:
          cal_exec.set_ndarray_ptr(arr, binding[1])
      return


  def join(self, hdl):
    # TODO - do something better to differentiate
    if len(hdl) == 2:
      # Join a kernel execution
      (th, prgm) = hdl
      cal_exec.join_stream(th)

      for arr in prgm._remote_bindings_data.values():
        binding = prgm._bindings[key]
        if isinstance(arr, extarray.extarray):
          arr.set_memory(bindings[1], arr.data_len * arr.itemsize)
        elif isinstance(arr, numpy.ndarray) and HAS_NUMPY:
          cal_exec.set_ndarray_ptr(arr, bindings[1])
    elif len(hdl) == 3:
      cal_exec.join_copy(self.ctx, hdl)
    return


# ------------------------------------------------------------
# Unit tests
# ------------------------------------------------------------

def TestCompileExec():
  import time
  SIZE = 1024
  kernel = ("il_ps_2_0\n" +
            "dcl_input_position_interp(linear_noperspective) v0\n" +
            "dcl_output_generic o0\n" +
            "dcl_output_generic o1\n" +
            #"dcl_output_generic o2\n" +
            "dcl_resource_id(0)_type(2d,unnorm)_fmtx(float)_fmty(float)_fmtz(float)_fmtw(float)\n" +
            #"mov r0, g[0]\n" +
            "sample_resource(0)_sampler(0) o0, v0.xyxx\n" +
            "mov g[0], r0\n" +
            "end\n")

  t1 = time.time()
  image = cal_exec.compile(kernel)
  t2 = time.time()
  print "compile time", t2 - t1

  input = cal_exec.alloc_remote(cal_exec.FMT_FLOAT32_4, SIZE, SIZE, 0)
  output = cal_exec.alloc_remote(cal_exec.FMT_FLOAT32_4, SIZE, SIZE, 0)
  #glob = cal_exec.alloc_remote(cal_exec.FMT_FLOAT32_4, 4096, 4096, cal_exec.GLOBAL_BUFFER)
  print "input", input
  print "output", output

  remote = {"o0": output, "i0": input}
  local = {"o1": (SIZE, SIZE, cal_exec.FMT_FLOAT32_4),
           "g[]": (4096, 4096, cal_exec.FMT_FLOAT32_4)}
  domain = (0, 0, SIZE, SIZE)
  print "remote bindings", remote
  print "local bindings", local

  # image, dev num, (x, y, w, h)
  t1 = time.time()
  cal_exec.run_stream(image, 0, domain, local, remote)
  t2 = time.time()
  print "run time", t2 - t1

  cal_exec.free(input)
  cal_exec.free(output)
  #cal_exec.free(glob)
  cal_exec.free_image(image)
  return


def TestRemoteAlloc():
  mem_handle = cal_exec.alloc_remote(cal_exec.FMT_FLOAT32_4, 1024, 1024, 0)
  print "mem handle", mem_handle
  cal_exec.free(mem_handle)
  return


def TestSimpleKernel():
  import corepy.arch.cal.isa as isa

  SIZE = 128

  proc = Processor(0)

  ext_input = proc.alloc_remote('f', 4, SIZE, SIZE)
  ext_output = proc.alloc_remote('f', 4, SIZE, SIZE)

  for i in xrange(0, SIZE * SIZE * 4):
    ext_input[i] = float(i + 1)
    ext_output[i] = 0.0

  # build and run the kernel
  prgm = Program()
  code = prgm.get_stream()  

  #code.add(isa.dcl_input('v0', USAGE=isa.usage.pos, INTERP='linear_noperspective'))
  code.add("dcl_input_position_interp(constant) v0.xy__")
  code.add(isa.dcl_output('o0', USAGE=isa.usage.generic))
  code.add(isa.dcl_resource(0, '2d', isa.fmt.float, UNNORM=True))
  code.add(isa.sample(0, 0, 'o0', 'v0.xy'))
  #code.add(isa.load(0, 'o0', 'v0.g'))

  domain = (0, 0, SIZE, SIZE)
  prgm.set_binding("o0", ext_output)
  prgm.set_binding("i0", ext_input)

  prgm.add(code)
  prgm.cache_code()
  prgm.print_code()

  proc.execute(prgm, domain)

  # Check the output
  for i in xrange(0, SIZE * SIZE * 4):
    if ext_output[i] != float(i + 1):
      print "ERROR index %d is %f, should be %f" % (i, ext_output[i], float(i + 1))

  proc.free(ext_input)
  proc.free(ext_output)
  return


def TestSimpleKernelNPy():
  import corepy.arch.cal.isa as isa

  SIZE = 128

  proc = Processor(0)

  arr_input = proc.alloc_remote_npy('f', 4, SIZE, SIZE)
  arr_output = proc.alloc_remote_npy('f', 4, SIZE, SIZE)

  #for i in xrange(0, SIZE * SIZE * 4):
  #  arr_input[i] = float(i + 1)
  #  arr_output[i] = 0.0
  #print arr_input.shape
  #print arr_output.shape
  #print type(arr_input.data)

  val = 0.0
  for i in xrange(0, SIZE):
    for j in xrange(0, SIZE):
      for k in xrange(0, 4):
        arr_input[i][j][k] = val
        arr_output[i][j][k] = 0.0
        val += 1.0

  # build and run the kernel
  prgm = Program()
  code = prgm.get_stream()  

  #code.add(isa.dcl_input('v0', USAGE=isa.usage.pos, INTERP='linear_noperspective'))
  code.add("dcl_input_position_interp(constant) v0.xy__")
  code.add(isa.dcl_output('o0', USAGE=isa.usage.generic))
  code.add(isa.dcl_resource(0, '2d', isa.fmt.float, UNNORM=True))
  code.add(isa.sample(0, 0, 'o0', 'v0.xy'))
  #code.add(isa.load(0, 'o0', 'v0.g'))

  domain = (0, 0, SIZE, SIZE)
  prgm.set_binding("o0", arr_output)
  prgm.set_binding("i0", arr_input)

  prgm.add(code)
  prgm.cache_code()
  prgm.print_code()

  proc.execute(prgm, domain)

  # Check the output
  val = 0.0
  for i in xrange(0, SIZE):
    for j in xrange(0, SIZE):
      for k in xrange(0, 4):
        if arr_output[i][j][k] != val:
          print "ERROR index %d is %f, should be %f" % (i, arr_output[i], val)
        val += 1.0

  return


def TestCopy():
  SIZE = 128
  proc = Processor(0)

  arr_inp = proc.alloc_remote('f', 4, SIZE, SIZE)
  arr_out = proc.alloc_remote('f', 4, SIZE, SIZE)
  local1 = proc.alloc_local('f', 4, SIZE, SIZE)
  local2 = proc.alloc_local('f', 4, SIZE, SIZE)

  arr_out.clear()
  for i in xrange(0, SIZE * SIZE * 4):
    arr_inp[i] = float(i)

  bytes = 16 * SIZE * SIZE
  mb = 1024.0 ** 2

  proc.copy(local1, arr_inp)
  proc.copy(local2, local1)
  proc.copy(arr_out, local2)

  for i in xrange(0, SIZE * SIZE * 4):
    if arr_out[i] != float(i):
      print "ERROR arr_out[%d] = %f" % (i, arr_out[i])
    assert(arr_out[i] == float(i))

  return


def TestCopyPerf():
  # If an operational error occurs, reduce the SIZE -- 4096 is 256mb
  SIZE = 4096
  proc1 = Processor(0)

  arr_inp = proc1.alloc_remote('f', 4, SIZE, SIZE)
  arr_out = proc1.alloc_remote('f', 4, SIZE, SIZE)
  local1 = proc1.alloc_local('f', 4, SIZE, SIZE)
  local2 = proc1.alloc_local('f', 4, SIZE, SIZE)

  import time
  
  bytes = 16 * SIZE * SIZE
  mb = 1024.0 ** 2

  proc1.copy(arr_out, arr_inp)
  proc1.copy(arr_out, arr_inp)
  t1 = time.time()
  proc1.copy(arr_out, arr_inp)
  t2 = time.time()

  print "remote->remote copy time", t2 - t1
  print "%f mbytes/sec" % ((float(bytes) / float(t2 - t1)) / mb)

  proc1.copy(local1, arr_inp)
  proc1.copy(local1, arr_inp)
  t1 = time.time()
  proc1.copy(local1, arr_inp)
  t2 = time.time()

  print "remote->local copy time", t2 - t1
  print "%f mbytes/sec" % ((float(bytes) / float(t2 - t1)) / mb)

  proc1.copy(local2, local1)
  proc1.copy(local2, local1)
  t1 = time.time()
  proc1.copy(local2, local1)
  t2 = time.time()

  print "local->local (intra GPU) copy time", t2 - t1
  print "%f mbytes/sec" % ((float(bytes) / float(t2 - t1)) / mb)

  proc1.copy(arr_out, local2)
  proc1.copy(arr_out, local2)
  t1 = time.time()
  proc1.copy(arr_out, local2)
  t2 = time.time()

  print "local->remote copy time", t2 - t1
  print "%f mbytes/sec" % ((float(bytes) / float(t2 - t1)) / mb)

  if N_GPUS >= 2:
    # Can't seem to allocate more memory than is on one GPU?
    proc1.free(local1)

    proc2 = Processor(1)
    local3 = proc2.alloc_local('f', 4, SIZE, SIZE)

    # Performance for cross-gpu copies seems to depend heavily on how many
    # priming copies are done -- sometimes doing more slows the timing down!
    proc1.copy(local3, local2)
    proc1.copy(local3, local2)
    t1 = time.time()
    proc1.copy(local3, local2)
    t2 = time.time()

    print "local->local (cross GPU, src copies) copy time", t2 - t1
    print "%f mbytes/sec" % ((float(bytes) / float(t2 - t1)) / mb)

    proc2.copy(local3, local2)
    proc2.copy(local3, local2)
    t1 = time.time()
    proc2.copy(local3, local2)
    t2 = time.time()

    print "local->local (cross GPU, dst copies) copy time", t2 - t1
    print "%f mbytes/sec" % ((float(bytes) / float(t2 - t1)) / mb)

  print
  return


if __name__ == '__main__':
  print "GPUs available:", N_GPUS
  #TestCompileExec()
  #TestRemoteAlloc()
  TestSimpleKernel()

  if HAS_NUMPY:
    TestSimpleKernelNPy()

  TestCopy()
  TestCopyPerf()

