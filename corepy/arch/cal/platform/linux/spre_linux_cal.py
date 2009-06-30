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

# ------------------------------------------------------------
# InstructionStream
# ------------------------------------------------------------

class InstructionStream(spe.InstructionStream):
  """
  CAL Instruction Stream.  
  """

  # Class attributes
  default_register_type = reg.TempRegister
  exec_module   = cal_exec
  instruction_type  = WORD_TYPE

  def __init__(self):
    spe.InstructionStream.__init__(self)

    #self._cached = False
    self.reset()
    return

  def __del__(self):
    if self._cached:
      cal_exec.free_image(self.render_code)
      self.render_string = None
    return


  def reset(self):
    spe.InstructionStream.reset(self)
    self._remote_bindings = {}
    self._remote_bindings_data = {}
    self._copy_bindings = {}
    self._copy_bindings_data = {}
    self._local_bindings = {}
    self._declare_registers = {}

    if self._cached:
      cal_exec.free_image(self.render_code)
      self.render_string = None

    return


  def make_executable(self):
    return 


  def create_register_files(self):
    # Each declarative RegisterFiles entry is:
    #   (file_id, register class, valid values)
    #for reg_type, cls, values in self.RegisterFiles:
    #  regs = [cls(value) for value in values]
    #  self._register_files[cls] = spe.RegisterFile(regs, reg_type)
    #  self._reg_type[reg_type] = cls
    self._register_files[reg.TempRegister] = spe.RegisterFile(reg.r, 'r')
    self._register_files[reg.LiteralRegister] = spe.RegisterFile(reg.l, 'l')
    self._reg_type['r'] = reg.TempRegister
    self._reg_type['l'] = reg.LiteralRegister
    return

  
  def acquire_register(self, type = None, reg = None):
    if isinstance(type, (list, tuple)):
      # if this is a LiteralRegister, acquire and set the value
      l = spe.InstructionStream.acquire_register(self, type='l', reg=reg)
      self.add(isa.dcl_literal(l, type[0], type[1], type[2], type[3]))
      return l
    else:
      return spe.InstructionStream.acquire_register(self, type=type, reg=reg)
      
  def release_register(self, register):
    if type(register) != reg.LiteralRegister:
      self._register_files[type(register)].release_register(register)
    # print 'release', str(self._register_files[type])
    return 

  
  # ------------------------------
  # Execute/ABI support
  # ------------------------------

  def _synthesize_prologue(self):
    self._prologue = 'il_ps_3_0\n'

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


  def cache_code(self):
    if self._cached == True:
      return
    render_string = ''
    
    self._synthesize_prologue()
    self._synthesize_epilogue()

    print "PROLOGUE", self._prologue
    
    for inst in self._instructions:
      if type(inst) == str:
        if inst[-1] != '\n':
          render_string += inst + '\n'
        else:
          render_string += inst
      else:
        render_string += inst.render() + '\n'
    self.render_string = self._prologue + render_string + self._epilogue

    print self.render_string
    self.render_code = cal_exec.compile(self.render_string)
    self._cached = True
    return

  def add(self, inst):
    if type(inst) == str:
      self._instructions.append(inst)
    else:
      spe.InstructionStream.add(self, inst)


  # ------------------------------
  # GPU memory binding management
  # ------------------------------

  def set_remote_binding(self, regname, arr, copy_local = False):
    if isinstance(arr, extarray.extarray) and hasattr(arr, "gpu_mem_handle"):
      binding = arr.gpu_mem_handle
    else:
      try:
        import numpy
      except ImportError:
        raise ImportError("alloc_remote_npy() requires NumPy")

      if not isinstance(arr, numpy.ndarray) and isinstance(arr.base, cal_exec.CALMemBuffer):
        raise Exception("Not NumPy with a GPU memory buffer")

      buf = arr.base
      binding = [buf.pointer, buf.pitch, buf.height, buf.format, buf.res]

    if isinstance(regname, (reg.CALRegister, reg.CALBuffer)):
      regname = regname.name

    if copy_local:
      self._copy_bindings_data[regname] = arr
      self._copy_bindings[regname] = binding
    else:
      self._remote_bindings_data[regname] = arr
      self._remote_bindings[regname] = binding
    #self._cached = False
    return

  def get_remote_binding(self, regname):
    return self._remote_bindings_data[regname]

  def set_local_binding(self, regname, rec):
    #local = {"o1": (SIZE, SIZE, cal_exec.FMT_FLOAT32_4),
    # TODO - better checking
    if not isinstance(rec, (tuple, list)) and len(rec) != 3:
      raise Exception("Invalid local binding record")

    if isinstance(regname, (reg.CALRegister, reg.CALBuffer)):
      regname = regname.name

    self._local_bindings[regname] = rec

  def get_local_binding(self, regname):
    return self._local_bindings[regname]

  def declare_register(regname, width, height, fmt, **kwargs):
    # Declare a register for the user in the prologue.
    # TODO - some error checking on the args here?
    self._declare_registers[regname] = (width, height, fmt, kwargs)
    return


class Processor(spe.Processor):
  exec_module = cal_exec

  def __init__(self, device):
    spe.Processor.__init__(self)

    if device < 0 or device > N_GPUS:
      raise Exception("Invalid device number %d" % device)

    self.device = device
    return


  def execute(self, code, domain = None, async = False):
    code.cache_code() 

    if domain is None:
      try:
        input = code.get_remote_binding("i0")
      except KeyError:
        raise Exception("No domain specified and no remote i0 register bound")

      domain = (0, 0, input.gpu_width, len(input) / input.gpu_width)

    if async:
      th = cal_exec.run_stream_async(code.render_code,
          self.device, domain, code._local_bindings, code._remote_bindings, code._copy_bindings)
      return (th, code)
    else:
      cal_exec.run_stream(code.render_code,
          self.device, domain, code._local_bindings, code._remote_bindings, code._copy_bindings)

      try:
        import numpy

        for (key, arr) in code._remote_bindings_data.items():
          if isinstance(arr, extarray.extarray):
            arr.set_memory(arr.gpu_mem_handle[0], arr.data_len * arr.itemsize)
          elif isinstance(arr, numpy.ndarray):
            cal_exec.set_ndarray_ptr(arr, code._remote_bindings[key][0])

        for (key, arr) in code._copy_bindings_data.items():
          if isinstance(arr, extarray.extarray):
            arr.set_memory(arr.gpu_mem_handle[0], arr.data_len * arr.itemsize)
          elif isinstance(arr, numpy.ndarray):
            cal_exec.set_ndarray_ptr(arr, code._remote_bindings[key][0])

      except ImportError:
        for arr in code._remote_bindings_data.values():
          arr.set_memory(arr.gpu_mem_handle[0], arr.data_len * arr.itemsize)
        for arr in code._copy_bindings_data.values():
          arr.set_memory(arr.gpu_mem_handle[0], arr.data_len * arr.itemsize)
      return


  def join(self, id):
    (th, code) = id
    cal_exec.join_stream(th)

    for arr in code._remote_bindings_data.values():
      arr.set_memory(arr.gpu_mem_handle[0], arr.data_len * arr.itemsize)
    return


  # TODO - need to deal with format types
  def alloc_remote(self, typecode, comps, width, height = 1, globl = False):
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

    if globl:
      globl = cal_exec.GLOBAL_BUFFER

    # Allocate and initialize the memory
    # TODO - more operand error checking
    mem = cal_exec.alloc_remote(self.device, fmt, width, height, globl)
    arr = extarray.extarray(typecode, 0)

    arr.data_len = mem[1] * height * comps
    arr.set_memory(mem[0], arr.data_len * 4)
    arr.gpu_mem_handle = mem
    arr.gpu_device = self.device
    arr.gpu_width = width
    arr.gpu_pitch = mem[1]
    return arr


  def alloc_remote_npy(self, typecode, comps, width, height = 1, globl = False):
    try:
      import numpy
    except ImportError:
      raise ImportError("alloc_remote_npy() requires NumPy")

    if typecode == 'f':
      if comps == 1:
        fmt = cal_exec.FMT_FLOAT32_1
      elif comps == 2:
        fmt = cal_exec.FMT_FLOAT32_2
      elif comps == 4:
        fmt = cal_exec.FMT_FLOAT32_4
      else:
        raise Exception("Number of components must be 1, 2, or 4")
      dtype = numpy.float32
    elif typecode == 'i':
      if comps == 1:
        fmt = cal_exec.FMT_SIGNED_INT32_1
      elif comps == 2:
        fmt = cal_exec.FMT_SIGNED_INT32_2
      elif comps == 4:
        fmt = cal_exec.FMT_SIGNED_INT32_4
      else:
        raise Exception("Number of components must be 1, 2, or 4")
      dtype = numpy.int32
    elif typecode == 'I':
      if comps == 1:
        fmt = cal_exec.FMT_UNSIGNED_INT32_1
      elif comps == 2:
        fmt = cal_exec.FMT_UNSIGNED_INT32_2
      elif comps == 4:
        fmt = cal_exec.FMT_UNSIGNED_INT32_4
      else:
        raise Exception("Number of components must be 1, 2, or 4")
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
  


  def free_remote(self, arr):
    if not (isinstance(arr, extarray.extarray) and hasattr(arr, "gpu_mem_handle")):
      raise Exception("Not a register or extarray with a GPU memory handle")

    cal_exec.free_remote(arr.gpu_mem_handle)

    del arr.gpu_mem_handle
    del arr.gpu_device
    del arr.gpu_width
    del arr.gpu_pitch

    arr.set_memory(0, 0)
    arr.data_len = 0
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

  cal_exec.free_remote(input)
  cal_exec.free_remote(output)
  #cal_exec.free_remote(glob)
  cal_exec.free_image(image)
  return


def TestRemoteAlloc():
  mem_handle = cal_exec.alloc_remote(cal_exec.FMT_FLOAT32_4, 1024, 1024, 0)
  print "mem handle", mem_handle
  cal_exec.free_remote(mem_handle)
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
  code = InstructionStream()  
  #code.add(isa.dcl_input('v0', USAGE=isa.usage.pos, INTERP='linear_noperspective'))
  code.add("dcl_input_position_interp(constant) v0.xy__")
  code.add(isa.dcl_output('o0', USAGE=isa.usage.generic))
  code.add(isa.dcl_resource(0, '2d', isa.fmt.float, UNNORM=True))
  code.add(isa.sample(0, 0, 'o0', 'v0.xy'))
  #code.add(isa.load(0, 'o0', 'v0.g'))
  code.cache_code()
  print code.render_string

  domain = (0, 0, SIZE, SIZE)
  code.set_remote_binding("o0", ext_output)
  code.set_remote_binding("i0", ext_input)

  proc.execute(code, domain)

  # Check the output
  for i in xrange(0, SIZE * SIZE * 4):
    if ext_output[i] != float(i + 1):
      print "ERROR index %d is %f, should be %f" % (i, ext_output[i], float(i + 1))


  proc.free_remote(ext_input)
  proc.free_remote(ext_output)
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
  print arr_input.shape
  print arr_output.shape
  print type(arr_input.data)

  val = 0.0
  for i in xrange(0, SIZE):
    for j in xrange(0, SIZE):
      for k in xrange(0, 4):
        arr_input[i][j][k] = val
        arr_output[i][j][k] = 0.0
        val += 1.0

  # build and run the kernel
  code = InstructionStream()  
  #code.add(isa.dcl_input('v0', USAGE=isa.usage.pos, INTERP='linear_noperspective'))
  code.add("dcl_input_position_interp(constant) v0.xy__")
  code.add(isa.dcl_output('o0', USAGE=isa.usage.generic))
  code.add(isa.dcl_resource(0, '2d', isa.fmt.float, UNNORM=True))
  code.add(isa.sample(0, 0, 'o0', 'v0.xy'))
  #code.add(isa.load(0, 'o0', 'v0.g'))
  code.cache_code()
  print code.render_string

  domain = (0, 0, SIZE, SIZE)
  code.set_remote_binding("o0", arr_output)
  code.set_remote_binding("i0", arr_input)

  proc.execute(code, domain)

  # Check the output
  val = 0.0
  for i in xrange(0, SIZE):
    for j in xrange(0, SIZE):
      for k in xrange(0, 4):
        if arr_output[i][j][k] != val:
          print "ERROR index %d is %f, should be %f" % (i, arr_output[i], val)
        val += 1.0

  return


if __name__ == '__main__':
  print "GPUs available:", N_GPUS
  #TestCompileExec()
  #TestRemoteAlloc()
  TestSimpleKernel()

  try:
    import numpy
    TestSimpleKernelNPy()
  except ImportError: pass

