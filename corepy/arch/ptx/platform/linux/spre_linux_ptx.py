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
SPE for the PTX GPU environment
"""

import corepy.lib.extarray as extarray
import corepy.spre.spe as spe
import corepy.arch.ptx.types.registers as reg
#import corepy.arch.ptx.isa as isa
import ptx_exec

# ------------------------------
# Constants
# ------------------------------

WORD_TYPE = 'f'           # array type that corresponds to 1 word
WORD_SIZE = 4             # size in bytes of one word
WORD_BITS = WORD_SIZE * 8 # number of bits in a word

INT_SIZES = {'b':1,  'c':1, 'h':2, 'i':4, 'B':1,  'H':2, 'I':4}

N_GPUS = ptx_exec.get_num_gpus()
HAS_NUMPY = ptx_exec.HAS_NUMPY

if HAS_NUMPY:
  import numpy


# ------------------------------------------------------------
# InstructionStream
# ------------------------------------------------------------

class InstructionStream(spe.InstructionStream):
  """
  PTX Instruction Stream.  
  """

  # Class attributes
  #default_register_type = reg.ptxRegister
  exec_module   = ptx_exec
  instruction_type  = WORD_TYPE

  def __init__(self, prgm, debug = False):
    spe.InstructionStream.__init__(self, prgm, debug)

  #  #self._cached = False
  #  #self.reset()
    return

  #def __del__(self):
  #  if self._cached:
  #    ptx_exec.free_image(self.render_code)
  #    self.render_string = None
  #  return


  #def reset(self):
  #  spe.InstructionStream.reset(self)
  #  self._bindings = {}
  #  self._bindings_data = {}
  #  #self._declare_registers = {}

  #  if self._cached:
  #    ptx_exec.free_image(self.render_code)
  #    self.render_string = None

  #  return


  #def make_executable(self):
  #  return 


  #def create_register_files(self):
    # Each declarative RegisterFiles entry is:
    #   (file_id, register class, valid values)
    #for reg_type, cls, values in self.RegisterFiles:
    #  regs = [cls(value) for value in values]
    #  self._register_files[cls] = spe.RegisterFile(regs, reg_type)
    #  self._reg_type[reg_type] = cls
  #  self._register_files[reg.TempRegister] = spe.RegisterFile(reg.r, 'r')
  #  self._register_files[reg.LiteralRegister] = spe.RegisterFile(reg.l, 'l')
  #  self._reg_type['r'] = reg.TempRegister
  #  self._reg_type['l'] = reg.LiteralRegister

  
# def acquire_register(self, space, rtype = None, reg = None):
#   return spe.InstructionStream.acquire_register(self, type=rtype, reg=reg)
#      
#  def release_register(self, register):
#    if type(register) != reg.LiteralRegister:
#      self._register_files[type(register)].release_register(register)
#    # print 'release', str(self._register_files[type])
#    return 

  
  def add(self, obj):
    if type(obj) == str:
      self.objects.append(obj)
    else:
      spe.InstructionStream.add(self, obj)


class Program(spe.Program):
  default_register_type = reg.ptxRegister_b32
  instruction_type  = WORD_TYPE

  stream_type = InstructionStream

  def __init__(self):
    spe.Program.__init__(self, None)

    # Array of literal register declarations
    # Added to by acquire_register(), rendered by synthesize_prologue()
    self._literals = []
    self.param_types = []
    self.params = []
    return


  def reset(self):
    spe.Program.reset(self)
    #self._bindings = {}
    #self._bindings_data = {}
    #self._declare_registers = {}
    self._declarations = []
    self._variable_count = 0

    if self._cached:
      #ptx_exec.free_image(self.render_code)
      ptx_exec.unload_module(self.render_code)
      self.render_string = None

    return

  def __del__(self):
    self.reset()


  def make_executable(self):
    return 


  def create_register_files(self):
    # Each declarative RegisterFiles entry is:
    #   (file_id, register class, valid values)
    #for reg_type, cls, values in self.RegisterFiles:
    #  regs = [cls(value) for value in values]
    #  self._register_files[cls] = spe.RegisterFile(regs, reg_type)
    #  self._reg_type[reg_type] = cls

    # TODO: CHANGE THIS BACK!
    #num_regs = reg.num_regs
    num_regs = 128

    rtypes = ('b8', 'b16', 'b32', 'b64', 'u8', 'u16', 'u32', 'u64', 's8', 's16', 's32', 's64', 'f16', 'f32', 'f64', 'pred')
    self._register_avail_bins = [[reg.__dict__['r' + str(i) + '_' + rtype] for rtype in rtypes] for i in xrange(num_regs)]

    self._reg_map = {reg.ptxRegister_b8: 0, reg.ptxRegister_b16: 1, reg.ptxRegister_b32: 2, reg.ptxRegister_b64: 3,
                     reg.ptxRegister_u8: 4, reg.ptxRegister_u16: 5, reg.ptxRegister_u32: 6, reg.ptxRegister_u64: 7,
                     reg.ptxRegister_s8: 8, reg.ptxRegister_s16: 9, reg.ptxRegister_s32: 10, reg.ptxRegister_s64: 11, 
                     reg.ptxRegister_f16: 12, reg.ptxRegister_f32: 13, reg.ptxRegister_f64: 14, reg.ptxRegister_pred: 15} 

    rclasses = (reg.ptxRegister_b8, reg.ptxRegister_b16, reg.ptxRegister_b32, reg.ptxRegister_b64,
                reg.ptxRegister_u8, reg.ptxRegister_u16, reg.ptxRegister_u32, reg.ptxRegister_u64,
                reg.ptxRegister_s8, reg.ptxRegister_s16, reg.ptxRegister_s32, reg.ptxRegister_s64, 
                reg.ptxRegister_f16, reg.ptxRegister_f32, reg.ptxRegister_f64, reg.ptxRegister_pred)
    
    for (reg_type, reg_class) in zip(rtypes, rclasses):
      self._reg_type[reg_type] = reg_class
      
    return

  
  def acquire_register(self, reg_type = None, reg_name = None):
    #print "Acquiring register of type ", reg_type
    r = spe.Program.acquire_register(self, reg_type = reg_type, reg_name = reg_name)
    stmt = '\t.' + str(r.space) + '.' + str(r.type) + ' ' + r.name + ';'
    if not stmt in self._declarations:
      self._declarations.append(stmt)
    return r

  def release_register(self, register):
    spe.Program.release_register(self, register)
    #self._register_files[type(register)].release_register(register)
    # print 'release', str(self._register_files[type])
    return 

  def add_variable(self, var_space = None, var_type = None, var_name = None):
    if var_space == 'reg':
      return self.acquire_register(var_type, var_name)
    else:
      if var_name == None:
        var_name = 'v' + str(self._variable_count)
        self._variable_count += 1
      var = reg.ptxVariable(var_space, var_type, var_name)
      self._declarations.append('\t.' + str(var.space) + '.' + str(var.type) + ' ' + var.name + ';')
  
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
      if not isinstance(hdl.base, ptx_exec.ptxmembuffer):
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
    # TODO: Generate proper version and target
    self._prologue =   '\t.version 1.4\n'
    #self._prologue += '\t.target sm_10, map_f64_to_f32\n'
    self._prologue += '\t.target sm_13\n'
    self._prologue += '\t.entry _main (\n'

    for i, param in enumerate(self.params):
      self._prologue += '\t.param ' + '.' + self.param_types[i] + ' ' + param
      if i < len(self.params) - 1:
        self._prologue += ',\n'
      else:
        self._prologue += ')\n'

    self._prologue += '\t{\n'

    self._prologue += '\n'.join(self._declarations) + '\n'

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
    self._epilogue = '\texit;\n\t}\n\n'
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
      elif isinstance(obj, spe.Label):
        render_string += obj.name + ':\n'
    return render_string


  def cache_code(self):
    if self._cached == True:
      return
    
    self._synthesize_prologue()
    self._synthesize_epilogue()

    render_string = ''
    for stream in self.objects:
      render_string = self._cache_code_S(render_string, stream.objects)

    self.render_string = self._prologue + render_string + self._epilogue

    #print self.render_string
    self.render_code = ptx_exec.compile(self.render_string)
    self._cached = True
    return


  ###########################
  # PTX-specific methods
  ###########################

  def add_parameter(self, ptype, name = ""):
    if name == "":
      name = "p" + str(len(self.params) + 1)

    # TODO: Parameter type checking
    self.param_types.append(ptype)
    self.params.append(name)
    return reg.ptxVariable('param', ptype, name)

# ------------------------------------------------------------
# Processor
# ------------------------------------------------------------

class DeviceMemory(object):
  """
  Handle representing a local GPU memory allocation.
  """

  def __init__(self, address, fmt, length):
    self.address = address
    self.fmt = fmt
    self.itemsize = int(fmt[1:]) / 8
    self.length = length
    self.size = length * self.itemsize
    return


class Processor(spe.Processor):
  exec_module = ptx_exec

  def __init__(self, device=0):
    """Create a new Processor representing a particular GPU in the system, 
       indexed by device."""
    spe.Processor.__init__(self)

    if device < 0 or device > N_GPUS:
      raise Exception("Invalid device number %d" % device)

    print "Creating ctx"
    self.ctx = ptx_exec.alloc_ctx(device)
    self.device = device
    return


  def __del__(self):
    print "Destroying ctx"
    ptx_exec.free_ctx(self.ctx)
    return


  # ------------------------------
  # Memory Management
  # ------------------------------

  # def _get_fmt(self, typecode, comps = 1):
#     if typecode == 'f':
#       if comps == 1:
#         fmt = ptx_exec.FMT_FLOAT32_1
#       elif comps == 2:
#         fmt = ptx_exec.FMT_FLOAT32_2
#       elif comps == 4:
#         fmt = ptx_exec.FMT_FLOAT32_4
#       else:
#         raise Exception("Number of components must be 1, 2, or 4")
#     elif typecode == 'i':
#       if comps == 1:
#         fmt = ptx_exec.FMT_SIGNED_INT32_1
#       elif comps == 2:
#         fmt = ptx_exec.FMT_SIGNED_INT32_2
#       elif comps == 4:
#         fmt = ptx_exec.FMT_SIGNED_INT32_4
#       else:
#         raise Exception("Number of components must be 1, 2, or 4")
#     elif typecode == 'I':
#       if comps == 1:
#         fmt = ptx_exec.FMT_UNSIGNED_INT32_1
#       elif comps == 2:
#         fmt = ptx_exec.FMT_UNSIGNED_INT32_2
#       elif comps == 4:
#         fmt = ptx_exec.FMT_UNSIGNED_INT32_4
#       else:
#         raise Exception("Number of components must be 1, 2, or 4")
#     else:
#       raise Exception("Unsupported data type: " + str(typecode))
#     return fmt
  

  def alloc_device(self, typecode, length, comps = 1):
    """
    Allocate local GPU memory and return a handle for copying/binding.

    Typecode is ptx typecode (u32, s32, f32, u64, etc.)
    """
    #fmt = self._get_fmt(typecode, comps)

    scalar_byte_width = int(typecode[1:])/8
        
    # Allocate GPU memory and create a DeviceMemory handle
    address = ptx_exec.alloc_device(length*scalar_byte_width*comps)

    return DeviceMemory(address, typecode, length)

  def alloc_host(self, typecode, length, comps = 1):
    """
    Allocate local GPU memory and return a handle for copying/binding.

    Typecode is ptx typecode (u32, s32, f32, u64, etc.)
    """
    #fmt = self._get_fmt(typecode, comps)


    array_typecode = ''
    
    # This might be clearer, but not very efficient...
    #type_conversion_table = {}
    #type_conversion_table['32'] = {'f': 'f', 'u': 'I', 's', 'i'}
    #type_conversion_table['64'] = {'f': 'd', 'u': 'L', 's', 'l'}
    #type_conversion_table['16'] = {'u': 'H', 's', 'h'}
    #type_conversion_table['8'] = {'u': 'B', 's', 'b'}
    #
    #if typecode == 'b':
    #  typecode = 'u'
    #array_typecode = type_conversion_table[typecode[0]][typecode[1:]]
    
    scalar_width = int(typecode[1:])
    if typecode[0] == 'f':
      if scalar_width == 32:
        array_typecode = 'f'
      elif scalar_width == 64:
        array_typecode = 'd'
    elif typecode[0] == 'u':
      if scalar_width == 32:
        array_typecode = 'I'
      elif scalar_width == 64:
        array_typecode = 'L'
      elif scalar_width == 16:
        array_typecode = 'H'
      elif scalar_width == 8:
        array_typecode = 'b'
    elif typecode[0] == 's':
      if scalar_width == 32:
        array_typecode = 'i'
      elif scalar_width == 64:
        array_typecode = 'l'
      elif scalar_width == 16:
        array_typecode = 'h'
      elif scalar_width == 8:
        array_typecode = 'B'

    if array_typecode == '':
      raise Exception('Unable to convert type')
          
    mem = ptx_exec.alloc_host(length*scalar_byte_width*comps)
    
    arr = extarray.extarray(array_typecode, 0)
    arr.data_len = scalar_width/4 * length * comps
    arr.set_memory(mem, arr.data_len * 4)
    arr.gpu_mem_handle = mem
#    arr.gpu_device = self.device
    arr.gpu_width = length
#     arr.gpu_pitch = mem[2]
#     arr.gpu_height = height
    return arr

#   def alloc_remote(self, typecode, comps, width, height = 1, globl = False):
#     """Allocate an ExtArray backed by remote (main) memory."""
#     fmt = self._get_fmt(typecode, comps)

#     if globl:
#       globl = ptx_exec.GLOBAL_BUFFER

#     # Allocate and initialize the memory
#     # TODO - more operand error checking
#     mem = ptx_exec.alloc_remote(self.device, fmt, width, height, globl)


#   def alloc_remote_npy(self, typecode, comps, width, height = 1, globl = False):
#     """Allocate a NumPy ndarray backed by remote (main) memory."""
#     if not HAS_NUMPY:
#       raise ImportError("NumPy array support requires NumPy installation")

#     fmt = self._get_fmt(typecode, comps)
#     if typecode == 'f':
#       dtype = numpy.float32
#     elif typecode == 'i':
#       dtype = numpy.int32
#     elif typecode == 'I':
#       dtype = numpy.uint32
#     else:
#       raise Exception("Unsupported data type: " + str(typecode))

#     if globl:
#       globl = ptx_exec.GLOBAL_BUFFER

#     buf = ptx_exec.calmembuffer(self.device, fmt, width, height, globl)
#     arr = numpy.frombuffer(buf, dtype=dtype)

#     if height == 1:
#       arr.shape = (width, comps)
#     else:
#       arr.shape = (buf.pitch, height, comps)

#     return arr


  def free_device(self, hdl):
    ptx_exec.free_device(hdl.address)

  def free_host(self, arr):
    ptx_exec.free_host(arr.buffer_info()[0])

  def free(self, hdl):
    #if not (isinstance(arr, extarray.extarray) and hasattr(arr, "gpu_mem_handle")):
    #  raise Exception("Not a register or extarray with a GPU memory handle")

    if isinstance(hdl, extarray.extarray):
      if not hasattr(hdl, "gpu_mem_handle"):
        raise TypeError("Not an extarray with a GPU memory handle")

      ptx_exec.free_remote(hdl.gpu_mem_handle)

      del hdl.gpu_mem_handle
      del hdl.gpu_device
      del hdl.gpu_width
      del hdl.gpu_pitch

      hdl.set_memory(0, 0)
      hdl.data_len = 0
    elif isinstance(hdl, LocalMemory):
      ptx_exec.free_local(hdl.binding)
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
      ptx_exec.copy_dtoh(dst.buffer_info()[0], src.address, src.length*src.itemsize)
    elif isinstance(dst, DeviceMemory):
      ptx_exec.copy_htod(dst.address, src.buffer_info()[0], src.buffer_info()[1]*src.itemsize)
    #elif isinstance(dst, numpy.ndarray):
    #  # NumPy array.. do we support it, and does it use a CAL buffer?
    #  if not HAS_NUMPY:
    #    raise ImportError("NumPy array support requires NumPy installation")
    #  if not isinstance(arr.base, ptx_exec.calmembuffer):
    #    raise TypeError("Not NumPy with a GPU memory buffer")

    ## Start the copy
    #hdl = ptx_exec.copy_async(self.ctx, dst_binding, src_binding)
    #
    #if async:
    #  return hdl
    #
    ## Not async, complete the copy here.
    #ptx_exec.join_copy(self.ctx, hdl)

    return
    

  def execute(self, prgm, threads = None, params = (), async = False):
    if not isinstance(prgm, Program):
      raise Exception("ERROR: Can only execute a Program, not %s" % type(prgm))

    prgm.cache_code() 

    if threads is None:
      threads = (1, 1, 1, 1, 1)
    else:
      if len(threads) == 2:
        threads = (threads[0], 1, 1, threads[1], 1)

      if len(threads) != 5:
        raise Exception("ERROR: Invalid thread block specification")


    # TODO: Support async
    if async:
      print "Asynchronous execution is not yet supported"
      return
    
      th = ptx_exec.run_stream_async(prgm.render_code,
          self.ctx, domain, prgm._bindings)
      return (th, prgm)
    else:

      # set up parameters here
      num_params = len(prgm.params)
      if num_params == 0:
        if params != () and params != []:
          raise Exception("Invalid parameters")
      elif num_params != len(params):
        raise Exception("Invalid parameters")
      else:
        # TODO: some sort of type checking? this old method ends up not being relevant...
        #for i, param in enumerate(params):  
        #  pt = prgrm.param_types[i]
        #  #  if !isinstance(param, prgm.params[i]):
        #  #    raise Exception("Invalid parameter type at parameter " + str(i))
        pass

      param_list = list(params)

      # Replace DeviceMemory parameters with their actual address
      # TODO - any other swaps than need to be done?
      for i in xrange(0, len(param_list)):
        if isinstance(param_list[i], DeviceMemory):
            param_list[i] = param_list[i].address

      type_num_tuple = tuple(map(ptx_exec.__dict__.__getitem__, prgm.param_types))
      ptx_exec.run_stream(prgm.render_code, threads, type_num_tuple, param_list)
      # ptx_exec.run_stream(prgm.render_code, self.ctx, threads, tuple(prgm.param_types), param_list)

      return
    
  def join(self, hdl):
    # TODO - do something better to differentiate
    if len(hdl) == 2:
      # Join a kernel execution
      (th, prgm) = hdl
      ptx_exec.join_stream(th)

      for arr in prgm._remote_bindings_data.values():
        binding = prgm._bindings[key]
        if isinstance(arr, extarray.extarray):
          arr.set_memory(bindings[1], arr.data_len * arr.itemsize)
        elif isinstance(arr, numpy.ndarray) and HAS_NUMPY:
          ptx_exec.set_ndarray_ptr(arr, bindings[1])
    elif len(hdl) == 3:
      ptx_exec.join_copy(self.ctx, hdl)
    return


# ------------------------------------------------------------
# Unit tests
# ------------------------------------------------------------

def TestCompileExec():
  import time
  #SIZE = 1024
  kernel = ("\t.version 1.4\n" +
            "\t.target sm_10, map_f64_to_f32\n" +
            "\t.entry _main () {\n" +
            "\t\tret;\n" + 
            "\t\texit;\n" + 
            "\t}\n" +
            "\n")
  print kernel

  #ctx = ptx_exec.alloc_ctx(0)
  t1 = time.time()
  module = ptx_exec.compile(kernel)
  t2 = time.time()
  print "compile time", t2 - t1

  #input = ptx_exec.alloc_remote(ptx_exec.FMT_FLOAT32_4, SIZE, SIZE, 0)
  #output = ptx_exec.alloc_remote(ptx_exec.FMT_FLOAT32_4, SIZE, SIZE, 0)
  ##glob = ptx_exec.alloc_remote(ptx_exec.FMT_FLOAT32_4, 4096, 4096, ptx_exec.GLOBAL_BUFFER)
  #print "input", input
  #print "output", output

  #remote = {"o0": output, "i0": input}
  #local = {"o1": (SIZE, SIZE, ptx_exec.FMT_FLOAT32_4),
  #         "g[]": (4096, 4096, ptx_exec.FMT_FLOAT32_4)}
  #domain = (0, 0, SIZE, SIZE)
  #print "remote bindings", remote
  #print "local bindings", local

  print "Executing..."
  # image, dev num, (x, y, w, h)
  t1 = time.time()
  ptx_exec.run_stream(module, (1, 1, 1, 1, 1), (), [])
  t2 = time.time()
  print "run time", t2 - t1

  #ptx_exec.free_ctx(ctx)
  
  return

def TestParams():
  import time
  #SIZE = 1024
  kernel = (
  '''
  .version 1.4
  .target sm_10, map_f64_to_f32
  .entry _main (
  .param .u64 __cudaparm__Z16addArrayOnDevicePfff_c,
  .param .f32 __cudaparm__Z16addArrayOnDevicePfff_a,
  .param .f32 __cudaparm__Z16addArrayOnDevicePfff_b)
  {
  .reg .u64 %rd<3>;
  .reg .f32 %f<6>;
  ld.param.f32    %f1, [__cudaparm__Z16addArrayOnDevicePfff_a];
  ld.param.f32    %f2, [__cudaparm__Z16addArrayOnDevicePfff_b];
  add.f32 %f3, %f1, %f2;
  
  mov.f32         %f4, %f3;
  ld.param.u64    %rd1, [__cudaparm__Z16addArrayOnDevicePfff_c];
  st.global.f32   [%rd1+0], %f4;

  exit;
  } // _Z16addArrayOnDevicePfff
  '''
  )

  t1 = time.time()
  module = ptx_exec.compile(kernel)
  t2 = time.time()
  print "compile time", t2 - t1

  a = 1.0
  b = 2.0

  ptx_mem_addr = ptx_exec.alloc_device(4)
  mem = extarray.extarray('f', 1)
  #mem.set_memory(ptx_mem_addr, 4)
  mem[0] = 5.0

  print ptx_mem_addr, type(ptx_mem_addr)
  print mem.buffer_info()[0], type(mem.buffer_info()[0])
  param_list = [ptx_mem_addr, a, b]
  # image, dev num, (x, y, w, h)

  ptx_exec.copy_htod(ptx_mem_addr, mem.buffer_info()[0], 4)
  t1 = time.time()
  ptx_exec.run_stream(module, (1, 1, 1, 1, 1), (ptx_exec.u64, ptx_exec.f32, ptx_exec.f32), param_list)
  t2 = time.time()
  print "run time", t2 - t1
  print "X", mem.buffer_info()[0], ptx_mem_addr
  ptx_exec.copy_dtoh(mem.buffer_info()[0], ptx_mem_addr, 4)

  print param_list
  print mem

  #ptx_exec.free(input)
  #ptx_exec.free(output)
  ##ptx_exec.free(glob)
  #ptx_exec.unload_module(image)
  return

def TestRemoteAlloc():
  mem_handle = ptx_exec.alloc_remote(ptx_exec.FMT_FLOAT32_4, 1024, 1024, 0)
  print "mem handle", mem_handle
  ptx_exec.free(mem_handle)
  return


def TestSimpleKernel():
  import corepy.arch.ptx.isa as isa
  import corepy.arch.ptx.types.registers as regs
  import time

  SIZE = 128

  proc = Processor(0)

  # build and run the kernel
  prgm = Program()
  code = prgm.get_stream()  

  _mem = prgm.add_parameter('u64', name='_mem')
  _a = prgm.add_parameter('f32', name='_a')
  _b = prgm.add_parameter('f32', name='_b')

#  rd1 = regs.ptxVariable('reg', 'u64', 'rd1')
#  r1 = regs.ptxVariable('reg', 'f32', 'f1')
#  r2 = regs.ptxVariable('reg', 'f32', 'f2')
#  r3 = regs.ptxVariable('reg', 'f32', 'f3')
#  r4 = regs.ptxVariable('reg', 'f32', 'f4')
#  code.add('  .reg .u64 rd1;')
#  code.add('  .reg .f32 f1;')
#  code.add('  .reg .f32 f2;')
#  code.add('  .reg .f32 f3;')
#  code.add('  .reg .f32 f4;')

  rd1 = prgm.acquire_register('u64')
  r1 = prgm.acquire_register('f32')
  r2 = prgm.acquire_register('f32')
  r3 = prgm.acquire_register('f32')
  r4 = prgm.acquire_register('f32')    
  v1 = prgm.add_variable('shared', 'f32') # don't need this, but let's test add_variable

#  import pdb
#  pdb.set_trace()
  #code.add(isa.add(r3, r2, r1))
  #code.add('add.f32 r3, r2, r1;')
  code.add(isa.ld('param', r1, regs.ptxAddress(_a)))
  code.add(isa.ld('param', r2, regs.ptxAddress(_b)))
  code.add(isa.add(r3, r2, r1))
  code.add(isa.add(r3, r3, 1.0))
  code.add(isa.mov(r4, r3))
  #temp = prgm.acquire_register('u32')
  #code.add(isa.cvt(temp, regs.tid.x))
  #code.add(isa.cvt(r4, temp, rnd='rn'))
  temp1 = prgm.acquire_register('u32')
  temp2 = prgm.acquire_register('u32')
  temp3 = prgm.acquire_register('u32')
  code.add(isa.mul(temp2, temp1, temp3, hlw='lo'))
  
  code.add(isa.ld('param', rd1, regs.ptxAddress(_mem)))
  code.add(isa.st('global', regs.ptxAddress(rd1), r4))
  prgm.add(code)

  prgm.cache_code()
#   prgm.render_string = (
#   '''
#   .version 1.4
#   .target sm_10, map_f64_to_f32
#   .entry _main (
#   .param .u64 __cudaparm__Z16addArrayOnDevicePfff_c,
#   .param .f32 __cudaparm__Z16addArrayOnDevicePfff_a,
#   .param .f32 __cudaparm__Z16addArrayOnDevicePfff_b)
#   {
#   .reg .u64 %rd<3>;
#   .reg .f32 %f<6>;
#   ld.param.f32    %f1, [__cudaparm__Z16addArrayOnDevicePfff_a];
#   ld.param.f32    %f2, [__cudaparm__Z16addArrayOnDevicePfff_b];
#   add.f32 %f3, %f1, %f2;
  
#   mov.f32         %f4, %f3;
#   ld.param.u64    %rd1, [__cudaparm__Z16addArrayOnDevicePfff_c];
#   st.global.f32   [%rd1+0], %f4;

#   exit;
#   } // _Z16addArrayOnDevicePfff
#   '''
#   )
#   prgm.render_code = ptx_exec.compile(prgm.render_string)

  ####
  #ptx_mem_addr = proc.alloc_device('f32', 1)
  ptx_mem_addr = ptx_exec.alloc_device(4)
  mem = extarray.extarray('f', 1)
  mem[0] = 5.0

  a = 1.0
  b = 2.0
  
  print mem.buffer_info()[0]
  param_list = [ptx_mem_addr, a, b]
  print map(type, param_list)
  #   # image, dev num, (x, y, w, h)

  #import pdb

  ptx_exec.copy_htod(ptx_mem_addr, mem.buffer_info()[0], 4)
  #kernel = prgm.render_string
  #module = ptx_exec.compile(kernel)
  t1 = time.time()
  #ptx_exec.run_stream(module, (1, 1, 1, 1, 1), (ptx_exec.u64, ptx_exec.f32, ptx_exec.f32), param_list)
  proc.execute(prgm, (1,1,1,1,1), param_list)
  t2 = time.time()
#  pdb.set_trace()
  print "run time", t2 - t1

  print "YY", mem.buffer_info()[0], ptx_mem_addr, type(mem.buffer_info()[0]), type(ptx_mem_addr)
  print int(ptx_mem_addr)
  print int(mem.buffer_info()[0])
  ptx_exec.copy_dtoh(mem.buffer_info()[0], ptx_mem_addr, 4)

  print param_list
  print mem
  ####

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

def TestParamsFull():
  import time
  import corepy.arch.ptx.isa as isa
  import corepy.arch.ptx.types.registers as regs

  proc = Processor(0)

  # build and run the kernel
  prgm = Program()
  code = prgm.get_stream()  

  _mem = prgm.add_parameter('u64', name='_mem')
  _a = prgm.add_parameter('f32', name='_a')
  _b = prgm.add_parameter('f32', name='_b')

  rd1 = prgm.acquire_register('u64')
  r1 = prgm.acquire_register('f32')
  r2 = prgm.acquire_register('f32')
  r3 = prgm.acquire_register('f32')
  r4 = prgm.acquire_register('f32')    
  v1 = prgm.add_variable('shared', 'f32') # don't need this, but let's test add_variable

  code.add(isa.ld('param', r1, regs.ptxAddress(_a)))
  code.add(isa.ld('param', r2, regs.ptxAddress(_b)))
  code.add(isa.add(r3, r2, r1))
  code.add(isa.add(r3, r3, 1.0))
  code.add(isa.mov(r4, r3))
  code.add(isa.ld('param', rd1, regs.ptxAddress(_mem)))
  code.add(isa.st('global', regs.ptxAddress(rd1), r4))
  prgm.add(code)

  prgm.cache_code()

  a = 1.0
  b = 2.0

  ptx_mem_addr = proc.alloc_device('f32', 1)
  mem = extarray.extarray('f', 1)
  mem[0] = 5.0

  param_list = [ptx_mem_addr.address, a, b]

  proc.copy(ptx_mem_addr, mem)
  prgm.cache_code()
  for i in range(20):
    t1 = time.time()
    proc.execute(prgm, (1, 1, 1, 1, 1), param_list)
    t2 = time.time()
    print "run time", t2 - t1
    print "#####"
  print "X", mem.buffer_info()[0], ptx_mem_addr.address
  proc.copy(mem, ptx_mem_addr)

  print param_list
  print mem

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
  # Definitely working ptx tests
#  TestCompileExec()
#  TestParams()
  #for i in range(100):
  #TestSimpleKernel()
  
  #for i in range(100):
  TestParamsFull()
  #  print "#####"
  #TestRemoteAlloc()

  #if HAS_NUMPY:
  #  TestSimpleKernelNPy()

  #TestCopy()
  #TestCopyPerf()

