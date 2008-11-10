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

__doc__ = """
Base classes for the Synthetic Programming Environment.
"""


import inspect

import corepy.lib.extarray as extarray

from syn_util import *
#import syn_util as util

__annoy__ = True

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------


def _extract_stack_info(stack):
  """
  Given a stack, save the static info for each frame (basically
  everything except for the reference to the frame).
  """

  stack_info = []
  for frame in stack: stack_info.append(frame[1:])
  return stack_info

def _first_user_frame(stack_info):
  """
  Find the first frame that's not from InstructionStream.

  TODO: This is somewhat of a hack at this point.  Abstract it so the
        corepy_conf.py has a regexp or something that the user can use
        to determine what files should be used for printing code.
  """
  import os.path

  idx = 0
  for i, frame in enumerate(stack_info):
    file = os.path.split(frame[0])[1]
    if (file != 'spe.py' and
        file[:5] != 'spre_' and
        file not in ('util.py', 'spu_extended.py', 'iterators.py', '__init__.py') and
        #file not in ('util.py', 'spu_extended.py', '__init__.py', 'dma.py') and
        file[-9:] != '_types.py'):
      idx = i
      break
    
  return stack_info[idx], file


# ------------------------------------------------------------
# Register Management
# ------------------------------------------------------------

class RegisterFile(object):
  """
  Manage a set of registers.
  """

  def __init__(self, registers, name = '<unknown>'):
    """
    Create a register file with registers identified by the values in
    registers.  registers is a sequence.
    """
    object.__init__(self)
    
    self._registers = registers[:]
    self._pool = None
    self._used = None
    self._name = name
    self.reset()
    return

  def get_used(self): return self._used.keys()
  
  def reset(self):
    self._pool = self._registers[:]
    self._used = {} # register: None
    return

  def acquire_register(self, reg = None):
    """
    Supply the user with a register or raise an exception if none are
    available.  If reg is set, return that register if it is available,
    otherwise raise an exception.

    Add the register to the used list.
    """

    # See if there are registers available
    if(len(self._pool) == 0):
      raise Exception('Out of registers!')

    # Get a register and mark that it's been used
    if reg is not None:
      if reg in self._pool:
        reg = self._pool[reg]
        del self._pool[self._pool.index(reg)]
      else:
        raise Exception('Register ' + str(reg) + ' is not available!')
    else:
      reg = self._pool.pop()

    self._used[reg] = None

    return reg

  def release_register(self, reg):
    """
    Return a register to the list of available registers.  If the
    register is already in the available list, a warning is printed
    (rather than an exception).
    """
    if reg in self._pool:
      # print 'Warning: release_register:', reg, 'already exists!'
      raise Exception('Warning: release_register from %s: %s already exists!' % (self._name, str(reg)))
    else:
      self._pool.append(reg)
    return 

  def __str__(self):
    s = '[%2d/%2d] ' % (len(self._pool), len(self._registers))
    for reg in self._registers[:-1]:
      if reg in self._pool:
        s += '%s  ' % str(reg)
      else:
        s += '%s* ' % str(reg)
    return s
  

class Register(object):
  def __init__(self, reg, code = None, name = None, prefix = ''):
    """
    Create a new register:
      reg is the architecture dependent value for the register, usually an int.
      code is the InstructionStream that created and owns this register.
    """
    if isinstance(code, str):
      raise Exception("Use the 'name' keyword argument to set the name of a register")
    
    self.reg = reg
    self.code = code
    self.name = name
    self.prefix = prefix
    return

  def __str__(self): 
    if self.name != None:
      return self.name
    else:
      return '$%d' % (self.reg)
      #return '%s%d' % (self.prefix, self.reg)

  def __eq__(self, other):
    if isinstance(other, Register):
      return other.reg == self.reg
    elif isinstance(other, int):
      return self.reg == other
    elif isinstance(other, str):
      return self.name == other
    else:
      raise Exception('Cannot compare Register against %s' % (type(other),))

class Immediate(object):
  def __init__(self, value, size_type, type_type = None):
    self.value = value
    self.size_type = size_type
    self.type_type = type_type
    self.reg_type = None
    return

  def __str__(self): return 'i%s' % str(self.value)

class Literal(Immediate): pass

class Type(object):
  def _cast(cls, other):
    newinst = None
    if isinstance(other, Expression):
      newinst = cls.expr_cls(other._inst, *other._operands, **other._koperands)
      print 'Casting to:', cls.expr_cls
      
    elif isinstance(other, Variable):
      newinst = cls(reg = other.reg, code = other.code)
      newinst.value = other.value

    return newinst

  cast = classmethod(_cast)


# ------------------------------------------------------------
# Variable
# ------------------------------------------------------------

class Variable(object):
  """
  Variable, along with Expression, forms the basis of user-defined
  type semantics in CorePy. Variable and Expression are based on the
  Interpreter design pattern.

  Variables abstract a register and are extended by
  library/user-defined types to support type-specific operations, such
  as support for operators and storage sharing between Python and
  synthetic programs. 

  The base variable class is fairly simple and supports simple
  register management and assignment through the var.v attribute.

  Becuase Python does not provide any mechanism to overload assignment
  directly, Variables have a special attribute, 'v', that can be used
  to assign the result of an expression to the register abstracted by
  the variable.

  Subclasses must implement the following methods:

    copy_register(self, other) - copy the value from other into the
      local register
    _set_literal_value(self, value) - store a literal value in the
      register 
      
  """

  def __init__(self, value = None, code = None, reg = None):
    """
    Set up the variable and initialize the register with value, if
    present.

    Most of the logic in the constructor is identifies the appropriate
    code instance to use and sets up the register.
    """
    super(Variable, self).__init__()

    if code is None and self.active_code is not None:
      code = self.active_code
    
    if reg is not None and not isinstance(reg, (Register, Variable)):
      raise Exception('reg must be a Register')
    if code is not None and not isinstance(code, InstructionStream):
      raise Exception('code must be an InstructionStream')
    
    if reg is None and code is None:
      raise Exception('Variables must be created with a register (reg) and/or code')
    elif reg is not None and code is not None and reg.code is not code:
      raise Exception('Registers must be from the same InstructionStream as the supplied code object')

    self.value = value
    self.acquired_register = False
    if reg is not None:
      if isinstance(reg, Variable):
        self.reg = reg.reg
      else:
        self.reg = reg
    else:
      self.reg = code.acquire_register(self.register_type_id)
      self.acquired_register = True
      
    if code is None:
      self.code = reg.code
    else:
      self.code = code

    # self.v triggers the assignement mechanism. 
    if value is not None:
      self.v = self.value

    self.assigned = False
    self.expression = None
    return

  def release_register(self, force = False):
    if self.reg is not None and (self.acquired_register or force):
      self.code.release_register(self.reg)
      self.reg = None
    else:
      raise Exception('Attempt to release register acquired from elsewhere.  Use force = True keyword to release from here.')
    return

  def __str__(self):
    # return '<%s reg = %s>' % (type(self), str(self.reg))
    return '%s' % str(self.reg) # for ASM formatting
    return '<%s>' % str(self.reg)

  # Assignment property
  def get_value(self): return self.value
  def _set_value(self, v): self.set_value(v)
  # def _set_literal_value(self, v): raise Exception('No method to set literal values for %s' % (type(self)))
  v = property(get_value, _set_value)

  def set_value(self, value):
    """
    Assignment method.  This method is called when a value is assigned
    to the .v property.
    """
    if isinstance(value, Variable):
      self.copy_register(value)
    elif isinstance(value, Expression):
      value.eval(self.code, reg = self.reg)
      if isinstance(value, Expression):
        value.release_registers(self.code)
    elif isinstance(value, self.literal_types):
      self._set_literal_value(value)
    else:
      raise Exception('Cannot set %s to %s' % (type(self), type(value)) )

    self.assigned = True
    self.expression = value

    return


# ------------------------------------------------------------
# Expression
# ------------------------------------------------------------

class Expression(object):
  """
  Expression manages delayed evaluation and work in conjunction with
  variables to support user-defined type semantics.

  An expression contains an instruction and its arguments - less the
  destination register - and calls the instruction with the arguments
  with the eval() method is invoked.

  The destination register is either provided as a value to eval or
  acquired by eval.
  """

  def __init__(self, inst, *operands, **koperands):
    """
    Collect the arguments for later use.
    """
    super(Expression, self).__init__()
    
    self._inst = inst
    self._operands = operands
    self._koperands = koperands

    self._acquired_register = None
    return

  def release_registers(self, code):
    """
    Release any acquired registers for this object and its children.
    """

    if self._acquired_register is not None:
      code.release_register(self._acquired_register)
      self._acquired_register = None

    for op in self._operands:
      if issubclass(type(op), Expression):
        op.release_registers(code)

    return

  def eval(self, code, reg = None):
    """
    Evaluate the instruction, using reg as the destination or
    acquiring a new register if reg is None.
    """
    target = reg

    if reg is None:
      target = code.acquire_register(self.register_type_id)
      self._acquired_register = target
      
    eval_ops = [target]
    for op in self._operands:
      if issubclass(type(op), Expression):
        eval_ops.append(op.eval(code))
      else:
        eval_ops.append(op)

    code.add(self._inst(*eval_ops, **self._koperands))

    return target

  def set_value(self, value): raise Exception("Cannot assign value (.v) to Expression")
  


# ------------------------------------------------------------
# Instruction
# ------------------------------------------------------------

def _expression_method(cls, *operands, **koperands):
  """
  Class 'factory' method for creating an expression for an Instruction
  class.  The type of the expression is determined by the type_cls
  keyword parameter, an optional type_cls attribute on the class
  itself, or if neither are present, the type defaults to Expression.
  """
  if koperands.has_key('type_cls'):
    expr_cls = koperands['type_cls'].expr_cls
    del koperands['type_cls']
    return expr_cls(cls, *operands, **koperands)
  elif hasattr(cls, 'type_cls'):
    raise Exception('TODO: Make sure this works correctly (comment out this exception and try again)')
    return cls.type_cls.expr_cls(cls, *operands, **koperands)
  else:
    return Expression(cls, *operands, **koperands)
  

class InstructionOperand(object):
  """
  Used for machine instructions.  
  """
  def __init__(self, name, default = None):
    self.name = name
    self.default = default
    return

  def check(self, value):
    if __annoy__:
      print 'You should really implement a check method for %s' % (self.name)
    return True

  def render(self, value):
    raise Exception('You should really implement a render method for %s' % (self.name))

class MachineInstruction(object):
  """
  Machine instruction class for general types of machine instructions.

  Most instructions are based on a few general classes of physical
  instructions.  MachineInstructions represent these physical
  instruction classes.

  A MachineInstruction is composed of two main elements: an operand
  type signature and a render method.  
  """
  
  signature = ()   # Operand signature
  opt_kw = ()      # Optional keyword arguments (e.g., on ppc.add RC, OE)

  def _render(params, operands):
    raise Exception("render() method not implemented")
  render = staticmethod(_render)
  

class Instruction(object):
  def __init__(self, *operands, **koperands):
    # Remember what the user passed in for debugging
    # TODO - not just for debugging! remove the underscore, too
    self._supplied_operands = operands
    self._supplied_koperands = koperands    

    self._operands = self.validate_operands(*operands, **koperands)

    # If active code is present, add ourself to it and remember that
    # we did so.  active_code_used is checked by InstructionStream
    # to avoid double adds from code.add(inst(...)) when active_code
    # is set.
    self.active_code_used = None    

    # Allow the user to create an instruction without adding it to
    # active code.  Used for debugging purposes.
    ignore_active = False
    if koperands.has_key('ignore_active'):
      ignore_active = koperands['ignore_active']
      del koperands['ignore_active']

    if self.active_code is not None and not ignore_active:
      self.active_code.add(self)
      self.active_code_used = self.active_code
    
    return

  def validate_operands(self, *operands, **koperands):
    ops = {} # name: value
    if len(operands) != len(self.machine_inst.signature):
      if len(operands) < len(self.machine_inst.signature):
        reason = 'Not enough '
      else:
        reason = 'Too many '
      raise Exception(reason + 'arguments supplied to instruction %s(%s).  Found arguments: (%s)' % (
        self.__class__.__name__,
        ', '.join([op_type.name for op_type in self.machine_inst.signature]),
        ', '.join([str(value) for value in operands])))

    iop = 0
    for op_type, value in zip(self.machine_inst.signature, operands):
      if op_type.check(value):
        # Store ops by name and position.
        ops[op_type.name] = value
        ops[iop] = value        
        iop += 1
      else:
        raise Exception("Operand validation failed for " + op_type.name )
        
    for op_type in self.machine_inst.opt_kw:
      kw = op_type.name
      if koperands.has_key(kw):
        if op_type.check(koperands[kw]):
          ops[kw] = koperands[kw]
      elif op_type.default is not None:
        ops[kw] = op_type.default

    return ops

  
  def __str__(self):
    operands = []
    for op in self._supplied_operands:
      operands.append(str(op))

    return '%s %s' % (self.__class__.__name__, ', '.join([str(op) for op in operands]))
    #return '%s(%s)' % (self.__class__.__name__, ', '.join([str(op) for op in operands]))


  # Expression method
  ex = classmethod(_expression_method)
  
  def render(self):
    bin = self.machine_inst.render(self.params, self._operands)
    return bin

  def set_position(self, pos):
    """Set the byte-offset position of this instruction in its
       InstructionStream"""
    self._operands["position"] = pos


def _sig_cmp(sig, user):
  if len(sig) != len(user):
    return False

  for s, u in zip(sig, user):
    # Test s against u, since user will be more specific and use __eq__()
    if not u == s:
      return False
  return True


class DispatchInstruction(Instruction):
  """
  DispatchInstruction is a one-to-many mapping of Instructions 
  to MachineInstructions.  When constructed, a DispatchInstruction
  matches the type signature of its arguments against the type
  signatures of the different MachineInstructions in the dispatch
  table.  If a match is found, the MachineInstruction is used as the
  instance's instruction attribute.*  If a match is not found, an
  execption is raised. 

  DisptachInstruction is intended for use with x86 instruction sets
  where there is a one-to-many mapping between instructions and their
  implmentations. 

  *Note that in Instruction, the instruction attribute is a static
  attribute.  DispatchInstruction abuses things a bit and creates an
  instance attribute for instruction.
  """
  type_id = [type]
  dispatch = ()

  def __init__(self, *operands, **koperands):
    # Get a list of the types of the operands
    type_func = self.type_id[0]
    op_types = [type_func(arg) for arg in operands]

    # Find the method that matches the operand type list
    instruction = None
    params = None
    for entry in self.dispatch:
#      print "[check] (%s) -> (%s)" % (
#        ','.join([str(arg_type.name) for arg_type in op_types],),
#        ','.join([str(arg_type.name) for arg_type in entry[0].signature],)
#        )

      if _sig_cmp(entry[0].signature, op_types):
        instruction = entry[0]
        params = entry[1]
        break

    if instruction is None:
      print op_types
      raise Exception("No instruction method found for operand types (%s)" % (
        ','.join([str(arg_type.name) for arg_type in op_types],)))
#     else:
#       print "(%s) -> (%s)" % (
#         ','.join([str(arg_type.name) for arg_type in op_types],),
#         ','.join([str(arg_type.name) for arg_type in instruction.signature],)
#         )
        

    # Set our instruction type
    self.machine_inst = instruction
    self.params = params
    
    # Call the main Instruction constructor
    Instruction.__init__(self, *operands, **koperands)
    return


class ExtendedInstruction(object):
  """
  An ExtendedInstruction is a simple synthetic component that 
  provides Instruction-compatible interfaces to operarations that
  require more than one instruction.

  For instance, on SPU, right-shift is constructed from a two
  instructions: 

    spu.sfi(temp, b, 0)
    spu.rotm(d, temp, a)

  ExtendedInstruction provides an Instruction-compatible interface and
  semantics for implementing operations that use more than one
  instruction.

  Subclasses implement the block() method to provide the instruction
  sequence for the ExtendedInstruction.  The calling environment
  (typically InstructionStream) sets the active code to the correct
  InstructionStream instance for the current evaluation of the
  instruction.  This allows the developer to call instructions
  directly, without using code.add(...).

  The complete block() method for the SPU right-shift operation is:

  def block(self):

    temp = self.get_active_code().acquire_register()
    spu.sfi(temp, b, 0)
    spu.rotm(d, temp, a)
    self.get_active_code().release_register(temp)

    return

  Subclasses of ExtendedInstruction must provide:
    isa_module [class attribute] - module that contains the active_code 
    block() [method] - the method that generates the instructions.
  """

  def __init__(self, *operands, **koperands):
    self.active_code_used = None
    active_code = self.get_active_code()

    self._operands = operands
    self._koperands = koperands

    if active_code is not None:
      active_code.add(self)
      self.active_code_used = active_code

    return

  def get_active_code(self):
    return self.isa_module.get_active_code()

  def set_active_code(self, code):
    return self.isa_module.set_active_code(code)

  ex = classmethod(_expression_method)

  def render(self):
    self.block(*self._operands, **self._koperands)
    return

  def block(self, *operands, **koperands): pass

# ------------------------------------------------------------
# InstructionStream
# ------------------------------------------------------------

class Label(object):
  def __init__(self, code, name):
    object.__init__(self)
    self.code = code
    self.name = name
    self.position = None
    return

  def __str__(self):
    return "Label(%s)" % (self.name)
    #return "Label(%s, %s)" % (self.name, self.position)
    #return "Label(%s, %s)" % (str(self.code), self.name)

  def set_position(self, pos):
    """Set the byte-offset position of this label in its
       InstructionStream.  Provided so that cache_code can simple call the
       set_position() without checking whether the object is an instruction or
       a label."""
    self.position = pos


class InstructionStream(object):
  """
  InstructionStream mantains ABI compliance and code cache
  """

  # Class parameters.  Sublasses must provide values for these.

  # Register file to pull a register from if no type is provided by the user
  default_register_type = None

  # Native size of the instruction or instruction components. This
  # will be 'I' for 32-bit instructions and 'B' for variable length 
  # instructions.
  instruction_type  = None
  
  def __init__(self):
    object.__init__(self)

    # Make sure subclasses provide property values
    if self.default_register_type is None:
      raise Exception("Subclasses must set default_register_type")
    if self.instruction_type is None:
      raise Exception("Subclasses must set instruction_type")
    
    # Major code sections
    self._prologue = []
    self._epilogue = []
    self._instructions = []
    self._labels = {}

    # Some default labels
    self.lbl_prologue = self.get_label("PROLOGUE")
    self.lbl_body = self.get_label("BODY")
    self.lbl_epilogue = self.get_label("EPILOGUE")

    # Debugging information
    self._stack_info = None
    self._debug = False
    
    # Register Files
    # Use RegisterFiles to create a set of register instances for this
    # instance of InstructionStream.
    self._register_files = {} # type: RegisterFile
    self._reg_type = {} # 'string': type, e.g. 'gp': GPRegister
    self.create_register_files()
    
    # Storage holds references to objects created by synthesizers
    # that may otherwise get garbage collected
    self._storage = None

    self._active_callback = None
    self.reset()
    return

  def make_executable(self):
    raise Exception("Required method not implemented by %s" % str(self.__class__))

  def create_register_files(self):
    raise Exception("Required method not implemented by %s" % str(self.__class__))  

  def set_debug(self, debug):
    self._debug = debug

  def get_debug(self):
    return self._debug

  debug = property(get_debug, set_debug)

 
  # TODO - AWF - what calls this, what does it do? 
  def set_active_callback(self, cb):
    self._active_callback = cb
  
  def __del__(self):
    print 'Destroying', self
    return
  
  def __setitem__(self, key, inst):
    """
    Allow the user to replace instructions by index.
    """
    self._instructions[key] = inst
    self.render_code = None
    self._cached = False
    return

  def __getitem__(self, v):
    return self._instructions[v]

  def __iter__(self):
    return self._instructions.__iter__()

  def get_inst(self, idx):
    """
    Return the instruction at idx.
    """
    return self._instructions[idx]

  def has_label(self, name):
    return self._labels.has_key(name)

  def get_label(self, name):
    if self._labels.has_key(name):
        return self._labels[name]
    lbl = Label(self, name)
    self._labels[name] = lbl
    return lbl
  
  def add_storage(self, s):
    """
    Add a reference to an object used during execution of the
    instruction stream.  This cache keeps locally allocated objects
    from being garbage collected once the stream is built.
    """
    self._storage.append(s)
    return

  def remove_storage(self, s):
    """
    Add a reference to an object used during execution of the
    instruction stream.  This cache keeps locally allocated objects
    from being garbage collected once the stream is built.
    """
    if s in self._storage:
      self._storage.remove(s)
    return

  def reset_storage(self):
    """
    Free the references to cached storage.
    """
    self._storage = []
    return

  def reset_code(self):
    """
    Clear the instruction stream.  This has the side effect of
    invalidating the code cache.
    """
    self._instructions = [self.lbl_body]
    self._labels = {}
    self._stack_info = []
    self.reset_cache()
    return

  def reset_cache(self):
    self._cached = False
    self.render_code = None
    return
    
  def reset(self):
    """
    Reset the instruction stream and storage and return all the
    registers to the register pools.
    """
    self.reset_code()
    self.reset_storage()

    for file in self._register_files.values():
      file.reset()
      
    self.reset_cache()
    return

  # ------------------------------
  # User register management
  # ------------------------------

  def acquire_register(self, type = None, reg = None):

    if type is None:
      type = self.default_register_type    
    elif isinstance(type, str):
      type = self._reg_type[type]

    reg = self._register_files[type].acquire_register(reg)

    # print 'acquire', str(self._register_files[type])
    return reg
    

  def release_register(self, reg):
    self._register_files[type(reg)].release_register(reg)
    # print 'release', str(self._register_files[type])
    return 
  

  def acquire_registers(self, n, type = None, reg = None):
    return [self.acquire_register(type, reg) for i in range(n)]

  def release_registers(self, regs):
    for reg in regs:
      self.release_register(reg)
    return
    
  # ------------------------------
  # Instruction management
  # ------------------------------

  def add(self, inst):
    """
    Add an instruction and return the current size of the instruction
    sequence.  The index of the current instruction is (size - 1).
    (TODO: this is confusing and probably should return the index of
           the just added instruction...but too much code depends on
           these sematnics right now...)
    """

    if isinstance(inst, Instruction):
      # Check to see if this instruction has already been added to
      # this InstructionStream via active code.  This is to prevent
      # double adds from:
      #   code.add(inst(a, b, c))
      # when active code is set.
      if inst.active_code_used is not self:
        self._instructions.append(inst)

        if self._debug:
          self._stack_info.append(_extract_stack_info(inspect.stack()))
        
    elif isinstance(inst, ExtendedInstruction):
      if inst.active_code_used is not self:
        old_active = inst.get_active_code()
        if old_active is not self:
          inst.set_active_code(self)
        # AWF - 'render' here on the extended inst is a misnomer.
        # What happens is that the extended inst's block method is called,
        # which then adds instructions to the active instruction stream.
        inst.render()
        if old_active is not self:
          inst.set_active_code(old_active)
    elif isinstance(inst, Label):
      self._instructions.append(inst)
    else:
      raise Exception('Unsupported instruction format: %s.  Instruction or int is required.' % type(inst))

    # Invalidate the cache
    self._cached = False
    self.render_code = None
    return len(self._instructions)

  def size(self): return len(self._instructions)
  def __len__(self): return self.size()

  # ------------------------------
  # Execute/ABI support
  # ------------------------------

  def _synthesize_prologue(self):
    raise Exception("Required method not implemented by %s" % str(self.__class__))

  def _synthesize_epilogue(self):
    raise Exception("Required method not implemented by %s" % str(self.__class__))


  def inst_addr(self):
    if self._cached == True:
      return self.render_code.buffer_info()[0]
    else:
      return None


  def _adjust_pass(self, inst_list):
    # Do adjustment passes until the rendered code stops changing
    change = True
    while change == True:
      change = False
      inst_len = 0
  
      for rec in inst_list:
        # Once a change occurs, lbl/inst positions need to be updated
        if change == True:
          rec[2].set_position(inst_len)

        if rec[0] == True:
          # Re-render the instruction, it has a label ref
          r = rec[2].render()
          if r != rec[1]:
            change = True
            rec[1] = r

        inst_len += len(rec[1])

    return inst_list


  def cache_code(self):
    """
    Fill in the epilogue and prologue.  This call freezes the code and
    any subsequent calls to acquire_register() or add() will unfreeze
    it.  Also perform alignment checks.  Once the checks are
    preformed, the code should not be modified.
    """

    if self._cached == True:
      return
    
    # HACK: Disable the current active code
    # NOTE: This may not work in the presence of multiple ISAs...
    active_callback = None
    if self._active_callback is not None:
      active_callback = self._active_callback
      active_callback(None)


    self._synthesize_prologue()
    self._synthesize_epilogue()

    render_code = extarray.extarray(self.instruction_type)

    # Note - TRAC ticket #19 has some background info and reference links on
    # the algorithms used here. https://svn.osl.iu.edu/trac/corepy/ticket/19

    if self.instruction_type == 'I':
      fwd_ref_list = []

      # Assumed below that 'I' type is 4 bytes
      for arr in (self._prologue, self._instructions, self._epilogue):
        for val in arr:
          if isinstance(val, (Instruction, ExtendedInstruction)):
            # Does this instruction reference any labels?
            lbl = None
            for k in val._operands.keys():
              if isinstance(val._operands[k], Label):
                lbl = val._operands[k]
                break

            if lbl == None: # No label reference, render the inst
              render_code.append(val.render())
            else: # Label reference
              assert(lbl.code == self)
              val.set_position(len(render_code) * 4)

              if lbl.position != None:  # Back reference, render the inst
                render_code.append(val.render())
              else: # Fill in a dummy instruction and save info to render later
                fwd_ref_list.append((val, len(render_code)))
                render_code.append(0xFFFFFFFF)
          elif isinstance(val, Label): # Label, fill in a zero-length slot
            val.set_position(len(render_code) * 4)

      # Render the instructions with forward label references
      for rec in fwd_ref_list:
        render_code[rec[1]] = rec[0].render()

    elif self.instruction_type == 'B':
      # inst_list is a list of tuples.  Each tuple contains a bool
      # indicating presence of a label reference, rendered code ([] if label),
      # and a label or instruction object.
      inst_list = []
      inst_len = 0

      for arr in (self._prologue, self._instructions, self._epilogue):
        for val in arr:
          if isinstance(val, (Instruction, ExtendedInstruction)):
            # Does this instruction reference any labels?
            lbl = None
            relref = False
            #iop = 0
            #for k in val._operands.keys():
            sig = val.machine_inst.signature
            #while val._operands.has_key(iop):
            for iop in xrange(0, len(sig)):
              opsig = sig[iop]
              #if isinstance(op, (int, long)):
              #  print "ops", val._operands
              #  print "op", op, iop, val.params, val.machine_inst.signature
              #  print "opsig", opsig
              if hasattr(opsig, "relative_op") and opsig.relative_op == True:
                op = val._operands[iop]
                if isinstance(op, Label):
                  lbl = op
                # This is a hack, but it works.  Some instructions can have
                # a relative offset that is not a label.  These insts need to be
                # re-rendered if instruction sizes change
                relref = True
              #iop += 1

            if lbl == None: # No label references
              val.set_position(inst_len)
              r = val.render()
              inst_list.append([relref, r, val])
              inst_len += len(r)
            else: # Instruction referencing a label.
              assert(lbl.code == self)
              val.set_position(inst_len)

              if lbl.position != None: # Back-reference, render the instruction
                r = val.render()
                inst_list.append([True, r, val])
                inst_len += len(r)
              else: # Fill in a dummy instruction, assuming 2-byte best case
                inst_list.append([True, [-1, -1], val])
                inst_len += 2
          elif isinstance(val, Label): # Label, fill in a zero-length slot
            val.set_position(inst_len)
            inst_list.append([False, [], val])

      inst_list = self._adjust_pass(inst_list)

      # Final loop, bring everything together into render_code
      for rec in inst_list:
        if isinstance(rec[2], (Instruction, ExtendedInstruction)):
          render_code.fromlist(rec[1])

    self.render_code = render_code
    self.make_executable()

    if active_callback is not None:
      active_callback(self)

    self._cached = True
    return


  # ------------------------------
  # Debugging
  # ------------------------------

  # Utility function to print an array of instructions, used by print_code()
  def _print_instructions(self, instrs, binary, hexad):
    offset = 0
    for inst in instrs:
      print '%4d %s' % (offset, str(inst))
      #print "%s" % (str(inst))
      if isinstance(inst, (Instruction, ExtendedInstruction)):
        render = inst.render()
        if self.instruction_type == 'I':
          offset += 4
          if binary or hexad:
            bin = DecToBin(render)
            hex = '%08x' % (render)
        else: #self.instruction_type == 'B'
          offset += len(render)
          if binary or hexad:
            bin = ''
            hex = ''
            for byte in render:
              bin += DecToBin(byte)[24:32]
              hex += '%02x' % (byte)

        if binary == True:
          print bin
        if hexad == True:
          print hex

  def print_code(self, pro = False, epi = False, binary = False, hex = False):
    """
    Print the user instruction stream.
    """

    if self._cached == False:
      self.cache_code()
    print 'code info:', self.render_code.buffer_info()[0], len(self.render_code)


    #if self._cached == True:
    #  print 'code info:', self.render_code.buffer_info()[0], len(self.render_code)
    #else:
    #  print 'code info: not cached/rendered'
    
    if pro:
      self._print_instructions(self._prologue, binary, hex)

    print 

    if self._debug:
      addr= self._code.buffer_info()[0]
      last = [None, None]
      for inst, dec, stack_info, i in zip(self._instructions, self._code, self._stack_info,
                                          range(0, self._code.buffer_info()[1])):
        user_frame, file = _first_user_frame(stack_info)

        # if file == 'spu_types.py':
        #  for frame in stack_info:
        #    print frame
            
        if last == [user_frame, file]:
          ssource = '  ""  ""'
        else:
          sdetails = '[%s:%s: %d]' % (file, user_frame[2], user_frame[1])
          sdetails += ' ' * (35 - len(sdetails))
          ssource = '%s %s' % (sdetails, user_frame[3][0][:-1]) # .strip())

        pipeline = ''
        if hasattr(inst, 'cycles'):
          if inst.cycles[0] == 0:
            pipeline = 'X ;'
          else:
            pipeline = ' X;'
          pipeline += '  %2d' % inst.cycles[1]
            
        saddr   = '0x%08X' % (addr + i * 4)
        sinst   = '%4d; %s' % (i, str(inst))
        sinst += ' ' * (40 - len(sinst))
        last = [user_frame, file]
        print saddr,';', pipeline, ';',  sinst, ';',  ssource
        if binary:
          print DecToBin(dec)
    else:
      self._print_instructions(self._instructions, binary, hex)

    print 

    if epi:
      self._print_instructions(self._epilogue, binary, hex)
    return


class Processor(object):
  """
  The processor class handles execution of InstructionStreams.

  The execute method takes an InstructionStream and an optional
  execution mode.  The 'void', 'fp', and 'int' modes execute the
  stream synchronously and return None, the value in fp_return, or the
  value in gp_return, respectively.  'async' mode executes the stream
  in a new thread and returns the thread id immediately.  
  """

  # Execution mode (return code type) constants
  MODE_VOID = 0
  MODE_INT = 1
  MODE_FP = 2

  def __init__(self):  object.__init__(self)
  
  def execute(self, code, mode = 'int', async = False, params = None, debug = False):
    """
    Execute the instruction stream in the code object.

    Execution modes are:

      'int'  - return the intetger value in register gp_return when
               execution is complete
      'fp'   - return the floating point value in register fp_return
               when execution is complete
      'void' - return None

    If async is True, a thread id and mode tuple is returned immediately
    and the code is executed asynchronously in its own thread.  The execution
    mode then controls what kind of value is returned from the join method.

    If debug is True, the buffer address and code length are printed
    to stdout before execution.
    """

    if len(code._instructions) == 0:
      return None

    if not code._cached:
      code.cache_code()

    if debug:
      print 'code info: 0x%x %d' % (
        code.inst_addr(),
        len(code.render_code))
      code.print_code(hex = True, pro = True, epi = True)
     
    addr = code.inst_addr()

    if params is None:
      params = self.exec_module.ExecParams()
    elif type(params) is not self.exec_module.ExecParams:
      # Backwards compatibility for list-style params
      _params = self.exec_module.ExecParams()
      _params.p1, _params.p2, _params.p3 = params
      params = _params

    if async:
      result = None
      if mode == 'void':
        result = self.exec_module.execute_int_async(addr, params)
        result.mode = self.MODE_VOID
      elif mode == 'int':
        result = self.exec_module.execute_int_async(addr, params)
        result.mode = self.MODE_INT
      elif mode == 'fp':
        result = self.exec_module.execute_fp_async(addr, params)
        result.mode = self.MODE_FP
      else:
        raise Exception('Unknown mode: ' + mode)
      return result
    else:
      if mode == 'int':
        return self.exec_module.execute_int(addr, params)
      elif mode == 'fp':
        return self.exec_module.execute_fp(addr, params)
      elif mode == 'void':
        self.exec_module.execute_int(addr, params)
      else:
        raise Exception('Unknown mode: ' + str(mode))
    return


  # ------------------------------
  # Thread control
  # ------------------------------

  def join(self, t):
    """
    'Join' thread t, blocking until t is complete.
    """
    # TODO - should have a way of returning the join error status
    if not isinstance(t, self.exec_module.ThreadInfo):
      raise Exception('Invalid thread handle: ' + str(t))
    if t.mode == self.MODE_INT:
      return self.exec_module.join_int(t)
    elif t.mode == self.MODE_FP:
      return self.exec_module.join_fp(t)
    elif t.mode == self.MODE_VOID:
      self.exec_module.join_int(t)
    else:
      raise Exception('Unknown mode: ' + str(t.mode))
    return

  def suspend(self, t):
    """
    Suspend execution of thread t.
    """
    if not isinstance(t, self.exec_module.ThreadInfo):
      raise Exception('Invalid thread handle: ' + str(t))
    return self.exec_module.suspend_async(t)

  def resume(self, t):
    """
    Resume exectuion of thread t.
    """
    if not isinstance(t, self.exec_module.ThreadInfo):
      raise Exception('Invalid thread handle: ' + str(t))
    return self.exec_module.resume_async(t)

  def cancel(self, t):
    """
    Cancel execution of thread t.
    """
    if not isinstance(t, self.exec_module.ThreadInfo):
      raise Exception('Invalid thread handle: ' + str(t))
    return self.exec_module.cancel_async(t)

