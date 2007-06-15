# Copyright 2006-2007 The Trustees of Indiana University.

# This software is available for evaluation purposes only.  It may not be
# redistirubted or used for any other purposes without express written
# permission from the authors.

# Authors:
#   Christopher Mueller (chemuell@cs.indiana.edu)
#   Andrew Lumsdaine    (lums@cs.indiana.edu)


__doc__ = """
Base classes for the Synthetic Programming Environment.
"""

import array

import inspect
import os.path
import traceback

from syn_util import *

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
  idx = 0
  for i, frame in enumerate(stack_info):
    file = os.path.split(frame[0])[1]
    if (file != 'spe.py' and
        file[:5] != 'spre_' and
        file not in ('util.py', 'spu_extended.py', 'iterators.py', '__init__.py') and
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
  def __init__(self, reg, code, prefix = 'r'):
    """
    Create a new register:
      reg is the architecture dependent value for the register, usually an int.
      code is the InstructionStream that created and owns this register.
    """
    self.reg = reg
    self.code = code
    self.prefix = prefix
    return

  def __str__(self): return '%s%d' % (self.prefix, self.reg)

  def __eq__(self, other):
    if isinstance(other, Register):
      return other.reg == self.reg
    elif isinstance(other, int):
      return self.reg == other
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
  

class Instruction(object):
  """
  Main user interface to machine instructions.

  Instruction supports to main modes of operation, an 'immediate' mode
  that creates the instruction and adds it to the active code (if
  active code is set) and a delayed evaluation mode.

  Immediate mode is the standard approach for using instructions.  See
  the comments in __init__ for more details.

  Delayed evaluation mode allows any instruction to be used in an
  expression. For instance, the spu.ai instruction can be insterted
  into an expression of SignedWord variables using the following code:

    r.v = x + y + spu.ai.ex(z, 12, type_cls=SignedWord)

  This tells ai to create an expression-compatible version of the
  instruction with a return type of SignedWord.  The argument list to
  the delayed form of the instruction consists of all arguments except
  for the destination operand (which in assembly order is the first
  operand). 
  """
  
  def __init__(self, *operands, **koperands):
    """
    The constructor for Instruction takes the underlying instructions
    operands as its operands.  By default, the operands are assumed to
    be in the same order as the assembly language specification for
    the instruction.

    The optional 'order' keyword argument can change the order the
    operands are interpreted in. This feature exists primarily to
    switch to machine order for an instruction, e.g.:
      spu.ai(b, a, d, order=spu.ai.machine_inst._machine_order)

    Compared this to the normal asm order:
      spu.ai(d, a, b) 
    """

    # Determin the order for the operands
    user_order = None
    if koperands.has_key('order'):
      user_order = koperands['order']
      del koperands['order']

    self._operands = koperands
    order = user_order or self.asm_order or self.machine_inst._machine_order

    if (order is self.asm_order) and len(operands) != len(self.asm_order):
      raise Exception("Not enough arguments supplied to %s.  %d required, %d supplied" % (
        self.machine_inst.name, len(self.asm_order), len(operands)))

    # if order is self.machine_inst._machine_order:
    #    print self.machine_inst.name,'using machine order'

    # Create the orded list of operands
    for op, field in zip(operands, order):

      # Check immediate operands to make sure they fit
      if (field.mask is not None and                     # is immediate? 
          ((op >= 0 and (field.mask & op) != op) or      # positive values
           (op < 0 and (field.mask & (-op)) != (-op)))): # negative values
        frame, file = _first_user_frame(_extract_stack_info(inspect.stack()))
        print '[%s:%s:%d] Warning: immediate operand %d does not fit in allocated bits' % (
          file, frame[2], frame[1], op)

      # Store the operand
      self._operands[field.name] = op

    # Check to make sure the user didn't intend to call the delayed
    # version of the instruction
    if koperands.has_key('type_cls'):
      print "Warning: Instruction created with keyword argument 'type_cls'.  Did you mean to use ex()?" 
      traceback.print_stack()

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

  def __getattr__(self, name):
    if self._operands.has_key(name):
      return self._operands[name]
    else:
      raise AttributeError, name
    
  def __str__(self):
    if self.asm_order is None:
      order = self.machine_inst._machine_order
    else:
      order = self.asm_order

    operands = []
    for field in order:
      if self._operands.has_key(field.name):
        operands.append(str(self._operands[field.name]))
        
    return '%s(%s)' % (self.machine_inst.name, ','.join([str(op) for op in operands]))

  ex = classmethod(_expression_method)
  
  def render(self):
    """
    Generate the binary form of the instruction by extracting the
    immediate and register values from the operands and passing them
    to the underlying machine instruction.
    """

    rendered_operands = {}
    for key in self._operands:
      op = self._operands[key]
      if isinstance(op, Register):
        rendered_operands[key] = op.reg
      elif isinstance(op, Immediate):
        rendered_operands[key] = op.value
      elif isinstance(op, Variable):
        rendered_operands[key] = op.reg.reg
      elif isinstance(op, (int, long)):
        rendered_operands[key] = op
      else:
        print op, isinstance(op, Variable)
        raise Exception('Unsupported operand type: %s = %s.  Register, Immediate, Variable, or int required.' % (type(op), str(op)))
      
    return self.machine_inst(**rendered_operands)


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


class InstructionStream(object):
  """
  InstructionStream mantains ABI compliance and code cach
  """

  def __init__(self):
    object.__init__(self)
    
    # Major code sections
    self._prologue = None
    self._code     = None
    self._epilogue = None
    self._instructions = None

    # Debugging information
    self._stack_info = None
    self._debug = False
    
    # Alignment parameters
    self._offset = 0
    
    # Register Files
    # Use RegisterFiles to create a set of register instances for this
    # instance of InstructionStream.
    # 
    # Each declarative RegisterFiles entry is:
    #   (file_id, register class, valid values)
    self._register_files = {} # type: RegisterFile
    self._reg_type = {} # 'cls': type
    for reg_type, cls, values in self.RegisterFiles:
      regs = [cls(value, self) for value in values]
      self._register_files[cls] = RegisterFile(regs, reg_type)
      self._reg_type[reg_type] = cls
      for reg in regs:
        reg.code = self
        
    # Storage holds references to objects created by synthesizers
    # that may otherwise get garbage collected
    self._storage = None

    # True if the all instruction streams have been created.
    self._cached = False

    self._active_callback = None
    self.reset()

    return

  def set_debug(self, debug): self._debug = debug
  def get_debug(self): return self._debug
  debug = property(get_debug, set_debug)
  
  def set_active_callback(self, cb): self._active_callback = cb
  
  def __del__(self):
    print 'Destroying', self
    return
  
  def __setitem__(self, key, inst):
    """
    Allow the user to replace instructions by index.
    """
    self._code[key + self._offset] = inst.render()
    self._instructions[key + self._offset] = inst


  def get_inst(self, idx):
    """
    Return the instruction at idx.
    """
    return self._instructions[idx + self._offset]
    

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
    self._code = array.array(self.instruction_type)
    self._instructions = []
    self._stack_info = []
    self.reset_cache()
    return

  def reset_cache(self): self._cached = False
    
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
    self._cached = False # Invalidate the cache

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
        self._code.append(inst.render())
        self._instructions.append(inst)

        if self._debug:
          self._stack_info.append(_extract_stack_info(inspect.stack()))
        
    elif isinstance(inst, ExtendedInstruction):
      if inst.active_code_used is not self:
        old_active = inst.get_active_code()
        if old_active is not self:
          inst.set_active_code(self)
        inst.render() # Calls add for each individual instruction
        if old_active is not self:
          inst.set_active_code(old_active)

    elif isinstance(inst, (int, long)):
      self._code.append(inst)
    else:
      raise Exception('Unsupported instruction format: %s.  Instruction or int is required.' % type(inst))

    # Invalidate the cache
    self._cached = False
    return len(self._code)

  def size(self): return len(self._code)


  # ------------------------------
  # Execute/ABI support
  # ------------------------------

  def _synthesize_prologue(self):
    """
    Create the prologue to  manage register preservation requirements
    from the ABI. 
    """
    return

  def _synthesize_epilogue(self):
    """
    Create the prologue to  manage register preservation requirements
    from the ABI. 
    """
    return


  def _align_instructions(self):
    """
    TODO: This needs to be replaced with something better.  Python's
    memory allocator is not honest about allocated memory if debug
    memory is turned on.  It's very easy to get off by 2 bytes and
    overwrite the sentinals.

    The solution will probably be to just bite the bullet and move
    all binary data over to our own aligned memory library.
    """
    a = self._code
    
    # If we're not aligned, make a new buffer that can be aligned and copy
    # the instructions into the buffer, starting at an aligned index
    if self.align != 0 and a.buffer_info()[0] % self.align != 0:
      n_extra = self.align / 4
      
      # print 'Warning: Unaligned array. Subsequent __setitem__ calls will not work. TODO: FIX THIS :)'
      aligned = array.array('I', range(len(a)) + range(n_extra))

      # print 'align: ', '0x%X' % a.buffer_info()[0], self.align
      offset = (aligned.buffer_info()[0] % (n_extra * 4)) / 4
      # print 'align: ', offset, '0x%X' % aligned.buffer_info()[0], self.align

      for i in range(offset):
        aligned[i] = 0
        
      for i in range(len(a)):
        aligned[i + offset] = a[i]
        
      self._offset = offset
      self._code = aligned
    
    self.inst_addr()
    

    return 

  def inst_addr(self):
    # print self._code
    addr = self._code.buffer_info()[0]
    # print self._code
    # print '0x%X 0x%X' % (addr, addr + self._offset * 4)
    return addr + self._offset * 4 # (addr % 16)
    

  def _check_alignment(self, buffer, name):
    """
    This should never return false for Python arrays...
    Until Linux.  Blah.
    """
    if self.align != 0 and buffer.buffer_info()[0] % self.align != 0:
      # print 'Warning: misaligned code:', name
      self._align_instructions()
      return False
    else:
      return True


  def add_return(self):
    """
    Add the architecture dependent code to return from a function.
    Used by cache_code to have the epilogue return.
    """
    return

  def add_jump(self, addr, reg):
    """
    Add the architecture dependent code to jump to a new instruction.
    Used by cache_code to chain the prologue, code, and epilogue.
    """
    return
    

  def cache_code(self):
    """
    Fill in the epilogue and prologue.  This call freezes the code and
    any subsequent calls to acquire_register() or add() will unfreeze
    it.  Also perform alignment checks.  Once the checks are
    preformed, the code should not be modified.
    """
    
    # HACK: Disable the current active code
    # NOTE: This may not work in the presence of multiple ISAs...
    active_callback = None
    if self._active_callback is not None:
      active_callback = self._active_callback
      active_callback(None)

    # Acquire a register to form the jump address in.  Acquire the
    # register before the prologue is created so the register is
    # properly saved and restored.
    jump_reg = self.acquire_register()

    # Generate the prologue
    self._synthesize_prologue()

    # Generate the epilogue
    self._synthesize_epilogue()

    # Connect the prologue -> code -> epilogue -> return (in reverse
    # order to avoid address changes from buffer reallocation)
    self._epilogue.add_return()
    self._check_alignment(self._epilogue._code, 'epilogue')

    # self.add_jump(self._epilogue._code.buffer_info()[0], jump_reg)
    self.add_jump(self._epilogue.inst_addr(), jump_reg)
    self._check_alignment(self._code,     'code')

    # self._prologue.add_jump(self._code.buffer_info()[0], jump_reg)
    self._prologue.add_jump(self.inst_addr(), jump_reg)
    self._check_alignment(self._prologue._code, 'prologue')

    # Finally, make everything executable
    for code in [self._prologue._code, self._code, self._epilogue._code]:
      self.exec_module.make_executable(code.buffer_info()[0], len(code))

    if active_callback is not None:
      active_callback(self)

    self.release_register(jump_reg)
    self._cached = True
    return


  # ------------------------------
  # Debugging
  # ------------------------------

  def print_code(self, pro = False, epi = False, binary = False):
    """
    Print the user instruction stream.
    """

    print 'code info:', self._code.buffer_info()[0], len(self._code)
    
    if pro:
      for inst, dec in zip(self._prologue._instructions, self._prologue._code):
        print str(inst)
        if binary:
          print DecToBin(dec)

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

        saddr   = '0x%08X' % (addr + i * 4)
        sinst   = '%4d %s' % (i, str(inst))
        sinst += ' ' * (40 - len(sinst))
        last = [user_frame, file]
        print saddr, sinst,  ssource
        if binary:
          print DecToBin(dec)
    else:
      for inst, dec, i in zip(self._instructions, self._code, range(0, self._code.buffer_info()[1])):
        print '%4d %s' % (i, str(inst))
        if binary:
          print DecToBin(dec)

    print 

    if epi:
      for inst, dec in zip(self._epilogue._instructions, self._epilogue._code):
        print str(inst)
        if binary:
          print DecToBin(dec)
      
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

  def __init__(self):  object.__init__(self)
  
  def execute(self, code, mode = 'int', debug = False, params = None):
    """
    Execute the instruction stream in the code object.

    Execution modes are:

      'int'  - return the intetger value in register gp_return when
               execution is complete
      'fp'   - return the floating point value in register fp_return
               when execution is complete
      'void' - return None
      'async'- execute the code in a new thread and return the thread
               id immediately

    If debug is True, the buffer address and code length are printed
    to stdout before execution.
    """

    if len(code._code) == 0:
      return None

    if not code._cached:
      code.cache_code()

    if debug:
      print 'code info: 0x%x 0x%x 0x%x %d' % (
        # code._prologue._code.buffer_info()[0],
        code._prologue.inst_addr(),
        # code._code.buffer_info()[0],
        code.inst_addr(),
        code._epilogue.inst_addr(),
        len(code._code))
      # print code._prologue._code
      # print code._code
      # print code._epilogue._code      
              
    
    # addr = code._prologue._code.buffer_info()[0]
    addr = code._prologue.inst_addr()

    result = None
    if mode == 'fp':
      result = self.exec_module.execute_fp(addr)
    elif mode == 'async':
      if params is None:
        result = self.exec_module.execute_async(addr)
      elif type(params) is self.exec_module.ExecParams:
        result = self.exec_module.execute_param_async(addr, params)
      else:
        # Backwards compatibility for list-style params
        _params = self.exec_module.ExecParams()
        _params.p1, _params.p2, _params.p3 = params
        result = self.exec_module.execute_param_async(addr, _params)
    elif mode == 'void':
      result = None
      self.exec_module.execute_void(addr)
    elif mode == 'int':
      if params is None:
        result = self.exec_module.execute_int(addr)
      elif type(params) is self.exec_module.ExecParams:
        result = self.exec_module.execute_param_int(addr, params)
      else:
        # Backwards compatibility for list-style params
        _params = self.exec_module.ExecParams()
        _params.p1, _params.p2, _params.p3 = params
        result = self.exec_module.execute_param_int(addr, _params)
    else:
      raise Exception('Unknown mode: ' + mode)

    return result


  # ------------------------------
  # Thread control
  # ------------------------------

  def join(self, t):
    """
    'Join' thread t, blocking until t is complete.
    """
    return self.exec_module.join_async(t)

  def suspend(self, t):
    """
    Suspend execution of thread t.
    """
    return self.exec_module.suspend_async(t)

  def resume(self, t):
    """
    Resume exectuion of thread t.
    """
    return self.exec_module.resume_async(t)

  def cancel(self, t):
    """
    Cancel exectuion of thread t.
    """
    return self.exec_module.cancel_async(t)
