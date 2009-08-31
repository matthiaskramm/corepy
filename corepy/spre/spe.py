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

__doc__ = """
Base classes for the Synthetic Programming Environment.
"""

import corepy.lib.extarray as extarray
import collections

from syn_util import *
#import syn_util as util


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
# Register
# ------------------------------------------------------------

class Register(object):
  def __init__(self, name):
    """Create a new register object."""
    self.name = name
    return

  def __str__(self): 
    if self.name != None:
      return self.name
    else:
      return object.__str__(self)
      #return '$%d' % (self.reg)
      #return '%s%d' % (self.prefix, self.reg)

  def __eq__(self, other):
    if isinstance(other, Register):
      return self.name == other.name
    elif isinstance(other, (int, long)):
      return self.reg == other
    elif isinstance(other, str):
      return self.name == other
    return False


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
      # print 'Casting to:', cls.expr_cls
      
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

    # TODO Should a variable always use the active code at the time of __init__
    # (like it is now), or follow the active code as it changes?
    if code is None and self.active_code is not None:
      code = self.active_code
    
    if reg is not None and not isinstance(reg, (Register, Variable)):
      raise Exception('reg argument must be a Register')
    if code is not None and not isinstance(code, InstructionStream):
      raise Exception('code argument must be an InstructionStream')
    
    if code is None:
      raise Exception('Variables require an InstructionStream to be either specified or set active')

    self.value = value
    self.acquired_register = False
    if reg is not None:
      if isinstance(reg, Variable):
        self.reg = reg.reg
      else:
        self.reg = reg
    else:
      self.reg = code.prgm.acquire_register(self.register_type_id)
      self.acquired_register = True
      
    self.code = code

    # self.v triggers the assignement mechanism. 
    if value is not None:
      self.v = self.value

    self.assigned = False
    self.expression = None
    return

  def release_register(self, force = False):
    if self.reg is not None and (self.acquired_register or force):
      self.code.prgm.release_register(self.reg)
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
      code.prgm.release_register(self._acquired_register)
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
      target = code.prgm.acquire_register(self.register_type_id)

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
    raise Exception('You should really implement a check method for %s' % (self.name))

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

    self.validate_operands(*operands, **koperands)

    # If active code is present, add ourself to it and remember that
    # we did so.  active_code_used is checked by InstructionStream
    # to avoid double adds from code.add(inst(...)) when active_code
    # is set.
    self.active_code_used = None    

    # Allow the user to create an instruction without adding it to active code.
    ignore_active = False
    if koperands.has_key('ignore_active'):
      ignore_active = koperands['ignore_active']
      del koperands['ignore_active']

    if self.active_code is not None and not ignore_active:
      self.active_code.add(self)
      self.active_code_used = self.active_code

    return

  def validate_operands(self, *operands, **koperands):
    dops = {} # name: value
    aops = []

    if len(operands) != len(self.machine_inst.signature):
      if len(operands) < len(self.machine_inst.signature):
        reason = 'Not enough '
      else:
        reason = 'Too many '
      raise Exception(reason + 'arguments supplied to instruction %s(%s).  Found arguments: (%s)' % (
        self.__class__.__name__,
        ', '.join([op_type.name for op_type in self.machine_inst.signature]),
        ', '.join([str(value) for value in operands])))

    #iop = 0
    for op_type, value in zip(self.machine_inst.signature, operands):
      if op_type.check(value):
        # Store ops by name and position.
        dops[op_type.name] = value
        aops.append(value)
        #ops[iop] = value        
        #iop += 1

      else:
        raise Exception("Operand validation failed for " + op_type.name )

    # TODO - include keyword operands in ordered array? 
    for op_type in self.machine_inst.opt_kw:
      kw = op_type.name
      if koperands.has_key(kw):
        if op_type.check(koperands[kw]):
          dops[kw] = koperands[kw]
      elif op_type.default is not None:
        dops[kw] = op_type.default

    self._operands = dops
    self._operand_iter = aops
    return


  def __contains__(self, val):
    return self._operand_iter.__contains__(val)

  def __iter__(self):
    """Iterate over the non-keyword operands, in order"""
    return self._operand_iter.__iter__()

  
  def __str__(self):
    operands = [str(op) for op in self._supplied_operands]
    return '%s %s' % (self.__class__.__name__, ', '.join(operands))
    #return '%s(%s)' % (self.__class__.__name__, ', '.join([str(op) for op in operands]))


  # Expression method
  ex = classmethod(_expression_method)
  
  def render(self):
    return self.machine_inst.render(self.params, self._operands)

  def set_position(self, pos):
    """Set the byte-offset position of this instruction in its
       InstructionStream"""
    self._operands["position"] = pos


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
#  type_id = [type]
  dispatch = ()

  def __init__(self, *operands, **koperands):
    # Attempt to find a dispatch entry that matches the operands.
    self.machine_inst = None

    for machine_inst, params in self.dispatch:
      #print "[check] (%s)" % (
      #  ','.join([str(arg_type.name) for arg_type in entry[0].signature],)),

      # entry[0].signature contains the fields to check
      if len(operands) != len(machine_inst.signature):
        # Signature is a different length, so it can't match.
        #print "LEN FAIL"
        continue

      match = True
      for i, field in enumerate(machine_inst.signature):
        if not field.check(operands[i]):
          match = False
          break

      if match:
        #print "MATCH"
        # Entire signature matched, break out.
        self.machine_inst = machine_inst
        self.params = params
        break


    if self.machine_inst is None:
      raise TypeError("Instruction %s does not support operands (%s)" % (
        type(self), ', '.join([str(op) for op in operands],)))


    # Skip the Instruction constructor and do mostly the same work here.
    # The operands have already been validated, so no need to re-do that.

    # Remember what the user passed in for debugging
    # TODO - not just for debugging! remove the underscore, too
    self._supplied_operands = operands
    self._supplied_koperands = koperands    

    # Do what validate_operands does, skipping the checks done already
    self._operands = {}
    self._operand_iter = []

    for i, op_type in enumerate(self.machine_inst.signature):
      # Store ops by name and position.
      self._operands[op_type.name] = self._operands[i] = operands[i]
      self._operand_iter.append(operands[i])
        
    for op_type in self.machine_inst.opt_kw:
      kw = op_type.name
      if koperands.has_key(kw):
        if op_type.check(koperands[kw]):
          self._operands[kw] = koperands[kw]
      elif op_type.default is not None:
        self._operands[kw] = op_type.default


    # If active code is present, add ourself to it and remember that
    # we did so.  active_code_used is checked by InstructionStream
    # to avoid double adds from code.add(inst(...)) when active_code
    # is set.
    self.active_code_used = None    

    # Allow the user to create an instruction without adding it to active code.
    ignore_active = False
    if koperands.has_key('ignore_active'):
      ignore_active = koperands['ignore_active']
      del koperands['ignore_active']

    if self.active_code is not None and not ignore_active:
      self.active_code.add(self)
      self.active_code_used = self.active_code

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
  def __init__(self, name):
    """Create a uniquely named label object.
       A unique number is prepended to the optionally specified label name.
    """
    self.position = None # Needed? not really
    self.name = name
    return

  def __str__(self):
    return "Label(%s)" % (self.name)
    #return "Label(%s, %s)" % (self.name, self.position)

  def set_position(self, pos):
    """Set the byte-offset position of this label in its
       InstructionStream.  Provided so that cache_code can simply call
       set_position() without checking whether the object is an instruction or
       a label."""
    self.position = pos
    return


class AlignStream(object):
  def __init__(self, prgm, align):
    self.align = align
    self.prgm = prgm
    self.position = None
    return

  def render(self):
    #return [i.render() for i in self.prgm._align_stream(self.position, self.align)]
    r = self.prgm._align_stream(self.position, self.align)
    ret = []
    for n in r:
      ret.extend(n.render())
    return ret
    

  def set_position(self, pos):
    """Set the byte-offset position of this AlignStream in its
       InstructionStream.  Provided so that cache_code can simply call
       set_position() without checking whether the object is an instruction or
       a label or alignment object.
       The position is used to determinate how much padding is needed to achieve
       the requested alignment."""
    self.position = pos
    return
  

class InstructionStream(object):
  """
  InstructionStream is a container for Instruction & Label objects
  """

  def __init__(self, prgm = None, debug = False):
    if not isinstance(prgm, Program):
      raise TypeError("A Program must be passed to InstructionStream.  Have you updated your code for the code composition changes?")

    self.prgm = prgm
 
    # TODO - set a default label for the body of this IS?

    # Debugging information
    self._stack_info = None
    self._debug = debug
    
    self._active_callback = None
    self.reset()
    return

  def set_debug(self, debug):
    self._debug = debug

  def get_debug(self):
    return self._debug

  debug = property(get_debug, set_debug)

  def set_active_callback(self, cb):
    self._active_callback = cb
  
  def reset(self):
    """
    Reset the instruction stream.
    """
    self.objects = []
    self.labels = []
    #self._objects = []
    #self._labels = []
    self._stack_info = []

    # This stream belongs to a program, which may have any code previously
    # added to this stream cached.  So, invalidate the program's cache.
    self.prgm._cached = False

    if self._debug:
      import inspect
      self._stack_info.append(_extract_stack_info(inspect.stack()))

    return


  # ------------------------------
  # Overloaded Operators
  # ------------------------------

  def __setitem__(self, key, obj):
    """
    Allow the user to replace instructions by index.
    """
    # If a label was replaced, remove it from the labels list.
    if isinstance(self.objects[key], Label):
      self.labels.remove(self.objects[key])

    self.objects[key] = obj

    # Clear the code cache
    self.render_code = None
    self._cached = False
    return

  def __getitem__(self, v):
    """Return the instruction at a particular index."""
    return self.objects[v]

  def __iter__(self):
    """Iterate over all the objects in the InstructionStream"""
    return self.objects.__iter__()


  def size(self): return len(self.objects)
  def __len__(self): return len(self.objects)

  # Overload addition to do the equivalent of code.add(other)
  def __add__(self, other):
    """Create and return a new IntructionStream consisting of the 'other'
       object concatenated to the contents of this stream."""
    code = self.prgm.get_stream()
    code.add(self)
    code.add(other)
    return code

  def __iadd__(self, other):
    """Add an object (e.g. Instruction, Label) to the stream."""
    self.add(other)
    return self


  # ------------------------------
  # Object Management
  # ------------------------------

  def add(self, obj):
    """
    Add an instruction and return the current size of the instruction
    sequence.  The index of the current instruction is (size - 1).
    (TODO: this is confusing and probably should return the index of
           the just added instruction...but too much code depends on
           these sematnics right now...)
    """

    if isinstance(obj, Instruction):
      # Check to see if this instruction has already been added to
      # this InstructionStream via active code.  This is to prevent
      # double adds from:
      #   code.add(inst(a, b, c))
      # when active code is set.
      if obj.active_code_used is not self:
        self.objects.append(obj)

        if self._debug:
          import inspect
          self._stack_info.append(_extract_stack_info(inspect.stack()))
    elif isinstance(obj, Label):
      if obj in self.labels:
        raise Exception('Label has already been added to the instruction stream; labels may only be added once.')

      self.objects.append(obj)
      self.labels.append(obj)

      if self._debug:
        import inspect
        self._stack_info.append(_extract_stack_info(inspect.stack()))
    elif isinstance(obj, ExtendedInstruction):
      if obj.active_code_used is not self:
        old_active = obj.get_active_code()
        if old_active is not self:
          obj.set_active_code(self)
        # AWF - 'render' here on the extended inst is a misnomer.
        # What happens is that the extended inst's block method is called,
        # which then adds instructions to the active instruction stream.
        obj.render()
        if old_active is not self:
          obj.set_active_code(old_active)
    elif isinstance(obj, InstructionStream):
      # Merge in the objects list of the substream.
      for subobj in obj:
        if isinstance(subobj, Label):
          self.labels.append(subobj)
        self.objects.append(subobj)
    else:
      raise Exception('Unsupported object: %s' % type(obj))

    # Clear the program's cache
    self.prgm._cached = False

    return len(self.objects)


  def align(self, align):
    """Insert no-op's into the stream to achieve a specified alignment"""
    self.objects.append(AlignStream(self.prgm, align))
    return


  class __type_iter(object):
    """Internal iterator that iterates over a list,
       only returning items of a particular type"""
    def __init__(self, obj_type, obj_list):
      self.obj_type = obj_type
      self.obj_list = obj_list
      return

    def __iter__(self):
      return self

    def next(self):
      for obj in self.obj_list:
        if isinstance(obj, self.obj_type):
          yield obj
      raise StopIteration
      return


  def inst_iter(self):
    """
    Return an iterator that iterates over the Instruction objects in the stream.
    """
    return __type_iter(Instruction, self.objects)


  def label_iter(self):
    """
    Return an iterator that iterates over the Label objects in the stream.
    """

    # Could use __type_iter, but we already have a list of labels to improve
    #  speed in other places.
    return self.labels


  def align_iter(self):
    """Return an iterator that iterates over the AlignStream's in the stream."""
    return __type_iter(AlignStream, self.objects)


  # ------------------------------
  # Debugging
  # ------------------------------

  def print_code(self):
    if not self._debug:
      import corepy.lib.printer as printer

      module = printer.Default(line_numbers = True)
      printer.PrintInstructionStream(self, module)
    else:
      addr = self.render_code.buffer_info()[0]
      last = (None, None)
      for i in xrange(0, len(self._objects)):
        inst = self._objects[i]
        stack_info = self._stack_info[i]
        
        user_frame, file = _first_user_frame(stack_info)

        # if file == 'spu_types.py':
        #  for frame in stack_info:
        #    print frame
            
        if last == (user_frame, file):
          ssource = '  ""  ""'
        else:
          sdetails = '%s:%s:%d ' % (file, user_frame[2], user_frame[1])
          sdetails += ' ' * (30 - len(sdetails))
          ssource = '%s %s' % (sdetails, user_frame[3][0][:-1]) # .strip())

#        TODO - this belongs in the SPU-specific code
#        pipeline = ''
#        if hasattr(inst, 'cycles'):
#          if inst.cycles[0] == 0:
#            pipeline = 'X ;'
#          else:
#            pipeline = ' X;'
#          pipeline += '  %2d' % inst.cycles[1]
            
        saddr   = '%08X' % (addr + i * 4)
        sinst   = '%4d %-30s' % (i, str(inst))
        last = (user_frame, file)
        print "%s %s %s" % (saddr, sinst, ssource)

    return


class Program(object):
  stream_type = None
  default_register_type = None

  # Native size of the instruction or instruction components. This
  # will be 'I' for 32-bit instructions and 'B' for variable length 
  # instructions.
  instruction_type = None

  def __init__(self, debug = False):
    # Make sure subclasses provide property values
    if self.default_register_type is None:
      raise Exception("Subclasses must set default_register_type")
    if self.stream_type is None:
      raise Exception("Subclasses must set stream_type")
    if self.instruction_type is None:
      raise Exception("Subclasses must set instruction_type")

    # Enable LRU-style register allocation by default.  This maximizes the
    # number of unique registers that get used, which reduces false register
    # dependences and increases performance.  However on some architectures it
    # may be better to use a MRU stack-style allocation to reduce the number
    # of unique registers used -- those arch's can do so by setting this
    # value to False.
    self.lru_reg_allocation = True

    # Some default labels
    self.lbl_prologue = Label("PROLOGUE")
    self.lbl_body = Label("BODY")
    self.lbl_epilogue = Label("EPILOGUE")
    self._builtin_labels = dict([(lbl.name, lbl) for lbl in (self.lbl_prologue, self.lbl_body, self.lbl_epilogue)])

    # Register Files
    # Use RegisterFiles to create a set of register instances for this
    # instance of InstructionStream.
    self._used_registers = {}
    self._register_avail_bins = []
    self._register_used_bins = []
    self._register_files = {} # type: RegisterFile
    self._register_pools = {}
    self._reg_type = {} # 'string': type, e.g. 'gp': GPRegister
    self.create_register_files()
    
    # Counter to use for generating unique labels
    self._label_counter = 0

    self._prologue = None
    self._epilogue = None

    self.reset()
    return

  def create_register_files(self):
    raise Exception("Required method not implemented by %s" % str(self.__class__))  


  def reset(self):
    """
    Reset the program, clear all storage, and return all the
    registers to the register pools.
    """
    self.objects = []
    self.labels = self._builtin_labels.copy()

    self._cached = False
    self.render_code = None

    for k in self._register_files.keys():
      self._register_pools[k] = collections.deque(self._register_files[k])
      self._used_registers[k] = {}

    self._register_avail_bins.extend(self._register_used_bins)
    self._register_used_bins = []

    # Free the references to cached storage
    self._storage_dict = {}
    self._storage_arr = []
    return


  # ------------------------------
  # Overloaded Operators
  # ------------------------------

  def __len__(self): return len(self.objects)

  def __iter__(self):
    return self.objects.__iter__()

  def __str__(self):
    import corepy.lib.printer as printer
    import cStringIO

    strfd = cStringIO.StringIO()
    module = printer.Default()
    printer.PrintProgram(self, module, fd = strfd)
    return strfd.getvalue()

  # Overload addition to do the equivalent of prgm.add(other)
  def __iadd__(self, other):
    """Add an object (e.g. InstructionStream) to the program."""
    self.add(other)
    return self


  # ------------------------------
  # Storage Management
  # ------------------------------

  def add_storage(self, key, val = None):
    """
    Add a reference to an object used during execution of the
    instruction stream.  This cache keeps locally allocated objects
    from being garbage collected once the stream is built.
    """
    if val is None:
      self._storage_arr.append(key)
    else:
      self._storage_dict[key] = val
    return

  def remove_storage(self, key):
    """
    Remove a storage reference by key.
    """

    if key in self._storage_arr:
      self._storage_arr.remove(key)
    else:
      try:
        del self._storage_dict[key]
      except KeyError:
        pass
    return

  def get_storage(self, key):
    try:
      return self._storage_dict[key]
    except KeyError:
      return None

  def append_storage(self, prgm):
    """
    Append all the other Program's storage.
    """

    self._storage_arr.extend(prgm._storage_arr)
    self._storage_dict.update(prgm._storage_dict)
    return


  # ------------------------------
  # InstructionStream Management
  # ------------------------------

  def get_stream(self):
    """Return a new InstructionStream object.
       For any code in the stream to be part of the Program, the stream must
       be added back to the program with code.add() or += operator."""
    return self.stream_type(self)


  def add(self, obj):
    if isinstance(obj, InstructionStream):
      self.objects.append(obj)
    else:
      raise Exception('Unsupported object: %s' % type(obj))

    self._cached = False
    self.render_code = None
    return len(self.objects)


  # ------------------------------
  # Label Management
  # ------------------------------

  def has_label(self, name):
    return self._labels.has_key(name)

  def get_label(self, name):
    try:
      return self.labels[name]
    except KeyError: pass

    lbl = Label(name)
    self.labels[name] = lbl
    return lbl

  def get_unique_label(self, name = ""):
    """
    Generate a unique label name and create/return a label with that name.
    """
    nr = self._label_counter
    self._label_counter += 1
    return self.get_label("_".join((name, str(nr))))


  # ------------------------------
  # Register management
  # ------------------------------

  # Register allocation now has two different modes, depending on whether a
  # register type is 'complex' (has the '_complex_reg' attr) or not.
  # The simple mode is the same mode that has been around for a long time,
  # and is used exclusively by the SPU, PPC, and CAL architectures.  The x86
  # architectures use simple mode for ST, MM, and XMM type registers, but use
  # the complex mode for general purpose registers.

  # Complex mode organizes registers into bins, or sets of registers.
  # Entire bins of registers are required/released at a time.  The purpose
  # of complex mode is to deal with x86's overlapping GP registers.  For
  # example, the lower 32 bits of the rbx register are used for the ebx
  # register, whose lower 16 bits are used for bx.  Therefore when allocating
  # registers of mixed sizes, we want to be careful not to allocate eax when
  # rax has been allocated -- major badness can/will occur.  We do this by
  # arranging rbx/ebx/bx/bl into a single bin, and acquire/releasing the
  # entire bin.  Which register from the bin gets returned depends on the
  # register type/width the user requested (gp8/gp16/gp32/gp64), defaulting to
  # 64bit on x86_64 and 32bit on x86.

  # Some profiling showed that acquire/release register were taking a
  # siginificant amount time during code generation.  This code has been
  # highly tuned/specialized for optimal performance.  For example the high-
  # speed 'deque' container is used for the register pools, and a dictionary
  # used to track used registers -- this was faster than using an array.

  def acquire_register(self, reg_type = None, reg_name = None):
    if reg_type is None:
      reg_type = self.default_register_type    
    elif isinstance(reg_type, str):
      reg_type = self._reg_type[reg_type]

    if hasattr(reg_type, '_complex_reg'):
      # Do complicated-style register bin allocations
      if reg_name is None:
        # Assign a new register name and remove the bin
        try:
          bin = self._register_avail_bins.pop()
        except IndexError:
          fmtstr = "No more registers of type %s available"
          raise Exception(fmtstr % str(reg_type))

        self._register_used_bins.append(bin)
        reg = bin[self._reg_map[reg_type]]
        # TODO - update used_registers?
        return reg
      else:
        # Register specified, find the bin and remove it
        def find_bin_reg():
          for b in self._register_avail_bins:
            for r in b:
              if r == reg_name:
                return (b, r)

        (bin, reg) = find_bin_reg()

        self._register_avail_bins.remove(bin)
        self._register_used_bins.append(bin)
        return reg
    # else: simple register allocation

    pool = self._register_pools[reg_type]

    if reg_name is not None:
      # TODO - right now, if a user wants to specify a reg that is not the
      # default type, they also have to specify reg_type.  Is it possible to
      # just specify the register name?
      # TODO - for this to be fast, need to quicky convert a str/int to a reg
      reg = None
      for r in pool:
        if r == reg_name:
          reg = r
          break

      if reg is None:
        raise Exception("Requested register %s not available" % str(reg_name))

      pool.remove(reg)
    else:
      try:
        reg = pool.pop()
      except IndexError:
        fmtstr = "No more registers of type %s available"
        raise Exception(fmtstr % str(reg_type))

    self._used_registers[reg_type][reg] = True
    return reg

  def release_register(self, reg):
    # TODO - perhaps mark a variable on a register to indicate that it has been
    #  released?  then we can error if we try to use it after releasing.

    if hasattr(type(reg), '_complex_reg'):
      # complex register, find the bin this reg belongs to and release it
      for bin in self._register_used_bins:
        if reg in bin:
          self._register_used_bins.remove(bin)
          self._register_avail_bins.append(bin)
          return

      raise Exception("Warning: register %s already released" % str(reg))
    else:
      #if not reg.acquired:
      #  raise Exception("Warning: register %s already released" % str(reg))
      # TODO - need to allow the arch to decide which side to append regs.
      #  left creates an LRU, minimizing reg reuse, while right creates a
      #  stack, maximizing reg reuse.
      pool = self._register_pools[type(reg)]

      if self.lru_reg_allocation:
        pool.appendleft(reg)
      else:
        pool.append(reg)
    return 


  def registers_available(self, reg_type = None):
    if reg_type is None:
      reg_type = self.default_register_type    
    elif isinstance(reg_type, str):
      reg_type = self._reg_type[reg_type]

    if hasattr(reg_type, '_complex_reg'):
      return len(self._register_avail_bins)

    return len(self._register_pools[reg_type]) 
 

  def acquire_registers(self, n, reg_type = None):
    return [self.acquire_register(reg_type) for i in xrange(n)]

  def release_registers(self, regs):
    for reg in regs:
      self.release_register(reg)
    return


  # ------------------------------
  # Execute/ABI support
  # ------------------------------

  def inst_addr(self):
    if self._cached:
      return self.render_code.buffer_info()[0]
    else:
      return None

  def make_executable(self):
    raise Exception("Required method not implemented by %s" % str(self.__class__))

  def _synthesize_prologue(self):
    raise Exception("Required method not implemented by %s" % str(self.__class__))

  def _synthesize_epilogue(self):
    raise Exception("Required method not implemented by %s" % str(self.__class__))


  # Note - TRAC ticket #19 has some background info and reference links on
  # the algorithms used here. https://svn.osl.iu.edu/trac/corepy/ticket/19

  def _cache_code_I(self, render_code, fwd_refs, stream):
    # Assumed below that 'I' type is 4 bytes
    for obj in stream:
      if isinstance(obj, Instruction):
        # Does this instruction reference any labels?
        lbl = None
        for op in obj._operand_iter:
          if isinstance(op, Label):
            lbl = op
            break

        if lbl is None: # No label reference, render the inst
          render_code.append(obj.render())
        else: # Label reference
          # Check that the label is in this stream
          if not lbl.name in self.labels:
            raise Exception("Label operand '%s' has not beed added to instruction stream" % lbl.name)

          obj.set_position(len(render_code) * 4)

          # TODO - could remove this conditional and always delay render
          # Could this eliminate the need to initially clear the position?
          # More beneficial here than in the 'B' case; instruction lengths
          # won't change so it's no big deal to delay right here.
          if lbl.position != None:  # Back reference, render the inst
            render_code.append(obj.render())
          else: # Fill in a dummy instruction and save info to render later
            fwd_refs.append((obj, len(render_code)))
            render_code.append(0xFFFFFFFF)
      elif isinstance(obj, Label): # Label, fill in a zero-length slot
        obj.set_position(len(render_code) * 4)
      elif isinstance(obj, AlignStream):
        # Call arch-specific alignment.
        # give it the desired alignment and current alignment.
        # should return an array of instructions to render
        insts = self._align_stream(len(render_code) * 4, obj.align)
        for i in insts:
          render_code.append(i.render())
    return

  def _resolve_label_refs_I(self, render_code, fwd_refs):
    # Render the instructions with forward label references
    for rec in fwd_refs:
      render_code[rec[1]] = rec[0].render()
    return


  def _cache_code_B(self, inst_list, inst_len, stream):
    # inst_list is a list of tuples.  Each tuple contains:
    # bool indicating presence of a label reference
    # rendered code ([] if label)
    # label or instruction object

    for obj in stream:
      if isinstance(obj, Instruction):
        # Does this instruction reference any labels?
        lbl = None
        relref = False
        sig = obj.machine_inst.signature

        #for iop in xrange(0, len(sig)):
        for iop, op in enumerate(obj._operand_iter):
          opsig = sig[iop]
          if hasattr(opsig, "relative_op") and opsig.relative_op == True:
            #op = obj._operands[iop]
            if isinstance(op, Label):
              lbl = op
            # This is a hack, but it works.  Some instructions can have
            # a relative offset that is not a label.  These insts need to be
            # re-rendered if instruction sizes change
            relref = True

        if lbl is None: # No label references
          obj.set_position(inst_len)
          r = obj.render()
          inst_list.append([relref, r, obj])
          inst_len += len(r)
        else: # Instruction referencing a label.
          if not lbl.name in self.labels:
            raise Exception("Label operand '%s' has not beed added to instruction stream" % lbl.name)
          obj.set_position(inst_len)

          if lbl.position != None: # Back-reference, render the instruction
            r = obj.render()
            inst_list.append([True, r, obj])
            inst_len += len(r)
          else: # Fill in a dummy instruction, assuming 2-byte best case
            inst_list.append([True, [-1, -1], obj])
            inst_len += 2
      elif isinstance(obj, Label): # Label, fill in a zero-length slot
        obj.set_position(inst_len)
        inst_list.append([False, [], obj])
      elif isinstance(obj, AlignStream):
        # Call arch-specific alignment.
        # give it the desired alignment and current alignment.
        # should return an array of instructions to render
        obj.set_position(inst_len)
        r = obj.render()
        inst_list.append([True, r, obj])
        inst_len += len(r)

    return inst_len

  def _resolve_label_refs_B(self, render_code, inst_list):
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

    # Final loop, bring everything together into render_code
    for rec in inst_list:
      if isinstance(rec[2], (Instruction, AlignStream)):
        render_code.fromlist(rec[1])
    return


  def cache_code(self):
    """
    Fill in the epilogue and prologue.  This call freezes the code and
    any subsequent calls to acquire_register() or add() will unfreeze
    it.  Once the checks are performed, the code should not be modified.
    """

    if self._cached == True:
      return


    self._synthesize_prologue()
    self._prologue.append(self.lbl_body)
    self._synthesize_epilogue()

    # We depend on label positions being None initially.  But if a label
    # is part of a cache operation more than once, we need to reset the
    # position back to None.
    # Maybe just do this on B type, and always delay render on I type?
    for lbl in self.labels.values():
      lbl.position = None

    if self.instruction_type == 'I':
      render_code = extarray.extarray('I')
      fwd_refs = [] # TODO - make fwd_refs a dict inst -> position
      self._cache_code_I(render_code, fwd_refs, self._prologue)

      # TODO - may want to do something different for non-IS objects
      for stream in self.objects:
        self._cache_code_I(render_code, fwd_refs, stream.objects)

      self._cache_code_I(render_code, fwd_refs, self._epilogue)

      self._resolve_label_refs_I(render_code, fwd_refs)
      self.render_code = render_code

      #self.render_code = self._cache_code_I()
    elif self.instruction_type == 'B':
      inst_list = []
      inst_len = 0

      inst_len = self._cache_code_B(inst_list, inst_len, self._prologue)

      # TODO - may want to do something different for non-IS objects
      for stream in self.objects:
        inst_len = self._cache_code_B(inst_list, inst_len, stream.objects)

      inst_len = self._cache_code_B(inst_list, inst_len, self._epilogue)

      self.render_code = extarray.extarray('B')
      self._resolve_label_refs_B(self.render_code, inst_list)

    self.make_executable()
    self._cached = True
    return


  def print_code(self, pro = False, epi = False, binary = False, hex = False):
    import corepy.lib.printer as printer
    if pro or epi or hex or binary:
      self.cache_code()
    module = printer.Default(show_prologue = pro, show_epilogue = epi,
                             show_binary = binary, show_hex = hex,
                             line_numbers = True)
    printer.PrintProgram(self, module)
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
  
  def execute(self, prgm, mode = 'int', async = False, params = None, debug = False):
    """
    Execute the code in the Program object.

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

    if not isinstance(prgm, Program):
      raise TypeError("Only Programs may be executed")

    if len(prgm) == 0:
      return None

    if not prgm._cached:
      prgm.cache_code()

    addr = prgm.inst_addr()

    if debug:
      print 'prgm info: 0x%x %d' % (addr, len(prgm.render_code))
      prgm.print_code(hex = True, pro = True, epi = True)
     

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

