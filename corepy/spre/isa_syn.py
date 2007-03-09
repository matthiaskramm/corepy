# Copyright 2006-2007 The Trustees of Indiana University.

# This software is available for evaluation purposes only.  It may not be
# redistirubted or used for any other purposes without express written
# permission from the authors.

# Authors:
#   Christopher Mueller (chemuell@cs.indiana.edu)
#   Andrew Lumsdaine    (lums@cs.indiana.edu)


import array
from syn_util import *

# Array type for an instruction
INSTRUCTION_TYPE = 'I' # 32-bit unsigned integer

__doc__=="""
ISA synthesize module.  To run in debug mode, set DEBUG_MODE to True. 
"""

DEBUG_MODE = False
# DEBUG_MODE = True


# ------------------------------------------------------------
# Instruction Fields
# ------------------------------------------------------------

class Field:
  """
  A Field is a callable object that generates properly formatted
  instruction fields.  The constructor takes a name, shift value and
  bit range and generates a new __call__ method specialized for those
  values.

  Field uses runtime code generation to create the specialized
  __call__ function.  The class properties callTemplate and codeTemplate
  are strings that contain the code for generating an instruction field.
  Subclasses override these values to create new Field types.
  The templates take parameters as string formating parameters.  The
  available parameters are 'name' and 'shift'.  The are passed using
  a dictionary to the string formatting operator '%'.

  callTemplate is a complete function that returns an value encoded in
  the proper Field format.

  codeTemplate returns a Python statement that can be combined with
  other statements using logical operators to form a complete
  instruction. The MachineInstruction classes uses sequences of codeTemplates
  connected by logical OR '|' to create its __call__ method.  This has
  the advantage of avoiding function calls while the instruction is
  generated.

  When a new Field is constructed, Field creates specialized
  instances of callTemplate and codeTemplate but substituting the
  proper values into the strings.  These new strings are available
  using the instance properties sCode and sFunc.

  As an example, on PowerPC the D field (destination register)
  is a 5-bit integer occupying bits [6,10] in an instruction.  On
  PowerPC is big-endian, so to place the value in the proper location
  in a 32-bit instruction it is shifted 21 bits to the left:

    Field('D', 21, [6,10])
    d.sCode is '(long(D) << 21)'
    d.sFunc is 'def CallFunc(value):\n  return (value << 21)\nself.__call__ = CallFunc\n' 

  Note the effect of partial specialization: rather than accessing the
  shift through a variable, it is encoded directly into the code.
  While this does not benefit __call__() that much, it has a huge
  benefit when MachineInstructions chain together multiple sCode statements.  

  SynthesizeField() and SynthesizeFields() are available to build
  Fields from lists of parameters and fields, respectively.
  """

  # Templates that partial specialize the __call__ method and shift
  # code using the shift and name values passed to the constructor.
  callTemplate = "def CallFunc(value):\n  return (value << %(shift)d)\nself.__call__ = CallFunc\n"
  codeTemplate = '(long(%(name)s) << %(shift)d)'

  def __init__(self, name, shift, bits, default = None, mask = None):
    self._shift = shift
    self.bits = bits
    self.width = bits[1] - bits[0] + 1
    self.name  = name
    self.has_default = False
    
    if default is None:
      self.param = name
    else:
      self.param = name + '=' + str(default)
      self.has_default = True

    self.mask = mask
    self.sCode = self.codeTemplate % {'name':name, 'shift':shift, 'mask':mask} 
    self.sFunc = self.callTemplate % {'name':name, 'shift':shift, 'mask':mask}
    code = compile(self.sFunc, '<Field %s>' % name, 'exec')
    exec code
    return
  
  def __call__(self, value):
    return (value << self._shift) # note that shift can be made constant at this point...


# ------------------------------
# Field subclasses
# ------------------------------

class Opcode(Field):
  """
  Syntactic sugar for opcodes.
  """
  callTemplate = "def CallFunc(value):\n  return (long(value) << %(shift)d)\nself.__call__ = CallFunc\n"


class MaskedField(Field):
  """
  Arbitrary mask applied before shift.
  TODO: Refactor other masked fields to this.
  """
  
  callTemplate = "def CallFunc(value):\n  return (value & 0x%(mask)x) << %(shift)d\nself.__call__ = CallFunc\n"
  codeTemplate = "(long(%(name)s) & 0x%(mask)x) << %(shift)d"


class MaskedField_16(Field):
  """
  Mask to only allow the first 16 bits through
  """
  
  callTemplate = "def CallFunc(value):\n  return (value & 0xFFFF)\nself.__call__ = CallFunc\n"
  codeTemplate = "long(%(name)s) & 0xFFFF"

class MaskedField_14(Field):
  """
  Mask out the bottom two bits
  """
  callTemplate = "def CallFunc(value):\n  return (value & 0xFFFC)\nself.__call__ = CallFunc\n"
  codeTemplate = "long(%(name)s) & 0xFFFC"

class MaskedField_LI(Field):
  """
  Mask out the bottom two bits
  """
  callTemplate = "def CallFunc(value):\n  return (value & 0xFFFFFF)\nself.__call__ = CallFunc\n"
  codeTemplate = "long(%(name)s) & 0xFFFFFF"
                                    
  
class SplitField(Field):
  """
  spr split-field format: A 10-bit number split into 5-bit chunks
  with the order reversed
    0x3E0 - upper 5 bits mask for a 10-bit number
    0x1F  - lower 5 bits mask for a 10-bit number
  """
  callTemplate = "def CallFunc(value):\n  return (((value & 0x3E0) >> 5) | ((value & 0x1F) << 5)) << %(shift)d\nself.__call__ = CallFunc\n"
  codeTemplate = "((((long(%(name)s) & 0x3E0) >> 5) | ((long(%(name)s) & 0x1F) << 5)) << %(shift)d)"



# ------------------------------
# Syntheise functions
# ------------------------------

def SynthesizeField(name, fieldtype, bits, default):
  """
  Synthesize a big-endian Field from a name, class, and bit range.
  """
  if type(bits) == int:
    bits = (bits, bits)

    
  shift = 31 - bits[1]

  if fieldtype is MaskedField:
    return fieldtype(name, shift, bits, mask=default)
  else:
    return fieldtype(name, shift, bits, default)
  
def SynthesizeFields(fields, g):
  """
  Synthesize a list of fields of the form:

  Fields = (
    ("name1",  (Class1, (bit_start1,bit_end1) [, default])),
    ("name2",  (Class2, (bit_start2,bit_end2) [, default])),
     ...
   )

  The new instances are placed in the dictionary g, which is typically
  the global environment for a module.  This populates the   module
  with a one function (callable object) for each Field.
  """
  for field in fields:
    name, params = field
    default = None
    if len(params) == 3: default = params[2]
    g[name] = SynthesizeField(name, params[0], params[1], default)
  return


# ------------------------------------------------------------
# Instuructions
# ------------------------------------------------------------

class _MachineInstruction:
  """
  MachineInstruction generates a callable object that returns a properly
  formatted instruction when called the the instruction's arguments. 

  (Note that there are two versions of MachineInstruction.  _MachineInstruction is
   the default version and _DebugMachineInstruction prints out details each
   time __call__ is called.  To use one or the other, assign the
   module level MachineInstruction to either)
   
  MachineInstructions are created by appending a sequence of Fields to the
  instance in the order they appear in the ISA.  As Fields are added,
  their generate statements are accumulated into a single statement
  joined by logical ORs (|).  The Field names are also used to
  accumulate a parameter list.  When the MachineInstruction instance is called
  for the first time, the statement is 'synthesized' into a function
  by printing the string to callTemplate and compiling it.  The
  last statement in callTempalte assignes the new function to the
  __call__() of the current instance.

  MachineInstruction has three callable states:
    state 0: when first created, an MachineInstruction is not callable
    state 1: once a Field has been added, __call__ is set to
             synthesize
    state 2: when synthesize is called, custom __call__ method is
             generated and replaces synthesize as __call__

  As an example, consider the addx instruction:
    addx = MachineInstruction('addx')
    addx.AppendOpcode(OPCD(31))  # Primary opcode
    addx.AppendField(D)          # Operands...
    addx.AppendField(A)
    addx.AppendField(B)
    addx.AppendOpcode(OPCD(266)) # Extended opcode
    addx(3, 4, 5) # call with D=r3, A=r4, B=r5

  In this case, the final __call__ method is:
    def __call__(self, rd, ra, rb):
      return (31 << 26) | (rd << 21) | (ra << 16) | (rb << 11) | (266 << 1);

  """

  # The call template
  # def call_name(param1, param2, ...):
  #   return statement1 | statement2 | ...
  # self.__call__ = Call_name
  callTemplate = "def Call_%(name)s(%(params)s):\n  return %(code)s\nself.__call__ = Call_%(name)s\n"

  def __init__(self, name):
    """
    Create a new instance of MachineInstruction 'name'.
    """
    self._fields = {} # name: Field
    self._inst = array.array(INSTRUCTION_TYPE, (0,))
    self._sFields = ''
    self.sCode   = ''
    self.name = name
    self._code = None
    self._opcodes = [] # opcodes in the order they were added
    self._machine_order = [] # name of the fields in order added
    
    self.generalize()
    return

  def generalize(self):
    """
    Invalicdate the current __call__function and reset it to
    synthesize().
    """
    self._code = None
    self.__call__ = self.synthesize
    return

  def AppendOpcode(self, opcode):
    """
    Append an opcode.  Opcodes are constants and do not have a
    corresponding parameter in the __call__ method.
    """
    self.generalize()
    
    if len(self._sFields) == 0:
      self.sCode += str(opcode)
    else:
      self.sCode += ' | ' + str(opcode)

    self._opcodes.append(opcode)
    return
  
  def AppendField(self, field):
    """
    Append a Field.  Add the name of the Field to the parameter list
    and append the Field's statement to the code for the MachineInstruction.
    """
    self.generalize()

    self._fields[field.param] = field
    self._machine_order.append(field)
      
    if len(self._sFields) == 0:
      self._sFields += field.param
      self.sCode += ' | ' + field.sCode 
    else:
      self._sFields += ', ' + field.param
      self.sCode += ' | ' + field.sCode
  
    return

  def synthesize(self, *args, **kargs):
    """
    Create a new __call__ method from the parameters and statements.
    """
    self._sCode = self.callTemplate % {'name':self.name, 'params':self._sFields, 'code':self.sCode}

    self._code = compile(self.callTemplate % {'name':self.name, 'params':self._sFields, 'code':self.sCode},
                         '<MachineInstruction %s>' % self.name, 'exec')
    exec self._code
    return self(*args, **kargs)


def _remove_default(param):
  return param.split('=')[0]

class _DebugMachineInstruction(_MachineInstruction):
  """
  Debug version of MachineInstruction
  """

  callTemplate = "def Call_%(name)s(%(params)s):\n  print 'code:', '%(name)s', '(%(params)s) [',\n  for p in '%(params)s'.split(', '): \n    if p != '': print p, '=', locals()[_remove_default(p)],\n  print ']: %(code)s'\n  print DecToBin(%(code)s)\n  return %(code)s\nself.__call__ = Call_%(name)s\n"

#     callTemplate = "def Call_%(name)s(%(params)s):
#   print 'code:', '%(name)s', '(%(params)s) [',
#   for p in '%(params)s'.split(', '): 
#     if p != '': print p, '=', locals()[p],
#   print ']: %(code)s'
#   print DecToBin(%(code)s)
#   return %(code)s
# self.__call__ = Call_%(name)s



# ------------------------------------------------------------
# Select the version of MachineInstruction to use
# ------------------------------------------------------------

if DEBUG_MODE:
  MachineInstruction = _DebugMachineInstruction
else:
  MachineInstruction = _MachineInstruction

def SynthesizeMachineInstruction(name, format, opcodes):
  """
  Create a new instance if MachineInstruction from a name, format, and an
  opcode 'factory'.  Extended opcodes can appear in multiple bit
  positions. The opcode factory maps a bit position to an opcode
  class. Bits that contain '0' can be specified by the integer 0 or a
  string of zeros with optional '_': '0_0000' means the next five bits
  are zero.  The '-' matches the syntax in the PEM.

  ADDX = (31, D, A, B, OE, 266, Rc),
  opcodes = {0:OPCD, 21:XO_1, 22:XO_2, 26:XO_3} 
  
  SyntheiszeMachineInstruction('addx', ADDX, opcodes)

  generates the sequence of instructions shown in the MachineInstruction
  docstring. 
  """

  inst = MachineInstruction(name)

  # keep track of how many bits are used to select the appropriate extended opcode
  ibits = 0 

  for elt in format:
    # print '  field:', elt
    if isinstance(elt, int):

      if elt == 0:
        # '0' bits
        ibits += 1
      else:
        # Opcodes

        inst.AppendOpcode(opcodes[ibits](elt))
        ibits += opcodes[ibits].width
    elif isinstance(elt, str):
      # Strings of '0' (reserved) bits
      ibits += len(elt)

      # Right now this is only used for AltiVec and there is only ever on '_' in the bits
      if elt.find('_'):
        ibits -= 1
    elif isinstance(elt, long):
      # TODO: This is a hack for SPU opcodes.  Fix it!
      inst.AppendOpcode(elt)
    else:
      inst.AppendField(elt)
      ibits += elt.width
          
  return inst


_current_instruction_stream = None
def set_code(code):
  global _current_instruction_stream
  _current_instruction_stream = code
  return

class coded_call:
  def __init__(self, func, name):
    self._func = func
    self._name = name
    return

  def __call__(self, *args, **kwargs):
    return _current_instruction_stream.add(self._func(*args, **kwargs))
    
def SynthesizeISA(isa, g, opcodes):
  """
  Create all the MachineInstructions for an ISA from a table of instructions
  and formats.  For example, 

  PPC_ISA = (
    ('addx',    (31, D, A, B, OE, 266, Rc)),
    ('addcx',   (31, D, A, B, OE, 10, Rc)),
    ...
  )
  
  """
  for inst in isa:
    name = inst[0]
    format = inst[1]['binary']
    g[name] = SynthesizeMachineInstruction(name, format, opcodes)
    
    g['c_' + name] = coded_call(g[name], name)
  return

