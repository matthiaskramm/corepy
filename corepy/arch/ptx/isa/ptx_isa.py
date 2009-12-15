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

# PTX instructions

# WARNING: The PTX ISA here is handled differently than ppc, spu, x86, x86_64, and cal!
# Because of the type suffixes for instructions (which may be paired with untyped
# operands, or intentionally differ from operand types, e.g. with small unsigned numbers
# being used as signed numbers), we don't have fixed types (like spu and cal) and we can't
# do dispatch by checking operand type (like with x86). So instead we determine the
# instruction type directly in the Instruction classes, and assign a MachineInstruction
# there.
# BDM

from corepy.spre.spe import Instruction, DispatchInstruction, Register, Variable

import corepy.arch.ptx.types.registers as regs
import corepy.lib.extarray as extarray
import ptx_insts

__doc__="""
ISA for NVIDIA's PTX
"""

op_types = ('b', 'u', 's', 'f', 'p')
op_widths = (1, 8, 16, 32, 64)

###############################################
# Utility functions
###############################################

def guess_type_from_operands_2(d, s0, wide=False):
  '''
  Given two ptxOperands that are variables, determine the most reasonable type (if possible) for the instruction.
  '''
  rtype = ''
  rwidth = 0

  t_d = d.getTypeCode()
  t_s0 = s0.getTypeCode()
  w_d = d.getWidth()
  w_s0 = s0.getWidth()
  i_s0 = s0.isImmediate()

  if (wide == True and
        ((w_d == 2*w_s0)
         or (i_s0))):
    rwidth = w_d / 2
  elif ((w_d == w_s0)
       or (i_s0)):
    rwidth = w_d
  else:
    raise Exception("Widths for all operands must match")

  if t_d == 'f':
    if t_s0 in ('f', 'b'):
        rtype = 'f'
    else:
      raise "Incompatible operand types"
  elif t_d == 'u' or t_d == 's':
    if t_s0 in ('u', 's', 'b'):
        rtype = t_d
    else:
      raise "Incompatible operand types"
  elif t_d == 'b':
    # this is the case we actually care about...
    if t_s0 in ('u', 's'):
      if t_s0 == 's':
        rtype = 's'
      elif t_s0 == 'u':
        rtype = 'u'
    elif t_s0 == 'f':
      rtype = 'f'

  t = rtype + str(rwidth)
  if t == 'p1':
    t = pred
  return t

def guess_type_from_operands_3(d, s0, s1, wide = False):
  '''
  Given three ptxOperands that are variables, determine the most reasonable type (if possible) for the instruction.
  '''
  rtype = ''
  rwidth = 0

  t_d = d.getTypeCode()
  t_s0 = s0.getTypeCode()
  t_s1 = s1.getTypeCode()
  w_d = d.getWidth()
  w_s0 = s0.getWidth()
  w_s1 = s1.getWidth()
  i_s0 = s0.isImmediate()
  i_s1 = s1.isImmediate()

  if (wide == True and
        ((w_d == 2*w_s0 and w_d == 2*w_s1)
         or (i_s0 and w_d == 2*w_s1)
         or (i_s1 and w_d == 2*w_s0)
         or (i_s0 and i_s1))):
    rwidth = w_d / 2
  elif ((w_d == w_s0 and w_s0 == w_s1)
       or (i_s0 and w_s1 == w_d)
       or (i_s1 and w_s0 == w_d)
       or (i_s0 and i_s1)):
    rwidth = w_d
  else:
    raise Exception("Widths for all operands must match")

  if t_d == 'f':
    if (t_s0 in ('f', 'b') or i_s0) and (t_s1 in ('f', 'b') or i_s1):
      rtype = 'f'
    else:
      raise Exception("Incompatible operand types")
  elif t_d == 'u' or t_d == 's':
    if (t_s0 in ('u', 's', 'b')) and (t_s1 in ('u', 's', 'b')):
      rtype = t_d
    else:
      raise Exception("Incompatible operand types")
  elif t_d == 'b':
    # this is the case we actually care about...
    if t_s0 == 's' or t_s1 == 's' or t_s0 == 'u' or t_s1 == 'u': # if int value
      if t_s0 == 's' or t_s1 == 's':
        if t_s0 in ('u', 's', 'b') and t_s1 in ('u', 's', 'b'):
          rtype = 's'
        else:
          raise Exception("Incompatible operand types")
      elif t_s0 in ('u', 'b') and t_s1 in ('u', 'b'):
        rtype = 'u'
      else:
        raise Exception("Incompatible operand types")
    elif t_s0 == 'f' or t_s1 == 'f': # if floating point value
      if (t_s0 in ('f', 'b') or i_s0) and (t_s1 in ('f', 'b') or i_s1):
        rtype = 'f'
      else:
        raise Exception("Incompatible operand types")
    else:
      rtype = 'u'
      #raise Exception("Could not guess instruction type - please specify")

  t = rtype + str(rwidth)
  if t == 'p1':
    t = pred
  return t


def guess_type_from_operands_4(d, s0, s1, s2, wide=False):
  '''
  Given four ptxOperands that are variables, determine the most reasonable type (if possible) for the instruction.
  '''
  rtype = ''
  rwidth = 0

  t_d = d.getTypeCode()
  t_s0 = s0.getTypeCode()
  t_s1 = s1.getTypeCode()
  t_s2 = s2.getTypeCode()
  w_d = d.getWidth()
  w_s0 = s0.getWidth()
  w_s1 = s1.getWidth()
  w_s2 = s2.getWidth()
  i_s0 = s0.isImmediate()
  i_s1 = s1.isImmediate()
  i_s2 = s2.isImmediate()

  # in the 4 operand case, (d, a, b, c), d and c are the same width 
  if (wide == True and
        ((w_d == 2*w_s0 and w_d == 2*w_s1 and w_d == w_s2)
         or (i_s0 and w_d == 2*w_s1 and w_d == w_s2)
         or (i_s1 and w_d == 2*w_s0 and w_d == w_s2)
         or (i_s0 and i_s1 and i_s2))):
    rwidth = w_d / 2
  elif ((w_d == w_s0 and w_d == w_s1 and w_d == w_s2)
       or (i_s0 and w_s1 == w_d and w_s2 == w_d)
       or (i_s1 and w_s0 == w_d and w_s1 == w_d)
       or (i_s0 and i_s1 and i_s2)):
    rwidth = w_d
  else:
    raise Exception("Widths for all operands must match")


  if t_d == 'f':
    if t_s0 in ('f', 'b') and t_s1 in ('f', 'b') and t_s2 in ('f', 'b'):
        rtype = 'f'
    else:
      raise Exception("Incompatible operand types")
  elif t_d == 'u' or t_d == 's':
    if t_s0 in ('u', 's', 'b') and t_s1 in ('u', 's', 'b') and t_s2 in ('u', 's', 'b'):
        rtype = t_d
    else:
      raise Exception("Incompatible operand types")
  elif t_d == 'b':
    # this is the case we actually care about...
    if t_s0 in ('u', 's') or t_s1 in ('u', 's') or t_s2 in ('u', 's'):
      if t_s0 == 's' or t_s1 == 's' or t_s2 == 's':
        if t_s0 in ('u', 's', 'b') and t_s1 in ('u', 's', 'b') and t_s2 in ('u', 's', 'b'):
          rtype = 's'
        else:
          raise Exception("Incompatible operand types")
      elif t_s0 in ('u', 'b') and t_s1 in ('u', 'b') or t_s2 in ('u', 'b'):
          rtype = 'u'
      else:
        raise Exception("Incompatible operand types")
    elif t_s0 == 'f' or t_s1 == 'f' or t_s2 == 'f':
      if t_s0 in ('f', 'b') and t_s1 in ('f', 'b') and t_s2 in ('f', 'b'):
        rtype = 'f'
      else:
        raise Exception("Incompatible operand types")
    else:
      rtype = 'u'
      #raise "Could not guess instruction type - please specify"

  t = rtype + str(rwidth)
  if t == 'p1':
    t = pred
  return t

def getArithmeticType2(*operands, **koperands):
  d  = ptxOperand(operands[0])
  s0 = ptxOperand(operands[1])

  if not (d.isRegister() and (s0.isRegister() or s0.isImmediate())):
    raise "All operands for arithmetic instructions must be registers or immediates"
  
  if 'inst_type' not in koperands:
    inst_type = guess_type_from_operands_2(d, s0)
  elif koperands['inst_type'] in ('u16', 's16', 'u32', 's32', 'f32', 'u64', 's64', 'f64'):
    inst_type = koperands['inst_type']
    # TODO?: check operand types here
  else:
    raise "Invalid type for this instruction"

  return inst_type

def getArithmeticType3(*operands, **koperands):
  d  = ptxOperand(operands[0])
  s0 = ptxOperand(operands[1])
  s1 = ptxOperand(operands[2])

  if not (d.isRegister() and (s0.isRegister() or s0.isImmediate()) and (s1.isRegister() or s1.isImmediate())):
    raise "All operands for arithmetic instructions must be registers"
  
  if 'inst_type' not in koperands:
    inst_type = guess_type_from_operands_3(d, s0, s1)
  elif koperands['inst_type'] in ('u16', 's16', 'u32', 's32', 'f32', 'u64', 's64', 'f64'):
    inst_type = koperands['inst_type']
    # TODO?: check operand types here
  else:
    raise "Invalid type for this instruction"

  return inst_type

def getArithmeticType4(*operands, **koperands):
  d  = ptxOperand(operands[0])
  s0 = ptxOperand(operands[1])
  s1 = ptxOperand(operands[2])
  s2 = ptxOperand(operands[3])

  if not (d.isRegister() and (s0.isRegister() or s0.isImmediate()) and (s1.isRegister() or s1.isImmediate()) \
          and (s2.isRegister() or s2.isImmediate())):
    raise "All operands for arithmetic instructions must be registers"
  
  if 'inst_type' not in koperands:
    inst_type = guess_type_from_operands_4(d, s0, s1, s2)
  elif koperands['inst_type'] in ('u16', 's16', 'u32', 's32', 'f32', 'u64', 's64', 'f64'):
    inst_type = koperands['inst_type']
    # TODO?: check operand types here
  else:
    raise "Invalid type for this instruction"

  return inst_type

def getType2(valid_types, *operands, **koperands):
  d  = ptxOperand(operands[0])
  s_or_a = ptxOperand(operands[1])
  
  if 'inst_type' not in koperands:
    inst_type = guess_type_from_operands_2(d, s_or_a)
  elif koperands['inst_type'] in valid_types:
    inst_type = koperands['inst_type']

  if inst_type not in valid_types:
    raise "Invalid type for this instruction"

  return inst_type

def getLogicType3(*operands, **koperands):
  d  = ptxOperand(operands[0])
  s0 = ptxOperand(operands[1])
  s1 = ptxOperand(operands[2])

  if not (d.isRegister() and (s0.isRegister() or s0.isImmediate()) and (s1.isRegister() or s1.isImmediate())):
    raise "All operands for logic instructions must be registers or immediates"
  
  if 'inst_type' not in koperands:
    if ((d.getWidth() == s0.getWidth() or s0.isImmediate()) and
        (s0.getWidth() == s1.getWidth() or s0.isImmediate() or s1.isImmediate())):
      inst_type = 'b' + str(d.getWidth())
    else:
      raise Exception("Widths for all operands must match")

  elif koperands['inst_type'] in ('b16', 'b32', 'b64', 'pred'):
    inst_type = koperands['inst_type']
    # TODO?: check operand types here
  else:
    raise "Invalid type for this instruction"

  return inst_type

def getLogicType2(*operands, **koperands):
  d  = ptxOperand(operands[0])
  s0 = ptxOperand(operands[1])

  if not (d.isRegister() and (s0.isRegister() or s0.isImmediate())):
    raise "All operands for logic instructions must be registers"
  
  if 'inst_type' not in koperands:
    if d.getWidth() == s0.getWidth() or s0.isImmediate():
      inst_type = 'b' + str(d.getWidth())
    else:
      raise Exception("Widths for all operands must match")

  elif koperands['inst_type'] in ('b16', 'b32', 'b64', 'pred'):
    inst_type = koperands['inst_type']
    # TODO?: check operand types here
  else:
    raise "Invalid type for this instruction"

  return inst_type

def isIntegerType(rtype):
  if rtype[0] == 's' or rtype[0] == 'u':
    return True
  else:
    return False

def isFloatType(rtype):
  if rtype[0] == 'f':
    return True
  else:
    return False

def getTypeWidth(rtype):
  return int(rtype[1:])

#######################################
# Important classes (other than Instructions)
######################################

class ptxOperand(object):
  def __init__(self, value):
    self.value = value

    self.is_immediate = False
    self.is_variable = False # though we'll also include anything you can write to directly
    self.is_address = False
    self.is_register = False

    # if this is a synthetic variable, we want to extract its register
    if isinstance(value, Variable):
      value = value.reg

    if isinstance(value, (int, long, float)):
      self.is_immediate = True
    elif isinstance(value, regs.ptxVariable):
      self.is_variable = True
      if value.space == 'reg':
        self.is_register = True
    elif isinstance(value, regs.ptxAddress):
      self.is_address = True
    else:
      raise Exception("Unrecognized operand type (does not appear to be an immediate, valid register, or valid address).")

    if self.is_variable:
      self.vtype = value.rtype
      self.width = value.rwidth
    elif self.is_address:
      self.vtype = 'u'
      self.width = 64 # this could actually be 32, but we don't use this anywhere, so...
    else:
      # is immediate
      # note that all immediates (except for 32 bit hex values for floating point...)
      # are considered 64 bit in PTX and conversion happens elsewhere - so don't set proper sizes
      # (and we totally cheat by ignoring the 32 bit hex values, which should be safe)
      if isinstance(value, float):
        self.vtype = 'f'
        self.width = 64
      elif isinstance(value, (int, long)):
        # by default integer literals are signed
        if value > 2**63-1:
          self.vtype = 'u'
        else:
          self.vtype = 's'
        self.width = 64

  def isImmediate(self):
    return self.is_immediate

  def isVariable(self):
    return self.is_variable

  def isAddress(self):
    return self.is_address

  def isRegister(self):
    return self.is_register

  def getTypeCode(self):
    return self.vtype

  def getWidth(self):
    return self.width

  def getType(self):
    if self.vtype != 'pred':
      return self.vtype + str(self.width)
    else:
      return 'pred'

###################################
# ptxInstruction
##################################

#class ptxInstruction(Instruction):
#  def __init__(self, *operands, **koperands):
#    self._operands = {}
#    Instruction.__init__(self, *operands, **koperands)
#    
#  def validate_operands(self, *operands, **koperands):
#    return check(*operands, **koperands)
#
#  def check(self, *operands, **koperands):
#    raise "Subclasses must implement the check method."
#
#  def render(self):
#    retval = self.params['opcode']

class ptxInstruction(Instruction):
  def __init__(self, *operands, **koperands):
    self._operands = {}
    Instruction.__init__(self, *operands, **koperands)
    
  def validate_operands(self, *operands, **koperands):
    Instruction.validate_operands(self, *operands, **koperands)

##################################
# General instruction classes
##################################

class ptxALU2Instruction(ptxInstruction):
  def __init__(self, *operands, **koperands):
    if 'type' not in self.params:
      inst_type = getArithmeticType2(*operands, **koperands)
    self.params['type'] =  inst_type
    ptxInstruction.__init__(self, *operands, **koperands)

class ptxALU3Instruction(ptxInstruction):
  def __init__(self, *operands, **koperands):
    if 'type' not in self.params:
      inst_type = getArithmeticType3(*operands, **koperands)
      self.params['type'] =  inst_type
    ptxInstruction.__init__(self, *operands, **koperands)

class ptxALU4Instruction(ptxInstruction):
  def __init__(self, *operands, **koperands):
    if 'type' not in self.params:
      inst_type = getArithmeticType4(*operands, **koperands)
      self.params['type'] =  inst_type
    ptxInstruction.__init__(self, *operands, **koperands)

class ptxLogic3Instruction(ptxInstruction):
  def __init__(self, *operands, **koperands):
    if 'type' not in self.params:
      inst_type = getLogicType3(*operands, **koperands)
      self.params['type'] =  inst_type
    ptxInstruction.__init__(self, *operands, **koperands)

class ptxLogic2Instruction(ptxInstruction):
  def __init__(self, *operands, **koperands):
    if 'type' not in self.params:
      inst_type = getLogicType2(*operands, **koperands)
      self.params['type'] =  inst_type
    ptxInstruction.__init__(self, *operands, **koperands)

class ptxMovementInstruction(ptxInstruction):
  def __init__(self, *operands, **koperands):
    #if 'type' not in self.params:
    #  inst_type = getType2(self.valid_types, *operands, **koperands)
    #  self.params['type'] =  inst_type
    if self.params['type'] not in self.valid_types:
      raise Exception("Invalid type for this instruction")
    ptxInstruction.__init__(self, *operands, **koperands)

##################################
# Specific instruction classes
##################################

# Arithmetic #####################

class add(ptxALU3Instruction):
  def __init__(self, *operands, **koperands):
    self.params = {'opcode': 'add'}

    inst_type = getArithmeticType3(*operands, **koperands)
    self.params['type'] =  inst_type

    if isIntegerType(inst_type):
      if 'cc' in koperands and koperands['cc'] == True:
        if inst_type not in ('u32', 's32'):
          raise "Not a valid operand type for use with '.cc' option."
        self.machine_inst = ptx_insts.cc_x3
      else:
        self.machine_inst = ptx_insts.s_x3
          
    elif isFloatType(inst_type):
      self.machine_inst = ptx_insts.r_s_x3

    ptxALU3Instruction.__init__(self, *operands, **koperands)

class addc(ptxALU3Instruction):
  def __init__(self, *operands, **koperands):
    self.params = {'opcode': 'addc'}
    self.machine_inst = cc_x3
    ptxALU3Instruction.__init__(self, *operands, **koperands)

class sub(ptxALU3Instruction):
  def __init__(self, *operands, **koperands):
    self.params = {'opcode': 'sub'}
    inst_type = getArithmeticType3(*operands, **koperands)
    self.params['type'] =  inst_type
    if isIntegerType(inst_type):
        self.machine_inst = ptx_insts.s_x3
    elif isFloatType(inst_type):
      self.machine_inst = ptx_insts.r_s_x3

    ptxALU3Instruction.__init__(self, *operands, **koperands)

class mul(ptxALU3Instruction):
  def __init__(self, *operands, **koperands):
    self.params = {'opcode': 'mul'}
    inst_type = getArithmeticType3(*operands, **koperands)
    self.params['type'] =  inst_type
    if isIntegerType(inst_type):
        self.machine_inst = ptx_insts.hlw_x3
    elif isFloatType(inst_type):
      self.machine_inst = ptx_insts.r_s_x3
    ptxALU3Instruction.__init__(self, *operands, **koperands)

class mad(ptxALU4Instruction):
  def __init__(self, *operands, **koperands):
    self.params = {'opcode': 'mad'}
    inst_type = getArithmeticType4(*operands, **koperands)
    self.params['type'] =  inst_type
    if isIntegerType(inst_type):
        self.machine_inst = ptx_insts.hlws_x4
    elif isFloatType(inst_type):
      self.machine_inst = ptx_insts.r_s_x4
    ptxALU4Instruction.__init__(self, *operands, **koperands)

class mul24(ptxALU3Instruction):
  def __init__(self, *operands, **koperands):
    self.params = {'opcode': 'mul24'}
    self.machine_inst = ptx_insts.hl_x3
    ptxALU3Instruction.__init__(self, *operands, **koperands)

class mad24(ptxALU4Instruction):
  def __init__(self, *operands, **koperands):
    self.params = {'opcode': 'mad24'}
    self.machine_inst = ptx_insts.hls_x4
    ptxALU4Instruction.__init__(self, *operands, **koperands)

class sad(ptxALU4Instruction):
  def __init__(self, *operands, **koperands):
    self.params = {'opcode': 'sad'}
    self.machine_inst = ptx_insts.x4
    ptxALU4Instruction.__init__(self, *operands, **koperands)

class div(ptxALU3Instruction):
  def __init__(self, *operands, **koperands):
    self.params = {'opcode': 'div'}
    self.machine_inst = ptx_insts.ws_x3
    ptxALU3Instruction.__init__(self, *operands, **koperands)

class rem(ptxALU3Instruction):
  def __init__(self, *operands, **koperands):
    self.params = {'opcode': 'rem'}
    self.machine_inst = ptx_insts.w_x3
    ptxALU3Instruction.__init__(self, *operands, **koperands)

class abs(ptxALU2Instruction):
  def __init__(self, *operands, **koperands):
    self.params = {'opcode': 'abs'}
    self.machine_inst = ptx_insts.x2
    ptxALU2Instruction.__init__(self, *operands, **koperands)

class neg(ptxALU2Instruction):
  def __init__(self, *operands, **koperands):
    self.params = {'opcode': 'neg'}
    self.machine_inst = ptx_insts.x2
    ptxALU2Instruction.__init__(self, *operands, **koperands)

class min(ptxALU2Instruction):
  def __init__(self, *operands, **koperands):
    self.params = {'opcode': 'min'}
    self.machine_inst = ptx_insts.x2
    ptxALU2Instruction.__init__(self, *operands, **koperands)

class max(ptxALU2Instruction):
  def __init__(self, *operands, **koperands):
    self.params = {'opcode': 'max'}
    self.machine_inst = ptx_insts.x2
    ptxALU2Instruction.__init__(self, *operands, **koperands)

# Comparison and Selection #####

class set(ptxMovementInstruction):
  def __init__(self, *operands, **koperands):
    self.params = {'opcode': 'set'}
    if operands[1] not in ('and', 'or', 'xor'):
      self.machine_inst = ptx_insts.compop_d_a_b
      d = ptxOperand(operands[1])
      s0 = ptxOperand(operands[2])
      s1 = ptxOperand(operands[3])
    else:
      self.machine_inst = ptx_insts.compop_boolop_d_a_b
      d = ptxOperand(operands[2])
      s0 = ptxOperand(operands[3])
      s1 = ptxOperand(operands[4])
    self.valid_stypes = ('b16', 'b32', 'b64', 'u16', 'u32', 'u64', 's16', 's32', 's64', 'f32', 'f64')
    self.valid_dtypes = ('u32', 's32', 'f32')
    self.params['dtype'] = d.getType()
    self.params['stype'] = guess_type_from_operands_2(s0, s1)
    if self.params['stype'] not in self.valid_stypes or \
           self.params['dtype'] not in self.valid_dtypes:
      raise Exception("Invalid type for this instruction")
    ptxInstruction.__init__(self, *operands, **koperands)

# TODO: figure out how to handle setp.compopop.type p|q, a, b form
#       maybe by using operator overloading again
#       or (more sanely) guessing from operands

class setp(ptxMovementInstruction):
  def __init__(self, *operands, **koperands):
    self.params = {'opcode': 'setp'}
    if operands[1] not in ('and', 'or', 'xor'):
      self.machine_inst = ptx_insts.compop_p_a_b
      s0 = ptxOperand(operands[2])
      s1 = ptxOperand(operands[3])
    else:
      self.machine_inst = ptx_insts.compop_boolop_p_a_b
      s0 = ptxOperand(operands[3])
      s1 = ptxOperand(operands[4])
    self.valid_types = ('b16', 'b32', 'b64', 'u16', 'u32', 'u64', 's16', 's32', 's64', 'f32', 'f64')
    self.params['type'] = guess_type_from_operands_2(s0, s1)
    if self.params['type'] not in self.valid_types:
      raise Exception("Invalid type for this instruction")
    ptxInstruction.__init__(self, *operands, **koperands)

class selp(ptxMovementInstruction):
  def __init__(self, *operands, **koperands):
    self.params = {'opcode': 'selp'}
    self.machine_inst = ptx_insts.x4
    d = ptxOperand(operands[0])
    s0 = ptxOperand(operands[1])
    s1 = ptxOperand(operands[2])
    s2 = ptxOperand(operands[3])
    self.valid_types = ('b16', 'b32', 'b64', 'u16', 'u32', 'u64', 's16', 's32', 's64', 'f32', 'f64')
    self.params['type'] = guess_type_from_operands_3(d, s0, s1)
    if self.params['type'] not in self.valid_types or s2.getType() != 'pred':
      raise Exception("Invalid type for this instruction")
    ptxInstruction.__init__(self, *operands, **koperands)

class slct(ptxMovementInstruction):
  def __init__(self, *operands, **koperands):
    self.params = {'opcode': 'slct'}
    self.machine_inst = ptx_insts.x3_x1
    d = ptxOperand(operands[0])
    s0 = ptxOperand(operands[1])
    s1 = ptxOperand(operands[2])
    s2 = ptxOperand(operands[3])
    self.valid_types = ('b16', 'b32', 'b64', 'u16', 'u32', 'u64', 's16', 's32', 's64', 'f32', 'f64')
    self.params['dtype'] = guess_type_from_operands_3(d, s0, s1)
    self.params['ctype'] = s2.getType()
    if self.params['dtype'] not in self.valid_types or self.params['ctype'] not in ('s32', 'f32'):
      raise Exception("Invalid type for this instruction")
    ptxInstruction.__init__(self, *operands, **koperands)

# Logic and Shift ##############

class and_(ptxLogic3Instruction):
  def __init__(self, *operands, **koperands):
    self.params = {'opcode': 'and'}
    self.machine_inst = ptx_insts.x3
    ptxLogic3Instruction.__init__(self, *operands, **koperands)

class or_(ptxLogic3Instruction):
  def __init__(self, *operands, **koperands):
    self.params = {'opcode': 'or'}
    self.machine_inst = ptx_insts.x3
    ptxLogic3Instruction.__init__(self, *operands, **koperands)

class xor(ptxLogic3Instruction):
  def __init__(self, *operands, **koperands):
    self.params = {'opcode': 'xor'}
    self.machine_inst = ptx_insts.x3
    ptxLogic3Instruction.__init__(self, *operands, **koperands)

class not_(ptxLogic2Instruction):
  def __init__(self, *operands, **koperands):
    self.params = {'opcode': 'not'}
    self.machine_inst = ptx_insts.x2
    ptxLogic2Instruction.__init__(self, *operands, **koperands)

class cnot(ptxInstruction):
  def __init__(self, *operands, **koperands):
    self.params = {'opcode': 'cnot'}
    self.machine_inst = ptx_insts.x2
    d = ptxOperand(operands[0])
    s0 = ptxOperand(operands[1])
    if d.getWidth() == s0.getWidth():
      if d.getWidth() in (16, 32, 64):
        self.params['type'] =  'b' + str(d.getWidth())
      else:
        raise ("Invalid type for this instruction")
    else:
      raise Exception("Widths for all operands must match")
    ptxInstruction.__init__(self, *operands, **koperands)

class shl(ptxInstruction):
  def __init__(self, *operands, **koperands):
    self.params = {'opcode': 'shl'}
    self.machine_inst = ptx_insts.x3
    d = ptxOperand(operands[0])
    s0 = ptxOperand(operands[1])
    s1 = ptxOperand(operands[2])
    if (d.getWidth() == s0.getWidth()) and (s1.getWidth() == 32 or s1.isImmediate()):
      if d.getWidth() in (16, 32, 64):
        self.params['type'] =  'b' + str(d.getWidth())
      else:
        raise Exception("Invalid type for this instruction")
    else:
      raise Exception("Widths for all operands must match")
    ptxInstruction.__init__(self, *operands, **koperands)

class shr(ptxInstruction):
  def __init__(self, *operands, **koperands):
    self.params = {'opcode': 'shr'}
    self.machine_inst = ptx_insts.x3
    d = ptxOperand(operands[0])
    s0 = ptxOperand(operands[1])
    s1 = ptxOperand(operands[2])
    inst_type = guess_type_from_operands_3(d, s0, s1)
    if inst_type in ('b16', 'b32', 'b64', 'u16', 'u32', 'u64', 's16', 's32', 's64'):
      self.params['type'] = inst_type
    else:
      raise Exception("Invalid type for this instruction")
    ptxInstruction.__init__(self, *operands, **koperands)

# Data Movement and Conversion #

class mov(ptxMovementInstruction):
  def __init__(self, *operands, **koperands):
    self.params = {'opcode': 'mov'}
    self.machine_inst = ptx_insts.d_sora
    d = ptxOperand(operands[0])
    self.params['type'] =  d.getType()
    self.valid_types = ('b16', 'b32', 'b64', 'u16', 'u32', 'u64', 's16', 's32', 's64', 'f32', 'f64', 'pred')
    ptxMovementInstruction.__init__(self, *operands, **koperands)

# TODO: Add vector support
class ld(ptxMovementInstruction):
  def __init__(self, *operands, **koperands):
    self.params = {'opcode': 'ld'}
    self.machine_inst = ptx_insts.space_d_a
    d = ptxOperand(operands[1])
    self.params['type'] =  d.getType()
    self.valid_types = ('b8', 'b16', 'b32', 'b64', 'u8', 'u16', 'u32', 'u64', 's8', 's16', 's32', 's64', 'f32', 'f64', 'pred')
    ptxInstruction.__init__(self, *operands, **koperands)

class st(ptxMovementInstruction):
  def __init__(self, *operands, **koperands):
    self.params = {'opcode': 'st'}
    self.machine_inst = ptx_insts.space_d_r
    r = ptxOperand(operands[2])
    self.params['type'] =  r.getType()
    self.valid_types = ('b8', 'b16', 'b32', 'b64', 'u8', 'u16', 'u32', 'u64', 's8', 's16', 's32', 's64', 'f32', 'f64', 'pred')
    ptxInstruction.__init__(self, *operands, **koperands)

class cvt(ptxMovementInstruction):
  def __init__(self, *operands, **koperands):
    self.params = {'opcode': 'cvt'}
    self.machine_inst = ptx_insts.r_s_d_a
    d = ptxOperand(operands[0])
    self.params['dtype'] =  d.getType()
    a = ptxOperand(operands[1])
    self.params['atype'] =  a.getType()
    self.valid_types = ('u8', 'u16', 'u32', 'u64', 's8', 's16', 's32', 's64', 'f16', 'f32', 'f64')
    ptxInstruction.__init__(self, *operands, **koperands)

# Texture Instruction ############

# TODO: tex instruction

# Control Flow Instructions ######

# Note that some aspects of control flow are implicit with corepy,
# so not all "instructions" mentioned in the manual are present
# here. Specifically, "@" is ommitted.
# (On the other hand, I originally was going to leave out "{" and "}",
# but since they do actually stand alone (as opposed to "@" which modifies
# other instructions) and they are necessary for certain things,
# might as well include them...



# TODO?: {
# TODO?: }
# TODO?: call


class bra(ptxInstruction):
  def __init__(self, *operands, **koperands):
    self.params = {'opcode': 'bra'}
    self.machine_inst = ptx_insts.uni_lorr
    ptxInstruction.__init__(self, *operands, **koperands)

class ret(ptxInstruction):
  def __init__(self, *operands, **koperands):
    self.params = {'opcode': 'ret'}
    self.machine_inst = ptx_insts.x0_uni
    ptxInstruction.__init__(self, *operands, **koperands)

class exit(ptxInstruction):
  def __init__(self, *operands, **koperands):
    self.params = {'opcode': 'exit'}
    self.machine_inst = ptx_insts.x0
    ptxInstruction.__init__(self, *operands, **koperands)


# Parallel Synchronization and Communication Instructions ###########

class bar_sync(ptxInstruction):
  def __init__(self, *operands, **koperands):
    self.params = {'opcode': 'bar.sync'}
    self.machine_inst = ptx_insts.bar
    ptxInstruction.__init__(self, *operands, **koperands)
  
class atom(ptxMovementInstruction):
  def __init__(self, *operands, **koperands):
    self.params = {'opcode': 'atom'}
    space = operands[0]
    if space not in ('global', 'shared'):
      raise Exception('Space for atomic reduction must be global or shared')
    operation = operands[1]
    if operation not in ('and', 'or', 'xor', 'cas', 'exch', 'add', 'inc', 'dec', 'min', 'max'):
      raise Exception('Operation for atomic reduction not recognized or not valid')

    d = ptxOperand(operands[2])
    a = ptxOperand(operands[3])
    s0 = ptxOperand(operands[4])

    if operation == 'cas':
      self.machine_inst = ptx_insts.space_op_d_a_b_c
    else:
      self.machine_inst = ptx_insts.space_op_d_a_b

    self.params['type'] =  d.getType()
    self.valid_types = ('b32', 'b64', 'u32', 'u64', 's32', 'f32')
    if self.params['type'] not in self.valid_types:
      raise Exception("Invalid type for this instruction")
    ptxInstruction.__init__(self, *operands, **koperands)

class red(ptxMovementInstruction):
  def __init__(self, *operands, **koperands):
    self.params = {'opcode': 'red'}
    self.machine_inst = ptx_insts.space_op_a_b

    space = operands[0]
    if space not in ('global', 'shared'):
      raise Exception('Space for atomic reduction must be global or shared')
    operation = operands[1]
    if operation not in ('and', 'or', 'xor', 'cas', 'exch', 'add', 'inc', 'dec', 'min', 'max'):
      raise Exception('Operation for atomic reduction not recognized or not valid')
    #a = ptxOperand(operands[3])
    s0 = ptxOperand(operands[4])

    self.params['type'] =  s0.getType()
    self.valid_types = ('b32', 'b64', 'u32', 'u64', 's32', 'f32')
    if self.params['type'] not in self.valid_types:
      raise Exception("Invalid type for this instruction")
    ptxInstruction.__init__(self, *operands, **koperands)

class vote(ptxMovementInstruction):
  def __init__(self, *operands, **koperands):
    self.params = {'opcode': 'vote'}
    self.machine_inst = ptx_insts.mode_x2
    mode = operands[0]
    if mode not in ('all', 'any', 'uni'):
      raise Exception('Mode for vote not valid')
    ptxInstruction.__init__(self, *operands, **koperands)

# Floating Point Instructions #######################################

class rcp(ptxALU2Instruction):
  def __init__(self, *operands, **koperands):
    self.params = {'opcode': 'rcp'}
    self.machine_inst = ptx_insts.x2
    ptxALU2Instruction.__init__(self, *operands, **koperands)

class sqrt(ptxALU2Instruction):
  def __init__(self, *operands, **koperands):
    self.params = {'opcode': 'sqrt'}
    self.machine_inst = ptx_insts.x2
    ptxALU2Instruction.__init__(self, *operands, **koperands)

class rsqrt(ptxALU2Instruction):
  def __init__(self, *operands, **koperands):
    self.params = {'opcode': 'rsqrt'}
    self.machine_inst = ptx_insts.x2
    ptxALU2Instruction.__init__(self, *operands, **koperands)

class sin(ptxALU2Instruction):
  def __init__(self, *operands, **koperands):
    self.params = {'opcode': 'sin', 'type': 'f32'}
    self.machine_inst = ptx_insts.x2
    ptxALU2Instruction.__init__(self, *operands, **koperands)

class cos(ptxALU2Instruction):
  def __init__(self, *operands, **koperands):
    self.params = {'opcode': 'cos', 'type': 'f32'}
    self.machine_inst = ptx_insts.x2
    ptxALU2Instruction.__init__(self, *operands, **koperands)

class lg2(ptxALU2Instruction):
  def __init__(self, *operands, **koperands):
    self.params = {'opcode': 'lg2', 'type': 'f32'}
    self.machine_inst = ptx_insts.x2
    ptxALU2Instruction.__init__(self, *operands, **koperands)

class ex2(ptxALU2Instruction):
  def __init__(self, *operands, **koperands):
    self.params = {'opcode': 'ex2', 'type': 'f32'}
    self.machine_inst = ptx_insts.x2
    ptxALU2Instruction.__init__(self, *operands, **koperands)

# Miscellaneous Instructions ######################################

class trap(ptxInstruction):
  def __init__(self, *operands, **koperands):
    self.params = {'opcode': 'trap'}
    self.machine_inst = ptx_insts.x0
    ptxInstruction.__init__(self, *operands, **koperands)

class brkpt(ptxInstruction):
  def __init__(self, *operands, **koperands):
    self.params = {'opcode': 'brkpt'}
    self.machine_inst = ptx_insts.x0
    ptxInstruction.__init__(self, *operands, **koperands)


































# class add(ptxInstruction):
#   def __init__(self, *operands, **koperands):
#     self.params = {'opcode': 'add'}
#     ptxInstruction.__init__(self, *operands, **koperands)

#   def check(self, *operands, **koperands):
#     d  = ptxOperand(operands[0])
#     s0 = ptxOperand(operands[1])
#     s1 = ptxOperand(operands[2])

#     inst_type = getArithmeticType3(*operands, **koperands)

#     self.params['type'] =  inst_type
# #     self._operands['d'] = d
# #     self._operands['s0'] = s0
# #     self._operands['s1'] = s1

# #     if isIntegerType(inst_type):
# #       if 'cc' in koperands and koperands['cc'] == True:
# #         if inst_type not in ('u32', 's32'):
# #           raise "Not a valid operand type for use with '.cc' option."
# #         self.machine_inst = cc_x3
# #         self._operands['cc'] =  True
# #       else:
# #         self.machine_inst = s_x3
# #         if 'sat' in koperands and koperands['sat'] == True:
# #           self._operands['sat'] =  True
          
# #     elif isFloatType(inst_type):
# #       self.machine_inst = r_s_x3
# #       if 'sat' in koperands and koperands['sat'] == True:
# #         self._operands['sat'] =  True
# #       if 'rnd' in koperands and koperands['rnd'] != None:
# #         if koperands['rnd] in ('rn', 'rz', 'rm', 'rp'):
# #           self._operands['rnd'] =  koperands['rnd']
          
#     return True

# class addc(ptxInstruction):
#   def __init__(self, *operands, **koperands):
#     self.params = {'opcode': 'addc'}
#     ptxInstruction.__init__(self, *operands, **koperands)

#   def check(self, *operands, **koperands):
#     d  = ptxOperand(operands[0])
#     s0 = ptxOperand(operands[1])
#     s1 = ptxOperand(operands[2])

#     inst_type = getArithmeticType3(*operands, **koperands)
#     if inst_type not in ('u32', 's32'):
#       raise "Invalid type for this instruction"        

#     self.params = {'opcode': 'addc', 'type': inst_type}
#     self.machine_inst = cc_x3

#     if 'cc' in koperands and koperands['cc'] == True:
#         self.params['cc'] =  True

#     return True

# class sub(add):
#   def __init__(self, *operands, **koperands):
#     self.params = {'opcode': 'sub'}
#     ptxInstruction.__init__(self, *operands, **koperands)

# class mul(ptxInstruction):
#   def __init__(self, *operands, **koperands):
#     self.params = {'opcode': 'mul'}
#     ptxInstruction.__init__(self, *operands, **koperands)

#   def check(self, *operands, **koperands):
#     d  = ptxOperand(operands[0])
#     s0 = ptxOperand(operands[1])
#     s1 = ptxOperand(operands[2])

#     inst_type = getArithmeticType3(*operands, **koperands)

#     self.params['type'] =  inst_type

#     if isIntegerType(inst_type):
#       self.machine_inst = hlw_x3
#       if 'hlw' in koperands and koperands['hlw'] != None:
#         if koperands['hlw'] == 'hi':
#           self.params['hlw'] =  'hi'
#         elif koperands['hlw'] == 'lo':
#           self.params['hlw'] =  'lo'
#         elif koperands['hlw'] == 'wide':
#           self.params['hlw'] =  'wide'
#         else:
#           raise "Invalid value for 'hlw'"
          
#     elif isFloatType(inst_type):
#       self.machine_inst = r_s_x3
#       if 'sat' in koperands and koperands['sat'] == True:
#         self.params['sat'] =  True
#       if 'rnd' in koperands and koperands['rnd'] != None:
#         if koperands['rnd] in ('rn', 'rz', 'rm', 'rp'):
#           self.params['rnd'] =  koperands['rnd']
          
#     return True

# class mad(ptxInstruction):
#   def __init__(self, *operands, **koperands):
#     self.params = {'opcode': 'mad'}
#     ptxInstruction.__init__(self, *operands, **koperands)

#   def check(self, *operands, **koperands):
#     d  = ptxOperand(operands[0])
#     s0 = ptxOperand(operands[1])
#     s1 = ptxOperand(operands[2])
#     s2 = ptxOperand(operands[3])

#     inst_type = getArithmeticType4(*operands, **koperands)

#     self.params['type'] =  inst_type

#     if isIntegerType(inst_type):
#       self.machine_inst = hlws_x4
#       if 'hlw' in koperands and koperands['hlw'] != None:
#         if koperands['hlw'] == 'hi':
#           self.params['hlw'] =  'hi'
#         elif koperands['hlw'] == 'lo':
#           self.params['hlw'] =  'lo'
#         elif koperands['hlw'] == 'wide':
#           self.params['hlw'] =  'wide'
#         else:
#           raise "Invalid value for 'hlw'"
#       if 'sat' in koperands and koperands['sat'] == True:
#         self.params['sat'] =  True

#     elif isFloatType(inst_type):
#       self.machine_inst = r_s_x4
#       if 'sat' in koperands and koperands['sat'] == True:
#         self.params['sat'] =  True
#       if 'rnd' in koperands and koperands['rnd'] != None:
#         if koperands['rnd] in ('rn', 'rz', 'rm', 'rp'):
#           self.params['rnd'] =  koperands['rnd']
          
#     return True


if __name__ == '__main__':
  import corepy.arch.ptx.isa as isa
  #import corepy.arch.ptx.platform as env

  #code = env.InstructionStream()
  #set_active_code(code)

  r1 = regs.ptxVariable('reg', 'u32', 'r1')
  r2 = regs.ptxVariable('reg', 'u32', 'r2')
  r3 = regs.ptxVariable('reg', 'u32', 'r3')
  r4 = regs.ptxVariable('reg', 'u32', 'r4')

  #x = add(r3, r2, r1, ignore_active = True)
  x = isa.add(r3, r2, r1)
  print x.render()
  y = isa.mov(r2, r1)
  print y.render()
  a = regs.ptxAddress(r4)
  z = isa.ld('param', r1, a)
  print z.render()

  f1 = regs.ptxVariable('reg', 'f32', 'f1')
  f2 = regs.ptxVariable('reg', 'f32', 'f2')
  f3 = regs.ptxVariable('reg', 'f32', 'f3')
  a = isa.add(f3, f2, f1)
  print a.render()
