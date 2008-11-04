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

from corepy.spre.spe import Register, Instruction, DispatchInstruction, InstructionOperand, Label
from corepy.arch.x86.types.registers import GPRegister8, GPRegister16, GPRegister32, FPRegister, MMXRegister, XMMRegister
from corepy.arch.x86.lib.memory import MemoryReference, MemRef

from x86_fields import *
from x86_insts import *

__annoy__ = True

__doc__="""
x86 Instruction Set Architecture (ISA).

To use, import this module and call the Instructions as Python
functions to generate a properly coded version.  For example, to
create an add instruction:

import corepy.arch.x86.isa as isa
import corepy.arch.x86.types.registers as regs

inst = isa.add(regs.eax, regs.ebx) # add ebx to eax

Operands are in the same order as presented in the architecture manuals.

For a complete reference and details for all instructions, please
referer to: 

'Intel 64 and IA-32 Architectures Software Developer's Manual' or
'AMD64 Architecture Programmer's Manual'.

URL (valid as of Sept 21, 2007):
    http://www.intel.com/products/processor/manuals/index.htm
    http://developer.amd.com/devguides.jsp
"""

# ------------------------------
# x86 Registers
# ------------------------------

# reg num: named register
gp8_map  = {0: al_t, 1: cl_t}
gp16_map = {0: ax_t, 2: dx_t}
gp32_map = {0: eax_t}
fp_map = {0: st0_t}

def x86_imm_operand_type(op):
  if isinstance(op, (int, long)):
    if op == 1:
      return one_t
    elif rel8off_t.fits(op):
      return rel8off_t
    elif imm8_t.fits(op):
      return imm8_t
    elif imm16_t.fits(op):
      return imm16_t
    elif rel32off_t.fits(op):
      return rel32off_t
    elif imm32_t.fits(op):
      return imm32_t
    else:
      raise Exception('int/long operand too large: %d' % op)

  return 

def x86_reg_operand_type(op):    
  if isinstance(op, GPRegister8):
    if op.reg in gp8_map.keys():
      return gp8_map[op.reg]
    else:
      return reg8_t
  elif isinstance(op, GPRegister16):
    if op.reg in gp16_map.keys():
      return gp16_map[op.reg]
    else:
      return reg16_t
  elif isinstance(op, GPRegister32):
    if op.reg in gp32_map.keys():
      return gp32_map[op.reg]
    else:
      return reg32_t
  elif isinstance(op, FPRegister):
    if op.reg in fp_map.keys():
      return fp_map[op.reg]
    else:
      return regst_t
  elif isinstance(op, MMXRegister):
    return mmx_t
  elif isinstance(op, XMMRegister):
    return xmm_t
  return

def x86_reloff_operand_type(op):
  if isinstance(op, (int, long)):
    if rel8off_t.fits(op):
      return rel8off_t
    elif rel32off_t.fits(op):
      return rel32off_t
    else:
      raise Exception('int/long operand too large: %d' % op)
  elif isinstance(op, Label):
    if lbl8off_t.fits(op):
      return lbl8off_t
    elif lbl16off_t.fits(op):
      return lbl16off_t
    elif lbl32off_t.fits(op):
      return lbl32off_t
    else:
      raise Exception('int/long operand too large: %s' % str(op))
  return

def x86_mem_operand_type(op):
  if isinstance(op, MemoryReference):
    if op.data_size == 32:
      return mem32_t
    elif op.data_size == 16:
      return mem16_t
    elif op.data_size == 8:
      return mem8_t
    if op.data_size == 64:
      return mem64_t
    if op.data_size == 80:
      return mem80_t
    if op.data_size == 128:
      return mem128_t
    if op.data_size == 228:
      return mem228_t
    if op.data_size == 512:
      return mem512_t
    if op.data_size == 752:
      return mem752_t
    if op.data_size != None and __annoy__:
        print "MemRef has size of %s bits, this is may not work as expected." % str(op.data_size)
    return mem_t
  return

def x86_type(op):
  t = x86_imm_operand_type(op)

  if t is None:
    t = x86_reg_operand_type(op)
    if t is None:
      t = x86_reloff_operand_type(op)
      if t is None:
        t = x86_mem_operand_type(op)
  return t

#def x86_one_type(op):
#  if op == 1:
#    return one_t
#  
#  t = x86_imm_operand_type(op)
#
#  if t is None:
#    t = x86_reg_operand_type(op)
#
#  return t

#def x86_reloff_type(op):
#  t = x86_reloff_operand_type(op)
#
#  if t is None:
#    t = x86_reg_operand_type(op)
#
#  return t


class x86Instruction(Instruction): pass
class x86DispatchInstruction(DispatchInstruction):
  type_id = [x86_type]


# ------------------------------
# x86 Instructions
# ------------------------------

# Currently 16bit versions of instructions have separate operand
# functions, and the size-override prefix is in the opcode, so protected
# (32bit default) mode is assumed here.


class adc(x86DispatchInstruction):
  dispatch = (
    (reg32_imm8,          {'opcode':[0x83],             'modrm':0x10}),
    (mem32_imm8,          {'opcode':[0x83],             'modrm':0x10}),
    (eax_imm32,           {'opcode':[0x15],             'modrm':None}),
    (reg32_imm32,         {'opcode':[0x81],             'modrm':0x10}),
    (mem32_imm32,         {'opcode':[0x81],             'modrm':0x10}),
    (reg32_reg32,         {'opcode':[0x11],             'modrm':None}),
    (mem32_reg32,         {'opcode':[0x11],             'modrm':None}),
    (reg32_mem32,         {'opcode':[0x13],             'modrm':None}),
    (reg16_imm8,          {'opcode':[0x66, 0x83],       'modrm':0x10}),
    (mem16_imm8,          {'opcode':[0x66, 0x83],       'modrm':0x10}),
    (ax_imm16,            {'opcode':[0x66, 0x15],       'modrm':None}),
    (reg16_imm16,         {'opcode':[0x66, 0x81],       'modrm':0x10}),
    (mem16_imm16,         {'opcode':[0x66, 0x81],       'modrm':0x10}),
    (reg16_reg16,         {'opcode':[0x66, 0x11],       'modrm':None}),
    (mem16_reg16,         {'opcode':[0x66, 0x11],       'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x13],       'modrm':None}),
    (al_imm8,             {'opcode':[0x14],             'modrm':None}),
    (reg8_imm8,           {'opcode':[0x80],             'modrm':0x10}),
    (mem8_imm8,           {'opcode':[0x80],             'modrm':0x10}),
    (reg8_reg8,           {'opcode':[0x10],             'modrm':None}),
    (mem8_reg8,           {'opcode':[0x10],             'modrm':None}),
    (reg8_mem8,           {'opcode':[0x12],             'modrm':None}))
  
class add(x86DispatchInstruction):
  dispatch = (
    (reg32_imm8,          {'opcode':[0x83],             'modrm':0x00}),
    (mem32_imm8,          {'opcode':[0x83],             'modrm':0x00}),
    (eax_imm32,           {'opcode':[0x05],             'modrm':None}),
    (reg32_imm32,         {'opcode':[0x81],             'modrm':0x00}),
    (mem32_imm32,         {'opcode':[0x81],             'modrm':0x00}),
    (reg32_reg32,         {'opcode':[0x01],             'modrm':None}),
    (mem32_reg32,         {'opcode':[0x01],             'modrm':None}),
    (reg32_mem32,         {'opcode':[0x03],             'modrm':None}),
    (reg16_imm8,          {'opcode':[0x66, 0x83],       'modrm':0x00}),
    (mem16_imm8,          {'opcode':[0x66, 0x83],       'modrm':0x00}),
    (ax_imm16,            {'opcode':[0x66, 0x05],       'modrm':None}),
    (reg16_imm16,         {'opcode':[0x66, 0x81],       'modrm':0x00}),
    (mem16_imm16,         {'opcode':[0x66, 0x81],       'modrm':0x00}),
    (reg16_reg16,         {'opcode':[0x66, 0x01],       'modrm':None}),
    (mem16_reg16,         {'opcode':[0x66, 0x01],       'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x03],       'modrm':None}),
    (al_imm8,             {'opcode':[0x04],             'modrm':None}),
    (reg8_imm8,           {'opcode':[0x80],             'modrm':0x00}),
    (mem8_imm8,           {'opcode':[0x80],             'modrm':0x00}),
    (reg8_reg8,           {'opcode':[0x00],             'modrm':None}),
    (mem8_reg8,           {'opcode':[0x00],             'modrm':None}),
    (reg8_mem8,           {'opcode':[0x02],             'modrm':None}))
    
class and_(x86DispatchInstruction):
  dispatch = (
    (reg32_imm8,          {'opcode':[0x83],             'modrm':0x20}),
    (mem32_imm8,          {'opcode':[0x83],             'modrm':0x20}),
    (eax_imm32,           {'opcode':[0x25],             'modrm':None}),
    (reg32_imm32,         {'opcode':[0x81],             'modrm':0x20}),
    (mem32_imm32,         {'opcode':[0x81],             'modrm':0x20}),
    (reg32_reg32,         {'opcode':[0x21],             'modrm':None}),
    (mem32_reg32,         {'opcode':[0x21],             'modrm':None}),
    (reg32_mem32,         {'opcode':[0x23],             'modrm':None}),
    (reg16_imm8,          {'opcode':[0x66, 0x83],       'modrm':0x20}),
    (mem16_imm8,          {'opcode':[0x66, 0x83],       'modrm':0x20}),
    (ax_imm16,            {'opcode':[0x66, 0x25],       'modrm':None}),
    (reg16_imm16,         {'opcode':[0x66, 0x81],       'modrm':0x20}),
    (mem16_imm16,         {'opcode':[0x66, 0x81],       'modrm':0x20}),
    (reg16_reg16,         {'opcode':[0x66, 0x21],       'modrm':None}),
    (mem16_reg16,         {'opcode':[0x66, 0x21],       'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x23],       'modrm':None}),
    (al_imm8,             {'opcode':[0x24],             'modrm':None}),
    (reg8_imm8,           {'opcode':[0x80],             'modrm':0x20}),
    (mem8_imm8,           {'opcode':[0x80],             'modrm':0x20}),
    (reg8_reg8,           {'opcode':[0x20],             'modrm':None}),
    (mem8_reg8,           {'opcode':[0x20],             'modrm':None}),
    (reg8_mem8,           {'opcode':[0x22],             'modrm':None}))

class bsf(x86DispatchInstruction):
  dispatch = (
    (reg32_reg32,         {'opcode':[0x0F, 0xBC],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0xBC],       'modrm':None}),
    (reg16_reg16,         {'opcode':[0x66, 0x0F, 0xBC], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0xBC], 'modrm':None}))
    
class bsr(x86DispatchInstruction):
  dispatch = (
    (reg32_reg32,         {'opcode':[0x0F, 0xBD],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0xBD],       'modrm':None}),
    (reg16_reg16,         {'opcode':[0x66, 0x0F, 0xBD], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0xBD], 'modrm':None}))
  
class bt(x86DispatchInstruction):
  dispatch = (
    (reg32_imm8,          {'opcode':[0x0F, 0xBA],       'modrm':0x20}),
    (mem32_imm8,          {'opcode':[0x0F, 0xBA],       'modrm':0x20}),
    (reg32_reg32,         {'opcode':[0x0F, 0xA3],       'modrm':None}),
    (mem32_reg32,         {'opcode':[0x0F, 0xA3],       'modrm':None}),
    (reg16_imm8,          {'opcode':[0x66, 0x0F, 0xBA], 'modrm':0x20}),
    (mem16_imm8,          {'opcode':[0x66, 0x0F, 0xBA], 'modrm':0x20}),
    (reg16_reg16,         {'opcode':[0x66, 0x0F, 0xA3], 'modrm':None}),
    (mem16_reg16,         {'opcode':[0x66, 0x0F, 0xA3], 'modrm':None}))
  
class btc(x86DispatchInstruction):
  dispatch = (
    (reg32_imm8,          {'opcode':[0x0F, 0xBA],       'modrm':0x38}),
    (mem32_imm8,          {'opcode':[0x0F, 0xBA],       'modrm':0x38}),
    (reg32_reg32,         {'opcode':[0x0F, 0xBB],       'modrm':None}),
    (mem32_reg32,         {'opcode':[0x0F, 0xBB],       'modrm':None}),
    (reg16_imm8,          {'opcode':[0x66, 0x0F, 0xBA], 'modrm':0x38}),
    (mem16_imm8,          {'opcode':[0x66, 0x0F, 0xBA], 'modrm':0x38}),
    (reg16_reg16,         {'opcode':[0x66, 0x0F, 0xBB], 'modrm':None}),
    (mem16_reg16,         {'opcode':[0x66, 0x0F, 0xBB], 'modrm':None}))
  
class btr(x86DispatchInstruction):
  dispatch = (
    (reg32_imm8,          {'opcode':[0x0F, 0xBA],       'modrm':0x30}),
    (mem32_imm8,          {'opcode':[0x0F, 0xBA],       'modrm':0x30}),
    (reg32_reg32,         {'opcode':[0x0F, 0xB3],       'modrm':None}),
    (mem32_reg32,         {'opcode':[0x0F, 0xB3],       'modrm':None}),
    (reg16_imm8,          {'opcode':[0x66, 0x0F, 0xBA], 'modrm':0x30}),
    (mem16_imm8,          {'opcode':[0x66, 0x0F, 0xBA], 'modrm':0x30}),
    (reg16_reg16,         {'opcode':[0x66, 0x0F, 0xB3], 'modrm':None}),
    (mem16_reg16,         {'opcode':[0x66, 0x0F, 0xB3], 'modrm':None}))
  
class bts(x86DispatchInstruction):
  dispatch = (
    (reg32_imm8,          {'opcode':[0x0F, 0xBA],       'modrm':0x28}),
    (mem32_imm8,          {'opcode':[0x0F, 0xBA],       'modrm':0x28}),
    (reg32_reg32,         {'opcode':[0x0F, 0xAB],       'modrm':None}),
    (mem32_reg32,         {'opcode':[0x0F, 0xAB],       'modrm':None}),
    (reg16_imm8,          {'opcode':[0x66, 0x0F, 0xBA], 'modrm':0x28}),
    (mem16_imm8,          {'opcode':[0x66, 0x0F, 0xBA], 'modrm':0x28}),
    (reg16_reg16,         {'opcode':[0x66, 0x0F, 0xAB], 'modrm':None}),
    (mem16_reg16,         {'opcode':[0x66, 0x0F, 0xAB], 'modrm':None}))
    
class bswap(x86Instruction):
  machine_inst = reg32
  params = {'opcode':[0x0F, 0xC8],       'modrm':None}
  
class call(x86DispatchInstruction):
  dispatch = (
    (lbl32off,            {'opcode':[0xE8],             'modrm':None}),
    (rel32off,            {'opcode':[0xE8],             'modrm':None}),
    (reg32,               {'opcode':[0xFF],             'modrm':0x10}),
    (mem32,               {'opcode':[0xFF],             'modrm':0x10}),
    (reg16,               {'opcode':[0x66, 0xFF],       'modrm':0x10}),
    (mem16,               {'opcode':[0x66, 0xFF],       'modrm':0x10}))
  
class cbw(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0x66, 0x98],       'modrm':None}
  
class cwde(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0x98],             'modrm':None}
  
class cwd(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0x66, 0x99],       'modrm':None}
  
class cdq(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0x99],             'modrm':None}
  
class clc(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0xF8],             'modrm':None}
  
class cld(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0xFC],             'modrm':None}
  
class clflush(x86Instruction):
  machine_inst = mem8
  params = {'opcode':[0x0F, 0xAE],       'modrm':0x38}
  
class cmc(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0xF5],             'modrm':None}
  
class cmovo(x86DispatchInstruction):
  dispatch = (
    (reg32_reg32_rev,     {'opcode':[0x0F, 0x40],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0x40],       'modrm':None}),
    (reg16_reg16_rev,     {'opcode':[0x66, 0x0F, 0x40], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0x40], 'modrm':None}))
    
class cmovno(x86DispatchInstruction):
  dispatch = (
    (reg32_reg32_rev,     {'opcode':[0x0F, 0x41],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0x41],       'modrm':None}),
    (reg16_reg16_rev,     {'opcode':[0x66, 0x0F, 0x41], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0x41], 'modrm':None}))
    
class cmovb(x86DispatchInstruction):
  dispatch = (
    (reg32_reg32_rev,     {'opcode':[0x0F, 0x42],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0x42],       'modrm':None}),
    (reg16_reg16_rev,     {'opcode':[0x66, 0x0F, 0x42], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0x42], 'modrm':None}))
    
class cmovc(x86DispatchInstruction):
  dispatch = (
    (reg32_reg32_rev,     {'opcode':[0x0F, 0x42],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0x42],       'modrm':None}),
    (reg16_reg16_rev,     {'opcode':[0x66, 0x0F, 0x42], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0x42], 'modrm':None}))
    
class cmovnae(x86DispatchInstruction):
  dispatch = (
    (reg32_reg32_rev,     {'opcode':[0x0F, 0x42],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0x42],       'modrm':None}),
    (reg16_reg16_rev,     {'opcode':[0x66, 0x0F, 0x42], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0x42], 'modrm':None}))
    
class cmovnb(x86DispatchInstruction):
  dispatch = (
    (reg32_reg32_rev,     {'opcode':[0x0F, 0x43],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0x43],       'modrm':None}),
    (reg16_reg16_rev,     {'opcode':[0x66, 0x0F, 0x43], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0x43], 'modrm':None}))
    
class cmovnc(x86DispatchInstruction):
  dispatch = (
    (reg32_reg32_rev,     {'opcode':[0x0F, 0x43],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0x43],       'modrm':None}),
    (reg16_reg16_rev,     {'opcode':[0x66, 0x0F, 0x43], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0x43], 'modrm':None}))
    
class cmovae(x86DispatchInstruction):
  dispatch = (
    (reg32_reg32_rev,     {'opcode':[0x0F, 0x43],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0x43],       'modrm':None}),
    (reg16_reg16_rev,     {'opcode':[0x66, 0x0F, 0x43], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0x43], 'modrm':None}))
  
class cmovz(x86DispatchInstruction):
  dispatch = (
    (reg32_reg32_rev,     {'opcode':[0x0F, 0x44],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0x44],       'modrm':None}),
    (reg16_reg16_rev,     {'opcode':[0x66, 0x0F, 0x44], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0x44], 'modrm':None}))
  
class cmove(x86DispatchInstruction):
  dispatch = (
    (reg32_reg32_rev,     {'opcode':[0x0F, 0x44],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0x44],       'modrm':None}),
    (reg16_reg16_rev,     {'opcode':[0x66, 0x0F, 0x44], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0x44], 'modrm':None}))
    
class cmovnz(x86DispatchInstruction):
  dispatch = (
    (reg32_reg32_rev,     {'opcode':[0x0F, 0x45],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0x45],       'modrm':None}),
    (reg16_reg16_rev,     {'opcode':[0x66, 0x0F, 0x45], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0x45], 'modrm':None}))
  
class cmovne(x86DispatchInstruction):
  dispatch = (
    (reg32_reg32_rev,     {'opcode':[0x0F, 0x45],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0x45],       'modrm':None}),
    (reg16_reg16_rev,     {'opcode':[0x66, 0x0F, 0x45], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0x45], 'modrm':None}))
    
class cmovbe(x86DispatchInstruction):
  dispatch = (
    (reg32_reg32_rev,     {'opcode':[0x0F, 0x46],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0x46],       'modrm':None}),
    (reg16_reg16_rev,     {'opcode':[0x66, 0x0F, 0x46], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0x46], 'modrm':None}))
    
class cmovna(x86DispatchInstruction):
  dispatch = (
    (reg32_reg32_rev,     {'opcode':[0x0F, 0x46],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0x46],       'modrm':None}),
    (reg16_reg16_rev,     {'opcode':[0x66, 0x0F, 0x46], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0x46], 'modrm':None}))
    
class cmovnbe(x86DispatchInstruction):
  dispatch = (
    (reg32_reg32_rev,     {'opcode':[0x0F, 0x47],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0x47],       'modrm':None}),
    (reg16_reg16_rev,     {'opcode':[0x66, 0x0F, 0x47], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0x47], 'modrm':None}))
    
class cmova(x86DispatchInstruction):
  dispatch = (
    (reg32_reg32_rev,     {'opcode':[0x0F, 0x47],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0x47],       'modrm':None}),
    (reg16_reg16_rev,     {'opcode':[0x66, 0x0F, 0x47], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0x47], 'modrm':None}))
    
class cmovs(x86DispatchInstruction):
  dispatch = (
    (reg32_reg32_rev,     {'opcode':[0x0F, 0x48],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0x48],       'modrm':None}),
    (reg16_reg16_rev,     {'opcode':[0x66, 0x0F, 0x48], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0x48], 'modrm':None}))
  
class cmovns(x86DispatchInstruction):
  dispatch = (
    (reg32_reg32_rev,     {'opcode':[0x0F, 0x49],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0x49],       'modrm':None}),
    (reg16_reg16_rev,     {'opcode':[0x66, 0x0F, 0x49], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0x49], 'modrm':None}))
    
class cmovp(x86DispatchInstruction):
  dispatch = (
    (reg32_reg32_rev,     {'opcode':[0x0F, 0x4A],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0x4A],       'modrm':None}),
    (reg16_reg16_rev,     {'opcode':[0x66, 0x0F, 0x4A], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0x4A], 'modrm':None}))
  
class cmovpe(x86DispatchInstruction):
  dispatch = (
    (reg32_reg32_rev,     {'opcode':[0x0F, 0x4A],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0x4A],       'modrm':None}),
    (reg16_reg16_rev,     {'opcode':[0x66, 0x0F, 0x4A], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0x4A], 'modrm':None}))
  
class cmovnp(x86DispatchInstruction):
  dispatch = (
    (reg32_reg32_rev,     {'opcode':[0x0F, 0x4B],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0x4B],       'modrm':None}),
    (reg16_reg16_rev,     {'opcode':[0x66, 0x0F, 0x4B], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0x4B], 'modrm':None}))
    
class cmovpo(x86DispatchInstruction):
  dispatch = (
    (reg32_reg32_rev,     {'opcode':[0x0F, 0x4B],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0x4B],       'modrm':None}),
    (reg16_reg16_rev,     {'opcode':[0x66, 0x0F, 0x4B], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0x4B], 'modrm':None}))
    
class cmovl(x86DispatchInstruction):
  dispatch = (
    (reg32_reg32_rev,     {'opcode':[0x0F, 0x4C],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0x4C],       'modrm':None}),
    (reg16_reg16_rev,     {'opcode':[0x66, 0x0F, 0x4C], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0x4C], 'modrm':None}))
  
class cmovnge(x86DispatchInstruction):
  dispatch = (
    (reg32_reg32_rev,     {'opcode':[0x0F, 0x4C],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0x4C],       'modrm':None}),
    (reg16_reg16_rev,     {'opcode':[0x66, 0x0F, 0x4C], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0x4C], 'modrm':None}))
    
class cmovnl(x86DispatchInstruction):
  dispatch = (
    (reg32_reg32_rev,     {'opcode':[0x0F, 0x4D],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0x4D],       'modrm':None}),
    (reg16_reg16_rev,     {'opcode':[0x66, 0x0F, 0x4D], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0x4D], 'modrm':None}))
  
class cmovge(x86DispatchInstruction):
  dispatch = (
    (reg32_reg32_rev,     {'opcode':[0x0F, 0x4D],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0x4D],       'modrm':None}),
    (reg16_reg16_rev,     {'opcode':[0x66, 0x0F, 0x4D], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0x4D], 'modrm':None}))
    
class cmovle(x86DispatchInstruction):
  dispatch = (
    (reg32_reg32_rev,     {'opcode':[0x0F, 0x4E],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0x4E],       'modrm':None}),
    (reg16_reg16_rev,     {'opcode':[0x66, 0x0F, 0x4E], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0x4E], 'modrm':None}))
    
class cmovng(x86DispatchInstruction):
  dispatch = (
    (reg32_reg32_rev,     {'opcode':[0x0F, 0x4E],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0x4E],       'modrm':None}),
    (reg16_reg16_rev,     {'opcode':[0x66, 0x0F, 0x4E], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0x4E], 'modrm':None}))
  
class cmovnle(x86DispatchInstruction):
  dispatch = (
    (reg32_reg32_rev,     {'opcode':[0x0F, 0x4F],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0x4F],       'modrm':None}),
    (reg16_reg16_rev,     {'opcode':[0x66, 0x0F, 0x4F], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0x4F], 'modrm':None}))
    
class cmovg(x86DispatchInstruction):
  dispatch = (
    (reg32_reg32_rev,     {'opcode':[0x0F, 0x4F],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0x4F],       'modrm':None}),
    (reg16_reg16_rev,     {'opcode':[0x66, 0x0F, 0x4F], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0x4F], 'modrm':None}))
    
class cmp(x86DispatchInstruction):
  dispatch = (
    (reg32_imm8,          {'opcode':[0x83],             'modrm':0x38}),
    (mem32_imm8,          {'opcode':[0x83],             'modrm':0x38}),
    (eax_imm32,           {'opcode':[0x3D],             'modrm':None}),
    (reg32_imm32,         {'opcode':[0x81],             'modrm':0x38}),
    (mem32_imm32,         {'opcode':[0x81],             'modrm':0x38}),
    (reg32_reg32,         {'opcode':[0x39],             'modrm':None}),
    (mem32_reg32,         {'opcode':[0x39],             'modrm':None}),
    (reg32_mem32,         {'opcode':[0x3B],             'modrm':None}),
    (reg16_imm8,          {'opcode':[0x66, 0x83],       'modrm':0x38}),
    (mem16_imm8,          {'opcode':[0x66, 0x83],       'modrm':0x38}),
    (ax_imm16,            {'opcode':[0x66, 0x3D],       'modrm':None}),
    (reg16_imm16,         {'opcode':[0x66, 0x81],       'modrm':0x38}),
    (mem16_imm16,         {'opcode':[0x66, 0x81],       'modrm':0x38}),
    (reg16_reg16,         {'opcode':[0x66, 0x39],       'modrm':None}),
    (mem16_reg16,         {'opcode':[0x66, 0x39],       'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x3B],       'modrm':None}),
    (al_imm8,             {'opcode':[0x3C],             'modrm':None}),
    (reg8_imm8,           {'opcode':[0x80],             'modrm':0x38}),
    (mem8_imm8,           {'opcode':[0x80],             'modrm':0x38}),
    (reg8_reg8,           {'opcode':[0x38],             'modrm':None}),
    (mem8_reg8,           {'opcode':[0x38],             'modrm':None}),
    (reg8_mem8,           {'opcode':[0x3A],             'modrm':None}))
    
class cmpsb(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0xA6],             'modrm':None}
  
class cmpsd(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0xA7],             'modrm':None}
  
class cmpsw(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0x66, 0xA7],       'modrm':None}
  
class cmpxchg(x86DispatchInstruction):
  dispatch = (
    (reg32_reg32,         {'opcode':[0x0F, 0xB1],       'modrm':None}),
    (mem32_reg32,         {'opcode':[0x0F, 0xB1],       'modrm':None}),
    (reg16_reg16,         {'opcode':[0x66, 0x0F, 0xB1], 'modrm':None}),
    (mem16_reg16,         {'opcode':[0x66, 0x0F, 0xB1], 'modrm':None}))
  
class cmpxchg8b(x86Instruction):
  machine_inst = mem64
  params = {'opcode':[0x0F, 0xC7],       'modrm':0x08}
  
class cpuid(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0x0F, 0xA2],       'modrm':None}
  
class crc32(x86DispatchInstruction):
  dispatch = (
    (reg32_reg32, {'opcode':[0xF2, 0x0F, 0x38, 0xF1], 'modrm':None}),
    (reg32_mem32, {'opcode':[0xF2, 0x0F, 0x38, 0xF1], 'modrm':None}),
    (reg32_reg16, {'opcode':[0xF2, 0x0F, 0x38, 0xF1], 'modrm':None}),
    (reg32_mem16, {'opcode':[0xF2, 0x0F, 0x38, 0xF1], 'modrm':None}),
    (reg32_reg8,  {'opcode':[0xF2, 0x0F, 0x38, 0xF0], 'modrm':None}),
    (reg32_mem8,  {'opcode':[0xF2, 0x0F, 0x38, 0xF0], 'modrm':None}))
  
class dec(x86DispatchInstruction):
  dispatch = (
    (reg32,               {'opcode':[0x48],             'modrm':None}),
    (mem32,               {'opcode':[0xFF],             'modrm':0x08}),
    (reg16,               {'opcode':[0x66, 0x48],       'modrm':None}),
    (mem16,               {'opcode':[0x66, 0xFF],       'modrm':0x08}),
    (reg8,                {'opcode':[0xFE],             'modrm':0x08}),
    (mem8,                {'opcode':[0xFE],             'modrm':0x08}))
    
class div(x86DispatchInstruction):
  dispatch = (
    (reg32,               {'opcode':[0xF7],             'modrm':0x30}),
    (mem32,               {'opcode':[0xF7],             'modrm':0x30}),
    (reg16,               {'opcode':[0x66, 0xF7],       'modrm':0x30}),
    (mem16,               {'opcode':[0x66, 0xF7],       'modrm':0x30}),
    (reg8,                {'opcode':[0xF6],             'modrm':0x30}),
    (mem8,                {'opcode':[0xF6],             'modrm':0x30}))
    
class enter(x86Instruction):
  machine_inst = imm16_imm8
  params = {'opcode':[0xC8],             'modrm':None}
  
class idiv(x86DispatchInstruction):
  dispatch = (
    (reg32,               {'opcode':[0xF7],             'modrm':0x38}),
    (mem32,               {'opcode':[0xF7],             'modrm':0x38}),
    (reg16,               {'opcode':[0x66, 0xF7],       'modrm':0x38}),
    (mem16,               {'opcode':[0x66, 0xF7],       'modrm':0x38}),
    (reg8,                {'opcode':[0xF6],             'modrm':0x38}),
    (mem8,                {'opcode':[0xF6],             'modrm':0x38}))
  
class imul(x86DispatchInstruction):
  dispatch = (
    (reg32,               {'opcode':[0xF7],             'modrm':0x28}),
    (mem32,               {'opcode':[0xF7],             'modrm':0x28}),
    (reg32_reg32,         {'opcode':[0x0F, 0xAF],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0xAF],       'modrm':None}),
    (reg32_reg32_imm8_rev,{'opcode':[0x6B],             'modrm':None}),
    (reg32_mem32_imm8,    {'opcode':[0x6B],             'modrm':None}),
    (reg32_reg32_imm32,   {'opcode':[0x69],             'modrm':None}),
    (reg32_mem32_imm32,   {'opcode':[0x69],             'modrm':None}),
    (reg16,               {'opcode':[0x66, 0xF7],       'modrm':0x28}),
    (mem16,               {'opcode':[0x66, 0xF7],       'modrm':0x28}),
    (reg16_reg16,         {'opcode':[0x66, 0x0F, 0xAF], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0xAF], 'modrm':None}),
    (reg16_reg16_imm8_rev,{'opcode':[0x66, 0x6B],       'modrm':None}),
    (reg16_mem16_imm8,    {'opcode':[0x66, 0x6B],       'modrm':None}),
    (reg16_reg16_imm16,   {'opcode':[0x66, 0x69],       'modrm':None}),
    (reg16_mem16_imm16,   {'opcode':[0x66, 0x69],       'modrm':None}),
    (reg8,                {'opcode':[0xF6],             'modrm':0x28}),
    (mem8,                {'opcode':[0xF6],             'modrm':0x28}))
  
class in_(x86DispatchInstruction):
  dispatch = (
    (eax_dx,              {'opcode':[0xED],             'modrm':None}),
    (ax_dx,               {'opcode':[0x66, 0xED],       'modrm':None}),
    (al_dx,               {'opcode':[0xEC],             'modrm':None}),
    (eax_imm8,            {'opcode':[0xE5],             'modrm':None}),
    (ax_imm8,             {'opcode':[0x66, 0xE5],       'modrm':None}),
    (al_imm8,             {'opcode':[0xE4],             'modrm':None}))
    
class inc(x86DispatchInstruction):
  dispatch = (
    (reg32,               {'opcode':[0x40],             'modrm':None}),
    (reg32,               {'opcode':[0xFF],             'modrm':0x00}),
    (mem32,               {'opcode':[0xFF],             'modrm':0x00}),
    (reg16,               {'opcode':[0x66, 0x40],       'modrm':None}),
    (reg16,               {'opcode':[0x66, 0xFF],       'modrm':0x00}),
    (mem16,               {'opcode':[0x66, 0xFF],       'modrm':0x00}),
    (reg8,                {'opcode':[0xFE],             'modrm':0x00}),
    (mem8,                {'opcode':[0xFE],             'modrm':0x00}))
    
class insb(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0x6C],             'modrm':None}
  
class insd(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0x6D],             'modrm':None}
  
class insw(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0x66, 0x6D],       'modrm':None}
  
class int_(x86Instruction):
  machine_inst = imm8
  params = {'opcode':[0xCD],             'modrm':None}
  
class int_3(x86Instruction):
  """NOTE - this is a special form of 'int 3' used for debugging; see the
     architecture manuals for more information."""
  machine_inst = no_op
  params = {'opcode':[0xCC],             'modrm':None}
  
class ja(x86DispatchInstruction):
  dispatch = (
    (lbl32_8off,          {'opcode':[[0x77], [0x0F, 0x87]], 'modrm':None}),
    (rel32_8off,          {'opcode':[[0x77], [0x0F, 0x87]], 'modrm':None}))
    
class jae(x86DispatchInstruction):
  dispatch = (
    (lbl32_8off,          {'opcode':[[0x73], [0x0F, 0x83]], 'modrm':None}),
    (rel32_8off,          {'opcode':[[0x73], [0x0F, 0x83]], 'modrm':None}))
    
class jb(x86DispatchInstruction):
  dispatch = (
    (lbl32_8off,          {'opcode':[[0x72], [0x0F, 0x82]], 'modrm':None}),
    (rel32_8off,          {'opcode':[[0x72], [0x0F, 0x82]], 'modrm':None}))
  
class jbe(x86DispatchInstruction):
  dispatch = (
    (lbl32_8off,          {'opcode':[[0x76], [0x0F, 0x86]], 'modrm':None}),
    (rel32_8off,          {'opcode':[[0x76], [0x0F, 0x86]], 'modrm':None}))
    
class jc(x86DispatchInstruction):
  dispatch = (
    (lbl32_8off,          {'opcode':[[0x72], [0x0F, 0x82]], 'modrm':None}),
    (rel32_8off,          {'opcode':[[0x72], [0x0F, 0x82]], 'modrm':None}))
    
class jcxz(x86DispatchInstruction):
  dispatch = (
    (lbl8off,             {'opcode':[0x67, 0xE3],       'modrm':None}),
    (rel8off,             {'opcode':[0x67, 0xE3],       'modrm':None}))
  
class je(x86DispatchInstruction):
  dispatch = (
    (lbl32_8off,          {'opcode':[[0x74], [0x0F, 0x84]], 'modrm':None}),
    (rel32_8off,          {'opcode':[[0x74], [0x0F, 0x84]], 'modrm':None}))
    
class jecxz(x86DispatchInstruction):
  dispatch = (
    (lbl8off,             {'opcode':[0xE3],             'modrm':None}),
    (rel8off,             {'opcode':[0xE3],             'modrm':None}))
  
class jg(x86DispatchInstruction):
  dispatch = (
    (lbl32_8off,          {'opcode':[[0x7F], [0x0F, 0x8F]], 'modrm':None}),
    (rel32_8off,          {'opcode':[[0x7F], [0x0F, 0x8F]], 'modrm':None}))
    
class jge(x86DispatchInstruction):
  dispatch = (
    (lbl32_8off,          {'opcode':[[0x7D], [0x0F, 0x8D]], 'modrm':None}),
    (rel32_8off,          {'opcode':[[0x7D], [0x0F, 0x8D]], 'modrm':None}))
    
class jl(x86DispatchInstruction):
  dispatch = (
    (lbl32_8off,          {'opcode':[[0x7C], [0x0F, 0x8C]], 'modrm':None}),
    (rel32_8off,          {'opcode':[[0x7C], [0x0F, 0x8C]], 'modrm':None}))
    
class jle(x86DispatchInstruction):
  dispatch = (
    (lbl32_8off,          {'opcode':[[0x7E], [0x0F, 0x8E]], 'modrm':None}),
    (rel32_8off,          {'opcode':[[0x7E], [0x0F, 0x8E]], 'modrm':None}))
    
class jmp(x86DispatchInstruction):
  dispatch = (
    (lbl32_8off,          {'opcode':[[0xEB], [0xE9]],   'modrm':None}),
    (rel32_8off,          {'opcode':[[0xEB], [0xE9]],   'modrm':None}),
    (reg32,               {'opcode':[0xFF],             'modrm':0x20}),
    (mem32,               {'opcode':[0xFF],             'modrm':0x20}),
    (reg16,               {'opcode':[0x66, 0xFF],       'modrm':0x20}),
    (mem16,               {'opcode':[0x66, 0xFF],       'modrm':0x20}))
    
class jna(x86DispatchInstruction):
  dispatch = (
    (lbl32_8off,          {'opcode':[[0x76], [0x0F, 0x86]], 'modrm':None}),
    (rel32_8off,          {'opcode':[[0x76], [0x0F, 0x86]], 'modrm':None}))
    
class jnae(x86DispatchInstruction):
  dispatch = (
    (lbl32_8off,          {'opcode':[[0x72], [0x0F, 0x82]], 'modrm':None}),
    (rel32_8off,          {'opcode':[[0x72], [0x0F, 0x82]], 'modrm':None}))
    
class jnb(x86DispatchInstruction):
  dispatch = (
    (lbl32_8off,          {'opcode':[[0x73], [0x0F, 0x83]], 'modrm':None}),
    (rel32_8off,          {'opcode':[[0x73], [0x0F, 0x83]], 'modrm':None}))
    
class jnbe(x86DispatchInstruction):
  dispatch = (
    (lbl32_8off,          {'opcode':[[0x77], [0x0F, 0x87]], 'modrm':None}),
    (rel32_8off,          {'opcode':[[0x77], [0x0F, 0x87]], 'modrm':None}))
    
class jnc(x86DispatchInstruction):
  dispatch = (
    (lbl32_8off,          {'opcode':[[0x73], [0x0F, 0x83]], 'modrm':None}),
    (rel32_8off,          {'opcode':[[0x73], [0x0F, 0x83]], 'modrm':None}))
    
class jne(x86DispatchInstruction):
  dispatch = (
    (lbl32_8off,          {'opcode':[[0x75], [0x0F, 0x85]], 'modrm':None}),
    (rel32_8off,          {'opcode':[[0x75], [0x0F, 0x85]], 'modrm':None}))
    
class jng(x86DispatchInstruction):
  dispatch = (
    (lbl32_8off,          {'opcode':[[0x7E], [0x0F, 0x8E]], 'modrm':None}),
    (rel32_8off,          {'opcode':[[0x7E], [0x0F, 0x8E]], 'modrm':None}))
    
class jnge(x86DispatchInstruction):
  dispatch = (
    (lbl32_8off,          {'opcode':[[0x7C], [0x0F, 0x8C]], 'modrm':None}),
    (rel32_8off,          {'opcode':[[0x7C], [0x0F, 0x8C]], 'modrm':None}))
    
class jnl(x86DispatchInstruction):
  dispatch = (
    (lbl32_8off,          {'opcode':[[0x7D], [0x0F, 0x8D]], 'modrm':None}),
    (rel32_8off,          {'opcode':[[0x7D], [0x0F, 0x8D]], 'modrm':None}))
    
class jnle(x86DispatchInstruction):
  dispatch = (
    (lbl32_8off,          {'opcode':[[0x7F], [0x0F, 0x8F]], 'modrm':None}),
    (rel32_8off,          {'opcode':[[0x7F], [0x0F, 0x8F]], 'modrm':None}))
    
class jno(x86DispatchInstruction):
  dispatch = (
    (lbl32_8off,          {'opcode':[[0x71], [0x0F, 0x81]], 'modrm':None}),
    (rel32_8off,          {'opcode':[[0x71], [0x0F, 0x81]], 'modrm':None}))
  
class jnp(x86DispatchInstruction):
  dispatch = (
    (lbl32_8off,          {'opcode':[[0x7B], [0x0F, 0x8B]], 'modrm':None}),
    (rel32_8off,          {'opcode':[[0x7B], [0x0F, 0x8B]], 'modrm':None}))
    
class jns(x86DispatchInstruction):
  dispatch = (
    (lbl32_8off,          {'opcode':[[0x79], [0x0F, 0x89]], 'modrm':None}),
    (rel32_8off,          {'opcode':[[0x79], [0x0F, 0x89]], 'modrm':None}))
  
class jnz(x86DispatchInstruction):
  dispatch = (
    (lbl32_8off,          {'opcode':[[0x75], [0x0F, 0x85]], 'modrm':None}),
    (rel32_8off,          {'opcode':[[0x75], [0x0F, 0x85]], 'modrm':None}))
    
class jo(x86DispatchInstruction):
  dispatch = (
    (lbl32_8off,          {'opcode':[[0x70], [0x0F, 0x80]], 'modrm':None}),
    (rel32_8off,          {'opcode':[[0x70], [0x0F, 0x80]], 'modrm':None}))
    
class jp(x86DispatchInstruction):
  dispatch = (
    (lbl32_8off,          {'opcode':[[0x7A], [0x0F, 0x8A]], 'modrm':None}),
    (rel32_8off,          {'opcode':[[0x7A], [0x0F, 0x8A]], 'modrm':None}))
    
class jpe(x86DispatchInstruction):
  dispatch = (
    (lbl32_8off,          {'opcode':[[0x7A], [0x0F, 0x8A]], 'modrm':None}),
    (rel32_8off,          {'opcode':[[0x7A], [0x0F, 0x8A]], 'modrm':None}))
    
class jpo(x86DispatchInstruction):
  dispatch = (
    (lbl32_8off,          {'opcode':[[0x7B], [0x0F, 0x8B]], 'modrm':None}),
    (rel32_8off,          {'opcode':[[0x7B], [0x0F, 0x8B]], 'modrm':None}))
    
class js(x86DispatchInstruction):
  dispatch = (
    (lbl32_8off,          {'opcode':[[0x78], [0x0F, 0x88]], 'modrm':None}),
    (rel32_8off,          {'opcode':[[0x78], [0x0F, 0x88]], 'modrm':None}))
    
class jz(x86DispatchInstruction):
  dispatch = (
    (lbl32_8off,          {'opcode':[[0x74], [0x0F, 0x84]], 'modrm':None}),
    (rel32_8off,          {'opcode':[[0x74], [0x0F, 0x84]], 'modrm':None}))
  
class lahf(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0x9F],             'modrm':None}
  
class lea(x86DispatchInstruction):
  dispatch = (
    (reg32_mem,           {'opcode':[0x8D],             'modrm':0x00}),
    (reg16_mem,           {'opcode':[0x66, 0x8D],       'modrm':0x00}))
    
class leave(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0xC9],             'modrm':None}
  
class lfence(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0x0F, 0xAE, 0xE8], 'modrm':None}
  
class lodsb(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0xAC],             'modrm':None}
  
class lodsd(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0xAD],             'modrm':None}
  
class lodsw(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0x66, 0xAD],       'modrm':None}
  
class loop(x86DispatchInstruction):
  dispatch = (
    (lbl8off,             {'opcode':[0xE2],             'modrm':None}),
    (rel8off,             {'opcode':[0xE2],             'modrm':None}))

class loope(x86DispatchInstruction):
  dispatch = (
    (lbl8off,             {'opcode':[0xE1],             'modrm':None}),
    (rel8off,             {'opcode':[0xE1],             'modrm':None}))

class loopne(x86DispatchInstruction):
  dispatch = (
    (lbl8off,             {'opcode':[0xE0],             'modrm':None}),
    (rel8off,             {'opcode':[0xE0],             'modrm':None}))

class loopnz(x86DispatchInstruction):
  dispatch = (
    (lbl8off,             {'opcode':[0xE0],             'modrm':None}),
    (rel8off,             {'opcode':[0xE0],             'modrm':None}))

class loopz(x86DispatchInstruction):
  dispatch = (
    (lbl8off,             {'opcode':[0xE1],             'modrm':None}),
    (rel8off,             {'opcode':[0xE1],             'modrm':None}))

class lzcnt(x86DispatchInstruction):
  dispatch = (
    (reg32_reg32,    {'opcode':[0xF3, 0x0F, 0xBD], 'modrm':None}),
    (reg32_mem32,    {'opcode':[0xF3, 0x0F, 0xBD], 'modrm':None}),
    (reg16_reg16,   {'opcode':[0x66, 0xF3, 0x0F, 0xBD], 'modrm':None}),
    (reg16_mem16,   {'opcode':[0x66, 0xF3, 0x0F, 0xBD], 'modrm':None}))
            
class mfence(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0x0F, 0xAE, 0xF0], 'modrm':None}
  
class mov(x86DispatchInstruction):
  dispatch = (
    (reg32_imm32,         {'opcode':[0xB8],             'modrm':None}),
    (mem32_imm32,         {'opcode':[0xC7],             'modrm':0x00}),
    (reg32_reg32,         {'opcode':[0x89],             'modrm':None}),
    (mem32_reg32,         {'opcode':[0x89],             'modrm':None}),
    (reg32_mem32,         {'opcode':[0x8B],             'modrm':None}),
    (reg16_imm16,         {'opcode':[0x66, 0xB8],       'modrm':None}),
    (mem16_imm16,         {'opcode':[0x66, 0xC7],       'modrm':0x00}),
    (reg16_reg16,         {'opcode':[0x66, 0x89],       'modrm':None}),
    (mem16_reg16,         {'opcode':[0x66, 0x89],       'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x8B],       'modrm':None}),
    (reg8_imm8,           {'opcode':[0xB0],             'modrm':None}),
    (mem8_imm8,           {'opcode':[0xC6],             'modrm':0x00}),
    (reg8_reg8,           {'opcode':[0x88],             'modrm':None}),
    (mem8_reg8,           {'opcode':[0x88],             'modrm':None}),
    (reg8_mem8,           {'opcode':[0x8A],             'modrm':None}))
    
class movnti(x86Instruction):
  machine_inst = mem32_reg32
  params = {'opcode':[0x0F, 0xC3],       'modrm':None}
  # SSE2!

class movsb(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0xA4],             'modrm':None}
  
class movsd(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0xA5],             'modrm':None}
  
class movsw(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0x66, 0xA5],       'modrm':None}
  
class movsx(x86DispatchInstruction):
  dispatch = (
    (reg32_reg8,          {'opcode':[0x0F, 0xBE], 'modrm':None}),
    (reg32_mem8,          {'opcode':[0x0F, 0xBE], 'modrm':None}),
    (reg32_reg16,         {'opcode':[0x0F, 0xBF], 'modrm':None}),
    (reg32_mem16,         {'opcode':[0x0F, 0xBF], 'modrm':None}),
    (reg16_reg8,          {'opcode':[0x66, 0x0F, 0xBE], 'modrm':None}),
    (reg16_mem8,          {'opcode':[0x66, 0x0F, 0xBE], 'modrm':None}))
  
class movzx(x86DispatchInstruction):
  dispatch = (
    (reg32_reg8,          {'opcode':[0x0F, 0xB6], 'modrm':None}),
    (reg32_mem8,          {'opcode':[0x0F, 0xB6], 'modrm':None}),
    (reg32_reg16,         {'opcode':[0x0F, 0xB7], 'modrm':None}),
    (reg32_mem16,         {'opcode':[0x0F, 0xB7], 'modrm':None}),
    (reg16_reg8,          {'opcode':[0x66, 0x0F, 0xB6], 'modrm':None}),
    (reg16_mem8,          {'opcode':[0x66, 0x0F, 0xB6], 'modrm':None}))
  
class mul(x86DispatchInstruction):
  dispatch = (
    (reg32,               {'opcode':[0xF7],             'modrm':0x20}),
    (mem32,               {'opcode':[0xF7],             'modrm':0x20}),
    (reg16,               {'opcode':[0x66, 0xF7],       'modrm':0x20}),
    (mem16,               {'opcode':[0x66, 0xF7],       'modrm':0x20}),
    (reg8,                {'opcode':[0xF6],             'modrm':0x20}),
    (mem8,                {'opcode':[0xF6],             'modrm':0x20}))
  
class neg(x86DispatchInstruction):
  dispatch = (
    (reg32,               {'opcode':[0xF7],             'modrm':0x18}),
    (mem32,               {'opcode':[0xF7],             'modrm':0x18}),
    (reg16,               {'opcode':[0x66, 0xF7],       'modrm':0x18}),
    (mem16,               {'opcode':[0x66, 0xF7],       'modrm':0x18}),
    (reg8,                {'opcode':[0xF6],             'modrm':0x18}),
    (mem8,                {'opcode':[0xF6],             'modrm':0x18}))
  
class nop(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0x90],             'modrm':None}
  
class not_(x86DispatchInstruction):
  dispatch = (
    (reg32,               {'opcode':[0xF7],             'modrm':0x10}),
    (mem32,               {'opcode':[0xF7],             'modrm':0x10}),
    (reg16,               {'opcode':[0x66, 0xF7],       'modrm':0x10}),
    (mem16,               {'opcode':[0x66, 0xF7],       'modrm':0x10}),
    (reg8,                {'opcode':[0xF6],             'modrm':0x10}),
    (mem8,                {'opcode':[0xF6],             'modrm':0x10}))
  
class or_(x86DispatchInstruction):
  dispatch = (
    (reg32_imm8,          {'opcode':[0x83],             'modrm':0x08}),
    (mem32_imm8,          {'opcode':[0x83],             'modrm':0x08}),
    (eax_imm32,           {'opcode':[0x0D],             'modrm':None}),
    (reg32_imm32,         {'opcode':[0x81],             'modrm':0x08}),
    (mem32_imm32,         {'opcode':[0x81],             'modrm':0x08}),
    (mem32_reg32,         {'opcode':[0x09],             'modrm':None}),
    (reg32_reg32,         {'opcode':[0x09],             'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0B],             'modrm':None}),
    (reg16_imm8,          {'opcode':[0x66, 0x83],       'modrm':0x08}),
    (mem16_imm8,          {'opcode':[0x66, 0x83],       'modrm':0x08}),
    (ax_imm16,            {'opcode':[0x66, 0x0D],       'modrm':None}),
    (reg16_imm16,         {'opcode':[0x66, 0x81],       'modrm':0x08}),
    (mem16_imm16,         {'opcode':[0x66, 0x81],       'modrm':0x08}),
    (mem16_reg16,         {'opcode':[0x66, 0x09],       'modrm':None}),
    (reg16_reg16,         {'opcode':[0x66, 0x09],       'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0B],       'modrm':None}),
    (al_imm8,             {'opcode':[0x0C],             'modrm':None}),
    (reg8_imm8,           {'opcode':[0x80],             'modrm':0x08}),
    (mem8_imm8,           {'opcode':[0x80],             'modrm':0x08}),
    (reg8_reg8,           {'opcode':[0x08],             'modrm':None}),
    (mem8_reg8,           {'opcode':[0x08],             'modrm':None}),
    (reg8_mem8,           {'opcode':[0x0A],             'modrm':None}))
    
class out(x86DispatchInstruction):
  dispatch = (
    (dx_eax,              {'opcode':[0xEF],             'modrm':None}),
    (dx_ax,               {'opcode':[0x66, 0xEF],       'modrm':None}),
    (dx_al,               {'opcode':[0xEE],             'modrm':None}),
    (imm8_eax,            {'opcode':[0xE7],             'modrm':None}),
    (imm8_ax,             {'opcode':[0x66, 0xE7],       'modrm':None}),
    (imm8_al,             {'opcode':[0xE6],             'modrm':None}))
    
class outsb(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0x6E],             'modrm':None}
  
class outsd(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0x6F],             'modrm':None}
  
class outsw(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0x66, 0x6F],       'modrm':None}
  
class pause(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0xF3, 0x90],       'modrm':None}
  
class pop(x86DispatchInstruction):
  dispatch = (
    (reg32,               {'opcode':[0x58],             'modrm':None}),
    (mem32,               {'opcode':[0x8F],             'modrm':0x00}),
    (reg16,               {'opcode':[0x66, 0x58],       'modrm':None}),
    (mem16,               {'opcode':[0x66, 0x8F],       'modrm':0x00}))
    
class popa(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0x66, 0x61],       'modrm':None}
  
class popad(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0x61],             'modrm':None}
  
class popcnt(x86DispatchInstruction):
  dispatch = (
    (reg32_reg32,         {'opcode':[0xF3, 0x0F, 0xB8], 'modrm':None}),
    (reg32_mem32,         {'opcode':[0xF3, 0x0F, 0xB8], 'modrm':None}),
    (reg16_reg16,    {'opcode':[0x66, 0xF3, 0x0F, 0xB8],'modrm':None}),
    (reg16_mem16,    {'opcode':[0x66, 0xF3, 0x0F, 0xB8],'modrm':None}))
  
class popf(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0x66, 0x9D],       'modrm':None}
  
class popfd(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0x9D],             'modrm':None}
  
class prefetch(x86Instruction):
  machine_inst = mem8
  params = {'opcode':[0x0F, 0x0D],       'modrm':0x00}
  
class prefetchnta(x86Instruction):
  machine_inst = mem8
  params = {'opcode':[0x0F, 0x18],       'modrm':0x00}
  
class prefetcht0(x86Instruction):
  machine_inst = mem8
  params = {'opcode':[0x0F, 0x18],       'modrm':0x08}
  
class prefetcht1(x86Instruction):
  machine_inst = mem8
  params = {'opcode':[0x0F, 0x18],       'modrm':0x10}
  
class prefetcht2(x86Instruction):
  machine_inst = mem8
  params = {'opcode':[0x0F, 0x18],       'modrm':0x18}
  
class prefetchw(x86Instruction):
  machine_inst = mem8
  params = {'opcode':[0x0F, 0x0D],       'modrm':0x08}
  
class push(x86DispatchInstruction):
  dispatch = (
    (reg32,               {'opcode':[0x50],             'modrm':None}),
    (mem32,               {'opcode':[0xFF],             'modrm':0x30}),
    (imm8,                {'opcode':[0x6A],             'modrm':None}),
    (imm16,               {'opcode':[0x66, 0x68],       'modrm':None}),
    (imm32,               {'opcode':[0x68],             'modrm':None}),
    (reg16,               {'opcode':[0x66, 0x50],       'modrm':None}),
    (mem16,               {'opcode':[0x66, 0xFF],       'modrm':0x30}))
    
class pusha(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0x66, 0x60],       'modrm':None}
  
class pushad(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0x60],             'modrm':None}
  
class pushf(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0x66, 0x9C],       'modrm':None}
  
class pushfd(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0x9C],             'modrm':None}
  
class rcl(x86DispatchInstruction):
  dispatch = (
    (reg32_1,             {'opcode':[0xD1],             'modrm':0x10}),
    (mem32_1,             {'opcode':[0xD1],             'modrm':0x10}),
    (reg32_cl,            {'opcode':[0xD3],             'modrm':0x10}),
    (mem32_cl,            {'opcode':[0xD3],             'modrm':0x10}),
    (reg32_imm8,          {'opcode':[0xC1],             'modrm':0x10}),
    (mem32_imm8,          {'opcode':[0xC1],             'modrm':0x10}),
    (reg16_1,             {'opcode':[0x66, 0xD1],       'modrm':0x10}),
    (mem16_1,             {'opcode':[0x66, 0xD1],       'modrm':0x10}),
    (reg16_cl,            {'opcode':[0x66, 0xD3],       'modrm':0x10}),
    (mem16_cl,            {'opcode':[0x66, 0xD3],       'modrm':0x10}),
    (reg16_imm8,          {'opcode':[0x66, 0xC1],       'modrm':0x10}),
    (mem16_imm8,          {'opcode':[0x66, 0xC1],       'modrm':0x10}),
    (reg8_1,              {'opcode':[0xD0],             'modrm':0x10}),
    (mem8_1,              {'opcode':[0xD0],             'modrm':0x10}),
    (reg8_cl,             {'opcode':[0xD2],             'modrm':0x10}),
    (mem8_cl,             {'opcode':[0xD2],             'modrm':0x10}),
    (reg8_imm8,           {'opcode':[0xC0],             'modrm':0x10}),
    (mem8_imm8,           {'opcode':[0xC0],             'modrm':0x10}))
    
class rcr(x86DispatchInstruction):
  dispatch = (
    (reg32_1,             {'opcode':[0xD1],             'modrm':0x18}),
    (mem32_1,             {'opcode':[0xD1],             'modrm':0x18}),
    (reg32_cl,            {'opcode':[0xD3],             'modrm':0x18}),
    (mem32_cl,            {'opcode':[0xD3],             'modrm':0x18}),
    (reg32_imm8,          {'opcode':[0xC1],             'modrm':0x18}),
    (mem32_imm8,          {'opcode':[0xC1],             'modrm':0x18}),
    (reg16_1,             {'opcode':[0x66, 0xD1],       'modrm':0x18}),
    (mem16_1,             {'opcode':[0x66, 0xD1],       'modrm':0x18}),
    (reg16_cl,            {'opcode':[0x66, 0xD3],       'modrm':0x18}),
    (mem16_cl,            {'opcode':[0x66, 0xD3],       'modrm':0x18}),
    (reg16_imm8,          {'opcode':[0x66, 0xC1],       'modrm':0x18}),
    (mem16_imm8,          {'opcode':[0x66, 0xC1],       'modrm':0x18}),
    (reg8_1,              {'opcode':[0xD0],             'modrm':0x18}),
    (mem8_1,              {'opcode':[0xD0],             'modrm':0x18}),
    (reg8_cl,             {'opcode':[0xD2],             'modrm':0x18}),
    (mem8_cl,             {'opcode':[0xD2],             'modrm':0x18}),
    (reg8_imm8,           {'opcode':[0xC0],             'modrm':0x18}),
    (mem8_imm8,           {'opcode':[0xC0],             'modrm':0x18}))
    
class rdtsc(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0x0F, 0x31],       'modrm':None}
  
class ret(x86DispatchInstruction):
  dispatch = (
    (no_op,               {'opcode':[0xC3],             'modrm':None}),
    (imm16,               {'opcode':[0xC2],             'modrm':None}))
    
class rol(x86DispatchInstruction):
  dispatch = (
    (reg32_1,             {'opcode':[0xD1],             'modrm':0x00}),
    (mem32_1,             {'opcode':[0xD1],             'modrm':0x00}),
    (reg32_cl,            {'opcode':[0xD3],             'modrm':0x00}),
    (mem32_cl,            {'opcode':[0xD3],             'modrm':0x00}),
    (reg32_imm8,          {'opcode':[0xC1],             'modrm':0x00}),
    (mem32_imm8,          {'opcode':[0xC1],             'modrm':0x00}),
    (reg16_1,             {'opcode':[0x66, 0xD1],       'modrm':0x00}),
    (mem16_1,             {'opcode':[0x66, 0xD1],       'modrm':0x00}),
    (reg16_cl,            {'opcode':[0x66, 0xD3],       'modrm':0x00}),
    (mem16_cl,            {'opcode':[0x66, 0xD3],       'modrm':0x00}),
    (reg16_imm8,          {'opcode':[0x66, 0xC1],       'modrm':0x00}),
    (mem16_imm8,          {'opcode':[0x66, 0xC1],       'modrm':0x00}),
    (reg8_1,              {'opcode':[0xD0],             'modrm':0x00}),
    (mem8_1,              {'opcode':[0xD0],             'modrm':0x00}),
    (reg8_cl,             {'opcode':[0xD2],             'modrm':0x00}),
    (mem8_cl,             {'opcode':[0xD2],             'modrm':0x00}),
    (reg8_imm8,           {'opcode':[0xC0],             'modrm':0x00}),
    (mem8_imm8,           {'opcode':[0xC0],             'modrm':0x00}))
    
class ror(x86DispatchInstruction):
  dispatch = (
    (reg32_1,             {'opcode':[0xD1],             'modrm':0x08}),
    (mem32_1,             {'opcode':[0xD1],             'modrm':0x08}),
    (reg32_cl,            {'opcode':[0xD3],             'modrm':0x08}),
    (mem32_cl,            {'opcode':[0xD3],             'modrm':0x08}),
    (reg32_imm8,          {'opcode':[0xC1],             'modrm':0x08}),
    (mem32_imm8,          {'opcode':[0xC1],             'modrm':0x08}),
    (reg16_1,             {'opcode':[0x66, 0xD1],       'modrm':0x08}),
    (mem16_1,             {'opcode':[0x66, 0xD1],       'modrm':0x08}),
    (reg16_cl,            {'opcode':[0x66, 0xD3],       'modrm':0x08}),
    (mem16_cl,            {'opcode':[0x66, 0xD3],       'modrm':0x08}),
    (reg16_imm8,          {'opcode':[0x66, 0xC1],       'modrm':0x08}),
    (mem16_imm8,          {'opcode':[0x66, 0xC1],       'modrm':0x08}),
    (reg8_1,              {'opcode':[0xD0],             'modrm':0x08}),
    (mem8_1,              {'opcode':[0xD0],             'modrm':0x08}),
    (reg8_cl,             {'opcode':[0xD2],             'modrm':0x08}),
    (mem8_cl,             {'opcode':[0xD2],             'modrm':0x08}),
    (reg8_imm8,           {'opcode':[0xC0],             'modrm':0x08}),
    (mem8_imm8,           {'opcode':[0xC0],             'modrm':0x08}))
    
class sahf(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0x9E],             'modrm':None}
  
class sal(x86DispatchInstruction):
  dispatch = (
    (reg32_1,             {'opcode':[0xD1],             'modrm':0x20}),
    (mem32_1,             {'opcode':[0xD1],             'modrm':0x20}),
    (reg32_cl,            {'opcode':[0xD3],             'modrm':0x20}),
    (mem32_cl,            {'opcode':[0xD3],             'modrm':0x20}),
    (reg32_imm8,          {'opcode':[0xC1],             'modrm':0x20}),
    (mem32_imm8,          {'opcode':[0xC1],             'modrm':0x20}),
    (reg16_1,             {'opcode':[0x66, 0xD1],       'modrm':0x20}),
    (mem16_1,             {'opcode':[0x66, 0xD1],       'modrm':0x20}),
    (reg16_cl,            {'opcode':[0x66, 0xD3],       'modrm':0x20}),
    (mem16_cl,            {'opcode':[0x66, 0xD3],       'modrm':0x20}),
    (reg16_imm8,          {'opcode':[0x66, 0xC1],       'modrm':0x20}),
    (mem16_imm8,          {'opcode':[0x66, 0xC1],       'modrm':0x20}),
    (reg8_1,              {'opcode':[0xD0],             'modrm':0x20}),
    (mem8_1,              {'opcode':[0xD0],             'modrm':0x20}),
    (reg8_cl,             {'opcode':[0xD2],             'modrm':0x20}),
    (mem8_cl,             {'opcode':[0xD2],             'modrm':0x20}),
    (reg8_imm8,           {'opcode':[0xC0],             'modrm':0x20}),
    (mem8_imm8,           {'opcode':[0xC0],             'modrm':0x20}))
    
class sar(x86DispatchInstruction):
  dispatch = (
    (reg32_1,             {'opcode':[0xD1],             'modrm':0x38}),
    (mem32_1,             {'opcode':[0xD1],             'modrm':0x38}),
    (reg32_cl,            {'opcode':[0xD3],             'modrm':0x38}),
    (mem32_cl,            {'opcode':[0xD3],             'modrm':0x38}),
    (reg32_imm8,          {'opcode':[0xC1],             'modrm':0x38}),
    (mem32_imm8,          {'opcode':[0xC1],             'modrm':0x38}),
    (reg16_1,             {'opcode':[0x66, 0xD1],       'modrm':0x38}),
    (mem16_1,             {'opcode':[0x66, 0xD1],       'modrm':0x38}),
    (reg16_cl,            {'opcode':[0x66, 0xD3],       'modrm':0x38}),
    (mem16_cl,            {'opcode':[0x66, 0xD3],       'modrm':0x38}),
    (reg16_imm8,          {'opcode':[0x66, 0xC1],       'modrm':0x38}),
    (mem16_imm8,          {'opcode':[0x66, 0xC1],       'modrm':0x38}),
    (reg8_1,              {'opcode':[0xD0],             'modrm':0x38}),
    (mem8_1,              {'opcode':[0xD0],             'modrm':0x38}),
    (reg8_cl,             {'opcode':[0xD2],             'modrm':0x38}),
    (mem8_cl,             {'opcode':[0xD2],             'modrm':0x38}),
    (reg8_imm8,           {'opcode':[0xC0],             'modrm':0x38}),
    (mem8_imm8,           {'opcode':[0xC0],             'modrm':0x38}))
    
class sbb(x86DispatchInstruction):
  dispatch = (
    (reg32_imm8,          {'opcode':[0x83],             'modrm':0x18}),
    (mem32_imm8,          {'opcode':[0x83],             'modrm':0x18}),
    (eax_imm32,           {'opcode':[0x1D],             'modrm':None}),
    (reg32_imm32,         {'opcode':[0x81],             'modrm':0x18}),
    (mem32_imm32,         {'opcode':[0x81],             'modrm':0x18}),
    (reg32_reg32,         {'opcode':[0x19],             'modrm':None}),
    (mem32_reg32,         {'opcode':[0x19],             'modrm':None}),
    (reg32_mem32,         {'opcode':[0x1B],             'modrm':None}),
    (reg16_imm8,          {'opcode':[0x66, 0x83],       'modrm':0x18}),
    (mem16_imm8,          {'opcode':[0x66, 0x83],       'modrm':0x18}),
    (ax_imm16,            {'opcode':[0x66, 0x1D],       'modrm':None}),
    (reg16_imm16,         {'opcode':[0x66, 0x81],       'modrm':0x18}),
    (mem16_imm16,         {'opcode':[0x66, 0x81],       'modrm':0x18}),
    (reg16_reg16,         {'opcode':[0x66, 0x19],       'modrm':None}),
    (mem16_reg16,         {'opcode':[0x66, 0x19],       'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x1B],       'modrm':None}),
    (al_imm8,             {'opcode':[0x1C],             'modrm':None}),
    (reg8_imm8,           {'opcode':[0x80],             'modrm':0x18}),
    (mem8_imm8,           {'opcode':[0x80],             'modrm':0x18}),
    (reg8_reg8,           {'opcode':[0x18],             'modrm':None}),
    (mem8_reg8,           {'opcode':[0x18],             'modrm':None}),
    (reg8_mem8,           {'opcode':[0x1A],             'modrm':None}))
    
class scasb(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0xAE],             'modrm':None}
  
class scasd(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0xAF],             'modrm':None}
  
class scasw(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0x66, 0xAF],       'modrm':None}
  
class seta(x86DispatchInstruction):
  dispatch = (
    (reg8,                {'opcode':[0x0F, 0x97],       'modrm':0x00}),
    (mem8,                {'opcode':[0x0F, 0x97],       'modrm':0x00}))
  
class setae(x86DispatchInstruction):
  dispatch = (
    (reg8,                {'opcode':[0x0F, 0x93],       'modrm':0x00}),
    (mem8,                {'opcode':[0x0F, 0x93],       'modrm':0x00}))
  
class setb(x86DispatchInstruction):
  dispatch = (
    (reg8,                {'opcode':[0x0F, 0x92],       'modrm':0x00}),
    (mem8,                {'opcode':[0x0F, 0x92],       'modrm':0x00}))
  
class setbe(x86DispatchInstruction):
  dispatch = (
    (reg8,                {'opcode':[0x0F, 0x96],       'modrm':0x00}),
    (mem8,                {'opcode':[0x0F, 0x96],       'modrm':0x00}))
  
class setc(x86DispatchInstruction):
  dispatch = (
    (reg8,                {'opcode':[0x0F, 0x92],       'modrm':0x00}),
    (mem8,                {'opcode':[0x0F, 0x92],       'modrm':0x00}))
  
class sete(x86DispatchInstruction):
  dispatch = (
    (reg8,                {'opcode':[0x0F, 0x94],       'modrm':0x00}),
    (mem8,                {'opcode':[0x0F, 0x94],       'modrm':0x00}))
  
class setg(x86DispatchInstruction):
  dispatch = (
    (reg8,                {'opcode':[0x0F, 0x9F],       'modrm':0x00}),
    (mem8,                {'opcode':[0x0F, 0x9F],       'modrm':0x00}))
  
class setge(x86DispatchInstruction):
  dispatch = (
    (reg8,                {'opcode':[0x0F, 0x9D],       'modrm':0x00}),
    (mem8,                {'opcode':[0x0F, 0x9D],       'modrm':0x00}))
  
class setl(x86DispatchInstruction):
  dispatch = (
    (reg8,                {'opcode':[0x0F, 0x9C],       'modrm':0x00}),
    (mem8,                {'opcode':[0x0F, 0x9C],       'modrm':0x00}))
  
class setle(x86DispatchInstruction):
  dispatch = (
    (reg8,                {'opcode':[0x0F, 0x9E],       'modrm':0x00}),
    (mem8,                {'opcode':[0x0F, 0x9E],       'modrm':0x00}))
  
class setna(x86DispatchInstruction):
  dispatch = (
    (reg8,                {'opcode':[0x0F, 0x96],       'modrm':0x00}),
    (mem8,                {'opcode':[0x0F, 0x96],       'modrm':0x00}))
  
class setnae(x86DispatchInstruction):
  dispatch = (
    (reg8,                {'opcode':[0x0F, 0x92],       'modrm':0x00}),
    (mem8,                {'opcode':[0x0F, 0x92],       'modrm':0x00}))
  
class setnb(x86DispatchInstruction):
  dispatch = (
    (reg8,                {'opcode':[0x0F, 0x93],       'modrm':0x00}),
    (mem8,                {'opcode':[0x0F, 0x93],       'modrm':0x00}))
  
class setnbe(x86DispatchInstruction):
  dispatch = (
    (reg8,                {'opcode':[0x0F, 0x97],       'modrm':0x00}),
    (mem8,                {'opcode':[0x0F, 0x97],       'modrm':0x00}))
  
class setnc(x86DispatchInstruction):
  dispatch = (
    (reg8,                {'opcode':[0x0F, 0x93],       'modrm':0x00}),
    (mem8,                {'opcode':[0x0F, 0x93],       'modrm':0x00}))
  
class setne(x86DispatchInstruction):
  dispatch = (
    (reg8,                {'opcode':[0x0F, 0x95],       'modrm':0x00}),
    (mem8,                {'opcode':[0x0F, 0x95],       'modrm':0x00}))
  
class setng(x86DispatchInstruction):
  dispatch = (
    (reg8,                {'opcode':[0x0F, 0x9E],       'modrm':0x00}),
    (mem8,                {'opcode':[0x0F, 0x9E],       'modrm':0x00}))
  
class setnge(x86DispatchInstruction):
  dispatch = (
    (reg8,                {'opcode':[0x0F, 0x9C],       'modrm':0x00}),
    (mem8,                {'opcode':[0x0F, 0x9C],       'modrm':0x00}))
  
class setnl(x86DispatchInstruction):
  dispatch = (
    (reg8,                {'opcode':[0x0F, 0x9D],       'modrm':0x00}),
    (mem8,                {'opcode':[0x0F, 0x9D],       'modrm':0x00}))
  
class setnle(x86DispatchInstruction):
  dispatch = (
    (reg8,                {'opcode':[0x0F, 0x9F],       'modrm':0x00}),
    (mem8,                {'opcode':[0x0F, 0x9F],       'modrm':0x00}))
  
class setno(x86DispatchInstruction):
  dispatch = (
    (reg8,                {'opcode':[0x0F, 0x91],       'modrm':0x00}),
    (mem8,                {'opcode':[0x0F, 0x91],       'modrm':0x00}))
  
class setnp(x86DispatchInstruction):
  dispatch = (
    (reg8,                {'opcode':[0x0F, 0x9B],       'modrm':0x00}),
    (mem8,                {'opcode':[0x0F, 0x9B],       'modrm':0x00}))
  
class setns(x86DispatchInstruction):
  dispatch = (
    (reg8,                {'opcode':[0x0F, 0x99],       'modrm':0x00}),
    (mem8,                {'opcode':[0x0F, 0x99],       'modrm':0x00}))
  
class setnz(x86DispatchInstruction):
  dispatch = (
    (reg8,                {'opcode':[0x0F, 0x95],       'modrm':0x00}),
    (mem8,                {'opcode':[0x0F, 0x95],       'modrm':0x00}))
  
class seto(x86DispatchInstruction):
  dispatch = (
    (reg8,                {'opcode':[0x0F, 0x90],       'modrm':0x00}),
    (mem8,                {'opcode':[0x0F, 0x90],       'modrm':0x00}))
  
class setp(x86DispatchInstruction):
  dispatch = (
    (reg8,                {'opcode':[0x0F, 0x9A],       'modrm':0x00}),
    (mem8,                {'opcode':[0x0F, 0x9A],       'modrm':0x00}))
  
class setpe(x86DispatchInstruction):
  dispatch = (
    (reg8,                {'opcode':[0x0F, 0x9A],       'modrm':0x00}),
    (mem8,                {'opcode':[0x0F, 0x9A],       'modrm':0x00}))
  
class setpo(x86DispatchInstruction):
  dispatch = (
    (reg8,                {'opcode':[0x0F, 0x9B],       'modrm':0x00}),
    (mem8,                {'opcode':[0x0F, 0x9B],       'modrm':0x00}))
  
class sets(x86DispatchInstruction):
  dispatch = (
    (reg8,                {'opcode':[0x0F, 0x98],       'modrm':0x00}),
    (mem8,                {'opcode':[0x0F, 0x98],       'modrm':0x00}))
  
class setz(x86DispatchInstruction):
  dispatch = (
    (reg8,                {'opcode':[0x0F, 0x94],       'modrm':0x00}),
    (mem8,                {'opcode':[0x0F, 0x94],       'modrm':0x00}))
  
class sfence(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0x0F, 0xAE, 0xF8], 'modrm':None}
  
class shl(x86DispatchInstruction):
  dispatch = (
    (reg32_1,             {'opcode':[0xD1],             'modrm':0x20}),
    (mem32_1,             {'opcode':[0xD1],             'modrm':0x20}),
    (reg32_cl,            {'opcode':[0xD3],             'modrm':0x20}),
    (mem32_cl,            {'opcode':[0xD3],             'modrm':0x20}),
    (reg32_imm8,          {'opcode':[0xC1],             'modrm':0x20}),
    (mem32_imm8,          {'opcode':[0xC1],             'modrm':0x20}),
    (reg16_1,             {'opcode':[0x66, 0xD1],       'modrm':0x20}),
    (mem16_1,             {'opcode':[0x66, 0xD1],       'modrm':0x20}),
    (reg16_cl,            {'opcode':[0x66, 0xD3],       'modrm':0x20}),
    (mem16_cl,            {'opcode':[0x66, 0xD3],       'modrm':0x20}),
    (reg16_imm8,          {'opcode':[0x66, 0xC1],       'modrm':0x20}),
    (mem16_imm8,          {'opcode':[0x66, 0xC1],       'modrm':0x20}),
    (reg8_1,              {'opcode':[0xD0],             'modrm':0x20}),
    (mem8_1,              {'opcode':[0xD0],             'modrm':0x20}),
    (reg8_cl,             {'opcode':[0xD2],             'modrm':0x20}),
    (mem8_cl,             {'opcode':[0xD2],             'modrm':0x20}),
    (reg8_imm8,           {'opcode':[0xC0],             'modrm':0x20}),
    (mem8_imm8,           {'opcode':[0xC0],             'modrm':0x20}))
    
class shld(x86DispatchInstruction):
  dispatch = (
    (reg32_reg32_imm8,    {'opcode':[0x0F, 0xA4],       'modrm':None}),
    (mem32_reg32_imm8,    {'opcode':[0x0F, 0xA4],       'modrm':None}),
    (reg32_reg32_cl,      {'opcode':[0x0F, 0xA5],       'modrm':None}),
    (mem32_reg32_cl,      {'opcode':[0x0F, 0xA5],       'modrm':None}),
    (reg16_reg16_imm8,    {'opcode':[0x66, 0x0F, 0xA4], 'modrm':None}),
    (mem16_reg16_imm8,    {'opcode':[0x66, 0x0F, 0xA4], 'modrm':None}),
    (reg16_reg16_cl,      {'opcode':[0x66, 0x0F, 0xA5], 'modrm':None}),
    (mem16_reg16_cl,      {'opcode':[0x66, 0x0F, 0xA5], 'modrm':None}))
    
class shr(x86DispatchInstruction):
  dispatch = (
    (reg32_1,             {'opcode':[0xD1],             'modrm':0x28}),
    (mem32_1,             {'opcode':[0xD1],             'modrm':0x28}),
    (reg32_cl,            {'opcode':[0xD3],             'modrm':0x28}),
    (mem32_cl,            {'opcode':[0xD3],             'modrm':0x28}),
    (reg32_imm8,          {'opcode':[0xC1],             'modrm':0x28}),
    (mem32_imm8,          {'opcode':[0xC1],             'modrm':0x28}),
    (reg16_1,             {'opcode':[0x66, 0xD1],       'modrm':0x28}),
    (mem16_1,             {'opcode':[0x66, 0xD1],       'modrm':0x28}),
    (reg16_cl,            {'opcode':[0x66, 0xD3],       'modrm':0x28}),
    (mem16_cl,            {'opcode':[0x66, 0xD3],       'modrm':0x28}),
    (reg16_imm8,          {'opcode':[0x66, 0xC1],       'modrm':0x28}),
    (mem16_imm8,          {'opcode':[0x66, 0xC1],       'modrm':0x28}),
    (reg8_1,              {'opcode':[0xD0],             'modrm':0x28}),
    (mem8_1,              {'opcode':[0xD0],             'modrm':0x28}),
    (reg8_cl,             {'opcode':[0xD2],             'modrm':0x28}),
    (mem8_cl,             {'opcode':[0xD2],             'modrm':0x28}),
    (reg8_imm8,           {'opcode':[0xC0],             'modrm':0x28}),
    (mem8_imm8,           {'opcode':[0xC0],             'modrm':0x28}))
    
class shrd(x86DispatchInstruction):
  dispatch = (
    (reg32_reg32_imm8,    {'opcode':[0x0F, 0xAC],       'modrm':None}),
    (mem32_reg32_imm8,    {'opcode':[0x0F, 0xAC],       'modrm':None}),
    (reg32_reg32_cl,      {'opcode':[0x0F, 0xAD],       'modrm':None}),
    (mem32_reg32_cl,      {'opcode':[0x0F, 0xAD],       'modrm':None}),
    (reg16_reg16_imm8,    {'opcode':[0x66, 0x0F, 0xAC], 'modrm':None}),
    (mem16_reg16_imm8,    {'opcode':[0x66, 0x0F, 0xAC], 'modrm':None}),
    (reg16_reg16_cl,      {'opcode':[0x66, 0x0F, 0xAD], 'modrm':None}),
    (mem16_reg16_cl,      {'opcode':[0x66, 0x0F, 0xAD], 'modrm':None}))
  
class stc(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0xF9],             'modrm':None}
  
class std(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0xFD],             'modrm':None}
  
class stosb(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0xAA],             'modrm':None}
  
class stosd(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0xAB],             'modrm':None}
  
class stosw(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0x66, 0xAB],       'modrm':None}
  
class sub(x86DispatchInstruction):
  dispatch = (
    (reg32_imm8,          {'opcode':[0x83],             'modrm':0x28}),
    (mem32_imm8,          {'opcode':[0x83],             'modrm':0x28}),
    (eax_imm32,           {'opcode':[0x2D],             'modrm':None}),
    (reg32_imm32,         {'opcode':[0x81],             'modrm':0x28}),
    (mem32_imm32,         {'opcode':[0x81],             'modrm':0x28}),
    (reg32_reg32,         {'opcode':[0x29],             'modrm':None}),
    (mem32_reg32,         {'opcode':[0x29],             'modrm':None}),
    (reg32_mem32,         {'opcode':[0x2B],             'modrm':None}),
    (ax_imm16,            {'opcode':[0x66, 0x2D],       'modrm':None}),
    (reg16_imm8,          {'opcode':[0x66, 0x83],       'modrm':0x28}),
    (mem16_imm8,          {'opcode':[0x66, 0x83],       'modrm':0x28}),
    (reg16_imm16,         {'opcode':[0x66, 0x81],       'modrm':0x28}),
    (mem16_imm16,         {'opcode':[0x66, 0x81],       'modrm':0x28}),
    (reg16_reg16,         {'opcode':[0x66, 0x29],       'modrm':None}),
    (mem16_reg16,         {'opcode':[0x66, 0x29],       'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x2B],       'modrm':None}),
    (al_imm8,             {'opcode':[0x2C],             'modrm':None}),
    (reg8_imm8,           {'opcode':[0x80],             'modrm':0x28}),
    (mem8_imm8,           {'opcode':[0x80],             'modrm':0x28}),
    (reg8_reg8,           {'opcode':[0x28],             'modrm':None}),
    (mem8_reg8,           {'opcode':[0x28],             'modrm':None}),
    (reg8_mem8,           {'opcode':[0x2A],             'modrm':None}))
    
class test(x86DispatchInstruction):
  dispatch = (
    (eax_imm32,           {'opcode':[0xA9],             'modrm':None}),
    (reg32_imm32,         {'opcode':[0xF7],             'modrm':0x00}),
    (mem32_imm32,         {'opcode':[0xF7],             'modrm':0x00}),
    (reg32_reg32,         {'opcode':[0x85],             'modrm':None}),
    (mem32_reg32,         {'opcode':[0x85],             'modrm':None}),
    (ax_imm16,            {'opcode':[0x66, 0xA9],       'modrm':None}),
    (reg16_imm16,         {'opcode':[0x66, 0xF7],       'modrm':0x00}),
    (mem16_imm16,         {'opcode':[0x66, 0xF7],       'modrm':0x00}),
    (reg16_reg16,         {'opcode':[0x66, 0x85],       'modrm':None}),
    (mem16_reg16,         {'opcode':[0x66, 0x85],       'modrm':None}),
    (al_imm8,             {'opcode':[0xA8],             'modrm':None}),
    (reg8_imm8,           {'opcode':[0xF6],             'modrm':0x00}),
    (mem8_imm8,           {'opcode':[0xF6],             'modrm':0x00}),
    (reg8_reg8,           {'opcode':[0x84],             'modrm':None}),
    (mem8_reg8,           {'opcode':[0x84],             'modrm':None}))
    
class ud2(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0x0F, 0x0B],       'modrm':None}
  
class xadd(x86DispatchInstruction):
  dispatch = (
    (reg32_reg32,         {'opcode':[0x0F, 0xC1],       'modrm':None}),
    (mem32_reg32,         {'opcode':[0x0F, 0xC1],       'modrm':None}),
    (reg16_reg16,         {'opcode':[0x66, 0x0F, 0xC1], 'modrm':None}),
    (mem16_reg16,         {'opcode':[0x66, 0x0F, 0xC1], 'modrm':None}),
    (reg8_reg8,           {'opcode':[0x0F, 0xC0],       'modrm':None}),
    (mem8_reg8,           {'opcode':[0x0F, 0xC0],       'modrm':None}))
    
class xchg(x86DispatchInstruction):
  dispatch = (
    (eax_reg32,           {'opcode':[0x90],             'modrm':None}),
    (reg32_eax,           {'opcode':[0x90],             'modrm':None}),
    (reg32_reg32,         {'opcode':[0x87],             'modrm':None}),
    (mem32_reg32,         {'opcode':[0x87],             'modrm':None}),
    (reg32_mem32,         {'opcode':[0x87],             'modrm':None}),
    (reg16_ax,            {'opcode':[0x66, 0x90],       'modrm':None}),
    (ax_reg16,            {'opcode':[0x66, 0x90],       'modrm':None}),
    (reg16_reg16,         {'opcode':[0x66, 0x87],       'modrm':None}),
    (mem16_reg16,         {'opcode':[0x66, 0x87],       'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x87],       'modrm':None}),
    (reg8_reg8,           {'opcode':[0x86],             'modrm':None}),
    (mem8_reg8,           {'opcode':[0x86],             'modrm':None}),
    (reg8_mem8,           {'opcode':[0x86],             'modrm':None}))
    
class xlatb(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0xD7],             'modrm':None}
  
class xor(x86DispatchInstruction):
  dispatch = (
    (reg32_imm8,          {'opcode':[0x83],             'modrm':0x30}),
    (mem32_imm8,          {'opcode':[0x83],             'modrm':0x30}),
    (eax_imm32,           {'opcode':[0x35],             'modrm':None}),
    (reg32_imm32,         {'opcode':[0x81],             'modrm':0x30}),
    (mem32_imm32,         {'opcode':[0x81],             'modrm':0x30}),
    (reg32_reg32,         {'opcode':[0x31],             'modrm':None}),
    (mem32_reg32,         {'opcode':[0x31],             'modrm':None}),
    (reg32_mem32,         {'opcode':[0x33],             'modrm':None}),
    (reg16_imm8,          {'opcode':[0x66, 0x83],       'modrm':0x30}),
    (mem16_imm8,          {'opcode':[0x66, 0x83],       'modrm':0x30}),
    (ax_imm16,            {'opcode':[0x66, 0x35],       'modrm':None}),
    (reg16_imm16,         {'opcode':[0x66, 0x81],       'modrm':0x30}),
    (mem16_imm16,         {'opcode':[0x66, 0x81],       'modrm':0x30}),
    (reg16_reg16,         {'opcode':[0x66, 0x31],       'modrm':None}),
    (mem16_reg16,         {'opcode':[0x66, 0x31],       'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x33],       'modrm':None}),
    (al_imm8,             {'opcode':[0x34],             'modrm':None}),
    (reg8_imm8,           {'opcode':[0x80],             'modrm':0x30}),
    (mem8_imm8,           {'opcode':[0x80],             'modrm':0x30}),
    (reg8_reg8,           {'opcode':[0x30],             'modrm':None}),
    (mem8_reg8,           {'opcode':[0x30],             'modrm':None}),
    (reg8_mem8,           {'opcode':[0x32],             'modrm':None}))


# X87_ISA = (
class f2xm1(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0xD9, 0xF0],       'modrm':None}
    
class fabs(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0xD9, 0xE1],       'modrm':None}
  
class fadd(x86DispatchInstruction):
  dispatch = (
    (st0_sti,    {'opcode':[0xD8, 0xC0],       'modrm':None}),
    (sti_st0,    {'opcode':[0xDC, 0xC0],       'modrm':None}),
    (mem32,      {'opcode':[0xD8],             'modrm':0x00}),
    (mem64,      {'opcode':[0xDC],             'modrm':0x00}))
      
class faddp(x86DispatchInstruction):
  dispatch = (
    (no_op,      {'opcode':[0xDE, 0xC1],       'modrm':None}),
    (sti_st0,    {'opcode':[0xDE, 0xC0],       'modrm':None}))
    
class fiadd(x86DispatchInstruction):
  dispatch = (
    (mem32,      {'opcode':[0xDA],             'modrm':0x00}),
    (mem16,      {'opcode':[0xDE],             'modrm':0x00}))
    
class fchs(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0xD9, 0xE0],       'modrm':None}
  
class fcmovb(x86Instruction):
  machine_inst = st0_sti
  params = {'opcode':[0xDA, 0xC0],       'modrm':None}
  
class fcmovbe(x86Instruction):
  machine_inst = st0_sti
  params = {'opcode':[0xDA, 0xD0],       'modrm':None}
  
class fcmove(x86Instruction):
  machine_inst = st0_sti
  params = {'opcode':[0xDA, 0xC8],       'modrm':None}
  
class fcmovnb(x86Instruction):
  machine_inst = st0_sti
  params = {'opcode':[0xDB, 0xC0],       'modrm':None}
  
class fcmovnbe(x86Instruction):
  machine_inst = st0_sti
  params = {'opcode':[0xDB, 0xD0],       'modrm':None}
  
class fcmovne(x86Instruction):
  machine_inst = st0_sti
  params = {'opcode':[0xDB, 0xC8],       'modrm':None}
  
class fcmovnu(x86Instruction):
  machine_inst = st0_sti
  params = {'opcode':[0xDB, 0xD8],       'modrm':None}
  
class fcmovu(x86Instruction):
  machine_inst = st0_sti
  params = {'opcode':[0xDA, 0xD8],       'modrm':None}
  
class fcom(x86DispatchInstruction):
  dispatch = (
    (no_op,      {'opcode':[0xD8, 0xD1],       'modrm':None}),
    (sti,        {'opcode':[0xD8, 0xD0],       'modrm':None}),
    (mem32,      {'opcode':[0xD8],             'modrm':0x10}),
    (mem64,      {'opcode':[0xDC],             'modrm':0x10}))
    
class fcomp(x86DispatchInstruction):
  dispatch = (
    (no_op,      {'opcode':[0xD8, 0xD9],       'modrm':None}),
    (sti,        {'opcode':[0xD8, 0xD8],       'modrm':None}),
    (mem32,      {'opcode':[0xD8],             'modrm':0x18}),
    (mem64,      {'opcode':[0xDC],             'modrm':0x18}))
    
class fcompp(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0xDE, 0xD9],       'modrm':None}
  
class fcomi(x86Instruction):
  machine_inst = st0_sti
  params = {'opcode':[0xDB, 0xF0],       'modrm':None}
  
class fcomip(x86Instruction):
  machine_inst = st0_sti
  params = {'opcode':[0xDF, 0xF0],       'modrm':None}
  
class fcos(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0xD9, 0xFF],       'modrm':None}
  
class fdecstp(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0xD9, 0xF6],       'modrm':None}
  
class fdiv(x86DispatchInstruction):
  dispatch = (
    (st0_sti,    {'opcode':[0xD8, 0xF0],       'modrm':None}),
    (sti_st0,    {'opcode':[0xDC, 0xF8],       'modrm':None}),
    (mem32,      {'opcode':[0xD8],             'modrm':0x30}),
    (mem64,      {'opcode':[0xDC],             'modrm':0x30}))
    
class fdivp(x86DispatchInstruction):
  dispatch = (
    (no_op,      {'opcode':[0xDE, 0xF9],       'modrm':None}),
    (sti_st0,    {'opcode':[0xDE, 0xF8],       'modrm':None}))
  
class fidiv(x86DispatchInstruction):
  dispatch = (
    (mem32,      {'opcode':[0xDA],             'modrm':0x30}),
    (mem16,      {'opcode':[0xDE],             'modrm':0x30}))
    
class fdivr(x86DispatchInstruction):
  dispatch = (
    (st0_sti,    {'opcode':[0xD8, 0xF8],       'modrm':None}),
    (sti_st0,    {'opcode':[0xDC, 0xF0],       'modrm':None}),
    (mem32,      {'opcode':[0xD8],             'modrm':0x38}),
    (mem64,      {'opcode':[0xDC],             'modrm':0x38}))
    
class fdivrp(x86DispatchInstruction):
  dispatch = (
    (no_op,      {'opcode':[0xDE, 0xF1],       'modrm':None}),
    (sti_st0,    {'opcode':[0xDE, 0xF0],       'modrm':None}))
  
class fidivr(x86DispatchInstruction):
  dispatch = (
    (mem32,      {'opcode':[0xDA],             'modrm':0x38}),
    (mem16,      {'opcode':[0xDE],             'modrm':0x38}))
  
class ffree(x86Instruction):
  machine_inst = sti
  params = {'opcode':[0xDD, 0xC0],       'modrm':None}
  
class ficom(x86DispatchInstruction):
  dispatch = (
    (mem32,      {'opcode':[0xDA],             'modrm':0x10}),
    (mem16,      {'opcode':[0xDE],             'modrm':0x10}))
    
class ficomp(x86DispatchInstruction):
  dispatch = (
    (mem32,      {'opcode':[0xDA],             'modrm':0x18}),
    (mem16,      {'opcode':[0xDE],             'modrm':0x18}))
    
class fild(x86DispatchInstruction):
  dispatch = (
    (mem64,      {'opcode':[0xDF],             'modrm':0x28}),
    (mem32,      {'opcode':[0xDB],             'modrm':0x00}),
    (mem16,      {'opcode':[0xDF],             'modrm':0x00}))
    
class fincstp(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0xD9, 0xF7],       'modrm':None}
  
class finit(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0x9B, 0xDB, 0xE3], 'modrm':None}
  
class fninit(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0xDB, 0xE3],       'modrm':None}
  
class fist(x86DispatchInstruction):
  dispatch = (
    (mem32,      {'opcode':[0xDB],             'modrm':0x10}),
    (mem16,      {'opcode':[0xDF],             'modrm':0x10}))
    
class fistp(x86DispatchInstruction):
  dispatch = (
    (mem64,      {'opcode':[0xDF],             'modrm':0x38}),
    (mem32,      {'opcode':[0xDB],             'modrm':0x18}),
    (mem16,      {'opcode':[0xDF],             'modrm':0x18}))
    
class fisttp(x86DispatchInstruction):
  dispatch = (
    (mem64,      {'opcode':[0xDD],             'modrm':0x08}),
    (mem32,      {'opcode':[0xDB],             'modrm':0x08}),
    (mem16,      {'opcode':[0xDF],             'modrm':0x08}))
  
class fld(x86DispatchInstruction):
  dispatch = (
    (sti,        {'opcode':[0xD9, 0xC0],       'modrm':None}),
    (mem80,      {'opcode':[0xDB],             'modrm':0x28}),
    (mem64,      {'opcode':[0xDD],             'modrm':0x00}),
    (mem32,      {'opcode':[0xD9],             'modrm':0x00}))
    
class fld1(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0xD9, 0xE8],       'modrm':None}
  
class fldcw(x86Instruction):
  machine_inst = mem16
  params = {'opcode':[0xD9],             'modrm':0x28}
  
class fldenv(x86Instruction):
  machine_inst = mem228
  params = {'opcode':[0xD9],             'modrm':0x20}
  
class fldl2e(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0xD9, 0xEA],       'modrm':None}
  
class fldl2t(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0xD9, 0xE9],       'modrm':None}
  
class fldlg2(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0xD9, 0xEC],       'modrm':None}
  
class fldln2(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0xD9, 0xED],       'modrm':None}
  
class fldpi(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0xD9, 0xEB],       'modrm':None}
  
class fldz(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0xD9, 0xEE],       'modrm':None}
  
class fmul(x86DispatchInstruction):
  dispatch = (
    (st0_sti,    {'opcode':[0xD8, 0xC8],       'modrm':None}),
    (sti_st0,    {'opcode':[0xDC, 0xC8],       'modrm':None}),
    (mem32,      {'opcode':[0xD8],             'modrm':0x08}),
    (mem64,      {'opcode':[0xDC],             'modrm':0x08}))
    
class fmulp(x86DispatchInstruction):
  dispatch = (
    (no_op,      {'opcode':[0xDE, 0xC9],       'modrm':None}),
    (sti_st0,    {'opcode':[0xDE, 0xC8],       'modrm':None}))
    
class fimul(x86DispatchInstruction):
  dispatch = (
    (mem32,      {'opcode':[0xDA],             'modrm':0x08}),
    (mem16,      {'opcode':[0xDE],             'modrm':0x08}))
    
class fnop(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0xD9, 0xD0],       'modrm':None}
  
class fpatan(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0xD9, 0xF3],       'modrm':None}
  
class fprem(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0xD9, 0xF8],       'modrm':None}
  
class fprem1(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0xD9, 0xF5],       'modrm':None}
  
class fptan(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0xD9, 0xF2],       'modrm':None}
  
class frndint(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0xD9, 0xFC],       'modrm':None}
  
class frstor(x86Instruction):
  machine_inst = mem752
  params = {'opcode':[0xDD],             'modrm':0x20}
  
class fsave(x86Instruction):
  machine_inst = mem752
  params = {'opcode':[0x9B, 0xDD],       'modrm':0x30}
  
class fnsave(x86Instruction):
  machine_inst = mem752
  params = {'opcode':[0xDD],             'modrm':0x30}
  
class fscale(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0xD9, 0xFD],       'modrm':None}
  
class fsin(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0xD9, 0xFE],       'modrm':None}
  
class fsincos(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0xD9, 0xFB],       'modrm':None}
  
class fsqrt(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0xD9, 0xFA],       'modrm':None}
  
class fst(x86DispatchInstruction):
  dispatch = (
    (sti,        {'opcode':[0xDD, 0xD0],       'modrm':None}),
    (mem64,      {'opcode':[0xDD],             'modrm':0x10}),
    (mem32,      {'opcode':[0xD9],             'modrm':0x10}))
    
class fstp(x86DispatchInstruction):
  dispatch = (
    (sti,        {'opcode':[0xDD, 0xD8],       'modrm':None}),
    (mem80,      {'opcode':[0xDB],             'modrm':0x38}),
    (mem64,      {'opcode':[0xDD],             'modrm':0x18}),
    (mem32,      {'opcode':[0xD9],             'modrm':0x18}))
    
class fstcw(x86Instruction):
  machine_inst = mem16
  params = {'opcode':[0x9B, 0xD9],       'modrm':0x38}
  
class fnstcw(x86Instruction):
  machine_inst = mem16
  params = {'opcode':[0xD9],             'modrm':0x38}
  
class fstenv(x86Instruction):
  machine_inst = mem228
  params = {'opcode':[0x9B, 0xD9],       'modrm':0x30}
  
class fnstenv(x86Instruction):
  machine_inst = mem228
  params = {'opcode':[0xD9],             'modrm':0x30}
  
class fstsw(x86DispatchInstruction):
  dispatch = (
    (ax,         {'opcode':[0x9B, 0xDF, 0xE0], 'modrm':None}),
    (mem16,      {'opcode':[0x9B, 0xDD],       'modrm':0x38}))
    
class fnstsw(x86DispatchInstruction):
  dispatch = (
    (ax,         {'opcode':[0xDF, 0xE0],       'modrm':None}),
    (mem16,      {'opcode':[0xDD],             'modrm':0x38}))
    
class fsub(x86DispatchInstruction):
  dispatch = (
    (st0_sti,    {'opcode':[0xD8, 0xE0],       'modrm':None}),
    (sti_st0,    {'opcode':[0xDC, 0xE8],       'modrm':None}),
    (mem32,      {'opcode':[0xD8],             'modrm':0x20}),
    (mem64,      {'opcode':[0xDC],             'modrm':0x20}))
    
class fsubp(x86DispatchInstruction):
  dispatch = (
    (no_op,      {'opcode':[0xDE, 0xE9],       'modrm':None}),
    (sti_st0,    {'opcode':[0xDE, 0xE8],       'modrm':None}))
  
class fisub(x86DispatchInstruction):
  dispatch = (
    (mem32,      {'opcode':[0xDA],             'modrm':0x20}),
    (mem16,      {'opcode':[0xDE],             'modrm':0x20}))
  
class fsubr(x86DispatchInstruction):
  dispatch = (
    (st0_sti,    {'opcode':[0xD8, 0xE8],       'modrm':None}),
    (sti_st0,    {'opcode':[0xDC, 0xE0],       'modrm':None}),
    (mem32,      {'opcode':[0xD8],             'modrm':0x28}),
    (mem64,      {'opcode':[0xDC],             'modrm':0x28}))
    
class fsubrp(x86DispatchInstruction):
  dispatch = (
    (no_op,      {'opcode':[0xDE, 0xE1],       'modrm':None}),
    (sti_st0,    {'opcode':[0xDE, 0xE0],       'modrm':None}))
    
class fisubr(x86DispatchInstruction):
  dispatch = (
    (mem32,      {'opcode':[0xDA],             'modrm':0x28}),
    (mem16,      {'opcode':[0xDE],             'modrm':0x28}))
  
class ftst(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0xD9, 0xE4],       'modrm':None}
  
class fucom(x86DispatchInstruction):
  dispatch = (
    (no_op,      {'opcode':[0xDD, 0xE1],       'modrm':None}),
    (sti,        {'opcode':[0xDD, 0xE0],       'modrm':None}))
  
class fucomp(x86DispatchInstruction):
  dispatch = (
    (no_op,      {'opcode':[0xDD, 0xE9],       'modrm':None}),
    (sti,        {'opcode':[0xDD, 0xE8],       'modrm':None}))
  
class fucompp(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0xDA, 0xE9],       'modrm':None}
  
class fucomi(x86Instruction):
  machine_inst = st0_sti
  params = {'opcode':[0xDB, 0xE8],       'modrm':None}
  
class fucomip(x86Instruction):
  machine_inst = st0_sti
  params = {'opcode':[0xDF, 0xE8],       'modrm':None}
  
class fwait(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0x9B],             'modrm':None}
  
class fxam(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0xD9, 0xE5],       'modrm':None}
  
class fxch(x86DispatchInstruction):
  dispatch = (
    (no_op,      {'opcode':[0xD9, 0xC9],       'modrm':None}),
    (sti,        {'opcode':[0xD9, 0xC8],       'modrm':None}))
    
class fxrstor(x86Instruction):
  machine_inst = mem512
  params = {'opcode':[0x0F, 0xAE],       'modrm':0x08}
 #sse?

class fxsave(x86Instruction):
  machine_inst = mem512
  params = {'opcode':[0x0F, 0xAE],       'modrm':0x00}
 #sse?

class fxtract(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0xD9, 0xF4],       'modrm':None}
  
class fyl2x(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0xD9, 0xF1],       'modrm':None}
  
class fyl2xp1(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0xD9, 0xF9],       'modrm':None}
  

#SSE_ISA = (
class addpd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,      {'opcode':[0x66, 0x0F, 0x58], 'modrm':None}),
    (xmm_mem128,   {'opcode':[0x66, 0x0F, 0x58], 'modrm':None}))
  arch_ext = 2

class addps(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,      {'opcode':[0x0F, 0x58], 'modrm':None}),
    (xmm_mem128,   {'opcode':[0x0F, 0x58], 'modrm':None}))
  arch_ext = 1

class addsd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,      {'opcode':[0xF2, 0x0F, 0x58], 'modrm':None}),
    (xmm_mem128,   {'opcode':[0xF2, 0x0F, 0x58], 'modrm':None}))
  arch_ext = 2

class addss(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,      {'opcode':[0xF3, 0x0F, 0x58], 'modrm':None}),
    (xmm_mem128,   {'opcode':[0xF3, 0x0F, 0x58], 'modrm':None}))
  arch_ext = 1

class addsubpd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,      {'opcode':[0x66, 0x0F, 0xD0], 'modrm':None}),
    (xmm_mem128,   {'opcode':[0x66, 0x0F, 0xD0], 'modrm':None}))
  arch_ext = 3

class addsubps(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,      {'opcode':[0xF2, 0x0F, 0xD0], 'modrm':None}),
    (xmm_mem128,   {'opcode':[0xF2, 0x0F, 0xD0], 'modrm':None}))
  arch_ext = 3

class andnpd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,      {'opcode':[0x66, 0x0F, 0x55], 'modrm':None}),
    (xmm_mem128,   {'opcode':[0x66, 0x0F, 0x55], 'modrm':None}))
  arch_ext = 2

class andnps(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,      {'opcode':[0x0F, 0x55], 'modrm':None}),
    (xmm_mem128,   {'opcode':[0x0F, 0x55], 'modrm':None}))
  arch_ext = 1

class andpd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,      {'opcode':[0x66, 0x0F, 0x54], 'modrm':None}),
    (xmm_mem128,   {'opcode':[0x66, 0x0F, 0x54], 'modrm':None}))
  arch_ext = 2

class andps(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,      {'opcode':[0x0F, 0x54], 'modrm':None}),
    (xmm_mem128,   {'opcode':[0x0F, 0x54], 'modrm':None}))
  arch_ext = 1

class blendpd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm8,    {'opcode':[0x66, 0x0F, 0x3A, 0x0D], 'modrm':None}),
    (xmm_mem128_imm8, {'opcode':[0x66, 0x0F, 0x3A, 0x0D], 'modrm':None}))
  arch_ext = 4

class blendps(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm8,    {'opcode':[0x66, 0x0F, 0x3A, 0x0C], 'modrm':None}),
    (xmm_mem128_imm8, {'opcode':[0x66, 0x0F, 0x3A, 0x0C], 'modrm':None}))
  arch_ext = 4

class blendvpd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x15], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x15], 'modrm':None}))
  arch_ext = 4

class blendvps(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x14], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x14], 'modrm':None}))
  arch_ext = 4

class cmpeqpd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0x66, 0x0F, 0xC2], 'modrm':None, 'imm':0}),
    (xmm_mem128_imm, {'opcode':[0x66, 0x0F, 0xC2], 'modrm':None, 'imm':0}))
  arch_ext = 2

class cmpeqps(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0x0F, 0xC2], 'modrm':None, 'imm':0}),
    (xmm_mem128_imm, {'opcode':[0x0F, 0xC2], 'modrm':None, 'imm':0}))
  arch_ext = 1

class cmpeqsd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0xF2, 0x0F, 0xC2], 'modrm':None, 'imm':0}),
    (xmm_mem64_imm,  {'opcode':[0xF2, 0x0F, 0xC2], 'modrm':None, 'imm':0}))
  arch_ext = 2

class cmpeqss(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0xF3, 0x0F, 0xC2], 'modrm':None, 'imm':0}),
    (xmm_mem32_imm,  {'opcode':[0xF3, 0x0F, 0xC2], 'modrm':None, 'imm':0}))
  arch_ext = 1

class cmplepd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0x66, 0x0F, 0xC2], 'modrm':None, 'imm':2}),
    (xmm_mem128_imm, {'opcode':[0x66, 0x0F, 0xC2], 'modrm':None, 'imm':2}))
  arch_ext = 2

class cmpleps(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0x0F, 0xC2], 'modrm':None, 'imm':2}),
    (xmm_mem128_imm, {'opcode':[0x0F, 0xC2], 'modrm':None, 'imm':2}))
  arch_ext = 1

class cmplesd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0xF2, 0x0F, 0xC2], 'modrm':None, 'imm':2}),
    (xmm_mem64_imm,  {'opcode':[0xF2, 0x0F, 0xC2], 'modrm':None, 'imm':2}))
  arch_ext = 2

class cmpless(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0xF3, 0x0F, 0xC2], 'modrm':None, 'imm':2}),
    (xmm_mem32_imm,  {'opcode':[0xF3, 0x0F, 0xC2], 'modrm':None, 'imm':2}))
  arch_ext = 1

class cmpltpd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0x66, 0x0F, 0xC2], 'modrm':None, 'imm':1}),
    (xmm_mem128_imm, {'opcode':[0x66, 0x0F, 0xC2], 'modrm':None, 'imm':1}))
  arch_ext = 2

class cmpltps(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0x0F, 0xC2], 'modrm':None, 'imm':1}),
    (xmm_mem128_imm, {'opcode':[0x0F, 0xC2], 'modrm':None, 'imm':1}))
  arch_ext = 1

class cmpltsd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0xF2, 0x0F, 0xC2], 'modrm':None, 'imm':1}),
    (xmm_mem64_imm,  {'opcode':[0xF2, 0x0F, 0xC2], 'modrm':None, 'imm':1}))
  arch_ext = 2

class cmpltss(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0xF3, 0x0F, 0xC2], 'modrm':None, 'imm':1}),
    (xmm_mem32_imm,  {'opcode':[0xF3, 0x0F, 0xC2], 'modrm':None, 'imm':1}))
  arch_ext = 1

class cmpneqpd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0x66, 0x0F, 0xC2], 'modrm':None, 'imm':4}),
    (xmm_mem128_imm, {'opcode':[0x66, 0x0F, 0xC2], 'modrm':None, 'imm':4}))
  arch_ext = 2

class cmpneqps(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0x0F, 0xC2], 'modrm':None, 'imm':4}),
    (xmm_mem128_imm, {'opcode':[0x0F, 0xC2], 'modrm':None, 'imm':4}))
  arch_ext = 1

class cmpneqsd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0xF2, 0x0F, 0xC2], 'modrm':None, 'imm':4}),
    (xmm_mem64_imm,  {'opcode':[0xF2, 0x0F, 0xC2], 'modrm':None, 'imm':4}))
  arch_ext = 2

class cmpneqss(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0xF3, 0x0F, 0xC2], 'modrm':None, 'imm':4}),
    (xmm_mem32_imm,  {'opcode':[0xF3, 0x0F, 0xC2], 'modrm':None, 'imm':4}))
  arch_ext = 1

class cmpnlepd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0x66, 0x0F, 0xC2], 'modrm':None, 'imm':6}),
    (xmm_mem128_imm, {'opcode':[0x66, 0x0F, 0xC2], 'modrm':None, 'imm':6}))
  arch_ext = 2

class cmpnleps(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0x0F, 0xC2], 'modrm':None, 'imm':6}),
    (xmm_mem128_imm, {'opcode':[0x0F, 0xC2], 'modrm':None, 'imm':6}))
  arch_ext = 1

class cmpnlesd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0xF2, 0x0F, 0xC2], 'modrm':None, 'imm':6}),
    (xmm_mem64_imm,  {'opcode':[0xF2, 0x0F, 0xC2], 'modrm':None, 'imm':6}))
  arch_ext = 2

class cmpnless(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0xF3, 0x0F, 0xC2], 'modrm':None, 'imm':6}),
    (xmm_mem32_imm,  {'opcode':[0xF3, 0x0F, 0xC2], 'modrm':None, 'imm':6}))
  arch_ext = 1

class cmpnltpd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0x66, 0x0F, 0xC2], 'modrm':None, 'imm':5}),
    (xmm_mem128_imm, {'opcode':[0x66, 0x0F, 0xC2], 'modrm':None, 'imm':5}))
  arch_ext = 2

class cmpnltps(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0x0F, 0xC2], 'modrm':None, 'imm':5}),
    (xmm_mem128_imm, {'opcode':[0x0F, 0xC2], 'modrm':None, 'imm':5}))
  arch_ext = 1

class cmpnltsd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0xF2, 0x0F, 0xC2], 'modrm':None, 'imm':5}),
    (xmm_mem64_imm,  {'opcode':[0xF2, 0x0F, 0xC2], 'modrm':None, 'imm':5}))
  arch_ext = 2

class cmpnltss(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0xF3, 0x0F, 0xC2], 'modrm':None, 'imm':5}),
    (xmm_mem32_imm,  {'opcode':[0xF3, 0x0F, 0xC2], 'modrm':None, 'imm':5}))
  arch_ext = 1

class cmpordpd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0x66, 0x0F, 0xC2], 'modrm':None, 'imm':7}),
    (xmm_mem128_imm, {'opcode':[0x66, 0x0F, 0xC2], 'modrm':None, 'imm':7}))
  arch_ext = 2

class cmpordps(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0x0F, 0xC2], 'modrm':None, 'imm':7}),
    (xmm_mem128_imm, {'opcode':[0x0F, 0xC2], 'modrm':None, 'imm':7}))
  arch_ext = 1

class cmpordsd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0xF2, 0x0F, 0xC2], 'modrm':None, 'imm':7}),
    (xmm_mem64_imm,  {'opcode':[0xF2, 0x0F, 0xC2], 'modrm':None, 'imm':7}))
  arch_ext = 2

class cmpordss(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0xF3, 0x0F, 0xC2], 'modrm':None, 'imm':7}),
    (xmm_mem32_imm,  {'opcode':[0xF3, 0x0F, 0xC2], 'modrm':None, 'imm':7}))
  arch_ext = 1

class cmpunordpd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0x66, 0x0F, 0xC2], 'modrm':None, 'imm':3}),
    (xmm_mem128_imm, {'opcode':[0x66, 0x0F, 0xC2], 'modrm':None, 'imm':3}))
  arch_ext = 2

class cmpunordps(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0x0F, 0xC2], 'modrm':None, 'imm':3}),
    (xmm_mem128_imm, {'opcode':[0x0F, 0xC2], 'modrm':None, 'imm':3}))
  arch_ext = 1

class cmpunordsd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0xF2, 0x0F, 0xC2], 'modrm':None, 'imm':3}),
    (xmm_mem64_imm,  {'opcode':[0xF2, 0x0F, 0xC2], 'modrm':None, 'imm':3}))
  arch_ext = 2

class cmpunordss(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0xF3, 0x0F, 0xC2], 'modrm':None, 'imm':3}),
    (xmm_mem32_imm,  {'opcode':[0xF3, 0x0F, 0xC2], 'modrm':None, 'imm':3}))
  arch_ext = 1

class cmppd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm8,    {'opcode':[0x66, 0x0F, 0xC2], 'modrm':None}),
    (xmm_mem128_imm8, {'opcode':[0x66, 0x0F, 0xC2], 'modrm':None}))
  arch_ext = 2

class cmpps(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm8,    {'opcode':[0x0F, 0xC2], 'modrm':None}),
    (xmm_mem128_imm8, {'opcode':[0x0F, 0xC2], 'modrm':None}))
  arch_ext = 1

class cmpsd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm8,    {'opcode':[0xF2, 0x0F, 0xC2], 'modrm':None}),
    (xmm_mem64_imm8,  {'opcode':[0xF2, 0x0F, 0xC2], 'modrm':None}))
  arch_ext = 2

class cmpss(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm8,    {'opcode':[0xF3, 0x0F, 0xC2], 'modrm':None}),
    (xmm_mem32_imm8,  {'opcode':[0xF3, 0x0F, 0xC2], 'modrm':None}))
  arch_ext = 1

class comisd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,      {'opcode':[0x66, 0x0F, 0x2F], 'modrm':None}),
    (xmm_mem64,    {'opcode':[0x66, 0x0F, 0x2F], 'modrm':None}))
  arch_ext = 2

class comiss(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,      {'opcode':[0x0F, 0x2F], 'modrm':None}),
    (xmm_mem64,    {'opcode':[0x0F, 0x2F], 'modrm':None}))
  arch_ext = 1

class cvtdq2pd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,      {'opcode':[0xF3, 0x0F, 0xE6], 'modrm':None}),
    (xmm_mem64,    {'opcode':[0xF3, 0x0F, 0xE6], 'modrm':None}))
  arch_ext = 2

class cvtdq2ps(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,      {'opcode':[0x0F, 0x5B], 'modrm':None}),
    (xmm_mem64,    {'opcode':[0x0F, 0x5B], 'modrm':None}))
  arch_ext = 2

class cvtpd2dq(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,      {'opcode':[0xF2, 0x0F, 0xE6], 'modrm':None}),
    (xmm_mem128,   {'opcode':[0xF2, 0x0F, 0xE6], 'modrm':None}))
  arch_ext = 2

class cvtpd2pi(x86DispatchInstruction):
  dispatch = (
    (mmx_xmm,      {'opcode':[0x66, 0x0F, 0x2D], 'modrm':None}),
    (mmx_mem128,   {'opcode':[0x66, 0x0F, 0x2D], 'modrm':None}))
  arch_ext = 2

class cvtpd2ps(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,      {'opcode':[0x66, 0x0F, 0x5A], 'modrm':None}),
    (xmm_mem128,   {'opcode':[0x66, 0x0F, 0x5A], 'modrm':None}))
  arch_ext = 2

class cvtpi2pd(x86DispatchInstruction):
  dispatch = (
    (xmm_mmx,      {'opcode':[0x66, 0x0F, 0x2A], 'modrm':None}),
    (xmm_mem64,    {'opcode':[0x66, 0x0F, 0x2A], 'modrm':None}))
  arch_ext = 2

class cvtpi2ps(x86DispatchInstruction):
  dispatch = (
    (xmm_mmx,      {'opcode':[0x0F, 0x2A], 'modrm':None}),
    (xmm_mem64,    {'opcode':[0x0F, 0x2A], 'modrm':None}))
  arch_ext = 2

class cvtps2dq(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,      {'opcode':[0x66, 0x0F, 0x5B], 'modrm':None}),
    (xmm_mem128,   {'opcode':[0x66, 0x0F, 0x5B], 'modrm':None}))
  arch_ext = 2

class cvtps2pd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,      {'opcode':[0x0F, 0x5A], 'modrm':None}),
    (xmm_mem64,    {'opcode':[0x0F, 0x5A], 'modrm':None}))
  arch_ext = 2

class cvtps2pi(x86DispatchInstruction):
  dispatch = (
    (mmx_xmm,      {'opcode':[0x0F, 0x2D], 'modrm':None}),
    (mmx_mem64,    {'opcode':[0x0F, 0x2D], 'modrm':None}))
  arch_ext = 2

#class cvtsd2si(x86Instruction):
#  machine_inst = reg32
#  params = {'opcode':[0xF2, 0x0F, 0x2D],'modrm':None}
#  arch_ext = 2

class cvtsd2si(x86DispatchInstruction):
  dispatch = (
    (reg32_xmm,      {'opcode':[0xF2, 0x0F, 0x2D], 'modrm':None}),
    (reg32_mem64,    {'opcode':[0xF2, 0x0F, 0x2D], 'modrm':None}))
  arch_ext = 2

class cvtsd2ss(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,      {'opcode':[0xF2, 0x0F, 0x5A], 'modrm':None}),
    (xmm_mem64,    {'opcode':[0xF2, 0x0F, 0x5A], 'modrm':None}))
  arch_ext = 2

class cvtsi2sd(x86DispatchInstruction):
  dispatch = (
    (xmm_reg32,      {'opcode':[0xF2, 0x0F, 0x2A], 'modrm':None}),
    (xmm_mem32,      {'opcode':[0xF2, 0x0F, 0x2A], 'modrm':None}))
  arch_ext = 2

class cvtsi2ss(x86DispatchInstruction):
  dispatch = (
    (xmm_reg32,      {'opcode':[0xF3, 0x0F, 0x2A], 'modrm':None}),
    (xmm_mem32,      {'opcode':[0xF3, 0x0F, 0x2A], 'modrm':None}))
  arch_ext = 1

class cvtss2sd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0xF3, 0x0F, 0x5A], 'modrm':None}),
    (xmm_mem32,      {'opcode':[0xF3, 0x0F, 0x5A], 'modrm':None}))
  arch_ext = 2

class cvtss2si(x86DispatchInstruction):
  dispatch = (
    (reg32_xmm,        {'opcode':[0xF3, 0x0F, 0x2D], 'modrm':None}),
    (reg32_mem32,      {'opcode':[0xF3, 0x0F, 0x2D], 'modrm':None}))
  arch_ext = 1

class cvttpd2dq(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xE6], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xE6], 'modrm':None}))
  arch_ext = 2

class cvttpd2pi(x86DispatchInstruction):
  dispatch = (
    (mmx_xmm,        {'opcode':[0x66, 0x0F, 0x2C], 'modrm':None}),
    (mmx_mem128,     {'opcode':[0x66, 0x0F, 0x2C], 'modrm':None}))
  arch_ext = 2

class cvttps2dq(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0xF3, 0x0F, 0x5B], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0xF3, 0x0F, 0x5B], 'modrm':None}))
  arch_ext = 2

class cvttps2pi(x86DispatchInstruction):
  dispatch = (
    (mmx_xmm,        {'opcode':[0x0F, 0x2C], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0x2C], 'modrm':None}))
  arch_ext = 2

class cvttsd2si(x86DispatchInstruction):
  dispatch = (
    (reg32_xmm,        {'opcode':[0xF2, 0x0F, 0x2C], 'modrm':None}),
    (reg32_mem64,      {'opcode':[0xF2, 0x0F, 0x2C], 'modrm':None}))
  arch_ext = 2

class cvttss2si(x86DispatchInstruction):
  dispatch = (
    (reg32_xmm,        {'opcode':[0xF3, 0x0F, 0x2C], 'modrm':None}),
    (reg32_mem32,      {'opcode':[0xF3, 0x0F, 0x2C], 'modrm':None}))
  arch_ext = 1

class divpd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0x5E], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0x5E], 'modrm':None}))
  arch_ext = 2

class divps(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x0F, 0x5E], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x0F, 0x5E], 'modrm':None}))
  arch_ext = 1

class divsd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0xF2, 0x0F, 0x5E], 'modrm':None}),
    (xmm_mem64,      {'opcode':[0xF2, 0x0F, 0x5E], 'modrm':None}))
  arch_ext = 2

class divss(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0xF3, 0x0F, 0x5E], 'modrm':None}),
    (xmm_mem64,      {'opcode':[0xF3, 0x0F, 0x5E], 'modrm':None}))
  arch_ext = 1

class dppd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm8,    {'opcode':[0x66, 0x0F, 0x3A, 0x41], 'modrm':None}),
    (xmm_mem128_imm8, {'opcode':[0x66, 0x0F, 0x3A, 0x41], 'modrm':None}))
  arch_ext = 4

class dpps(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm8,    {'opcode':[0x66, 0x0F, 0x3A, 0x40], 'modrm':None}),
    (xmm_mem128_imm8, {'opcode':[0x66, 0x0F, 0x3A, 0x40], 'modrm':None}))
  arch_ext = 4

class emms(x86Instruction):
  machine_inst = no_op
  params = {'opcode':[0x0F, 0x77],'modrm':None}
  arch_ext = 0

class extractps(x86DispatchInstruction):
  dispatch = (
    (reg32_xmm_imm8_rev, {'opcode':[0x66, 0x0F, 0x3A, 0x17], 'modrm':None}),
    (mem32_xmm_imm8,     {'opcode':[0x66, 0x0F, 0x3A, 0x17], 'modrm':None}))
 # TODO - ugh, this make the printer not emit 'dword' for the mem32 case
 #arch_ext = 4

class extrq(x86DispatchInstruction):
  dispatch = (
    (xmm_imm8_imm8, {'opcode':[0x66, 0x0F, 0x78], 'modrm':0x00}),
    (xmm_xmm,       {'opcode':[0x66, 0x0F, 0x79], 'modrm':None}))
  arch_ext = 4

class haddpd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0x7C], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0x7C], 'modrm':None}))
  arch_ext = 3

class haddps(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0xF2, 0x0F, 0x7C], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0xF2, 0x0F, 0x7C], 'modrm':None}))
  arch_ext = 3

class hsubpd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0x7D], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0x7D], 'modrm':None}))
  arch_ext = 3

class hsubps(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0xF2, 0x0F, 0x7D], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0xF2, 0x0F, 0x7D], 'modrm':None}))
  arch_ext = 3

class insertps(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm8,    {'opcode':[0x66, 0x0F, 0x3A, 0x21], 'modrm':None}),
    (xmm_mem32_imm8,  {'opcode':[0x66, 0x0F, 0x3A, 0x21], 'modrm':None}))
  arch_ext = 4

class insertq(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm8_imm8, {'opcode':[0xF2, 0x0F, 0x78], 'modrm':None}),
    (xmm_xmm,           {'opcode':[0xF2, 0x0F, 0x79], 'modrm':None}))
  arch_ext = 4

class lddqu(x86Instruction):
  machine_inst = xmm_mem128
  params = {'opcode':[0xF2, 0x0F, 0xF0],'modrm':None}
  arch_ext = 3

class ldmxcsr(x86Instruction):
  machine_inst = mem32
  params = {'opcode':[0x0F, 0xAE],'modrm':0x10}
  arch_ext = 1

class maskmovdqu(x86Instruction):
  machine_inst = xmm_xmm
  params = {'opcode':[0x66, 0x0F, 0xF7],'modrm':None}
  arch_ext = 2

class maskmovq(x86Instruction):
  machine_inst = mmx_mmx
  params = {'opcode':[0x0F, 0xF7],'modrm':None}
  arch_ext = 1

class maxpd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0x5F], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0x5F], 'modrm':None}))
  arch_ext = 2

class maxps(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x0F, 0x5F], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x0F, 0x5F], 'modrm':None}))
  arch_ext = 1

class maxsd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0xF2, 0x0F, 0x5F], 'modrm':None}),
    (xmm_mem64,      {'opcode':[0xF2, 0x0F, 0x5F], 'modrm':None}))
  arch_ext = 2

class maxss(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0xF3, 0x0F, 0x5F], 'modrm':None}),
    (xmm_mem32,      {'opcode':[0xF3, 0x0F, 0x5F], 'modrm':None}))
  arch_ext = 1

class minpd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0x5D], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0x5D], 'modrm':None}))
  arch_ext = 2

class minps(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x0F, 0x5D], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x0F, 0x5D], 'modrm':None}))
  arch_ext = 1

class minsd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0xF2, 0x0F, 0x5D], 'modrm':None}),
    (xmm_mem64,      {'opcode':[0xF2, 0x0F, 0x5D], 'modrm':None}))
  arch_ext = 2

class minss(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0xF3, 0x0F, 0x5D], 'modrm':None}),
    (xmm_mem32,      {'opcode':[0xF3, 0x0F, 0x5D], 'modrm':None}))
  arch_ext = 1

class movapd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0x28], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0x28], 'modrm':None}),
    (mem128_xmm,     {'opcode':[0x66, 0x0F, 0x29], 'modrm':None}))
  arch_ext = 2

class movaps(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x0F, 0x28], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x0F, 0x28], 'modrm':None}),
    (mem128_xmm,     {'opcode':[0x0F, 0x29], 'modrm':None}))
  arch_ext = 1

class movd(x86DispatchInstruction):
  dispatch = (
    (xmm_reg32,        {'opcode':[0x66, 0x0F, 0x6E], 'modrm':None}),
    (xmm_mem32,        {'opcode':[0x66, 0x0F, 0x6E], 'modrm':None}),
    (mem32_xmm,        {'opcode':[0x66, 0x0F, 0x7E], 'modrm':None}),
    (reg32_xmm_rev,    {'opcode':[0x66, 0x0F, 0x7E], 'modrm':None}),
    (mmx_reg32,        {'opcode':[0x0F, 0x6E], 'modrm':None}),
    (mmx_mem32,        {'opcode':[0x0F, 0x6E], 'modrm':None}),
    (mem32_mmx,        {'opcode':[0x0F, 0x7E], 'modrm':None}),
    (reg32_mmx_rev,    {'opcode':[0x0F, 0x7E], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class movddup(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0xF2, 0x0F, 0x12], 'modrm':None}),
    (xmm_mem64,      {'opcode':[0xF2, 0x0F, 0x12], 'modrm':None}))
  arch_ext = 3

class movdq2q(x86Instruction):
  machine_inst = mmx_xmm
  params = {'opcode':[0xF2, 0x0F, 0xD6],'modrm':None}
  arch_ext = 2

class movdqa(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0x6F], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0x6F], 'modrm':None}),
    (mem128_xmm,     {'opcode':[0x66, 0x0F, 0x7F], 'modrm':None}))
  arch_ext = 2

class movdqu(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0xF3, 0x0F, 0x6F], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0xF3, 0x0F, 0x6F], 'modrm':None}),
    (mem128_xmm,     {'opcode':[0xF3, 0x0F, 0x7F], 'modrm':None}))
  arch_ext = 2

class movhlps(x86Instruction):
  machine_inst = xmm_xmm
  params = {'opcode':[0x0F, 0x12],'modrm':None}
  arch_ext = 1

class movhpd(x86DispatchInstruction):
  dispatch = (
    (xmm_mem64,      {'opcode':[0x66, 0x0F, 0x16], 'modrm':None}),
    (mem64_xmm,      {'opcode':[0x66, 0x0F, 0x17], 'modrm':None}))
  arch_ext = 2

class movhps(x86DispatchInstruction):
  dispatch = (
    (xmm_mem64,      {'opcode':[0x0F, 0x16], 'modrm':None}),
    (mem64_xmm,      {'opcode':[0x0F, 0x17], 'modrm':None}))
  arch_ext = 1

class movlhps(x86Instruction):
  machine_inst = xmm_xmm
  params = {'opcode':[0x0F, 0x16],'modrm':None}
  arch_ext = 1

class movlpd(x86DispatchInstruction):
  dispatch = (
    (xmm_mem64,      {'opcode':[0x66, 0x0F, 0x12], 'modrm':None}),
    (mem64_xmm,      {'opcode':[0x66, 0x0F, 0x13], 'modrm':None}))
  arch_ext = 2

class movlps(x86DispatchInstruction):
  dispatch = (
    (xmm_mem64,      {'opcode':[0x0F, 0x12], 'modrm':None}),
    (mem64_xmm,      {'opcode':[0x0F, 0x13], 'modrm':None}))
  arch_ext = 1

class movmskpd(x86Instruction):
  machine_inst = reg32_xmm
  params = {'opcode':[0x66, 0x0F, 0x50],'modrm':None}
  arch_ext = 2

class movmskps(x86Instruction):
  machine_inst = reg32_xmm
  params = {'opcode':[0x0F, 0x50],'modrm':None}
  arch_ext = 2

class movntdq(x86Instruction):
  machine_inst = mem128_xmm
  params = {'opcode':[0x66, 0x0F, 0xE7],'modrm':None}
  arch_ext = 2

class movntdqa(x86Instruction):
  machine_inst = xmm_mem128
  params = {'opcode':[0x66, 0x0F, 0x38, 0x2A], 'modrm':None}
  arch_ext = 4

class movntpd(x86Instruction):
  machine_inst = mem128_xmm
  params = {'opcode':[0x66, 0x0F, 0x2B],'modrm':None}
  arch_ext = 2

class movntps(x86Instruction):
  machine_inst = mem128_xmm
  params = {'opcode':[0x0F, 0x2B],'modrm':None}
  arch_ext = 2

class movntq(x86Instruction):
  machine_inst = mem64_mmx
  params = {'opcode':[0x0F, 0xE7],'modrm':None}
  arch_ext = 1

class movntsd(x86Instruction):
  machine_inst = mem64_xmm
  params = {'opcode':[0xF2, 0x0F, 0x2B], 'modrm':None}
  arch_ext = 4

class movntss(x86Instruction):
  machine_inst = mem32_xmm
  params = {'opcode':[0xF3, 0x0F, 0x2B], 'modrm':None}
  arch_ext = 4

class movq(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,          {'opcode':[0xF3, 0x0F, 0x7E], 'modrm':None}),
    (xmm_mem64,        {'opcode':[0xF3, 0x0F, 0x7E], 'modrm':None}),
    (mem64_xmm,        {'opcode':[0x66, 0x0F, 0xD6], 'modrm':None}),
    (mmx_mmx,          {'opcode':[0x0F, 0x6F], 'modrm':None}),
    (mmx_mem64,        {'opcode':[0x0F, 0x6F], 'modrm':None}),
    (mem64_mmx,        {'opcode':[0x0F, 0x7F], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class movq2dq(x86Instruction):
  machine_inst = xmm_mmx
  params = {'opcode':[0xF3, 0x0F, 0xD6],'modrm':None}
  arch_ext = 2

class movsd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0xF2, 0x0F, 0x10], 'modrm':None}),
    (xmm_mem64,      {'opcode':[0xF2, 0x0F, 0x10], 'modrm':None}),
    (mem64_xmm,      {'opcode':[0xF2, 0x0F, 0x11], 'modrm':None}))
  arch_ext = 2

class movshdup(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0xF3, 0x0F, 0x16], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0xF3, 0x0F, 0x16], 'modrm':None}))
  arch_ext = 3

class movsldup(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0xF3, 0x0F, 0x12], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0xF3, 0x0F, 0x12], 'modrm':None}))
  arch_ext = 3

class movss(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0xF3, 0x0F, 0x10], 'modrm':None}),
    (xmm_mem32,      {'opcode':[0xF3, 0x0F, 0x10], 'modrm':None}),
    (mem32_xmm,      {'opcode':[0xF3, 0x0F, 0x11], 'modrm':None}))
  arch_ext = 1

class movupd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0x10], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0x10], 'modrm':None}),
    (mem128_xmm,     {'opcode':[0x66, 0x0F, 0x11], 'modrm':None}))
  arch_ext = 2

class movups(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x0F, 0x10], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x0F, 0x10], 'modrm':None}),
    (mem128_xmm,     {'opcode':[0x0F, 0x11], 'modrm':None}))
  arch_ext = 1

class mpsadbw(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm8,    {'opcode':[0x66, 0x0F, 0x3A, 0x42], 'modrm':None}),
    (xmm_mem128_imm8, {'opcode':[0x66, 0x0F, 0x3A, 0x42], 'modrm':None}))
  arch_ext = 4

class mulpd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0x59], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0x59], 'modrm':None}))
  arch_ext = 2

class mulps(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x0F, 0x59], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x0F, 0x59], 'modrm':None}))
  arch_ext = 1

class mulsd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0xF2, 0x0F, 0x59], 'modrm':None}),
    (xmm_mem64,      {'opcode':[0xF2, 0x0F, 0x59], 'modrm':None}))
  arch_ext = 2

class mulss(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0xF3, 0x0F, 0x59], 'modrm':None}),
    (xmm_mem32,      {'opcode':[0xF3, 0x0F, 0x59], 'modrm':None}))
  arch_ext = 1

class orpd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0x56], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0x56], 'modrm':None}))
  arch_ext = 2

class orps(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x0F, 0x56], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x0F, 0x56], 'modrm':None}))
  arch_ext = 1

class pabsb(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x1C], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x1C], 'modrm':None}),
    (mmx_mmx,    {'opcode':[0x0F, 0x38, 0x1C], 'modrm':None}),
    (mmx_mem64,  {'opcode':[0x0F, 0x38, 0x1C], 'modrm':None}))
  arch_ext = 3

class pabsd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x1E], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x1E], 'modrm':None}),
    (mmx_mmx,    {'opcode':[0x0F, 0x38, 0x1E], 'modrm':None}),
    (mmx_mem64,  {'opcode':[0x0F, 0x38, 0x1E], 'modrm':None}))
  arch_ext = 3

class pabsw(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x1D], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x1D], 'modrm':None}),
    (mmx_mmx,    {'opcode':[0x0F, 0x38, 0x1D], 'modrm':None}),
    (mmx_mem64,  {'opcode':[0x0F, 0x38, 0x1D], 'modrm':None}))
  arch_ext = 3

class packssdw(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0x6B], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0x6B], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0x6B], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0x6B], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class packsswb(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0x63], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0x63], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0x63], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0x63], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class packusdw(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x2B], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x2B], 'modrm':None}))
  arch_ext = 4

class packuswb(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0x67], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0x67], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0x67], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0x67], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class paddb(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xFC], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xFC], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xFC], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xFC], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class paddd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xFE], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xFE], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xFE], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xFE], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class paddq(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xD4], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xD4], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xD4], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xD4], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class paddsb(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xEC], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xEC], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xEC], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xEC], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class paddsw(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xED], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xED], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xED], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xED], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class paddusb(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xDC], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xDC], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xDC], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xDC], 'modrm':None}))
  arch_ext = 0

class paddusw(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xDD], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xDD], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xDD], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xDD], 'modrm':None}))
  arch_ext = 0

class paddw(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xFD], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xFD], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xFD], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xFD], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class palignr(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm8,   {'opcode':[0x66, 0x0F, 0x3A, 0x0F], 'modrm':None}),
    (xmm_mem128_imm8,{'opcode':[0x66, 0x0F, 0x3A, 0x0F], 'modrm':None}),
    (mmx_mmx_imm8,   {'opcode':[0x0F, 0x3A, 0x0F], 'modrm':None}),
    (mmx_mem64_imm8, {'opcode':[0x0F, 0x3A, 0x0F], 'modrm':None}))
  arch_ext = 3

class pand(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xDB], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xDB], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xDB], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xDB], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class pandn(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xDF], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xDF], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xDF], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xDF], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class pavgb(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xE0], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xE0], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xE0], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xE0], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class pavgw(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xE3], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xE3], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xE3], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xE3], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class pblendvb(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x10], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x10], 'modrm':None}))
  arch_ext = 4

class pblendw(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm8,    {'opcode':[0x66, 0x0F, 0x3A, 0x0E], 'modrm':None}),
    (xmm_mem128_imm8, {'opcode':[0x66, 0x0F, 0x3A, 0x0E], 'modrm':None}))
  arch_ext = 4

class pcmpeqb(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0x74], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0x74], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0x74], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0x74], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class pcmpeqd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0x76], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0x76], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0x76], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0x76], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class pcmpeqq(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x29], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x29], 'modrm':None}))
  arch_ext = 4

class pcmpeqw(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0x75], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0x75], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0x75], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0x75], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class pcmpestri(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm8,    {'opcode':[0x66, 0x0F, 0x3A, 0x61], 'modrm':None}),
    (xmm_mem128_imm8, {'opcode':[0x66, 0x0F, 0x3A, 0x61], 'modrm':None}))
  arch_ext = 4

class pcmpestrm(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm8,    {'opcode':[0x66, 0x0F, 0x3A, 0x60], 'modrm':None}),
    (xmm_mem128_imm8, {'opcode':[0x66, 0x0F, 0x3A, 0x60], 'modrm':None}))
  arch_ext = 4

class pcmpgtb(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0x64], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0x64], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0x64], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0x64], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class pcmpgtd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0x66], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0x66], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0x66], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0x66], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class pcmpgtw(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0x65], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0x65], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0x65], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0x65], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class pcmpgtq(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x37], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x37], 'modrm':None}))
  arch_ext = 4

class pcmpistri(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm8,    {'opcode':[0x66, 0x0F, 0x3A, 0x63], 'modrm':None}),
    (xmm_mem128_imm8, {'opcode':[0x66, 0x0F, 0x3A, 0x63], 'modrm':None}))
  arch_ext = 4

class pcmpistrm(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm8,    {'opcode':[0x66, 0x0F, 0x3A, 0x62], 'modrm':None}),
    (xmm_mem128_imm8, {'opcode':[0x66, 0x0F, 0x3A, 0x62], 'modrm':None}))
  arch_ext = 4

class pextrb(x86DispatchInstruction):
  dispatch = (
    (reg32_xmm_imm8_rev, {'opcode':[0x66, 0x0F, 0x3A, 0x14], 'modrm':None}),
    (mem8_xmm_imm8,  {'opcode':[0x66, 0x0F, 0x3A, 0x14], 'modrm':None}))
  arch_ext = 4

class pextrd(x86DispatchInstruction):
  dispatch = (
    (reg32_xmm_imm8_rev, {'opcode':[0x66, 0x0F, 0x3A, 0x16], 'modrm':None}),
    (mem32_xmm_imm8, {'opcode':[0x66, 0x0F, 0x3A, 0x16], 'modrm':None}))
  arch_ext = 4

class pextrw(x86DispatchInstruction):
  dispatch = (
    (reg32_xmm_imm8, {'opcode':[0x66, 0x0F, 0xC5], 'modrm':None}),
    (mem16_xmm_imm8, {'opcode':[0x66, 0x0F, 0x3A, 0x15], 'modrm':None}),
    (reg32_mmx_imm8, {'opcode':[0x0F, 0xC5], 'modrm':None}))
  arch_ext = 1

class phaddsw(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x03], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x03], 'modrm':None}),
    (mmx_mmx,    {'opcode':[0x0F, 0x38, 0x03], 'modrm':None}),
    (mmx_mem64,  {'opcode':[0x0F, 0x38, 0x03], 'modrm':None}))
  arch_ext = 3

class phaddw(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x01], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x01], 'modrm':None}),
    (mmx_mmx,    {'opcode':[0x0F, 0x38, 0x01], 'modrm':None}),
    (mmx_mem64,  {'opcode':[0x0F, 0x38, 0x01], 'modrm':None}))
  arch_ext = 3

class phaddd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x02], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x02], 'modrm':None}),
    (mmx_mmx,    {'opcode':[0x0F, 0x38, 0x02], 'modrm':None}),
    (mmx_mem64,  {'opcode':[0x0F, 0x38, 0x02], 'modrm':None}))
  arch_ext = 3

class phminposuw(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x41], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x41], 'modrm':None}))
  arch_ext = 4

class phminposuw(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x41], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x41], 'modrm':None}))
  arch_ext = 4

class phsubsw(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x07], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x07], 'modrm':None}),
    (mmx_mmx,    {'opcode':[0x0F, 0x38, 0x07], 'modrm':None}),
    (mmx_mem64,  {'opcode':[0x0F, 0x38, 0x07], 'modrm':None}))
  arch_ext = 3

class phsubw(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x05], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x05], 'modrm':None}),
    (mmx_mmx,    {'opcode':[0x0F, 0x38, 0x05], 'modrm':None}),
    (mmx_mem64,  {'opcode':[0x0F, 0x38, 0x05], 'modrm':None}))
  arch_ext = 3

class phsubd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x06], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x06], 'modrm':None}),
    (mmx_mmx,    {'opcode':[0x0F, 0x38, 0x06], 'modrm':None}),
    (mmx_mem64,  {'opcode':[0x0F, 0x38, 0x06], 'modrm':None}))
  arch_ext = 3

class pinsrb(x86DispatchInstruction):
  dispatch = (
    (xmm_reg32_imm8, {'opcode':[0x66, 0x0F, 0x3A, 0x20], 'modrm':None}),
    (xmm_mem8_imm8,  {'opcode':[0x66, 0x0F, 0x3A, 0x20], 'modrm':None}))
  arch_ext = 4

class pinsrd(x86DispatchInstruction):
  dispatch = (
    (xmm_reg32_imm8, {'opcode':[0x66, 0x0F, 0x3A, 0x22], 'modrm':None}),
    (xmm_mem32_imm8, {'opcode':[0x66, 0x0F, 0x3A, 0x22], 'modrm':None}))
  arch_ext = 4

class pinsrw(x86DispatchInstruction):
  dispatch = (
    (xmm_reg32_imm8,     {'opcode':[0x66, 0x0F, 0xC4], 'modrm':None}),
    (xmm_mem16_imm8,     {'opcode':[0x66, 0x0F, 0xC4], 'modrm':None}),
    (mmx_reg32_imm8,     {'opcode':[0x0F, 0xC4], 'modrm':None}),
    (mmx_mem16_imm8,     {'opcode':[0x0F, 0xC4], 'modrm':None}))
  arch_ext = 1

class pmaddubsw(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x04], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x04], 'modrm':None}),
    (mmx_mmx,    {'opcode':[0x0F, 0x38, 0x04], 'modrm':None}),
    (mmx_mem64,  {'opcode':[0x0F, 0x38, 0x04], 'modrm':None}))
  arch_ext = 3

class pmaddwd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xF5], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xF5], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xF5], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xF5], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class pmaxsb(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x3C], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x3C], 'modrm':None}))
  arch_ext = 4

class pmaxsd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x3D], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x3D], 'modrm':None}))
  arch_ext = 4

class pmaxsw(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xEE], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xEE], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xEE], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xEE], 'modrm':None}))
  arch_ext = 1

class pmaxub(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xDE], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xDE], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xDE], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xDE], 'modrm':None}))
  arch_ext = 1

class pmaxud(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x3F], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x3F], 'modrm':None}))
  arch_ext = 4

class pmaxuw(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x3E], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x3E], 'modrm':None}))
  arch_ext = 4

class pminsb(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x38], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x38], 'modrm':None}))
  arch_ext = 4

class pminsd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x39], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x39], 'modrm':None}))
  arch_ext = 4

class pminsw(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xEA], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xEA], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xEA], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xEA], 'modrm':None}))
  arch_ext = 1

class pminub(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xDA], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xDA], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xDA], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xDA], 'modrm':None}))
  arch_ext = 1

class pminud(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x3B], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x3B], 'modrm':None}))
  arch_ext = 4

class pminuw(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x3A], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x3A], 'modrm':None}))
  arch_ext = 4

class pmovmskb(x86DispatchInstruction):
  dispatch = (
    (reg32_xmm,     {'opcode':[0x66, 0x0F, 0xD7], 'modrm':None}),
    (reg32_mmx,     {'opcode':[0x0F, 0xD7], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 1

class pmovsxbw(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,   {'opcode':[0x66, 0x0F, 0x38, 0x20], 'modrm':None}),
    (xmm_mem64, {'opcode':[0x66, 0x0F, 0x38, 0x20], 'modrm':None}))
  arch_ext = 4

class pmovsxbd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,   {'opcode':[0x66, 0x0F, 0x38, 0x21], 'modrm':None}),
    (xmm_mem32, {'opcode':[0x66, 0x0F, 0x38, 0x21], 'modrm':None}))
  arch_ext = 4

class pmovsxbq(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,   {'opcode':[0x66, 0x0F, 0x38, 0x22], 'modrm':None}),
    (xmm_mem16, {'opcode':[0x66, 0x0F, 0x38, 0x22], 'modrm':None}))
  arch_ext = 4

class pmovsxwd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,   {'opcode':[0x66, 0x0F, 0x38, 0x23], 'modrm':None}),
    (xmm_mem64, {'opcode':[0x66, 0x0F, 0x38, 0x23], 'modrm':None}))
  arch_ext = 4

class pmovsxwq(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,   {'opcode':[0x66, 0x0F, 0x38, 0x24], 'modrm':None}),
    (xmm_mem32, {'opcode':[0x66, 0x0F, 0x38, 0x24], 'modrm':None}))
  arch_ext = 4

class pmovsxdq(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,   {'opcode':[0x66, 0x0F, 0x38, 0x25], 'modrm':None}),
    (xmm_mem64, {'opcode':[0x66, 0x0F, 0x38, 0x25], 'modrm':None}))
  arch_ext = 4

class pmovzxbw(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,   {'opcode':[0x66, 0x0F, 0x38, 0x30], 'modrm':None}),
    (xmm_mem64, {'opcode':[0x66, 0x0F, 0x38, 0x30], 'modrm':None}))
  arch_ext = 4

class pmovzxbd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,   {'opcode':[0x66, 0x0F, 0x38, 0x31], 'modrm':None}),
    (xmm_mem32, {'opcode':[0x66, 0x0F, 0x38, 0x31], 'modrm':None}))
  arch_ext = 4

class pmovzxbq(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,   {'opcode':[0x66, 0x0F, 0x38, 0x32], 'modrm':None}),
    (xmm_mem16, {'opcode':[0x66, 0x0F, 0x38, 0x32], 'modrm':None}))
  arch_ext = 4

class pmovzxwd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,   {'opcode':[0x66, 0x0F, 0x38, 0x33], 'modrm':None}),
    (xmm_mem64, {'opcode':[0x66, 0x0F, 0x38, 0x33], 'modrm':None}))
  arch_ext = 4

class pmovzxwq(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,   {'opcode':[0x66, 0x0F, 0x38, 0x34], 'modrm':None}),
    (xmm_mem32, {'opcode':[0x66, 0x0F, 0x38, 0x34], 'modrm':None}))
  arch_ext = 4

class pmovzxdq(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,   {'opcode':[0x66, 0x0F, 0x38, 0x35], 'modrm':None}),
    (xmm_mem64, {'opcode':[0x66, 0x0F, 0x38, 0x35], 'modrm':None}))
  arch_ext = 4

class pmuldq(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x28], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x28], 'modrm':None}))
  arch_ext = 4

class pmulhrsw(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x0B], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x0B], 'modrm':None}),
    (mmx_mmx,    {'opcode':[0x0F, 0x38, 0x0B], 'modrm':None}),
    (mmx_mem64,  {'opcode':[0x0F, 0x38, 0x0B], 'modrm':None}))
  arch_ext = 3

class pmulhuw(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xE4], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xE4], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xE4], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xE4], 'modrm':None}))
  arch_ext = 1

class pmulhw(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xE5], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xE5], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xE5], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xE5], 'modrm':None}))
  arch_ext = 1

class pmulld(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x40], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x40], 'modrm':None}))
  arch_ext = 4

class pmullw(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xD5], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xD5], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xD5], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xD5], 'modrm':None}))
  arch_ext = 2 # and 0

class pmuludq(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xF4], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xF4], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xF4], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xF4], 'modrm':None}))
  arch_ext = 2

class por(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xEB], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xEB], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xEB], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xEB], 'modrm':None}))
  arch_ext = 2 # and 0

class psadbw(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xF6], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xF6], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xF6], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xF6], 'modrm':None}))
  arch_ext = 1

class psignb(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x08], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x08], 'modrm':None}),
    (mmx_mmx,    {'opcode':[0x0F, 0x38, 0x08], 'modrm':None}),
    (mmx_mem64,  {'opcode':[0x0F, 0x38, 0x08], 'modrm':None}))
  arch_ext = 3

class psignd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x0A], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x0A], 'modrm':None}),
    (mmx_mmx,    {'opcode':[0x0F, 0x38, 0x0A], 'modrm':None}),
    (mmx_mem64,  {'opcode':[0x0F, 0x38, 0x0A], 'modrm':None}))
  arch_ext = 3

class psignw(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x09], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x09], 'modrm':None}),
    (mmx_mmx,    {'opcode':[0x0F, 0x38, 0x09], 'modrm':None}),
    (mmx_mem64,  {'opcode':[0x0F, 0x38, 0x09], 'modrm':None}))
  arch_ext = 3

class pshufb(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x00], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x00], 'modrm':None}),
    (mmx_mmx,    {'opcode':[0x0F, 0x38, 0x00], 'modrm':None}),
    (mmx_mem64,  {'opcode':[0x0F, 0x38, 0x00], 'modrm':None}))
  arch_ext = 3

class pshufd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm8,    {'opcode':[0x66, 0x0F, 0x70], 'modrm':None}),
    (xmm_mem128_imm8, {'opcode':[0x66, 0x0F, 0x70], 'modrm':None}))
  arch_ext = 2

class pshufhw(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm8,    {'opcode':[0xF3, 0x0F, 0x70], 'modrm':None}),
    (xmm_mem128_imm8, {'opcode':[0xF3, 0x0F, 0x70], 'modrm':None}))
  arch_ext = 2

class pshuflw(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm8,    {'opcode':[0xF2, 0x0F, 0x70], 'modrm':None}),
    (xmm_mem128_imm8, {'opcode':[0xF2, 0x0F, 0x70], 'modrm':None}))
  arch_ext = 2

class pshufw(x86DispatchInstruction):
  dispatch = (
    (mmx_mmx_imm8,    {'opcode':[0x0F, 0x70], 'modrm':None}),
    (mmx_mem64_imm8,  {'opcode':[0x0F, 0x70], 'modrm':None}))
  arch_ext = 1

class pslld(x86DispatchInstruction):
  dispatch = (
    (xmm_imm8,   {'opcode':[0x66, 0x0F, 0x72], 'modrm':0x30}),
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0xF2], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0xF2], 'modrm':None}),
    (mmx_imm8,   {'opcode':[0x0F, 0x72], 'modrm':0x30}),
    (mmx_mmx,    {'opcode':[0x0F, 0xF2], 'modrm':None}),
    (mmx_mem128, {'opcode':[0x0F, 0xF2], 'modrm':None}))
  arch_ext = 2 # and 0 and 1

class pslldq(x86Instruction):
  machine_inst = xmm_imm8
  params = {'opcode':[0x66, 0x0F, 0x73],'modrm':0x38}
  arch_ext = 1

class psllq(x86DispatchInstruction):
  dispatch = (
    (xmm_imm8,   {'opcode':[0x66, 0x0F, 0x73], 'modrm':0x30}),
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0xF3], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0xF3], 'modrm':None}),
    (mmx_imm8,   {'opcode':[0x0F, 0x73], 'modrm':0x30}),
    (mmx_mmx,    {'opcode':[0x0F, 0xF3], 'modrm':None}),
    (mmx_mem128, {'opcode':[0x0F, 0xF3], 'modrm':None}))
  arch_ext = 2 # and 0 and 1

class psllw(x86DispatchInstruction):
  dispatch = (
    (xmm_imm8,   {'opcode':[0x66, 0x0F, 0x71], 'modrm':0x30}),
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0xF1], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0xF1], 'modrm':None}),
    (mmx_imm8,   {'opcode':[0x0F, 0x71], 'modrm':0x30}),
    (mmx_mmx,    {'opcode':[0x0F, 0xF1], 'modrm':None}),
    (mmx_mem128, {'opcode':[0x0F, 0xF1], 'modrm':None}))
  arch_ext = 2 # and 0 and 1

class psrad(x86DispatchInstruction):
  dispatch = (
    (xmm_imm8,   {'opcode':[0x66, 0x0F, 0x72], 'modrm':0x20}),
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0xE2], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0xE2], 'modrm':None}),
    (mmx_imm8,   {'opcode':[0x0F, 0x72], 'modrm':0x20}),
    (mmx_mmx,    {'opcode':[0x0F, 0xE2], 'modrm':None}),
    (mmx_mem128, {'opcode':[0x0F, 0xE2], 'modrm':None}))
  arch_ext = 2 # and 0 and 1

class psraw(x86DispatchInstruction):
  dispatch = (
    (xmm_imm8,   {'opcode':[0x66, 0x0F, 0x71], 'modrm':0x20}),
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0xE1], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0xE1], 'modrm':None}),
    (mmx_imm8,   {'opcode':[0x0F, 0x71], 'modrm':0x20}),
    (mmx_mmx,    {'opcode':[0x0F, 0xE1], 'modrm':None}),
    (mmx_mem128, {'opcode':[0x0F, 0xE1], 'modrm':None}))
  arch_ext = 2 # and 0 and 1

class psrld(x86DispatchInstruction):
  dispatch = (
    (xmm_imm8,   {'opcode':[0x66, 0x0F, 0x72], 'modrm':0x10}),
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0xD2], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0xD2], 'modrm':None}),
    (mmx_imm8,   {'opcode':[0x0F, 0x72], 'modrm':0x10}),
    (mmx_mmx,    {'opcode':[0x0F, 0xD2], 'modrm':None}),
    (mmx_mem128, {'opcode':[0x0F, 0xD2], 'modrm':None}))
  arch_ext = 2 # and 0 and 1

class psrldq(x86Instruction):
  machine_inst = xmm_imm8
  params = {'opcode':[0x66, 0x0F, 0x73],'modrm':0x18}
  arch_ext = 1

class psrlq(x86DispatchInstruction):
  dispatch = (
    (xmm_imm8,   {'opcode':[0x66, 0x0F, 0x73], 'modrm':0x10}),
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0xD3], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0xD3], 'modrm':None}),
    (mmx_imm8,   {'opcode':[0x0F, 0x73], 'modrm':0x10}),
    (mmx_mmx,    {'opcode':[0x0F, 0xD3], 'modrm':None}),
    (mmx_mem128, {'opcode':[0x0F, 0xD3], 'modrm':None}))
  arch_ext = 2 # and 0 and 1

class psrlw(x86DispatchInstruction):
  dispatch = (
    (xmm_imm8,   {'opcode':[0x66, 0x0F, 0x71], 'modrm':0x10}),
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0xD1], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0xD1], 'modrm':None}),
    (mmx_imm8,   {'opcode':[0x0F, 0x71], 'modrm':0x10}),
    (mmx_mmx,    {'opcode':[0x0F, 0xD1], 'modrm':None}),
    (mmx_mem128, {'opcode':[0x0F, 0xD1], 'modrm':None}))
  arch_ext = 2 # and 0 and 1

class psubb(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xF8], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xF8], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xF8], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xF8], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class psubd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xFA], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xFA], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xFA], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xFA], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class psubq(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xFB], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xFB], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xFB], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xFB], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class psubsb(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xE8], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xE8], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xE8], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xE8], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class psubsw(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xE9], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xE9], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xE9], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xE9], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class psubusb(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xD8], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xD8], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xD8], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xD8], 'modrm':None}))
  arch_ext = 0

class psubusw(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xD9], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xD9], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xD9], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xD9], 'modrm':None}))
  arch_ext = 0

class psubw(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xF9], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xF9], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xF9], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xF9], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class punpckhbw(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0x68], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0x68], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0x68], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0x68], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class punpckhdq(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0x6A], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0x6A], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0x6A], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0x6A], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class punpckhqdq(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0x6D], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0x6D], 'modrm':None}))
  arch_ext = 2

class punpckhwd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0x69], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0x69], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0x69], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0x69], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class punpcklbw(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0x60], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0x60], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0x60], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0x60], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class punpckldq(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0x62], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0x62], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0x62], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0x62], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class punpcklqdq(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0x6C], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0x6C], 'modrm':None}))
  arch_ext = 2

class punpcklwd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0x61], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0x61], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0x61], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0x61], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class pxor(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xEF], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xEF], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xEF], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xEF], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class rcpps(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x0F, 0x53], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x0F, 0x53], 'modrm':None}))
  arch_ext = 1

class rcpss(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0xF3, 0x0F, 0x53], 'modrm':None}),
    (xmm_mem32,      {'opcode':[0xF3, 0x0F, 0x53], 'modrm':None}))
  arch_ext = 2

class roundpd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm8,    {'opcode':[0x66, 0x0F, 0x3A, 0x09], 'modrm':None}),
    (xmm_mem128_imm8, {'opcode':[0x66, 0x0F, 0x3A, 0x09], 'modrm':None}))
  arch_ext = 4

class roundps(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm8,    {'opcode':[0x66, 0x0F, 0x3A, 0x08], 'modrm':None}),
    (xmm_mem128_imm8, {'opcode':[0x66, 0x0F, 0x3A, 0x08], 'modrm':None}))
  arch_ext = 4

class roundsd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm8,    {'opcode':[0x66, 0x0F, 0x3A, 0x0B], 'modrm':None}),
    (xmm_mem128_imm8, {'opcode':[0x66, 0x0F, 0x3A, 0x0B], 'modrm':None}))
  arch_ext = 4

class roundss(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm8,    {'opcode':[0x66, 0x0F, 0x3A, 0x0A], 'modrm':None}),
    (xmm_mem128_imm8, {'opcode':[0x66, 0x0F, 0x3A, 0x0A], 'modrm':None}))
  arch_ext = 4

class rsqrtps(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x0F, 0x52], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x0F, 0x52], 'modrm':None}))
  arch_ext = 1

class rsqrtss(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0xF3, 0x0F, 0x52], 'modrm':None}),
    (xmm_mem32,      {'opcode':[0xF3, 0x0F, 0x52], 'modrm':None}))
  arch_ext = 2

class shufpd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm8,    {'opcode':[0x66, 0x0F, 0xC6], 'modrm':None}),
    (xmm_mem128_imm8, {'opcode':[0x66, 0x0F, 0xC6], 'modrm':None}))
  arch_ext = 2

class shufps(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm8,    {'opcode':[0x0F, 0xC6], 'modrm':None}),
    (xmm_mem128_imm8, {'opcode':[0x0F, 0xC6], 'modrm':None}))
  arch_ext = 1

class sqrtpd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x51], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x51], 'modrm':None}))
  arch_ext = 2

class sqrtps(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x0F, 0x51], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x0F, 0x51], 'modrm':None}))
  arch_ext = 1

class sqrtsd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0xF2, 0x0F, 0x51], 'modrm':None}),
    (xmm_mem128, {'opcode':[0xF2, 0x0F, 0x51], 'modrm':None}))
  arch_ext = 2

class sqrtss(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0xF3, 0x0F, 0x51], 'modrm':None}),
    (xmm_mem128, {'opcode':[0xF3, 0x0F, 0x51], 'modrm':None}))
  arch_ext = 1

class stmxcsr(x86Instruction):
  machine_inst = mem32
  params = {'opcode':[0x0F, 0xAE],'modrm':0x18}
  arch_ext = 1

class subpd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x5C], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x5C], 'modrm':None}))
  arch_ext = 2

class subps(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x0F, 0x5C], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x0F, 0x5C], 'modrm':None}))
  arch_ext = 1

class subsd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0xF2, 0x0F, 0x5C], 'modrm':None}),
    (xmm_mem128, {'opcode':[0xF2, 0x0F, 0x5C], 'modrm':None}))
  arch_ext = 2

class subss(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0xF3, 0x0F, 0x5C], 'modrm':None}),
    (xmm_mem128, {'opcode':[0xF3, 0x0F, 0x5C], 'modrm':None}))
  arch_ext = 1

class ucomisd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x2E], 'modrm':None}),
    (xmm_mem64,  {'opcode':[0x66, 0x0F, 0x2E], 'modrm':None}))
  arch_ext = 2

class ucomiss(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x0F, 0x2E], 'modrm':None}),
    (xmm_mem32,  {'opcode':[0x0F, 0x2E], 'modrm':None}))
  arch_ext = 1

class unpckhpd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x15], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x15], 'modrm':None}))
  arch_ext = 1

class unpckhps(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x0F, 0x15], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x0F, 0x15], 'modrm':None}))
  arch_ext = 1

class unpcklpd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x14], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x14], 'modrm':None}))
  arch_ext = 1

class unpcklps(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x0F, 0x14], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x0F, 0x14], 'modrm':None}))
  arch_ext = 1

class xorpd(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x57], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x57], 'modrm':None}))
  arch_ext = 2

class xorps(x86DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x0F, 0x57], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x0F, 0x57], 'modrm':None}))
  arch_ext = 1


if __name__=='__main__':
  r1 = GPRegister32(0)
  r2 = GPRegister32(9)

  a1 = add(r1, 9)
  a2 = add(r1, 70000)

  a3 = add(r2, 9)
  a4 = add(r2, 70000)

  for a in [a1, a2, a3, a4]:
    print a.render()
