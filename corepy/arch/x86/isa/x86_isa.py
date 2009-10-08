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

from corepy.spre.spe import Instruction, DispatchInstruction

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


# ------------------------------
# x86 Instructions
# ------------------------------

# Currently 16bit versions of instructions have separate operand
# functions, and the size-override prefix is in the opcode, so protected
# (32bit default) mode is assumed here.


class adc(DispatchInstruction):
  dispatch = (
    (reg32_simm8,         {'opcode':[0x83],             'modrm':0x10}),
    (mem32_simm8,         {'opcode':[0x83],             'modrm':0x10}),
    (eax_imm32,           {'opcode':[0x15],             'modrm':None}),
    (reg32_imm32,         {'opcode':[0x81],             'modrm':0x10}),
    (mem32_imm32,         {'opcode':[0x81],             'modrm':0x10}),
    (reg32_reg32,         {'opcode':[0x11],             'modrm':None}),
    (mem32_reg32,         {'opcode':[0x11],             'modrm':None}),
    (reg32_mem32,         {'opcode':[0x13],             'modrm':None}),
    (reg16_simm8,         {'opcode':[0x66, 0x83],       'modrm':0x10}),
    (mem16_simm8,         {'opcode':[0x66, 0x83],       'modrm':0x10}),
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
  
class add(DispatchInstruction):
  dispatch = (
    (reg32_simm8,         {'opcode':[0x83],             'modrm':0x00}),
    (mem32_simm8,         {'opcode':[0x83],             'modrm':0x00}),
    (eax_imm32,           {'opcode':[0x05],             'modrm':None}),
    (reg32_imm32,         {'opcode':[0x81],             'modrm':0x00}),
    (mem32_imm32,         {'opcode':[0x81],             'modrm':0x00}),
    (reg32_reg32,         {'opcode':[0x01],             'modrm':None}),
    (mem32_reg32,         {'opcode':[0x01],             'modrm':None}),
    (reg32_mem32,         {'opcode':[0x03],             'modrm':None}),
    (reg16_simm8,         {'opcode':[0x66, 0x83],       'modrm':0x00}),
    (mem16_simm8,         {'opcode':[0x66, 0x83],       'modrm':0x00}),
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
    
class and_(DispatchInstruction):
  dispatch = (
    (reg32_simm8,         {'opcode':[0x83],             'modrm':0x20}),
    (mem32_simm8,         {'opcode':[0x83],             'modrm':0x20}),
    (eax_imm32,           {'opcode':[0x25],             'modrm':None}),
    (reg32_imm32,         {'opcode':[0x81],             'modrm':0x20}),
    (mem32_imm32,         {'opcode':[0x81],             'modrm':0x20}),
    (reg32_reg32,         {'opcode':[0x21],             'modrm':None}),
    (mem32_reg32,         {'opcode':[0x21],             'modrm':None}),
    (reg32_mem32,         {'opcode':[0x23],             'modrm':None}),
    (reg16_simm8,         {'opcode':[0x66, 0x83],       'modrm':0x20}),
    (mem16_simm8,         {'opcode':[0x66, 0x83],       'modrm':0x20}),
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

class bsf(DispatchInstruction):
  dispatch = (
    (reg32_reg32,         {'opcode':[0x0F, 0xBC],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0xBC],       'modrm':None}),
    (reg16_reg16,         {'opcode':[0x66, 0x0F, 0xBC], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0xBC], 'modrm':None}))
    
class bsr(DispatchInstruction):
  dispatch = (
    (reg32_reg32,         {'opcode':[0x0F, 0xBD],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0xBD],       'modrm':None}),
    (reg16_reg16,         {'opcode':[0x66, 0x0F, 0xBD], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0xBD], 'modrm':None}))
  
class bt(DispatchInstruction):
  dispatch = (
    (reg32_imm8,          {'opcode':[0x0F, 0xBA],       'modrm':0x20}),
    (mem32_imm8,          {'opcode':[0x0F, 0xBA],       'modrm':0x20}),
    (reg32_reg32,         {'opcode':[0x0F, 0xA3],       'modrm':None}),
    (mem32_reg32,         {'opcode':[0x0F, 0xA3],       'modrm':None}),
    (reg16_imm8,          {'opcode':[0x66, 0x0F, 0xBA], 'modrm':0x20}),
    (mem16_imm8,          {'opcode':[0x66, 0x0F, 0xBA], 'modrm':0x20}),
    (reg16_reg16,         {'opcode':[0x66, 0x0F, 0xA3], 'modrm':None}),
    (mem16_reg16,         {'opcode':[0x66, 0x0F, 0xA3], 'modrm':None}))
  
class btc(DispatchInstruction):
  dispatch = (
    (reg32_imm8,          {'opcode':[0x0F, 0xBA],       'modrm':0x38}),
    (mem32_imm8,          {'opcode':[0x0F, 0xBA],       'modrm':0x38}),
    (reg32_reg32,         {'opcode':[0x0F, 0xBB],       'modrm':None}),
    (mem32_reg32,         {'opcode':[0x0F, 0xBB],       'modrm':None}),
    (reg16_imm8,          {'opcode':[0x66, 0x0F, 0xBA], 'modrm':0x38}),
    (mem16_imm8,          {'opcode':[0x66, 0x0F, 0xBA], 'modrm':0x38}),
    (reg16_reg16,         {'opcode':[0x66, 0x0F, 0xBB], 'modrm':None}),
    (mem16_reg16,         {'opcode':[0x66, 0x0F, 0xBB], 'modrm':None}))
  
class btr(DispatchInstruction):
  dispatch = (
    (reg32_imm8,          {'opcode':[0x0F, 0xBA],       'modrm':0x30}),
    (mem32_imm8,          {'opcode':[0x0F, 0xBA],       'modrm':0x30}),
    (reg32_reg32,         {'opcode':[0x0F, 0xB3],       'modrm':None}),
    (mem32_reg32,         {'opcode':[0x0F, 0xB3],       'modrm':None}),
    (reg16_imm8,          {'opcode':[0x66, 0x0F, 0xBA], 'modrm':0x30}),
    (mem16_imm8,          {'opcode':[0x66, 0x0F, 0xBA], 'modrm':0x30}),
    (reg16_reg16,         {'opcode':[0x66, 0x0F, 0xB3], 'modrm':None}),
    (mem16_reg16,         {'opcode':[0x66, 0x0F, 0xB3], 'modrm':None}))
  
class bts(DispatchInstruction):
  dispatch = (
    (reg32_imm8,          {'opcode':[0x0F, 0xBA],       'modrm':0x28}),
    (mem32_imm8,          {'opcode':[0x0F, 0xBA],       'modrm':0x28}),
    (reg32_reg32,         {'opcode':[0x0F, 0xAB],       'modrm':None}),
    (mem32_reg32,         {'opcode':[0x0F, 0xAB],       'modrm':None}),
    (reg16_imm8,          {'opcode':[0x66, 0x0F, 0xBA], 'modrm':0x28}),
    (mem16_imm8,          {'opcode':[0x66, 0x0F, 0xBA], 'modrm':0x28}),
    (reg16_reg16,         {'opcode':[0x66, 0x0F, 0xAB], 'modrm':None}),
    (mem16_reg16,         {'opcode':[0x66, 0x0F, 0xAB], 'modrm':None}))
    
class bswap(Instruction):
  machine_inst = reg32
  params = {'opcode':[0x0F, 0xC8],       'modrm':None}
  
class call(DispatchInstruction):
  dispatch = (
    (lbl32off,            {'opcode':[0xE8],             'modrm':None}),
    (rel32off,            {'opcode':[0xE8],             'modrm':None}),
    (reg32,               {'opcode':[0xFF],             'modrm':0x10}),
    (mem32,               {'opcode':[0xFF],             'modrm':0x10}),
    (reg16,               {'opcode':[0x66, 0xFF],       'modrm':0x10}),
    (mem16,               {'opcode':[0x66, 0xFF],       'modrm':0x10}))
  
class cbw(Instruction):
  machine_inst = no_op
  params = {'opcode':[0x66, 0x98],       'modrm':None}
  
class cwde(Instruction):
  machine_inst = no_op
  params = {'opcode':[0x98],             'modrm':None}
  
class cwd(Instruction):
  machine_inst = no_op
  params = {'opcode':[0x66, 0x99],       'modrm':None}
  
class cdq(Instruction):
  machine_inst = no_op
  params = {'opcode':[0x99],             'modrm':None}
  
class clc(Instruction):
  machine_inst = no_op
  params = {'opcode':[0xF8],             'modrm':None}
  
class cld(Instruction):
  machine_inst = no_op
  params = {'opcode':[0xFC],             'modrm':None}
  
class clflush(Instruction):
  machine_inst = mem8
  params = {'opcode':[0x0F, 0xAE],       'modrm':0x38}
  
class cmc(Instruction):
  machine_inst = no_op
  params = {'opcode':[0xF5],             'modrm':None}
  
class cmovo(DispatchInstruction):
  dispatch = (
    (reg32_reg32_rev,     {'opcode':[0x0F, 0x40],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0x40],       'modrm':None}),
    (reg16_reg16_rev,     {'opcode':[0x66, 0x0F, 0x40], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0x40], 'modrm':None}))
    
class cmovno(DispatchInstruction):
  dispatch = (
    (reg32_reg32_rev,     {'opcode':[0x0F, 0x41],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0x41],       'modrm':None}),
    (reg16_reg16_rev,     {'opcode':[0x66, 0x0F, 0x41], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0x41], 'modrm':None}))
    
class cmovb(DispatchInstruction):
  dispatch = (
    (reg32_reg32_rev,     {'opcode':[0x0F, 0x42],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0x42],       'modrm':None}),
    (reg16_reg16_rev,     {'opcode':[0x66, 0x0F, 0x42], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0x42], 'modrm':None}))
    
class cmovc(DispatchInstruction):
  dispatch = (
    (reg32_reg32_rev,     {'opcode':[0x0F, 0x42],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0x42],       'modrm':None}),
    (reg16_reg16_rev,     {'opcode':[0x66, 0x0F, 0x42], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0x42], 'modrm':None}))
    
class cmovnae(DispatchInstruction):
  dispatch = (
    (reg32_reg32_rev,     {'opcode':[0x0F, 0x42],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0x42],       'modrm':None}),
    (reg16_reg16_rev,     {'opcode':[0x66, 0x0F, 0x42], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0x42], 'modrm':None}))
    
class cmovnb(DispatchInstruction):
  dispatch = (
    (reg32_reg32_rev,     {'opcode':[0x0F, 0x43],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0x43],       'modrm':None}),
    (reg16_reg16_rev,     {'opcode':[0x66, 0x0F, 0x43], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0x43], 'modrm':None}))
    
class cmovnc(DispatchInstruction):
  dispatch = (
    (reg32_reg32_rev,     {'opcode':[0x0F, 0x43],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0x43],       'modrm':None}),
    (reg16_reg16_rev,     {'opcode':[0x66, 0x0F, 0x43], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0x43], 'modrm':None}))
    
class cmovae(DispatchInstruction):
  dispatch = (
    (reg32_reg32_rev,     {'opcode':[0x0F, 0x43],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0x43],       'modrm':None}),
    (reg16_reg16_rev,     {'opcode':[0x66, 0x0F, 0x43], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0x43], 'modrm':None}))
  
class cmovz(DispatchInstruction):
  dispatch = (
    (reg32_reg32_rev,     {'opcode':[0x0F, 0x44],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0x44],       'modrm':None}),
    (reg16_reg16_rev,     {'opcode':[0x66, 0x0F, 0x44], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0x44], 'modrm':None}))
  
class cmove(DispatchInstruction):
  dispatch = (
    (reg32_reg32_rev,     {'opcode':[0x0F, 0x44],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0x44],       'modrm':None}),
    (reg16_reg16_rev,     {'opcode':[0x66, 0x0F, 0x44], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0x44], 'modrm':None}))
    
class cmovnz(DispatchInstruction):
  dispatch = (
    (reg32_reg32_rev,     {'opcode':[0x0F, 0x45],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0x45],       'modrm':None}),
    (reg16_reg16_rev,     {'opcode':[0x66, 0x0F, 0x45], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0x45], 'modrm':None}))
  
class cmovne(DispatchInstruction):
  dispatch = (
    (reg32_reg32_rev,     {'opcode':[0x0F, 0x45],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0x45],       'modrm':None}),
    (reg16_reg16_rev,     {'opcode':[0x66, 0x0F, 0x45], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0x45], 'modrm':None}))
    
class cmovbe(DispatchInstruction):
  dispatch = (
    (reg32_reg32_rev,     {'opcode':[0x0F, 0x46],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0x46],       'modrm':None}),
    (reg16_reg16_rev,     {'opcode':[0x66, 0x0F, 0x46], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0x46], 'modrm':None}))
    
class cmovna(DispatchInstruction):
  dispatch = (
    (reg32_reg32_rev,     {'opcode':[0x0F, 0x46],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0x46],       'modrm':None}),
    (reg16_reg16_rev,     {'opcode':[0x66, 0x0F, 0x46], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0x46], 'modrm':None}))
    
class cmovnbe(DispatchInstruction):
  dispatch = (
    (reg32_reg32_rev,     {'opcode':[0x0F, 0x47],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0x47],       'modrm':None}),
    (reg16_reg16_rev,     {'opcode':[0x66, 0x0F, 0x47], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0x47], 'modrm':None}))
    
class cmova(DispatchInstruction):
  dispatch = (
    (reg32_reg32_rev,     {'opcode':[0x0F, 0x47],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0x47],       'modrm':None}),
    (reg16_reg16_rev,     {'opcode':[0x66, 0x0F, 0x47], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0x47], 'modrm':None}))
    
class cmovs(DispatchInstruction):
  dispatch = (
    (reg32_reg32_rev,     {'opcode':[0x0F, 0x48],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0x48],       'modrm':None}),
    (reg16_reg16_rev,     {'opcode':[0x66, 0x0F, 0x48], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0x48], 'modrm':None}))
  
class cmovns(DispatchInstruction):
  dispatch = (
    (reg32_reg32_rev,     {'opcode':[0x0F, 0x49],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0x49],       'modrm':None}),
    (reg16_reg16_rev,     {'opcode':[0x66, 0x0F, 0x49], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0x49], 'modrm':None}))
    
class cmovp(DispatchInstruction):
  dispatch = (
    (reg32_reg32_rev,     {'opcode':[0x0F, 0x4A],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0x4A],       'modrm':None}),
    (reg16_reg16_rev,     {'opcode':[0x66, 0x0F, 0x4A], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0x4A], 'modrm':None}))
  
class cmovpe(DispatchInstruction):
  dispatch = (
    (reg32_reg32_rev,     {'opcode':[0x0F, 0x4A],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0x4A],       'modrm':None}),
    (reg16_reg16_rev,     {'opcode':[0x66, 0x0F, 0x4A], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0x4A], 'modrm':None}))
  
class cmovnp(DispatchInstruction):
  dispatch = (
    (reg32_reg32_rev,     {'opcode':[0x0F, 0x4B],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0x4B],       'modrm':None}),
    (reg16_reg16_rev,     {'opcode':[0x66, 0x0F, 0x4B], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0x4B], 'modrm':None}))
    
class cmovpo(DispatchInstruction):
  dispatch = (
    (reg32_reg32_rev,     {'opcode':[0x0F, 0x4B],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0x4B],       'modrm':None}),
    (reg16_reg16_rev,     {'opcode':[0x66, 0x0F, 0x4B], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0x4B], 'modrm':None}))
    
class cmovl(DispatchInstruction):
  dispatch = (
    (reg32_reg32_rev,     {'opcode':[0x0F, 0x4C],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0x4C],       'modrm':None}),
    (reg16_reg16_rev,     {'opcode':[0x66, 0x0F, 0x4C], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0x4C], 'modrm':None}))
  
class cmovnge(DispatchInstruction):
  dispatch = (
    (reg32_reg32_rev,     {'opcode':[0x0F, 0x4C],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0x4C],       'modrm':None}),
    (reg16_reg16_rev,     {'opcode':[0x66, 0x0F, 0x4C], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0x4C], 'modrm':None}))
    
class cmovnl(DispatchInstruction):
  dispatch = (
    (reg32_reg32_rev,     {'opcode':[0x0F, 0x4D],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0x4D],       'modrm':None}),
    (reg16_reg16_rev,     {'opcode':[0x66, 0x0F, 0x4D], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0x4D], 'modrm':None}))
  
class cmovge(DispatchInstruction):
  dispatch = (
    (reg32_reg32_rev,     {'opcode':[0x0F, 0x4D],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0x4D],       'modrm':None}),
    (reg16_reg16_rev,     {'opcode':[0x66, 0x0F, 0x4D], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0x4D], 'modrm':None}))
    
class cmovle(DispatchInstruction):
  dispatch = (
    (reg32_reg32_rev,     {'opcode':[0x0F, 0x4E],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0x4E],       'modrm':None}),
    (reg16_reg16_rev,     {'opcode':[0x66, 0x0F, 0x4E], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0x4E], 'modrm':None}))
    
class cmovng(DispatchInstruction):
  dispatch = (
    (reg32_reg32_rev,     {'opcode':[0x0F, 0x4E],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0x4E],       'modrm':None}),
    (reg16_reg16_rev,     {'opcode':[0x66, 0x0F, 0x4E], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0x4E], 'modrm':None}))
  
class cmovnle(DispatchInstruction):
  dispatch = (
    (reg32_reg32_rev,     {'opcode':[0x0F, 0x4F],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0x4F],       'modrm':None}),
    (reg16_reg16_rev,     {'opcode':[0x66, 0x0F, 0x4F], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0x4F], 'modrm':None}))
    
class cmovg(DispatchInstruction):
  dispatch = (
    (reg32_reg32_rev,     {'opcode':[0x0F, 0x4F],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0x4F],       'modrm':None}),
    (reg16_reg16_rev,     {'opcode':[0x66, 0x0F, 0x4F], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0x4F], 'modrm':None}))
    
class cmp(DispatchInstruction):
  dispatch = (
    (reg32_simm8,         {'opcode':[0x83],             'modrm':0x38}),
    (mem32_simm8,         {'opcode':[0x83],             'modrm':0x38}),
    (eax_imm32,           {'opcode':[0x3D],             'modrm':None}),
    (reg32_imm32,         {'opcode':[0x81],             'modrm':0x38}),
    (mem32_imm32,         {'opcode':[0x81],             'modrm':0x38}),
    (reg32_reg32,         {'opcode':[0x39],             'modrm':None}),
    (mem32_reg32,         {'opcode':[0x39],             'modrm':None}),
    (reg32_mem32,         {'opcode':[0x3B],             'modrm':None}),
    (reg16_simm8,         {'opcode':[0x66, 0x83],       'modrm':0x38}),
    (mem16_simm8,         {'opcode':[0x66, 0x83],       'modrm':0x38}),
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
    
class cmpsb(Instruction):
  machine_inst = no_op
  params = {'opcode':[0xA6],             'modrm':None}
  
class cmpsd(Instruction):
  machine_inst = no_op
  params = {'opcode':[0xA7],             'modrm':None}
  
class cmpsw(Instruction):
  machine_inst = no_op
  params = {'opcode':[0x66, 0xA7],       'modrm':None}
  
class cmpxchg(DispatchInstruction):
  dispatch = (
    (reg32_reg32,         {'opcode':[0x0F, 0xB1],       'modrm':None}),
    (mem32_reg32,         {'opcode':[0x0F, 0xB1],       'modrm':None}),
    (reg16_reg16,         {'opcode':[0x66, 0x0F, 0xB1], 'modrm':None}),
    (mem16_reg16,         {'opcode':[0x66, 0x0F, 0xB1], 'modrm':None}))
  
class cmpxchg8b(Instruction):
  machine_inst = mem64
  params = {'opcode':[0x0F, 0xC7],       'modrm':0x08}
  
class cpuid(Instruction):
  machine_inst = no_op
  params = {'opcode':[0x0F, 0xA2],       'modrm':None}
  
class crc32(DispatchInstruction):
  dispatch = (
    (reg32_reg32, {'opcode':[0xF2, 0x0F, 0x38, 0xF1], 'modrm':None}),
    (reg32_mem32, {'opcode':[0xF2, 0x0F, 0x38, 0xF1], 'modrm':None}),
    (reg32_reg16, {'opcode':[0xF2, 0x0F, 0x38, 0xF1], 'modrm':None}),
    (reg32_mem16, {'opcode':[0xF2, 0x0F, 0x38, 0xF1], 'modrm':None}),
    (reg32_reg8,  {'opcode':[0xF2, 0x0F, 0x38, 0xF0], 'modrm':None}),
    (reg32_mem8,  {'opcode':[0xF2, 0x0F, 0x38, 0xF0], 'modrm':None}))
  
class dec(DispatchInstruction):
  dispatch = (
    (reg32,               {'opcode':[0x48],             'modrm':None}),
    (mem32,               {'opcode':[0xFF],             'modrm':0x08}),
    (reg16,               {'opcode':[0x66, 0x48],       'modrm':None}),
    (mem16,               {'opcode':[0x66, 0xFF],       'modrm':0x08}),
    (reg8,                {'opcode':[0xFE],             'modrm':0x08}),
    (mem8,                {'opcode':[0xFE],             'modrm':0x08}))
    
class div(DispatchInstruction):
  dispatch = (
    (reg32,               {'opcode':[0xF7],             'modrm':0x30}),
    (mem32,               {'opcode':[0xF7],             'modrm':0x30}),
    (reg16,               {'opcode':[0x66, 0xF7],       'modrm':0x30}),
    (mem16,               {'opcode':[0x66, 0xF7],       'modrm':0x30}),
    (reg8,                {'opcode':[0xF6],             'modrm':0x30}),
    (mem8,                {'opcode':[0xF6],             'modrm':0x30}))
    
class enter(Instruction):
  machine_inst = imm16_imm8
  params = {'opcode':[0xC8],             'modrm':None}
  
class idiv(DispatchInstruction):
  dispatch = (
    (reg32,               {'opcode':[0xF7],             'modrm':0x38}),
    (mem32,               {'opcode':[0xF7],             'modrm':0x38}),
    (reg16,               {'opcode':[0x66, 0xF7],       'modrm':0x38}),
    (mem16,               {'opcode':[0x66, 0xF7],       'modrm':0x38}),
    (reg8,                {'opcode':[0xF6],             'modrm':0x38}),
    (mem8,                {'opcode':[0xF6],             'modrm':0x38}))
  
class imul(DispatchInstruction):
  dispatch = (
    (reg32,               {'opcode':[0xF7],             'modrm':0x28}),
    (mem32,               {'opcode':[0xF7],             'modrm':0x28}),
    (reg32_reg32,         {'opcode':[0x0F, 0xAF],       'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0F, 0xAF],       'modrm':None}),
    (reg32_reg32_simm8_rev,{'opcode':[0x6B],             'modrm':None}),
    (reg32_mem32_simm8,    {'opcode':[0x6B],             'modrm':None}),
    (reg32_reg32_imm32,   {'opcode':[0x69],             'modrm':None}),
    (reg32_mem32_imm32,   {'opcode':[0x69],             'modrm':None}),
    (reg16,               {'opcode':[0x66, 0xF7],       'modrm':0x28}),
    (mem16,               {'opcode':[0x66, 0xF7],       'modrm':0x28}),
    (reg16_reg16,         {'opcode':[0x66, 0x0F, 0xAF], 'modrm':None}),
    (reg16_mem16,         {'opcode':[0x66, 0x0F, 0xAF], 'modrm':None}),
    (reg16_reg16_simm8_rev,{'opcode':[0x66, 0x6B],       'modrm':None}),
    (reg16_mem16_simm8,    {'opcode':[0x66, 0x6B],       'modrm':None}),
    (reg16_reg16_imm16,   {'opcode':[0x66, 0x69],       'modrm':None}),
    (reg16_mem16_imm16,   {'opcode':[0x66, 0x69],       'modrm':None}),
    (reg8,                {'opcode':[0xF6],             'modrm':0x28}),
    (mem8,                {'opcode':[0xF6],             'modrm':0x28}))
  
class in_(DispatchInstruction):
  dispatch = (
    (eax_dx,              {'opcode':[0xED],             'modrm':None}),
    (ax_dx,               {'opcode':[0x66, 0xED],       'modrm':None}),
    (al_dx,               {'opcode':[0xEC],             'modrm':None}),
    (eax_imm8,            {'opcode':[0xE5],             'modrm':None}),
    (ax_imm8,             {'opcode':[0x66, 0xE5],       'modrm':None}),
    (al_imm8,             {'opcode':[0xE4],             'modrm':None}))
    
class inc(DispatchInstruction):
  dispatch = (
    (reg32,               {'opcode':[0x40],             'modrm':None}),
    (reg32,               {'opcode':[0xFF],             'modrm':0x00}),
    (mem32,               {'opcode':[0xFF],             'modrm':0x00}),
    (reg16,               {'opcode':[0x66, 0x40],       'modrm':None}),
    (reg16,               {'opcode':[0x66, 0xFF],       'modrm':0x00}),
    (mem16,               {'opcode':[0x66, 0xFF],       'modrm':0x00}),
    (reg8,                {'opcode':[0xFE],             'modrm':0x00}),
    (mem8,                {'opcode':[0xFE],             'modrm':0x00}))
    
class insb(Instruction):
  machine_inst = no_op
  params = {'opcode':[0x6C],             'modrm':None}
  
class insd(Instruction):
  machine_inst = no_op
  params = {'opcode':[0x6D],             'modrm':None}
  
class insw(Instruction):
  machine_inst = no_op
  params = {'opcode':[0x66, 0x6D],       'modrm':None}
  
class int_(Instruction):
  machine_inst = imm8
  params = {'opcode':[0xCD],             'modrm':None}
  
class int_3(Instruction):
  """NOTE - this is a special form of 'int 3' used for debugging; see the
     architecture manuals for more information."""
  machine_inst = no_op
  params = {'opcode':[0xCC],             'modrm':None}
  
class ja(DispatchInstruction):
  dispatch = (
    (lbl32_8off,          {'opcode':[[0x77], [0x0F, 0x87]], 'modrm':None}),
    (rel32_8off,          {'opcode':[[0x77], [0x0F, 0x87]], 'modrm':None}))
    
class jae(DispatchInstruction):
  dispatch = (
    (lbl32_8off,          {'opcode':[[0x73], [0x0F, 0x83]], 'modrm':None}),
    (rel32_8off,          {'opcode':[[0x73], [0x0F, 0x83]], 'modrm':None}))
    
class jb(DispatchInstruction):
  dispatch = (
    (lbl32_8off,          {'opcode':[[0x72], [0x0F, 0x82]], 'modrm':None}),
    (rel32_8off,          {'opcode':[[0x72], [0x0F, 0x82]], 'modrm':None}))
  
class jbe(DispatchInstruction):
  dispatch = (
    (lbl32_8off,          {'opcode':[[0x76], [0x0F, 0x86]], 'modrm':None}),
    (rel32_8off,          {'opcode':[[0x76], [0x0F, 0x86]], 'modrm':None}))
    
class jc(DispatchInstruction):
  dispatch = (
    (lbl32_8off,          {'opcode':[[0x72], [0x0F, 0x82]], 'modrm':None}),
    (rel32_8off,          {'opcode':[[0x72], [0x0F, 0x82]], 'modrm':None}))
    
class jcxz(DispatchInstruction):
  dispatch = (
    (lbl8off,             {'opcode':[0x67, 0xE3],       'modrm':None}),
    (rel8off,             {'opcode':[0x67, 0xE3],       'modrm':None}))
  
class je(DispatchInstruction):
  dispatch = (
    (lbl32_8off,          {'opcode':[[0x74], [0x0F, 0x84]], 'modrm':None}),
    (rel32_8off,          {'opcode':[[0x74], [0x0F, 0x84]], 'modrm':None}))
    
class jecxz(DispatchInstruction):
  dispatch = (
    (lbl8off,             {'opcode':[0xE3],             'modrm':None}),
    (rel8off,             {'opcode':[0xE3],             'modrm':None}))
  
class jg(DispatchInstruction):
  dispatch = (
    (lbl32_8off,          {'opcode':[[0x7F], [0x0F, 0x8F]], 'modrm':None}),
    (rel32_8off,          {'opcode':[[0x7F], [0x0F, 0x8F]], 'modrm':None}))
    
class jge(DispatchInstruction):
  dispatch = (
    (lbl32_8off,          {'opcode':[[0x7D], [0x0F, 0x8D]], 'modrm':None}),
    (rel32_8off,          {'opcode':[[0x7D], [0x0F, 0x8D]], 'modrm':None}))
    
class jl(DispatchInstruction):
  dispatch = (
    (lbl32_8off,          {'opcode':[[0x7C], [0x0F, 0x8C]], 'modrm':None}),
    (rel32_8off,          {'opcode':[[0x7C], [0x0F, 0x8C]], 'modrm':None}))
    
class jle(DispatchInstruction):
  dispatch = (
    (lbl32_8off,          {'opcode':[[0x7E], [0x0F, 0x8E]], 'modrm':None}),
    (rel32_8off,          {'opcode':[[0x7E], [0x0F, 0x8E]], 'modrm':None}))
    
class jmp(DispatchInstruction):
  dispatch = (
    (lbl32_8off,          {'opcode':[[0xEB], [0xE9]],   'modrm':None}),
    (rel32_8off,          {'opcode':[[0xEB], [0xE9]],   'modrm':None}),
    (reg32,               {'opcode':[0xFF],             'modrm':0x20}),
    (mem32,               {'opcode':[0xFF],             'modrm':0x20}),
    (reg16,               {'opcode':[0x66, 0xFF],       'modrm':0x20}),
    (mem16,               {'opcode':[0x66, 0xFF],       'modrm':0x20}))
    
class jna(DispatchInstruction):
  dispatch = (
    (lbl32_8off,          {'opcode':[[0x76], [0x0F, 0x86]], 'modrm':None}),
    (rel32_8off,          {'opcode':[[0x76], [0x0F, 0x86]], 'modrm':None}))
    
class jnae(DispatchInstruction):
  dispatch = (
    (lbl32_8off,          {'opcode':[[0x72], [0x0F, 0x82]], 'modrm':None}),
    (rel32_8off,          {'opcode':[[0x72], [0x0F, 0x82]], 'modrm':None}))
    
class jnb(DispatchInstruction):
  dispatch = (
    (lbl32_8off,          {'opcode':[[0x73], [0x0F, 0x83]], 'modrm':None}),
    (rel32_8off,          {'opcode':[[0x73], [0x0F, 0x83]], 'modrm':None}))
    
class jnbe(DispatchInstruction):
  dispatch = (
    (lbl32_8off,          {'opcode':[[0x77], [0x0F, 0x87]], 'modrm':None}),
    (rel32_8off,          {'opcode':[[0x77], [0x0F, 0x87]], 'modrm':None}))
    
class jnc(DispatchInstruction):
  dispatch = (
    (lbl32_8off,          {'opcode':[[0x73], [0x0F, 0x83]], 'modrm':None}),
    (rel32_8off,          {'opcode':[[0x73], [0x0F, 0x83]], 'modrm':None}))
    
class jne(DispatchInstruction):
  dispatch = (
    (lbl32_8off,          {'opcode':[[0x75], [0x0F, 0x85]], 'modrm':None}),
    (rel32_8off,          {'opcode':[[0x75], [0x0F, 0x85]], 'modrm':None}))
    
class jng(DispatchInstruction):
  dispatch = (
    (lbl32_8off,          {'opcode':[[0x7E], [0x0F, 0x8E]], 'modrm':None}),
    (rel32_8off,          {'opcode':[[0x7E], [0x0F, 0x8E]], 'modrm':None}))
    
class jnge(DispatchInstruction):
  dispatch = (
    (lbl32_8off,          {'opcode':[[0x7C], [0x0F, 0x8C]], 'modrm':None}),
    (rel32_8off,          {'opcode':[[0x7C], [0x0F, 0x8C]], 'modrm':None}))
    
class jnl(DispatchInstruction):
  dispatch = (
    (lbl32_8off,          {'opcode':[[0x7D], [0x0F, 0x8D]], 'modrm':None}),
    (rel32_8off,          {'opcode':[[0x7D], [0x0F, 0x8D]], 'modrm':None}))
    
class jnle(DispatchInstruction):
  dispatch = (
    (lbl32_8off,          {'opcode':[[0x7F], [0x0F, 0x8F]], 'modrm':None}),
    (rel32_8off,          {'opcode':[[0x7F], [0x0F, 0x8F]], 'modrm':None}))
    
class jno(DispatchInstruction):
  dispatch = (
    (lbl32_8off,          {'opcode':[[0x71], [0x0F, 0x81]], 'modrm':None}),
    (rel32_8off,          {'opcode':[[0x71], [0x0F, 0x81]], 'modrm':None}))
  
class jnp(DispatchInstruction):
  dispatch = (
    (lbl32_8off,          {'opcode':[[0x7B], [0x0F, 0x8B]], 'modrm':None}),
    (rel32_8off,          {'opcode':[[0x7B], [0x0F, 0x8B]], 'modrm':None}))
    
class jns(DispatchInstruction):
  dispatch = (
    (lbl32_8off,          {'opcode':[[0x79], [0x0F, 0x89]], 'modrm':None}),
    (rel32_8off,          {'opcode':[[0x79], [0x0F, 0x89]], 'modrm':None}))
  
class jnz(DispatchInstruction):
  dispatch = (
    (lbl32_8off,          {'opcode':[[0x75], [0x0F, 0x85]], 'modrm':None}),
    (rel32_8off,          {'opcode':[[0x75], [0x0F, 0x85]], 'modrm':None}))
    
class jo(DispatchInstruction):
  dispatch = (
    (lbl32_8off,          {'opcode':[[0x70], [0x0F, 0x80]], 'modrm':None}),
    (rel32_8off,          {'opcode':[[0x70], [0x0F, 0x80]], 'modrm':None}))
    
class jp(DispatchInstruction):
  dispatch = (
    (lbl32_8off,          {'opcode':[[0x7A], [0x0F, 0x8A]], 'modrm':None}),
    (rel32_8off,          {'opcode':[[0x7A], [0x0F, 0x8A]], 'modrm':None}))
    
class jpe(DispatchInstruction):
  dispatch = (
    (lbl32_8off,          {'opcode':[[0x7A], [0x0F, 0x8A]], 'modrm':None}),
    (rel32_8off,          {'opcode':[[0x7A], [0x0F, 0x8A]], 'modrm':None}))
    
class jpo(DispatchInstruction):
  dispatch = (
    (lbl32_8off,          {'opcode':[[0x7B], [0x0F, 0x8B]], 'modrm':None}),
    (rel32_8off,          {'opcode':[[0x7B], [0x0F, 0x8B]], 'modrm':None}))
    
class js(DispatchInstruction):
  dispatch = (
    (lbl32_8off,          {'opcode':[[0x78], [0x0F, 0x88]], 'modrm':None}),
    (rel32_8off,          {'opcode':[[0x78], [0x0F, 0x88]], 'modrm':None}))
    
class jz(DispatchInstruction):
  dispatch = (
    (lbl32_8off,          {'opcode':[[0x74], [0x0F, 0x84]], 'modrm':None}),
    (rel32_8off,          {'opcode':[[0x74], [0x0F, 0x84]], 'modrm':None}))
  
class lahf(Instruction):
  machine_inst = no_op
  params = {'opcode':[0x9F],             'modrm':None}
  
class lea(DispatchInstruction):
  dispatch = (
    (reg32_mem,           {'opcode':[0x8D],             'modrm':0x00}),
    (reg16_mem,           {'opcode':[0x66, 0x8D],       'modrm':0x00}))
    
class leave(Instruction):
  machine_inst = no_op
  params = {'opcode':[0xC9],             'modrm':None}
  
class lfence(Instruction):
  machine_inst = no_op
  params = {'opcode':[0x0F, 0xAE, 0xE8], 'modrm':None}
  
class lodsb(Instruction):
  machine_inst = no_op
  params = {'opcode':[0xAC],             'modrm':None}
  
class lodsd(Instruction):
  machine_inst = no_op
  params = {'opcode':[0xAD],             'modrm':None}
  
class lodsw(Instruction):
  machine_inst = no_op
  params = {'opcode':[0x66, 0xAD],       'modrm':None}
  
class loop(DispatchInstruction):
  dispatch = (
    (lbl8off,             {'opcode':[0xE2],             'modrm':None}),
    (rel8off,             {'opcode':[0xE2],             'modrm':None}))

class loope(DispatchInstruction):
  dispatch = (
    (lbl8off,             {'opcode':[0xE1],             'modrm':None}),
    (rel8off,             {'opcode':[0xE1],             'modrm':None}))

class loopne(DispatchInstruction):
  dispatch = (
    (lbl8off,             {'opcode':[0xE0],             'modrm':None}),
    (rel8off,             {'opcode':[0xE0],             'modrm':None}))

class loopnz(DispatchInstruction):
  dispatch = (
    (lbl8off,             {'opcode':[0xE0],             'modrm':None}),
    (rel8off,             {'opcode':[0xE0],             'modrm':None}))

class loopz(DispatchInstruction):
  dispatch = (
    (lbl8off,             {'opcode':[0xE1],             'modrm':None}),
    (rel8off,             {'opcode':[0xE1],             'modrm':None}))

class lzcnt(DispatchInstruction):
  dispatch = (
    (reg32_reg32,    {'opcode':[0xF3, 0x0F, 0xBD], 'modrm':None}),
    (reg32_mem32,    {'opcode':[0xF3, 0x0F, 0xBD], 'modrm':None}),
    (reg16_reg16,   {'opcode':[0x66, 0xF3, 0x0F, 0xBD], 'modrm':None}),
    (reg16_mem16,   {'opcode':[0x66, 0xF3, 0x0F, 0xBD], 'modrm':None}))
            
class mfence(Instruction):
  machine_inst = no_op
  params = {'opcode':[0x0F, 0xAE, 0xF0], 'modrm':None}
  
class mov(DispatchInstruction):
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
    
class movnti(Instruction):
  machine_inst = mem32_reg32
  params = {'opcode':[0x0F, 0xC3],       'modrm':None}
  # SSE2!

class movsb(Instruction):
  machine_inst = no_op
  params = {'opcode':[0xA4],             'modrm':None}
  
class movsd(Instruction):
  machine_inst = no_op
  params = {'opcode':[0xA5],             'modrm':None}
  
class movsw(Instruction):
  machine_inst = no_op
  params = {'opcode':[0x66, 0xA5],       'modrm':None}
  
class movsx(DispatchInstruction):
  dispatch = (
    (reg32_reg8,          {'opcode':[0x0F, 0xBE], 'modrm':None}),
    (reg32_mem8,          {'opcode':[0x0F, 0xBE], 'modrm':None}),
    (reg32_reg16,         {'opcode':[0x0F, 0xBF], 'modrm':None}),
    (reg32_mem16,         {'opcode':[0x0F, 0xBF], 'modrm':None}),
    (reg16_reg8,          {'opcode':[0x66, 0x0F, 0xBE], 'modrm':None}),
    (reg16_mem8,          {'opcode':[0x66, 0x0F, 0xBE], 'modrm':None}))
  
class movzx(DispatchInstruction):
  dispatch = (
    (reg32_reg8,          {'opcode':[0x0F, 0xB6], 'modrm':None}),
    (reg32_mem8,          {'opcode':[0x0F, 0xB6], 'modrm':None}),
    (reg32_reg16,         {'opcode':[0x0F, 0xB7], 'modrm':None}),
    (reg32_mem16,         {'opcode':[0x0F, 0xB7], 'modrm':None}),
    (reg16_reg8,          {'opcode':[0x66, 0x0F, 0xB6], 'modrm':None}),
    (reg16_mem8,          {'opcode':[0x66, 0x0F, 0xB6], 'modrm':None}))
  
class mul(DispatchInstruction):
  dispatch = (
    (reg32,               {'opcode':[0xF7],             'modrm':0x20}),
    (mem32,               {'opcode':[0xF7],             'modrm':0x20}),
    (reg16,               {'opcode':[0x66, 0xF7],       'modrm':0x20}),
    (mem16,               {'opcode':[0x66, 0xF7],       'modrm':0x20}),
    (reg8,                {'opcode':[0xF6],             'modrm':0x20}),
    (mem8,                {'opcode':[0xF6],             'modrm':0x20}))
  
class neg(DispatchInstruction):
  dispatch = (
    (reg32,               {'opcode':[0xF7],             'modrm':0x18}),
    (mem32,               {'opcode':[0xF7],             'modrm':0x18}),
    (reg16,               {'opcode':[0x66, 0xF7],       'modrm':0x18}),
    (mem16,               {'opcode':[0x66, 0xF7],       'modrm':0x18}),
    (reg8,                {'opcode':[0xF6],             'modrm':0x18}),
    (mem8,                {'opcode':[0xF6],             'modrm':0x18}))
  
class nop(Instruction):
  machine_inst = no_op
  params = {'opcode':[0x90],             'modrm':None}
  
class not_(DispatchInstruction):
  dispatch = (
    (reg32,               {'opcode':[0xF7],             'modrm':0x10}),
    (mem32,               {'opcode':[0xF7],             'modrm':0x10}),
    (reg16,               {'opcode':[0x66, 0xF7],       'modrm':0x10}),
    (mem16,               {'opcode':[0x66, 0xF7],       'modrm':0x10}),
    (reg8,                {'opcode':[0xF6],             'modrm':0x10}),
    (mem8,                {'opcode':[0xF6],             'modrm':0x10}))
  
class or_(DispatchInstruction):
  dispatch = (
    (reg32_simm8,         {'opcode':[0x83],             'modrm':0x08}),
    (mem32_simm8,         {'opcode':[0x83],             'modrm':0x08}),
    (eax_imm32,           {'opcode':[0x0D],             'modrm':None}),
    (reg32_imm32,         {'opcode':[0x81],             'modrm':0x08}),
    (mem32_imm32,         {'opcode':[0x81],             'modrm':0x08}),
    (mem32_reg32,         {'opcode':[0x09],             'modrm':None}),
    (reg32_reg32,         {'opcode':[0x09],             'modrm':None}),
    (reg32_mem32,         {'opcode':[0x0B],             'modrm':None}),
    (reg16_simm8,         {'opcode':[0x66, 0x83],       'modrm':0x08}),
    (mem16_simm8,         {'opcode':[0x66, 0x83],       'modrm':0x08}),
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
    
class out(DispatchInstruction):
  dispatch = (
    (dx_eax,              {'opcode':[0xEF],             'modrm':None}),
    (dx_ax,               {'opcode':[0x66, 0xEF],       'modrm':None}),
    (dx_al,               {'opcode':[0xEE],             'modrm':None}),
    (imm8_eax,            {'opcode':[0xE7],             'modrm':None}),
    (imm8_ax,             {'opcode':[0x66, 0xE7],       'modrm':None}),
    (imm8_al,             {'opcode':[0xE6],             'modrm':None}))
    
class outsb(Instruction):
  machine_inst = no_op
  params = {'opcode':[0x6E],             'modrm':None}
  
class outsd(Instruction):
  machine_inst = no_op
  params = {'opcode':[0x6F],             'modrm':None}
  
class outsw(Instruction):
  machine_inst = no_op
  params = {'opcode':[0x66, 0x6F],       'modrm':None}
  
class pause(Instruction):
  machine_inst = no_op
  params = {'opcode':[0xF3, 0x90],       'modrm':None}
  
class pop(DispatchInstruction):
  dispatch = (
    (reg32,               {'opcode':[0x58],             'modrm':None}),
    (mem32,               {'opcode':[0x8F],             'modrm':0x00}),
    (reg16,               {'opcode':[0x66, 0x58],       'modrm':None}),
    (mem16,               {'opcode':[0x66, 0x8F],       'modrm':0x00}))
    
class popa(Instruction):
  machine_inst = no_op
  params = {'opcode':[0x66, 0x61],       'modrm':None}
  
class popad(Instruction):
  machine_inst = no_op
  params = {'opcode':[0x61],             'modrm':None}
  
class popcnt(DispatchInstruction):
  dispatch = (
    (reg32_reg32,         {'opcode':[0xF3, 0x0F, 0xB8], 'modrm':None}),
    (reg32_mem32,         {'opcode':[0xF3, 0x0F, 0xB8], 'modrm':None}),
    (reg16_reg16,    {'opcode':[0x66, 0xF3, 0x0F, 0xB8],'modrm':None}),
    (reg16_mem16,    {'opcode':[0x66, 0xF3, 0x0F, 0xB8],'modrm':None}))
  
class popf(Instruction):
  machine_inst = no_op
  params = {'opcode':[0x66, 0x9D],       'modrm':None}
  
class popfd(Instruction):
  machine_inst = no_op
  params = {'opcode':[0x9D],             'modrm':None}
  
class prefetch(Instruction):
  machine_inst = mem8
  params = {'opcode':[0x0F, 0x0D],       'modrm':0x00}
  
class prefetchnta(Instruction):
  machine_inst = mem8
  params = {'opcode':[0x0F, 0x18],       'modrm':0x00}
  
class prefetcht0(Instruction):
  machine_inst = mem8
  params = {'opcode':[0x0F, 0x18],       'modrm':0x08}
  
class prefetcht1(Instruction):
  machine_inst = mem8
  params = {'opcode':[0x0F, 0x18],       'modrm':0x10}
  
class prefetcht2(Instruction):
  machine_inst = mem8
  params = {'opcode':[0x0F, 0x18],       'modrm':0x18}
  
class prefetchw(Instruction):
  machine_inst = mem8
  params = {'opcode':[0x0F, 0x0D],       'modrm':0x08}
  
class push(DispatchInstruction):
  dispatch = (
    (reg32,               {'opcode':[0x50],             'modrm':None}),
    (mem32,               {'opcode':[0xFF],             'modrm':0x30}),
    # TODO - add keyword arg to override operand size?
    #(simm8,               {'opcode':[0x6A],             'modrm':None}),
    #(imm16,               {'opcode':[0x66, 0x68],       'modrm':None}),
    (imm32,               {'opcode':[0x68],             'modrm':None}),
    (reg16,               {'opcode':[0x66, 0x50],       'modrm':None}),
    (mem16,               {'opcode':[0x66, 0xFF],       'modrm':0x30}))
    
class pusha(Instruction):
  machine_inst = no_op
  params = {'opcode':[0x66, 0x60],       'modrm':None}
  
class pushad(Instruction):
  machine_inst = no_op
  params = {'opcode':[0x60],             'modrm':None}
  
class pushf(Instruction):
  machine_inst = no_op
  params = {'opcode':[0x66, 0x9C],       'modrm':None}
  
class pushfd(Instruction):
  machine_inst = no_op
  params = {'opcode':[0x9C],             'modrm':None}
  
class rcl(DispatchInstruction):
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
    
class rcr(DispatchInstruction):
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
    
class rdtsc(Instruction):
  machine_inst = no_op
  params = {'opcode':[0x0F, 0x31],       'modrm':None}
  
class ret(DispatchInstruction):
  dispatch = (
    (no_op,               {'opcode':[0xC3],             'modrm':None}),
    (imm16,               {'opcode':[0xC2],             'modrm':None}))
    
class rol(DispatchInstruction):
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
    
class ror(DispatchInstruction):
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
    
class sahf(Instruction):
  machine_inst = no_op
  params = {'opcode':[0x9E],             'modrm':None}
  
class sal(DispatchInstruction):
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
    
class sar(DispatchInstruction):
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
    
class sbb(DispatchInstruction):
  dispatch = (
    (reg32_simm8,         {'opcode':[0x83],             'modrm':0x18}),
    (mem32_simm8,         {'opcode':[0x83],             'modrm':0x18}),
    (eax_imm32,           {'opcode':[0x1D],             'modrm':None}),
    (reg32_imm32,         {'opcode':[0x81],             'modrm':0x18}),
    (mem32_imm32,         {'opcode':[0x81],             'modrm':0x18}),
    (reg32_reg32,         {'opcode':[0x19],             'modrm':None}),
    (mem32_reg32,         {'opcode':[0x19],             'modrm':None}),
    (reg32_mem32,         {'opcode':[0x1B],             'modrm':None}),
    (reg16_simm8,         {'opcode':[0x66, 0x83],       'modrm':0x18}),
    (mem16_simm8,         {'opcode':[0x66, 0x83],       'modrm':0x18}),
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
    
class scasb(Instruction):
  machine_inst = no_op
  params = {'opcode':[0xAE],             'modrm':None}
  
class scasd(Instruction):
  machine_inst = no_op
  params = {'opcode':[0xAF],             'modrm':None}
  
class scasw(Instruction):
  machine_inst = no_op
  params = {'opcode':[0x66, 0xAF],       'modrm':None}
  
class seta(DispatchInstruction):
  dispatch = (
    (reg8,                {'opcode':[0x0F, 0x97],       'modrm':0x00}),
    (mem8,                {'opcode':[0x0F, 0x97],       'modrm':0x00}))
  
class setae(DispatchInstruction):
  dispatch = (
    (reg8,                {'opcode':[0x0F, 0x93],       'modrm':0x00}),
    (mem8,                {'opcode':[0x0F, 0x93],       'modrm':0x00}))
  
class setb(DispatchInstruction):
  dispatch = (
    (reg8,                {'opcode':[0x0F, 0x92],       'modrm':0x00}),
    (mem8,                {'opcode':[0x0F, 0x92],       'modrm':0x00}))
  
class setbe(DispatchInstruction):
  dispatch = (
    (reg8,                {'opcode':[0x0F, 0x96],       'modrm':0x00}),
    (mem8,                {'opcode':[0x0F, 0x96],       'modrm':0x00}))
  
class setc(DispatchInstruction):
  dispatch = (
    (reg8,                {'opcode':[0x0F, 0x92],       'modrm':0x00}),
    (mem8,                {'opcode':[0x0F, 0x92],       'modrm':0x00}))
  
class sete(DispatchInstruction):
  dispatch = (
    (reg8,                {'opcode':[0x0F, 0x94],       'modrm':0x00}),
    (mem8,                {'opcode':[0x0F, 0x94],       'modrm':0x00}))
  
class setg(DispatchInstruction):
  dispatch = (
    (reg8,                {'opcode':[0x0F, 0x9F],       'modrm':0x00}),
    (mem8,                {'opcode':[0x0F, 0x9F],       'modrm':0x00}))
  
class setge(DispatchInstruction):
  dispatch = (
    (reg8,                {'opcode':[0x0F, 0x9D],       'modrm':0x00}),
    (mem8,                {'opcode':[0x0F, 0x9D],       'modrm':0x00}))
  
class setl(DispatchInstruction):
  dispatch = (
    (reg8,                {'opcode':[0x0F, 0x9C],       'modrm':0x00}),
    (mem8,                {'opcode':[0x0F, 0x9C],       'modrm':0x00}))
  
class setle(DispatchInstruction):
  dispatch = (
    (reg8,                {'opcode':[0x0F, 0x9E],       'modrm':0x00}),
    (mem8,                {'opcode':[0x0F, 0x9E],       'modrm':0x00}))
  
class setna(DispatchInstruction):
  dispatch = (
    (reg8,                {'opcode':[0x0F, 0x96],       'modrm':0x00}),
    (mem8,                {'opcode':[0x0F, 0x96],       'modrm':0x00}))
  
class setnae(DispatchInstruction):
  dispatch = (
    (reg8,                {'opcode':[0x0F, 0x92],       'modrm':0x00}),
    (mem8,                {'opcode':[0x0F, 0x92],       'modrm':0x00}))
  
class setnb(DispatchInstruction):
  dispatch = (
    (reg8,                {'opcode':[0x0F, 0x93],       'modrm':0x00}),
    (mem8,                {'opcode':[0x0F, 0x93],       'modrm':0x00}))
  
class setnbe(DispatchInstruction):
  dispatch = (
    (reg8,                {'opcode':[0x0F, 0x97],       'modrm':0x00}),
    (mem8,                {'opcode':[0x0F, 0x97],       'modrm':0x00}))
  
class setnc(DispatchInstruction):
  dispatch = (
    (reg8,                {'opcode':[0x0F, 0x93],       'modrm':0x00}),
    (mem8,                {'opcode':[0x0F, 0x93],       'modrm':0x00}))
  
class setne(DispatchInstruction):
  dispatch = (
    (reg8,                {'opcode':[0x0F, 0x95],       'modrm':0x00}),
    (mem8,                {'opcode':[0x0F, 0x95],       'modrm':0x00}))
  
class setng(DispatchInstruction):
  dispatch = (
    (reg8,                {'opcode':[0x0F, 0x9E],       'modrm':0x00}),
    (mem8,                {'opcode':[0x0F, 0x9E],       'modrm':0x00}))
  
class setnge(DispatchInstruction):
  dispatch = (
    (reg8,                {'opcode':[0x0F, 0x9C],       'modrm':0x00}),
    (mem8,                {'opcode':[0x0F, 0x9C],       'modrm':0x00}))
  
class setnl(DispatchInstruction):
  dispatch = (
    (reg8,                {'opcode':[0x0F, 0x9D],       'modrm':0x00}),
    (mem8,                {'opcode':[0x0F, 0x9D],       'modrm':0x00}))
  
class setnle(DispatchInstruction):
  dispatch = (
    (reg8,                {'opcode':[0x0F, 0x9F],       'modrm':0x00}),
    (mem8,                {'opcode':[0x0F, 0x9F],       'modrm':0x00}))
  
class setno(DispatchInstruction):
  dispatch = (
    (reg8,                {'opcode':[0x0F, 0x91],       'modrm':0x00}),
    (mem8,                {'opcode':[0x0F, 0x91],       'modrm':0x00}))
  
class setnp(DispatchInstruction):
  dispatch = (
    (reg8,                {'opcode':[0x0F, 0x9B],       'modrm':0x00}),
    (mem8,                {'opcode':[0x0F, 0x9B],       'modrm':0x00}))
  
class setns(DispatchInstruction):
  dispatch = (
    (reg8,                {'opcode':[0x0F, 0x99],       'modrm':0x00}),
    (mem8,                {'opcode':[0x0F, 0x99],       'modrm':0x00}))
  
class setnz(DispatchInstruction):
  dispatch = (
    (reg8,                {'opcode':[0x0F, 0x95],       'modrm':0x00}),
    (mem8,                {'opcode':[0x0F, 0x95],       'modrm':0x00}))
  
class seto(DispatchInstruction):
  dispatch = (
    (reg8,                {'opcode':[0x0F, 0x90],       'modrm':0x00}),
    (mem8,                {'opcode':[0x0F, 0x90],       'modrm':0x00}))
  
class setp(DispatchInstruction):
  dispatch = (
    (reg8,                {'opcode':[0x0F, 0x9A],       'modrm':0x00}),
    (mem8,                {'opcode':[0x0F, 0x9A],       'modrm':0x00}))
  
class setpe(DispatchInstruction):
  dispatch = (
    (reg8,                {'opcode':[0x0F, 0x9A],       'modrm':0x00}),
    (mem8,                {'opcode':[0x0F, 0x9A],       'modrm':0x00}))
  
class setpo(DispatchInstruction):
  dispatch = (
    (reg8,                {'opcode':[0x0F, 0x9B],       'modrm':0x00}),
    (mem8,                {'opcode':[0x0F, 0x9B],       'modrm':0x00}))
  
class sets(DispatchInstruction):
  dispatch = (
    (reg8,                {'opcode':[0x0F, 0x98],       'modrm':0x00}),
    (mem8,                {'opcode':[0x0F, 0x98],       'modrm':0x00}))
  
class setz(DispatchInstruction):
  dispatch = (
    (reg8,                {'opcode':[0x0F, 0x94],       'modrm':0x00}),
    (mem8,                {'opcode':[0x0F, 0x94],       'modrm':0x00}))
  
class sfence(Instruction):
  machine_inst = no_op
  params = {'opcode':[0x0F, 0xAE, 0xF8], 'modrm':None}
  
class shl(DispatchInstruction):
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
    
class shld(DispatchInstruction):
  dispatch = (
    (reg32_reg32_imm8,    {'opcode':[0x0F, 0xA4],       'modrm':None}),
    (mem32_reg32_imm8,    {'opcode':[0x0F, 0xA4],       'modrm':None}),
    (reg32_reg32_cl,      {'opcode':[0x0F, 0xA5],       'modrm':None}),
    (mem32_reg32_cl,      {'opcode':[0x0F, 0xA5],       'modrm':None}),
    (reg16_reg16_imm8,    {'opcode':[0x66, 0x0F, 0xA4], 'modrm':None}),
    (mem16_reg16_imm8,    {'opcode':[0x66, 0x0F, 0xA4], 'modrm':None}),
    (reg16_reg16_cl,      {'opcode':[0x66, 0x0F, 0xA5], 'modrm':None}),
    (mem16_reg16_cl,      {'opcode':[0x66, 0x0F, 0xA5], 'modrm':None}))
    
class shr(DispatchInstruction):
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
    
class shrd(DispatchInstruction):
  dispatch = (
    (reg32_reg32_imm8,    {'opcode':[0x0F, 0xAC],       'modrm':None}),
    (mem32_reg32_imm8,    {'opcode':[0x0F, 0xAC],       'modrm':None}),
    (reg32_reg32_cl,      {'opcode':[0x0F, 0xAD],       'modrm':None}),
    (mem32_reg32_cl,      {'opcode':[0x0F, 0xAD],       'modrm':None}),
    (reg16_reg16_imm8,    {'opcode':[0x66, 0x0F, 0xAC], 'modrm':None}),
    (mem16_reg16_imm8,    {'opcode':[0x66, 0x0F, 0xAC], 'modrm':None}),
    (reg16_reg16_cl,      {'opcode':[0x66, 0x0F, 0xAD], 'modrm':None}),
    (mem16_reg16_cl,      {'opcode':[0x66, 0x0F, 0xAD], 'modrm':None}))
  
class stc(Instruction):
  machine_inst = no_op
  params = {'opcode':[0xF9],             'modrm':None}
  
class std(Instruction):
  machine_inst = no_op
  params = {'opcode':[0xFD],             'modrm':None}
  
class stosb(Instruction):
  machine_inst = no_op
  params = {'opcode':[0xAA],             'modrm':None}
  
class stosd(Instruction):
  machine_inst = no_op
  params = {'opcode':[0xAB],             'modrm':None}
  
class stosw(Instruction):
  machine_inst = no_op
  params = {'opcode':[0x66, 0xAB],       'modrm':None}
  
class sub(DispatchInstruction):
  dispatch = (
    (reg32_simm8,         {'opcode':[0x83],             'modrm':0x28}),
    (mem32_simm8,         {'opcode':[0x83],             'modrm':0x28}),
    (eax_imm32,           {'opcode':[0x2D],             'modrm':None}),
    (reg32_imm32,         {'opcode':[0x81],             'modrm':0x28}),
    (mem32_imm32,         {'opcode':[0x81],             'modrm':0x28}),
    (reg32_reg32,         {'opcode':[0x29],             'modrm':None}),
    (mem32_reg32,         {'opcode':[0x29],             'modrm':None}),
    (reg32_mem32,         {'opcode':[0x2B],             'modrm':None}),
    (ax_imm16,            {'opcode':[0x66, 0x2D],       'modrm':None}),
    (reg16_simm8,         {'opcode':[0x66, 0x83],       'modrm':0x28}),
    (mem16_simm8,         {'opcode':[0x66, 0x83],       'modrm':0x28}),
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
    
class test(DispatchInstruction):
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
    
class ud2(Instruction):
  machine_inst = no_op
  params = {'opcode':[0x0F, 0x0B],       'modrm':None}
  
class xadd(DispatchInstruction):
  dispatch = (
    (reg32_reg32,         {'opcode':[0x0F, 0xC1],       'modrm':None}),
    (mem32_reg32,         {'opcode':[0x0F, 0xC1],       'modrm':None}),
    (reg16_reg16,         {'opcode':[0x66, 0x0F, 0xC1], 'modrm':None}),
    (mem16_reg16,         {'opcode':[0x66, 0x0F, 0xC1], 'modrm':None}),
    (reg8_reg8,           {'opcode':[0x0F, 0xC0],       'modrm':None}),
    (mem8_reg8,           {'opcode':[0x0F, 0xC0],       'modrm':None}))
    
class xchg(DispatchInstruction):
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
    
class xlatb(Instruction):
  machine_inst = no_op
  params = {'opcode':[0xD7],             'modrm':None}
  
class xor(DispatchInstruction):
  dispatch = (
    (reg32_simm8,         {'opcode':[0x83],             'modrm':0x30}),
    (mem32_simm8,         {'opcode':[0x83],             'modrm':0x30}),
    (eax_imm32,           {'opcode':[0x35],             'modrm':None}),
    (reg32_imm32,         {'opcode':[0x81],             'modrm':0x30}),
    (mem32_imm32,         {'opcode':[0x81],             'modrm':0x30}),
    (reg32_reg32,         {'opcode':[0x31],             'modrm':None}),
    (mem32_reg32,         {'opcode':[0x31],             'modrm':None}),
    (reg32_mem32,         {'opcode':[0x33],             'modrm':None}),
    (reg16_simm8,         {'opcode':[0x66, 0x83],       'modrm':0x30}),
    (mem16_simm8,         {'opcode':[0x66, 0x83],       'modrm':0x30}),
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
class f2xm1(Instruction):
  machine_inst = no_op
  params = {'opcode':[0xD9, 0xF0],       'modrm':None}
    
class fabs(Instruction):
  machine_inst = no_op
  params = {'opcode':[0xD9, 0xE1],       'modrm':None}
  
class fadd(DispatchInstruction):
  dispatch = (
    (st0_sti,    {'opcode':[0xD8, 0xC0],       'modrm':None}),
    (sti_st0,    {'opcode':[0xDC, 0xC0],       'modrm':None}),
    (mem32,      {'opcode':[0xD8],             'modrm':0x00}),
    (mem64,      {'opcode':[0xDC],             'modrm':0x00}))
      
class faddp(DispatchInstruction):
  dispatch = (
    (no_op,      {'opcode':[0xDE, 0xC1],       'modrm':None}),
    (sti_st0,    {'opcode':[0xDE, 0xC0],       'modrm':None}))
    
class fiadd(DispatchInstruction):
  dispatch = (
    (mem32,      {'opcode':[0xDA],             'modrm':0x00}),
    (mem16,      {'opcode':[0xDE],             'modrm':0x00}))
    
class fchs(Instruction):
  machine_inst = no_op
  params = {'opcode':[0xD9, 0xE0],       'modrm':None}
  
class fcmovb(Instruction):
  machine_inst = st0_sti
  params = {'opcode':[0xDA, 0xC0],       'modrm':None}
  
class fcmovbe(Instruction):
  machine_inst = st0_sti
  params = {'opcode':[0xDA, 0xD0],       'modrm':None}
  
class fcmove(Instruction):
  machine_inst = st0_sti
  params = {'opcode':[0xDA, 0xC8],       'modrm':None}
  
class fcmovnb(Instruction):
  machine_inst = st0_sti
  params = {'opcode':[0xDB, 0xC0],       'modrm':None}
  
class fcmovnbe(Instruction):
  machine_inst = st0_sti
  params = {'opcode':[0xDB, 0xD0],       'modrm':None}
  
class fcmovne(Instruction):
  machine_inst = st0_sti
  params = {'opcode':[0xDB, 0xC8],       'modrm':None}
  
class fcmovnu(Instruction):
  machine_inst = st0_sti
  params = {'opcode':[0xDB, 0xD8],       'modrm':None}
  
class fcmovu(Instruction):
  machine_inst = st0_sti
  params = {'opcode':[0xDA, 0xD8],       'modrm':None}
  
class fcom(DispatchInstruction):
  dispatch = (
    (no_op,      {'opcode':[0xD8, 0xD1],       'modrm':None}),
    (sti,        {'opcode':[0xD8, 0xD0],       'modrm':None}),
    (mem32,      {'opcode':[0xD8],             'modrm':0x10}),
    (mem64,      {'opcode':[0xDC],             'modrm':0x10}))
    
class fcomp(DispatchInstruction):
  dispatch = (
    (no_op,      {'opcode':[0xD8, 0xD9],       'modrm':None}),
    (sti,        {'opcode':[0xD8, 0xD8],       'modrm':None}),
    (mem32,      {'opcode':[0xD8],             'modrm':0x18}),
    (mem64,      {'opcode':[0xDC],             'modrm':0x18}))
    
class fcompp(Instruction):
  machine_inst = no_op
  params = {'opcode':[0xDE, 0xD9],       'modrm':None}
  
class fcomi(Instruction):
  machine_inst = st0_sti
  params = {'opcode':[0xDB, 0xF0],       'modrm':None}
  
class fcomip(Instruction):
  machine_inst = st0_sti
  params = {'opcode':[0xDF, 0xF0],       'modrm':None}
  
class fcos(Instruction):
  machine_inst = no_op
  params = {'opcode':[0xD9, 0xFF],       'modrm':None}
  
class fdecstp(Instruction):
  machine_inst = no_op
  params = {'opcode':[0xD9, 0xF6],       'modrm':None}
  
class fdiv(DispatchInstruction):
  dispatch = (
    (st0_sti,    {'opcode':[0xD8, 0xF0],       'modrm':None}),
    (sti_st0,    {'opcode':[0xDC, 0xF8],       'modrm':None}),
    (mem32,      {'opcode':[0xD8],             'modrm':0x30}),
    (mem64,      {'opcode':[0xDC],             'modrm':0x30}))
    
class fdivp(DispatchInstruction):
  dispatch = (
    (no_op,      {'opcode':[0xDE, 0xF9],       'modrm':None}),
    (sti_st0,    {'opcode':[0xDE, 0xF8],       'modrm':None}))
  
class fidiv(DispatchInstruction):
  dispatch = (
    (mem32,      {'opcode':[0xDA],             'modrm':0x30}),
    (mem16,      {'opcode':[0xDE],             'modrm':0x30}))
    
class fdivr(DispatchInstruction):
  dispatch = (
    (st0_sti,    {'opcode':[0xD8, 0xF8],       'modrm':None}),
    (sti_st0,    {'opcode':[0xDC, 0xF0],       'modrm':None}),
    (mem32,      {'opcode':[0xD8],             'modrm':0x38}),
    (mem64,      {'opcode':[0xDC],             'modrm':0x38}))
    
class fdivrp(DispatchInstruction):
  dispatch = (
    (no_op,      {'opcode':[0xDE, 0xF1],       'modrm':None}),
    (sti_st0,    {'opcode':[0xDE, 0xF0],       'modrm':None}))
  
class fidivr(DispatchInstruction):
  dispatch = (
    (mem32,      {'opcode':[0xDA],             'modrm':0x38}),
    (mem16,      {'opcode':[0xDE],             'modrm':0x38}))
  
class ffree(Instruction):
  machine_inst = sti
  params = {'opcode':[0xDD, 0xC0],       'modrm':None}
  
class ficom(DispatchInstruction):
  dispatch = (
    (mem32,      {'opcode':[0xDA],             'modrm':0x10}),
    (mem16,      {'opcode':[0xDE],             'modrm':0x10}))
    
class ficomp(DispatchInstruction):
  dispatch = (
    (mem32,      {'opcode':[0xDA],             'modrm':0x18}),
    (mem16,      {'opcode':[0xDE],             'modrm':0x18}))
    
class fild(DispatchInstruction):
  dispatch = (
    (mem64,      {'opcode':[0xDF],             'modrm':0x28}),
    (mem32,      {'opcode':[0xDB],             'modrm':0x00}),
    (mem16,      {'opcode':[0xDF],             'modrm':0x00}))
    
class fincstp(Instruction):
  machine_inst = no_op
  params = {'opcode':[0xD9, 0xF7],       'modrm':None}
  
class finit(Instruction):
  machine_inst = no_op
  params = {'opcode':[0x9B, 0xDB, 0xE3], 'modrm':None}
  
class fninit(Instruction):
  machine_inst = no_op
  params = {'opcode':[0xDB, 0xE3],       'modrm':None}
  
class fist(DispatchInstruction):
  dispatch = (
    (mem32,      {'opcode':[0xDB],             'modrm':0x10}),
    (mem16,      {'opcode':[0xDF],             'modrm':0x10}))
    
class fistp(DispatchInstruction):
  dispatch = (
    (mem64,      {'opcode':[0xDF],             'modrm':0x38}),
    (mem32,      {'opcode':[0xDB],             'modrm':0x18}),
    (mem16,      {'opcode':[0xDF],             'modrm':0x18}))
    
class fisttp(DispatchInstruction):
  dispatch = (
    (mem64,      {'opcode':[0xDD],             'modrm':0x08}),
    (mem32,      {'opcode':[0xDB],             'modrm':0x08}),
    (mem16,      {'opcode':[0xDF],             'modrm':0x08}))
  
class fld(DispatchInstruction):
  dispatch = (
    (sti,        {'opcode':[0xD9, 0xC0],       'modrm':None}),
    (mem80,      {'opcode':[0xDB],             'modrm':0x28}),
    (mem64,      {'opcode':[0xDD],             'modrm':0x00}),
    (mem32,      {'opcode':[0xD9],             'modrm':0x00}))
    
class fld1(Instruction):
  machine_inst = no_op
  params = {'opcode':[0xD9, 0xE8],       'modrm':None}
  
class fldcw(Instruction):
  machine_inst = mem16
  params = {'opcode':[0xD9],             'modrm':0x28}
  
class fldenv(Instruction):
  machine_inst = mem228
  params = {'opcode':[0xD9],             'modrm':0x20}
  
class fldl2e(Instruction):
  machine_inst = no_op
  params = {'opcode':[0xD9, 0xEA],       'modrm':None}
  
class fldl2t(Instruction):
  machine_inst = no_op
  params = {'opcode':[0xD9, 0xE9],       'modrm':None}
  
class fldlg2(Instruction):
  machine_inst = no_op
  params = {'opcode':[0xD9, 0xEC],       'modrm':None}
  
class fldln2(Instruction):
  machine_inst = no_op
  params = {'opcode':[0xD9, 0xED],       'modrm':None}
  
class fldpi(Instruction):
  machine_inst = no_op
  params = {'opcode':[0xD9, 0xEB],       'modrm':None}
  
class fldz(Instruction):
  machine_inst = no_op
  params = {'opcode':[0xD9, 0xEE],       'modrm':None}
  
class fmul(DispatchInstruction):
  dispatch = (
    (st0_sti,    {'opcode':[0xD8, 0xC8],       'modrm':None}),
    (sti_st0,    {'opcode':[0xDC, 0xC8],       'modrm':None}),
    (mem32,      {'opcode':[0xD8],             'modrm':0x08}),
    (mem64,      {'opcode':[0xDC],             'modrm':0x08}))
    
class fmulp(DispatchInstruction):
  dispatch = (
    (no_op,      {'opcode':[0xDE, 0xC9],       'modrm':None}),
    (sti_st0,    {'opcode':[0xDE, 0xC8],       'modrm':None}))
    
class fimul(DispatchInstruction):
  dispatch = (
    (mem32,      {'opcode':[0xDA],             'modrm':0x08}),
    (mem16,      {'opcode':[0xDE],             'modrm':0x08}))
    
class fnop(Instruction):
  machine_inst = no_op
  params = {'opcode':[0xD9, 0xD0],       'modrm':None}
  
class fpatan(Instruction):
  machine_inst = no_op
  params = {'opcode':[0xD9, 0xF3],       'modrm':None}
  
class fprem(Instruction):
  machine_inst = no_op
  params = {'opcode':[0xD9, 0xF8],       'modrm':None}
  
class fprem1(Instruction):
  machine_inst = no_op
  params = {'opcode':[0xD9, 0xF5],       'modrm':None}
  
class fptan(Instruction):
  machine_inst = no_op
  params = {'opcode':[0xD9, 0xF2],       'modrm':None}
  
class frndint(Instruction):
  machine_inst = no_op
  params = {'opcode':[0xD9, 0xFC],       'modrm':None}
  
class frstor(Instruction):
  machine_inst = mem752
  params = {'opcode':[0xDD],             'modrm':0x20}
  
class fsave(Instruction):
  machine_inst = mem752
  params = {'opcode':[0x9B, 0xDD],       'modrm':0x30}
  
class fnsave(Instruction):
  machine_inst = mem752
  params = {'opcode':[0xDD],             'modrm':0x30}
  
class fscale(Instruction):
  machine_inst = no_op
  params = {'opcode':[0xD9, 0xFD],       'modrm':None}
  
class fsin(Instruction):
  machine_inst = no_op
  params = {'opcode':[0xD9, 0xFE],       'modrm':None}
  
class fsincos(Instruction):
  machine_inst = no_op
  params = {'opcode':[0xD9, 0xFB],       'modrm':None}
  
class fsqrt(Instruction):
  machine_inst = no_op
  params = {'opcode':[0xD9, 0xFA],       'modrm':None}
  
class fst(DispatchInstruction):
  dispatch = (
    (sti,        {'opcode':[0xDD, 0xD0],       'modrm':None}),
    (mem64,      {'opcode':[0xDD],             'modrm':0x10}),
    (mem32,      {'opcode':[0xD9],             'modrm':0x10}))
    
class fstp(DispatchInstruction):
  dispatch = (
    (sti,        {'opcode':[0xDD, 0xD8],       'modrm':None}),
    (mem80,      {'opcode':[0xDB],             'modrm':0x38}),
    (mem64,      {'opcode':[0xDD],             'modrm':0x18}),
    (mem32,      {'opcode':[0xD9],             'modrm':0x18}))
    
class fstcw(Instruction):
  machine_inst = mem16
  params = {'opcode':[0x9B, 0xD9],       'modrm':0x38}
  
class fnstcw(Instruction):
  machine_inst = mem16
  params = {'opcode':[0xD9],             'modrm':0x38}
  
class fstenv(Instruction):
  machine_inst = mem228
  params = {'opcode':[0x9B, 0xD9],       'modrm':0x30}
  
class fnstenv(Instruction):
  machine_inst = mem228
  params = {'opcode':[0xD9],             'modrm':0x30}
  
class fstsw(DispatchInstruction):
  dispatch = (
    (ax,         {'opcode':[0x9B, 0xDF, 0xE0], 'modrm':None}),
    (mem16,      {'opcode':[0x9B, 0xDD],       'modrm':0x38}))
    
class fnstsw(DispatchInstruction):
  dispatch = (
    (ax,         {'opcode':[0xDF, 0xE0],       'modrm':None}),
    (mem16,      {'opcode':[0xDD],             'modrm':0x38}))
    
class fsub(DispatchInstruction):
  dispatch = (
    (st0_sti,    {'opcode':[0xD8, 0xE0],       'modrm':None}),
    (sti_st0,    {'opcode':[0xDC, 0xE8],       'modrm':None}),
    (mem32,      {'opcode':[0xD8],             'modrm':0x20}),
    (mem64,      {'opcode':[0xDC],             'modrm':0x20}))
    
class fsubp(DispatchInstruction):
  dispatch = (
    (no_op,      {'opcode':[0xDE, 0xE9],       'modrm':None}),
    (sti_st0,    {'opcode':[0xDE, 0xE8],       'modrm':None}))
  
class fisub(DispatchInstruction):
  dispatch = (
    (mem32,      {'opcode':[0xDA],             'modrm':0x20}),
    (mem16,      {'opcode':[0xDE],             'modrm':0x20}))
  
class fsubr(DispatchInstruction):
  dispatch = (
    (st0_sti,    {'opcode':[0xD8, 0xE8],       'modrm':None}),
    (sti_st0,    {'opcode':[0xDC, 0xE0],       'modrm':None}),
    (mem32,      {'opcode':[0xD8],             'modrm':0x28}),
    (mem64,      {'opcode':[0xDC],             'modrm':0x28}))
    
class fsubrp(DispatchInstruction):
  dispatch = (
    (no_op,      {'opcode':[0xDE, 0xE1],       'modrm':None}),
    (sti_st0,    {'opcode':[0xDE, 0xE0],       'modrm':None}))
    
class fisubr(DispatchInstruction):
  dispatch = (
    (mem32,      {'opcode':[0xDA],             'modrm':0x28}),
    (mem16,      {'opcode':[0xDE],             'modrm':0x28}))
  
class ftst(Instruction):
  machine_inst = no_op
  params = {'opcode':[0xD9, 0xE4],       'modrm':None}
  
class fucom(DispatchInstruction):
  dispatch = (
    (no_op,      {'opcode':[0xDD, 0xE1],       'modrm':None}),
    (sti,        {'opcode':[0xDD, 0xE0],       'modrm':None}))
  
class fucomp(DispatchInstruction):
  dispatch = (
    (no_op,      {'opcode':[0xDD, 0xE9],       'modrm':None}),
    (sti,        {'opcode':[0xDD, 0xE8],       'modrm':None}))
  
class fucompp(Instruction):
  machine_inst = no_op
  params = {'opcode':[0xDA, 0xE9],       'modrm':None}
  
class fucomi(Instruction):
  machine_inst = st0_sti
  params = {'opcode':[0xDB, 0xE8],       'modrm':None}
  
class fucomip(Instruction):
  machine_inst = st0_sti
  params = {'opcode':[0xDF, 0xE8],       'modrm':None}
  
class fwait(Instruction):
  machine_inst = no_op
  params = {'opcode':[0x9B],             'modrm':None}
  
class fxam(Instruction):
  machine_inst = no_op
  params = {'opcode':[0xD9, 0xE5],       'modrm':None}
  
class fxch(DispatchInstruction):
  dispatch = (
    (no_op,      {'opcode':[0xD9, 0xC9],       'modrm':None}),
    (sti,        {'opcode':[0xD9, 0xC8],       'modrm':None}))
    
class fxrstor(Instruction):
  machine_inst = mem4096
  params = {'opcode':[0x0F, 0xAE],       'modrm':0x08}
 #sse?

class fxsave(Instruction):
  machine_inst = mem4096
  params = {'opcode':[0x0F, 0xAE],       'modrm':0x00}
 #sse?

class fxtract(Instruction):
  machine_inst = no_op
  params = {'opcode':[0xD9, 0xF4],       'modrm':None}
  
class fyl2x(Instruction):
  machine_inst = no_op
  params = {'opcode':[0xD9, 0xF1],       'modrm':None}
  
class fyl2xp1(Instruction):
  machine_inst = no_op
  params = {'opcode':[0xD9, 0xF9],       'modrm':None}
  

#SSE_ISA = (
class addpd(DispatchInstruction):
  dispatch = (
    (xmm_xmm,      {'opcode':[0x66, 0x0F, 0x58], 'modrm':None}),
    (xmm_mem128,   {'opcode':[0x66, 0x0F, 0x58], 'modrm':None}))
  arch_ext = 2

class addps(DispatchInstruction):
  dispatch = (
    (xmm_xmm,      {'opcode':[0x0F, 0x58], 'modrm':None}),
    (xmm_mem128,   {'opcode':[0x0F, 0x58], 'modrm':None}))
  arch_ext = 1

class addsd(DispatchInstruction):
  dispatch = (
    (xmm_xmm,      {'opcode':[0xF2, 0x0F, 0x58], 'modrm':None}),
    (xmm_mem64,   {'opcode':[0xF2, 0x0F, 0x58], 'modrm':None}))
  arch_ext = 2

class addss(DispatchInstruction):
  dispatch = (
    (xmm_xmm,      {'opcode':[0xF3, 0x0F, 0x58], 'modrm':None}),
    (xmm_mem128,   {'opcode':[0xF3, 0x0F, 0x58], 'modrm':None}))
  arch_ext = 1

class addsubpd(DispatchInstruction):
  dispatch = (
    (xmm_xmm,      {'opcode':[0x66, 0x0F, 0xD0], 'modrm':None}),
    (xmm_mem128,   {'opcode':[0x66, 0x0F, 0xD0], 'modrm':None}))
  arch_ext = 3

class addsubps(DispatchInstruction):
  dispatch = (
    (xmm_xmm,      {'opcode':[0xF2, 0x0F, 0xD0], 'modrm':None}),
    (xmm_mem128,   {'opcode':[0xF2, 0x0F, 0xD0], 'modrm':None}))
  arch_ext = 3

class andnpd(DispatchInstruction):
  dispatch = (
    (xmm_xmm,      {'opcode':[0x66, 0x0F, 0x55], 'modrm':None}),
    (xmm_mem128,   {'opcode':[0x66, 0x0F, 0x55], 'modrm':None}))
  arch_ext = 2

class andnps(DispatchInstruction):
  dispatch = (
    (xmm_xmm,      {'opcode':[0x0F, 0x55], 'modrm':None}),
    (xmm_mem128,   {'opcode':[0x0F, 0x55], 'modrm':None}))
  arch_ext = 1

class andpd(DispatchInstruction):
  dispatch = (
    (xmm_xmm,      {'opcode':[0x66, 0x0F, 0x54], 'modrm':None}),
    (xmm_mem128,   {'opcode':[0x66, 0x0F, 0x54], 'modrm':None}))
  arch_ext = 2

class andps(DispatchInstruction):
  dispatch = (
    (xmm_xmm,      {'opcode':[0x0F, 0x54], 'modrm':None}),
    (xmm_mem128,   {'opcode':[0x0F, 0x54], 'modrm':None}))
  arch_ext = 1

class blendpd(DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm8,    {'opcode':[0x66, 0x0F, 0x3A, 0x0D], 'modrm':None}),
    (xmm_mem128_imm8, {'opcode':[0x66, 0x0F, 0x3A, 0x0D], 'modrm':None}))
  arch_ext = 4

class blendps(DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm8,    {'opcode':[0x66, 0x0F, 0x3A, 0x0C], 'modrm':None}),
    (xmm_mem128_imm8, {'opcode':[0x66, 0x0F, 0x3A, 0x0C], 'modrm':None}))
  arch_ext = 4

class blendvpd(DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x15], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x15], 'modrm':None}))
  arch_ext = 4

class blendvps(DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x14], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x14], 'modrm':None}))
  arch_ext = 4

class cmpeqpd(DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0x66, 0x0F, 0xC2], 'modrm':None, 'imm':0}),
    (xmm_mem128_imm, {'opcode':[0x66, 0x0F, 0xC2], 'modrm':None, 'imm':0}))
  arch_ext = 2

class cmpeqps(DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0x0F, 0xC2], 'modrm':None, 'imm':0}),
    (xmm_mem128_imm, {'opcode':[0x0F, 0xC2], 'modrm':None, 'imm':0}))
  arch_ext = 1

class cmpeqsd(DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0xF2, 0x0F, 0xC2], 'modrm':None, 'imm':0}),
    (xmm_mem64_imm,  {'opcode':[0xF2, 0x0F, 0xC2], 'modrm':None, 'imm':0}))
  arch_ext = 2

class cmpeqss(DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0xF3, 0x0F, 0xC2], 'modrm':None, 'imm':0}),
    (xmm_mem32_imm,  {'opcode':[0xF3, 0x0F, 0xC2], 'modrm':None, 'imm':0}))
  arch_ext = 1

class cmplepd(DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0x66, 0x0F, 0xC2], 'modrm':None, 'imm':2}),
    (xmm_mem128_imm, {'opcode':[0x66, 0x0F, 0xC2], 'modrm':None, 'imm':2}))
  arch_ext = 2

class cmpleps(DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0x0F, 0xC2], 'modrm':None, 'imm':2}),
    (xmm_mem128_imm, {'opcode':[0x0F, 0xC2], 'modrm':None, 'imm':2}))
  arch_ext = 1

class cmplesd(DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0xF2, 0x0F, 0xC2], 'modrm':None, 'imm':2}),
    (xmm_mem64_imm,  {'opcode':[0xF2, 0x0F, 0xC2], 'modrm':None, 'imm':2}))
  arch_ext = 2

class cmpless(DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0xF3, 0x0F, 0xC2], 'modrm':None, 'imm':2}),
    (xmm_mem32_imm,  {'opcode':[0xF3, 0x0F, 0xC2], 'modrm':None, 'imm':2}))
  arch_ext = 1

class cmpltpd(DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0x66, 0x0F, 0xC2], 'modrm':None, 'imm':1}),
    (xmm_mem128_imm, {'opcode':[0x66, 0x0F, 0xC2], 'modrm':None, 'imm':1}))
  arch_ext = 2

class cmpltps(DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0x0F, 0xC2], 'modrm':None, 'imm':1}),
    (xmm_mem128_imm, {'opcode':[0x0F, 0xC2], 'modrm':None, 'imm':1}))
  arch_ext = 1

class cmpltsd(DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0xF2, 0x0F, 0xC2], 'modrm':None, 'imm':1}),
    (xmm_mem64_imm,  {'opcode':[0xF2, 0x0F, 0xC2], 'modrm':None, 'imm':1}))
  arch_ext = 2

class cmpltss(DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0xF3, 0x0F, 0xC2], 'modrm':None, 'imm':1}),
    (xmm_mem32_imm,  {'opcode':[0xF3, 0x0F, 0xC2], 'modrm':None, 'imm':1}))
  arch_ext = 1

class cmpneqpd(DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0x66, 0x0F, 0xC2], 'modrm':None, 'imm':4}),
    (xmm_mem128_imm, {'opcode':[0x66, 0x0F, 0xC2], 'modrm':None, 'imm':4}))
  arch_ext = 2

class cmpneqps(DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0x0F, 0xC2], 'modrm':None, 'imm':4}),
    (xmm_mem128_imm, {'opcode':[0x0F, 0xC2], 'modrm':None, 'imm':4}))
  arch_ext = 1

class cmpneqsd(DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0xF2, 0x0F, 0xC2], 'modrm':None, 'imm':4}),
    (xmm_mem64_imm,  {'opcode':[0xF2, 0x0F, 0xC2], 'modrm':None, 'imm':4}))
  arch_ext = 2

class cmpneqss(DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0xF3, 0x0F, 0xC2], 'modrm':None, 'imm':4}),
    (xmm_mem32_imm,  {'opcode':[0xF3, 0x0F, 0xC2], 'modrm':None, 'imm':4}))
  arch_ext = 1

class cmpnlepd(DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0x66, 0x0F, 0xC2], 'modrm':None, 'imm':6}),
    (xmm_mem128_imm, {'opcode':[0x66, 0x0F, 0xC2], 'modrm':None, 'imm':6}))
  arch_ext = 2

class cmpnleps(DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0x0F, 0xC2], 'modrm':None, 'imm':6}),
    (xmm_mem128_imm, {'opcode':[0x0F, 0xC2], 'modrm':None, 'imm':6}))
  arch_ext = 1

class cmpnlesd(DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0xF2, 0x0F, 0xC2], 'modrm':None, 'imm':6}),
    (xmm_mem64_imm,  {'opcode':[0xF2, 0x0F, 0xC2], 'modrm':None, 'imm':6}))
  arch_ext = 2

class cmpnless(DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0xF3, 0x0F, 0xC2], 'modrm':None, 'imm':6}),
    (xmm_mem32_imm,  {'opcode':[0xF3, 0x0F, 0xC2], 'modrm':None, 'imm':6}))
  arch_ext = 1

class cmpnltpd(DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0x66, 0x0F, 0xC2], 'modrm':None, 'imm':5}),
    (xmm_mem128_imm, {'opcode':[0x66, 0x0F, 0xC2], 'modrm':None, 'imm':5}))
  arch_ext = 2

class cmpnltps(DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0x0F, 0xC2], 'modrm':None, 'imm':5}),
    (xmm_mem128_imm, {'opcode':[0x0F, 0xC2], 'modrm':None, 'imm':5}))
  arch_ext = 1

class cmpnltsd(DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0xF2, 0x0F, 0xC2], 'modrm':None, 'imm':5}),
    (xmm_mem64_imm,  {'opcode':[0xF2, 0x0F, 0xC2], 'modrm':None, 'imm':5}))
  arch_ext = 2

class cmpnltss(DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0xF3, 0x0F, 0xC2], 'modrm':None, 'imm':5}),
    (xmm_mem32_imm,  {'opcode':[0xF3, 0x0F, 0xC2], 'modrm':None, 'imm':5}))
  arch_ext = 1

class cmpordpd(DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0x66, 0x0F, 0xC2], 'modrm':None, 'imm':7}),
    (xmm_mem128_imm, {'opcode':[0x66, 0x0F, 0xC2], 'modrm':None, 'imm':7}))
  arch_ext = 2

class cmpordps(DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0x0F, 0xC2], 'modrm':None, 'imm':7}),
    (xmm_mem128_imm, {'opcode':[0x0F, 0xC2], 'modrm':None, 'imm':7}))
  arch_ext = 1

class cmpordsd(DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0xF2, 0x0F, 0xC2], 'modrm':None, 'imm':7}),
    (xmm_mem64_imm,  {'opcode':[0xF2, 0x0F, 0xC2], 'modrm':None, 'imm':7}))
  arch_ext = 2

class cmpordss(DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0xF3, 0x0F, 0xC2], 'modrm':None, 'imm':7}),
    (xmm_mem32_imm,  {'opcode':[0xF3, 0x0F, 0xC2], 'modrm':None, 'imm':7}))
  arch_ext = 1

class cmpunordpd(DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0x66, 0x0F, 0xC2], 'modrm':None, 'imm':3}),
    (xmm_mem128_imm, {'opcode':[0x66, 0x0F, 0xC2], 'modrm':None, 'imm':3}))
  arch_ext = 2

class cmpunordps(DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0x0F, 0xC2], 'modrm':None, 'imm':3}),
    (xmm_mem128_imm, {'opcode':[0x0F, 0xC2], 'modrm':None, 'imm':3}))
  arch_ext = 1

class cmpunordsd(DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0xF2, 0x0F, 0xC2], 'modrm':None, 'imm':3}),
    (xmm_mem64_imm,  {'opcode':[0xF2, 0x0F, 0xC2], 'modrm':None, 'imm':3}))
  arch_ext = 2

class cmpunordss(DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm,    {'opcode':[0xF3, 0x0F, 0xC2], 'modrm':None, 'imm':3}),
    (xmm_mem32_imm,  {'opcode':[0xF3, 0x0F, 0xC2], 'modrm':None, 'imm':3}))
  arch_ext = 1

class cmppd(DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm8,    {'opcode':[0x66, 0x0F, 0xC2], 'modrm':None}),
    (xmm_mem128_imm8, {'opcode':[0x66, 0x0F, 0xC2], 'modrm':None}))
  arch_ext = 2

class cmpps(DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm8,    {'opcode':[0x0F, 0xC2], 'modrm':None}),
    (xmm_mem128_imm8, {'opcode':[0x0F, 0xC2], 'modrm':None}))
  arch_ext = 1

class cmpsd(DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm8,    {'opcode':[0xF2, 0x0F, 0xC2], 'modrm':None}),
    (xmm_mem64_imm8,  {'opcode':[0xF2, 0x0F, 0xC2], 'modrm':None}))
  arch_ext = 2

class cmpss(DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm8,    {'opcode':[0xF3, 0x0F, 0xC2], 'modrm':None}),
    (xmm_mem32_imm8,  {'opcode':[0xF3, 0x0F, 0xC2], 'modrm':None}))
  arch_ext = 1

class comisd(DispatchInstruction):
  dispatch = (
    (xmm_xmm,      {'opcode':[0x66, 0x0F, 0x2F], 'modrm':None}),
    (xmm_mem64,    {'opcode':[0x66, 0x0F, 0x2F], 'modrm':None}))
  arch_ext = 2

class comiss(DispatchInstruction):
  dispatch = (
    (xmm_xmm,      {'opcode':[0x0F, 0x2F], 'modrm':None}),
    (xmm_mem64,    {'opcode':[0x0F, 0x2F], 'modrm':None}))
  arch_ext = 1

class cvtdq2pd(DispatchInstruction):
  dispatch = (
    (xmm_xmm,      {'opcode':[0xF3, 0x0F, 0xE6], 'modrm':None}),
    (xmm_mem64,    {'opcode':[0xF3, 0x0F, 0xE6], 'modrm':None}))
  arch_ext = 2

class cvtdq2ps(DispatchInstruction):
  dispatch = (
    (xmm_xmm,      {'opcode':[0x0F, 0x5B], 'modrm':None}),
    (xmm_mem64,    {'opcode':[0x0F, 0x5B], 'modrm':None}))
  arch_ext = 2

class cvtpd2dq(DispatchInstruction):
  dispatch = (
    (xmm_xmm,      {'opcode':[0xF2, 0x0F, 0xE6], 'modrm':None}),
    (xmm_mem128,   {'opcode':[0xF2, 0x0F, 0xE6], 'modrm':None}))
  arch_ext = 2

class cvtpd2pi(DispatchInstruction):
  dispatch = (
    (mmx_xmm,      {'opcode':[0x66, 0x0F, 0x2D], 'modrm':None}),
    (mmx_mem128,   {'opcode':[0x66, 0x0F, 0x2D], 'modrm':None}))
  arch_ext = 2

class cvtpd2ps(DispatchInstruction):
  dispatch = (
    (xmm_xmm,      {'opcode':[0x66, 0x0F, 0x5A], 'modrm':None}),
    (xmm_mem128,   {'opcode':[0x66, 0x0F, 0x5A], 'modrm':None}))
  arch_ext = 2

class cvtpi2pd(DispatchInstruction):
  dispatch = (
    (xmm_mmx,      {'opcode':[0x66, 0x0F, 0x2A], 'modrm':None}),
    (xmm_mem64,    {'opcode':[0x66, 0x0F, 0x2A], 'modrm':None}))
  arch_ext = 2

class cvtpi2ps(DispatchInstruction):
  dispatch = (
    (xmm_mmx,      {'opcode':[0x0F, 0x2A], 'modrm':None}),
    (xmm_mem64,    {'opcode':[0x0F, 0x2A], 'modrm':None}))
  arch_ext = 2

class cvtps2dq(DispatchInstruction):
  dispatch = (
    (xmm_xmm,      {'opcode':[0x66, 0x0F, 0x5B], 'modrm':None}),
    (xmm_mem128,   {'opcode':[0x66, 0x0F, 0x5B], 'modrm':None}))
  arch_ext = 2

class cvtps2pd(DispatchInstruction):
  dispatch = (
    (xmm_xmm,      {'opcode':[0x0F, 0x5A], 'modrm':None}),
    (xmm_mem64,    {'opcode':[0x0F, 0x5A], 'modrm':None}))
  arch_ext = 2

class cvtps2pi(DispatchInstruction):
  dispatch = (
    (mmx_xmm,      {'opcode':[0x0F, 0x2D], 'modrm':None}),
    (mmx_mem64,    {'opcode':[0x0F, 0x2D], 'modrm':None}))
  arch_ext = 2

#class cvtsd2si(Instruction):
#  machine_inst = reg32
#  params = {'opcode':[0xF2, 0x0F, 0x2D],'modrm':None}
#  arch_ext = 2

class cvtsd2si(DispatchInstruction):
  dispatch = (
    (reg32_xmm,      {'opcode':[0xF2, 0x0F, 0x2D], 'modrm':None}),
    (reg32_mem64,    {'opcode':[0xF2, 0x0F, 0x2D], 'modrm':None}))
  arch_ext = 2

class cvtsd2ss(DispatchInstruction):
  dispatch = (
    (xmm_xmm,      {'opcode':[0xF2, 0x0F, 0x5A], 'modrm':None}),
    (xmm_mem64,    {'opcode':[0xF2, 0x0F, 0x5A], 'modrm':None}))
  arch_ext = 2

class cvtsi2sd(DispatchInstruction):
  dispatch = (
    (xmm_reg32,      {'opcode':[0xF2, 0x0F, 0x2A], 'modrm':None}),
    (xmm_mem32,      {'opcode':[0xF2, 0x0F, 0x2A], 'modrm':None}))
  arch_ext = 2

class cvtsi2ss(DispatchInstruction):
  dispatch = (
    (xmm_reg32,      {'opcode':[0xF3, 0x0F, 0x2A], 'modrm':None}),
    (xmm_mem32,      {'opcode':[0xF3, 0x0F, 0x2A], 'modrm':None}))
  arch_ext = 1

class cvtss2sd(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0xF3, 0x0F, 0x5A], 'modrm':None}),
    (xmm_mem32,      {'opcode':[0xF3, 0x0F, 0x5A], 'modrm':None}))
  arch_ext = 2

class cvtss2si(DispatchInstruction):
  dispatch = (
    (reg32_xmm,        {'opcode':[0xF3, 0x0F, 0x2D], 'modrm':None}),
    (reg32_mem32,      {'opcode':[0xF3, 0x0F, 0x2D], 'modrm':None}))
  arch_ext = 1

class cvttpd2dq(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xE6], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xE6], 'modrm':None}))
  arch_ext = 2

class cvttpd2pi(DispatchInstruction):
  dispatch = (
    (mmx_xmm,        {'opcode':[0x66, 0x0F, 0x2C], 'modrm':None}),
    (mmx_mem128,     {'opcode':[0x66, 0x0F, 0x2C], 'modrm':None}))
  arch_ext = 2

class cvttps2dq(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0xF3, 0x0F, 0x5B], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0xF3, 0x0F, 0x5B], 'modrm':None}))
  arch_ext = 2

class cvttps2pi(DispatchInstruction):
  dispatch = (
    (mmx_xmm,        {'opcode':[0x0F, 0x2C], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0x2C], 'modrm':None}))
  arch_ext = 2

class cvttsd2si(DispatchInstruction):
  dispatch = (
    (reg32_xmm,        {'opcode':[0xF2, 0x0F, 0x2C], 'modrm':None}),
    (reg32_mem64,      {'opcode':[0xF2, 0x0F, 0x2C], 'modrm':None}))
  arch_ext = 2

class cvttss2si(DispatchInstruction):
  dispatch = (
    (reg32_xmm,        {'opcode':[0xF3, 0x0F, 0x2C], 'modrm':None}),
    (reg32_mem32,      {'opcode':[0xF3, 0x0F, 0x2C], 'modrm':None}))
  arch_ext = 1

class divpd(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0x5E], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0x5E], 'modrm':None}))
  arch_ext = 2

class divps(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x0F, 0x5E], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x0F, 0x5E], 'modrm':None}))
  arch_ext = 1

class divsd(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0xF2, 0x0F, 0x5E], 'modrm':None}),
    (xmm_mem64,      {'opcode':[0xF2, 0x0F, 0x5E], 'modrm':None}))
  arch_ext = 2

class divss(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0xF3, 0x0F, 0x5E], 'modrm':None}),
    (xmm_mem64,      {'opcode':[0xF3, 0x0F, 0x5E], 'modrm':None}))
  arch_ext = 1

class dppd(DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm8,    {'opcode':[0x66, 0x0F, 0x3A, 0x41], 'modrm':None}),
    (xmm_mem128_imm8, {'opcode':[0x66, 0x0F, 0x3A, 0x41], 'modrm':None}))
  arch_ext = 4

class dpps(DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm8,    {'opcode':[0x66, 0x0F, 0x3A, 0x40], 'modrm':None}),
    (xmm_mem128_imm8, {'opcode':[0x66, 0x0F, 0x3A, 0x40], 'modrm':None}))
  arch_ext = 4

class emms(Instruction):
  machine_inst = no_op
  params = {'opcode':[0x0F, 0x77],'modrm':None}
  arch_ext = 0

class extractps(DispatchInstruction):
  dispatch = (
    (reg32_xmm_imm8_rev, {'opcode':[0x66, 0x0F, 0x3A, 0x17], 'modrm':None}),
    (mem32_xmm_imm8,     {'opcode':[0x66, 0x0F, 0x3A, 0x17], 'modrm':None}))
 # TODO - ugh, this make the printer not emit 'dword' for the mem32 case
 #arch_ext = 4

class extrq(DispatchInstruction):
  dispatch = (
    (xmm_imm8_imm8, {'opcode':[0x66, 0x0F, 0x78], 'modrm':0x00}),
    (xmm_xmm,       {'opcode':[0x66, 0x0F, 0x79], 'modrm':None}))
  arch_ext = 4

class haddpd(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0x7C], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0x7C], 'modrm':None}))
  arch_ext = 3

class haddps(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0xF2, 0x0F, 0x7C], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0xF2, 0x0F, 0x7C], 'modrm':None}))
  arch_ext = 3

class hsubpd(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0x7D], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0x7D], 'modrm':None}))
  arch_ext = 3

class hsubps(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0xF2, 0x0F, 0x7D], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0xF2, 0x0F, 0x7D], 'modrm':None}))
  arch_ext = 3

class insertps(DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm8,    {'opcode':[0x66, 0x0F, 0x3A, 0x21], 'modrm':None}),
    (xmm_mem32_imm8,  {'opcode':[0x66, 0x0F, 0x3A, 0x21], 'modrm':None}))
  arch_ext = 4

class insertq(DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm8_imm8, {'opcode':[0xF2, 0x0F, 0x78], 'modrm':None}),
    (xmm_xmm,           {'opcode':[0xF2, 0x0F, 0x79], 'modrm':None}))
  arch_ext = 4

class lddqu(Instruction):
  machine_inst = xmm_mem128
  params = {'opcode':[0xF2, 0x0F, 0xF0],'modrm':None}
  arch_ext = 3

class ldmxcsr(Instruction):
  machine_inst = mem32
  params = {'opcode':[0x0F, 0xAE],'modrm':0x10}
  arch_ext = 1

class maskmovdqu(Instruction):
  machine_inst = xmm_xmm
  params = {'opcode':[0x66, 0x0F, 0xF7],'modrm':None}
  arch_ext = 2

class maskmovq(Instruction):
  machine_inst = mmx_mmx
  params = {'opcode':[0x0F, 0xF7],'modrm':None}
  arch_ext = 1

class maxpd(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0x5F], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0x5F], 'modrm':None}))
  arch_ext = 2

class maxps(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x0F, 0x5F], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x0F, 0x5F], 'modrm':None}))
  arch_ext = 1

class maxsd(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0xF2, 0x0F, 0x5F], 'modrm':None}),
    (xmm_mem64,      {'opcode':[0xF2, 0x0F, 0x5F], 'modrm':None}))
  arch_ext = 2

class maxss(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0xF3, 0x0F, 0x5F], 'modrm':None}),
    (xmm_mem32,      {'opcode':[0xF3, 0x0F, 0x5F], 'modrm':None}))
  arch_ext = 1

class minpd(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0x5D], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0x5D], 'modrm':None}))
  arch_ext = 2

class minps(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x0F, 0x5D], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x0F, 0x5D], 'modrm':None}))
  arch_ext = 1

class minsd(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0xF2, 0x0F, 0x5D], 'modrm':None}),
    (xmm_mem64,      {'opcode':[0xF2, 0x0F, 0x5D], 'modrm':None}))
  arch_ext = 2

class minss(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0xF3, 0x0F, 0x5D], 'modrm':None}),
    (xmm_mem32,      {'opcode':[0xF3, 0x0F, 0x5D], 'modrm':None}))
  arch_ext = 1

class movapd(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0x28], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0x28], 'modrm':None}),
    (mem128_xmm,     {'opcode':[0x66, 0x0F, 0x29], 'modrm':None}))
  arch_ext = 2

class movaps(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x0F, 0x28], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x0F, 0x28], 'modrm':None}),
    (mem128_xmm,     {'opcode':[0x0F, 0x29], 'modrm':None}))
  arch_ext = 1

class movd(DispatchInstruction):
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

class movddup(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0xF2, 0x0F, 0x12], 'modrm':None}),
    (xmm_mem64,      {'opcode':[0xF2, 0x0F, 0x12], 'modrm':None}))
  arch_ext = 3

class movdq2q(Instruction):
  machine_inst = mmx_xmm
  params = {'opcode':[0xF2, 0x0F, 0xD6],'modrm':None}
  arch_ext = 2

class movdqa(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0x6F], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0x6F], 'modrm':None}),
    (mem128_xmm,     {'opcode':[0x66, 0x0F, 0x7F], 'modrm':None}))
  arch_ext = 2

class movdqu(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0xF3, 0x0F, 0x6F], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0xF3, 0x0F, 0x6F], 'modrm':None}),
    (mem128_xmm,     {'opcode':[0xF3, 0x0F, 0x7F], 'modrm':None}))
  arch_ext = 2

class movhlps(Instruction):
  machine_inst = xmm_xmm
  params = {'opcode':[0x0F, 0x12],'modrm':None}
  arch_ext = 1

class movhpd(DispatchInstruction):
  dispatch = (
    (xmm_mem64,      {'opcode':[0x66, 0x0F, 0x16], 'modrm':None}),
    (mem64_xmm,      {'opcode':[0x66, 0x0F, 0x17], 'modrm':None}))
  arch_ext = 2

class movhps(DispatchInstruction):
  dispatch = (
    (xmm_mem64,      {'opcode':[0x0F, 0x16], 'modrm':None}),
    (mem64_xmm,      {'opcode':[0x0F, 0x17], 'modrm':None}))
  arch_ext = 1

class movlhps(Instruction):
  machine_inst = xmm_xmm
  params = {'opcode':[0x0F, 0x16],'modrm':None}
  arch_ext = 1

class movlpd(DispatchInstruction):
  dispatch = (
    (xmm_mem64,      {'opcode':[0x66, 0x0F, 0x12], 'modrm':None}),
    (mem64_xmm,      {'opcode':[0x66, 0x0F, 0x13], 'modrm':None}))
  arch_ext = 2

class movlps(DispatchInstruction):
  dispatch = (
    (xmm_mem64,      {'opcode':[0x0F, 0x12], 'modrm':None}),
    (mem64_xmm,      {'opcode':[0x0F, 0x13], 'modrm':None}))
  arch_ext = 1

class movmskpd(Instruction):
  machine_inst = reg32_xmm
  params = {'opcode':[0x66, 0x0F, 0x50],'modrm':None}
  arch_ext = 2

class movmskps(Instruction):
  machine_inst = reg32_xmm
  params = {'opcode':[0x0F, 0x50],'modrm':None}
  arch_ext = 2

class movntdq(Instruction):
  machine_inst = mem128_xmm
  params = {'opcode':[0x66, 0x0F, 0xE7],'modrm':None}
  arch_ext = 2

class movntdqa(Instruction):
  machine_inst = xmm_mem128
  params = {'opcode':[0x66, 0x0F, 0x38, 0x2A], 'modrm':None}
  arch_ext = 4

class movntpd(Instruction):
  machine_inst = mem128_xmm
  params = {'opcode':[0x66, 0x0F, 0x2B],'modrm':None}
  arch_ext = 2

class movntps(Instruction):
  machine_inst = mem128_xmm
  params = {'opcode':[0x0F, 0x2B],'modrm':None}
  arch_ext = 2

class movntq(Instruction):
  machine_inst = mem64_mmx
  params = {'opcode':[0x0F, 0xE7],'modrm':None}
  arch_ext = 1

class movntsd(Instruction):
  machine_inst = mem64_xmm
  params = {'opcode':[0xF2, 0x0F, 0x2B], 'modrm':None}
  arch_ext = 4

class movntss(Instruction):
  machine_inst = mem32_xmm
  params = {'opcode':[0xF3, 0x0F, 0x2B], 'modrm':None}
  arch_ext = 4

class movq(DispatchInstruction):
  dispatch = (
    (xmm_xmm,          {'opcode':[0xF3, 0x0F, 0x7E], 'modrm':None}),
    (xmm_mem64,        {'opcode':[0xF3, 0x0F, 0x7E], 'modrm':None}),
    (mem64_xmm,        {'opcode':[0x66, 0x0F, 0xD6], 'modrm':None}),
    (mmx_mmx,          {'opcode':[0x0F, 0x6F], 'modrm':None}),
    (mmx_mem64,        {'opcode':[0x0F, 0x6F], 'modrm':None}),
    (mem64_mmx,        {'opcode':[0x0F, 0x7F], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class movq2dq(Instruction):
  machine_inst = xmm_mmx
  params = {'opcode':[0xF3, 0x0F, 0xD6],'modrm':None}
  arch_ext = 2

class movsd(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0xF2, 0x0F, 0x10], 'modrm':None}),
    (xmm_mem64,      {'opcode':[0xF2, 0x0F, 0x10], 'modrm':None}),
    (mem64_xmm,      {'opcode':[0xF2, 0x0F, 0x11], 'modrm':None}))
  arch_ext = 2

class movshdup(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0xF3, 0x0F, 0x16], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0xF3, 0x0F, 0x16], 'modrm':None}))
  arch_ext = 3

class movsldup(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0xF3, 0x0F, 0x12], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0xF3, 0x0F, 0x12], 'modrm':None}))
  arch_ext = 3

class movss(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0xF3, 0x0F, 0x10], 'modrm':None}),
    (xmm_mem32,      {'opcode':[0xF3, 0x0F, 0x10], 'modrm':None}),
    (mem32_xmm,      {'opcode':[0xF3, 0x0F, 0x11], 'modrm':None}))
  arch_ext = 1

class movupd(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0x10], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0x10], 'modrm':None}),
    (mem128_xmm,     {'opcode':[0x66, 0x0F, 0x11], 'modrm':None}))
  arch_ext = 2

class movups(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x0F, 0x10], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x0F, 0x10], 'modrm':None}),
    (mem128_xmm,     {'opcode':[0x0F, 0x11], 'modrm':None}))
  arch_ext = 1

class mpsadbw(DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm8,    {'opcode':[0x66, 0x0F, 0x3A, 0x42], 'modrm':None}),
    (xmm_mem128_imm8, {'opcode':[0x66, 0x0F, 0x3A, 0x42], 'modrm':None}))
  arch_ext = 4

class mulpd(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0x59], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0x59], 'modrm':None}))
  arch_ext = 2

class mulps(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x0F, 0x59], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x0F, 0x59], 'modrm':None}))
  arch_ext = 1

class mulsd(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0xF2, 0x0F, 0x59], 'modrm':None}),
    (xmm_mem64,      {'opcode':[0xF2, 0x0F, 0x59], 'modrm':None}))
  arch_ext = 2

class mulss(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0xF3, 0x0F, 0x59], 'modrm':None}),
    (xmm_mem32,      {'opcode':[0xF3, 0x0F, 0x59], 'modrm':None}))
  arch_ext = 1

class orpd(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0x56], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0x56], 'modrm':None}))
  arch_ext = 2

class orps(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x0F, 0x56], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x0F, 0x56], 'modrm':None}))
  arch_ext = 1

class pabsb(DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x1C], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x1C], 'modrm':None}),
    (mmx_mmx,    {'opcode':[0x0F, 0x38, 0x1C], 'modrm':None}),
    (mmx_mem64,  {'opcode':[0x0F, 0x38, 0x1C], 'modrm':None}))
  arch_ext = 3

class pabsd(DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x1E], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x1E], 'modrm':None}),
    (mmx_mmx,    {'opcode':[0x0F, 0x38, 0x1E], 'modrm':None}),
    (mmx_mem64,  {'opcode':[0x0F, 0x38, 0x1E], 'modrm':None}))
  arch_ext = 3

class pabsw(DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x1D], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x1D], 'modrm':None}),
    (mmx_mmx,    {'opcode':[0x0F, 0x38, 0x1D], 'modrm':None}),
    (mmx_mem64,  {'opcode':[0x0F, 0x38, 0x1D], 'modrm':None}))
  arch_ext = 3

class packssdw(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0x6B], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0x6B], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0x6B], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0x6B], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class packsswb(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0x63], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0x63], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0x63], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0x63], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class packusdw(DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x2B], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x2B], 'modrm':None}))
  arch_ext = 4

class packuswb(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0x67], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0x67], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0x67], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0x67], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class paddb(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xFC], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xFC], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xFC], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xFC], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class paddd(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xFE], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xFE], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xFE], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xFE], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class paddq(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xD4], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xD4], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xD4], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xD4], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class paddsb(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xEC], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xEC], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xEC], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xEC], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class paddsw(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xED], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xED], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xED], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xED], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class paddusb(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xDC], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xDC], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xDC], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xDC], 'modrm':None}))
  arch_ext = 0

class paddusw(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xDD], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xDD], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xDD], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xDD], 'modrm':None}))
  arch_ext = 0

class paddw(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xFD], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xFD], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xFD], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xFD], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class palignr(DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm8,   {'opcode':[0x66, 0x0F, 0x3A, 0x0F], 'modrm':None}),
    (xmm_mem128_imm8,{'opcode':[0x66, 0x0F, 0x3A, 0x0F], 'modrm':None}),
    (mmx_mmx_imm8,   {'opcode':[0x0F, 0x3A, 0x0F], 'modrm':None}),
    (mmx_mem64_imm8, {'opcode':[0x0F, 0x3A, 0x0F], 'modrm':None}))
  arch_ext = 3

class pand(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xDB], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xDB], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xDB], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xDB], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class pandn(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xDF], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xDF], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xDF], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xDF], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class pavgb(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xE0], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xE0], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xE0], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xE0], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class pavgw(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xE3], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xE3], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xE3], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xE3], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class pblendvb(DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x10], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x10], 'modrm':None}))
  arch_ext = 4

class pblendw(DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm8,    {'opcode':[0x66, 0x0F, 0x3A, 0x0E], 'modrm':None}),
    (xmm_mem128_imm8, {'opcode':[0x66, 0x0F, 0x3A, 0x0E], 'modrm':None}))
  arch_ext = 4

class pcmpeqb(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0x74], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0x74], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0x74], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0x74], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class pcmpeqd(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0x76], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0x76], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0x76], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0x76], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class pcmpeqq(DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x29], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x29], 'modrm':None}))
  arch_ext = 4

class pcmpeqw(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0x75], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0x75], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0x75], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0x75], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class pcmpestri(DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm8,    {'opcode':[0x66, 0x0F, 0x3A, 0x61], 'modrm':None}),
    (xmm_mem128_imm8, {'opcode':[0x66, 0x0F, 0x3A, 0x61], 'modrm':None}))
  arch_ext = 4

class pcmpestrm(DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm8,    {'opcode':[0x66, 0x0F, 0x3A, 0x60], 'modrm':None}),
    (xmm_mem128_imm8, {'opcode':[0x66, 0x0F, 0x3A, 0x60], 'modrm':None}))
  arch_ext = 4

class pcmpgtb(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0x64], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0x64], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0x64], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0x64], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class pcmpgtd(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0x66], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0x66], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0x66], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0x66], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class pcmpgtw(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0x65], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0x65], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0x65], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0x65], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class pcmpgtq(DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x37], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x37], 'modrm':None}))
  arch_ext = 4

class pcmpistri(DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm8,    {'opcode':[0x66, 0x0F, 0x3A, 0x63], 'modrm':None}),
    (xmm_mem128_imm8, {'opcode':[0x66, 0x0F, 0x3A, 0x63], 'modrm':None}))
  arch_ext = 4

class pcmpistrm(DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm8,    {'opcode':[0x66, 0x0F, 0x3A, 0x62], 'modrm':None}),
    (xmm_mem128_imm8, {'opcode':[0x66, 0x0F, 0x3A, 0x62], 'modrm':None}))
  arch_ext = 4

class pextrb(DispatchInstruction):
  dispatch = (
    (reg32_xmm_imm8_rev, {'opcode':[0x66, 0x0F, 0x3A, 0x14], 'modrm':None}),
    (mem8_xmm_imm8,  {'opcode':[0x66, 0x0F, 0x3A, 0x14], 'modrm':None}))
  arch_ext = 4

class pextrd(DispatchInstruction):
  dispatch = (
    (reg32_xmm_imm8_rev, {'opcode':[0x66, 0x0F, 0x3A, 0x16], 'modrm':None}),
    (mem32_xmm_imm8, {'opcode':[0x66, 0x0F, 0x3A, 0x16], 'modrm':None}))
  arch_ext = 4

class pextrw(DispatchInstruction):
  dispatch = (
    (reg32_xmm_imm8, {'opcode':[0x66, 0x0F, 0xC5], 'modrm':None}),
    (mem16_xmm_imm8, {'opcode':[0x66, 0x0F, 0x3A, 0x15], 'modrm':None}),
    (reg32_mmx_imm8, {'opcode':[0x0F, 0xC5], 'modrm':None}))
  arch_ext = 1

class phaddsw(DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x03], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x03], 'modrm':None}),
    (mmx_mmx,    {'opcode':[0x0F, 0x38, 0x03], 'modrm':None}),
    (mmx_mem64,  {'opcode':[0x0F, 0x38, 0x03], 'modrm':None}))
  arch_ext = 3

class phaddw(DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x01], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x01], 'modrm':None}),
    (mmx_mmx,    {'opcode':[0x0F, 0x38, 0x01], 'modrm':None}),
    (mmx_mem64,  {'opcode':[0x0F, 0x38, 0x01], 'modrm':None}))
  arch_ext = 3

class phaddd(DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x02], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x02], 'modrm':None}),
    (mmx_mmx,    {'opcode':[0x0F, 0x38, 0x02], 'modrm':None}),
    (mmx_mem64,  {'opcode':[0x0F, 0x38, 0x02], 'modrm':None}))
  arch_ext = 3

class phminposuw(DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x41], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x41], 'modrm':None}))
  arch_ext = 4

class phminposuw(DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x41], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x41], 'modrm':None}))
  arch_ext = 4

class phsubsw(DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x07], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x07], 'modrm':None}),
    (mmx_mmx,    {'opcode':[0x0F, 0x38, 0x07], 'modrm':None}),
    (mmx_mem64,  {'opcode':[0x0F, 0x38, 0x07], 'modrm':None}))
  arch_ext = 3

class phsubw(DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x05], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x05], 'modrm':None}),
    (mmx_mmx,    {'opcode':[0x0F, 0x38, 0x05], 'modrm':None}),
    (mmx_mem64,  {'opcode':[0x0F, 0x38, 0x05], 'modrm':None}))
  arch_ext = 3

class phsubd(DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x06], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x06], 'modrm':None}),
    (mmx_mmx,    {'opcode':[0x0F, 0x38, 0x06], 'modrm':None}),
    (mmx_mem64,  {'opcode':[0x0F, 0x38, 0x06], 'modrm':None}))
  arch_ext = 3

class pinsrb(DispatchInstruction):
  dispatch = (
    (xmm_reg32_imm8, {'opcode':[0x66, 0x0F, 0x3A, 0x20], 'modrm':None}),
    (xmm_mem8_imm8,  {'opcode':[0x66, 0x0F, 0x3A, 0x20], 'modrm':None}))
  arch_ext = 4

class pinsrd(DispatchInstruction):
  dispatch = (
    (xmm_reg32_imm8, {'opcode':[0x66, 0x0F, 0x3A, 0x22], 'modrm':None}),
    (xmm_mem32_imm8, {'opcode':[0x66, 0x0F, 0x3A, 0x22], 'modrm':None}))
  arch_ext = 4

class pinsrw(DispatchInstruction):
  dispatch = (
    (xmm_reg32_imm8,     {'opcode':[0x66, 0x0F, 0xC4], 'modrm':None}),
    (xmm_mem16_imm8,     {'opcode':[0x66, 0x0F, 0xC4], 'modrm':None}),
    (mmx_reg32_imm8,     {'opcode':[0x0F, 0xC4], 'modrm':None}),
    (mmx_mem16_imm8,     {'opcode':[0x0F, 0xC4], 'modrm':None}))
  arch_ext = 1

class pmaddubsw(DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x04], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x04], 'modrm':None}),
    (mmx_mmx,    {'opcode':[0x0F, 0x38, 0x04], 'modrm':None}),
    (mmx_mem64,  {'opcode':[0x0F, 0x38, 0x04], 'modrm':None}))
  arch_ext = 3

class pmaddwd(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xF5], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xF5], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xF5], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xF5], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class pmaxsb(DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x3C], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x3C], 'modrm':None}))
  arch_ext = 4

class pmaxsd(DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x3D], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x3D], 'modrm':None}))
  arch_ext = 4

class pmaxsw(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xEE], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xEE], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xEE], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xEE], 'modrm':None}))
  arch_ext = 1

class pmaxub(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xDE], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xDE], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xDE], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xDE], 'modrm':None}))
  arch_ext = 1

class pmaxud(DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x3F], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x3F], 'modrm':None}))
  arch_ext = 4

class pmaxuw(DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x3E], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x3E], 'modrm':None}))
  arch_ext = 4

class pminsb(DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x38], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x38], 'modrm':None}))
  arch_ext = 4

class pminsd(DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x39], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x39], 'modrm':None}))
  arch_ext = 4

class pminsw(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xEA], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xEA], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xEA], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xEA], 'modrm':None}))
  arch_ext = 1

class pminub(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xDA], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xDA], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xDA], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xDA], 'modrm':None}))
  arch_ext = 1

class pminud(DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x3B], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x3B], 'modrm':None}))
  arch_ext = 4

class pminuw(DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x3A], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x3A], 'modrm':None}))
  arch_ext = 4

class pmovmskb(DispatchInstruction):
  dispatch = (
    (reg32_xmm,     {'opcode':[0x66, 0x0F, 0xD7], 'modrm':None}),
    (reg32_mmx,     {'opcode':[0x0F, 0xD7], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 1

class pmovsxbw(DispatchInstruction):
  dispatch = (
    (xmm_xmm,   {'opcode':[0x66, 0x0F, 0x38, 0x20], 'modrm':None}),
    (xmm_mem64, {'opcode':[0x66, 0x0F, 0x38, 0x20], 'modrm':None}))
  arch_ext = 4

class pmovsxbd(DispatchInstruction):
  dispatch = (
    (xmm_xmm,   {'opcode':[0x66, 0x0F, 0x38, 0x21], 'modrm':None}),
    (xmm_mem32, {'opcode':[0x66, 0x0F, 0x38, 0x21], 'modrm':None}))
  arch_ext = 4

class pmovsxbq(DispatchInstruction):
  dispatch = (
    (xmm_xmm,   {'opcode':[0x66, 0x0F, 0x38, 0x22], 'modrm':None}),
    (xmm_mem16, {'opcode':[0x66, 0x0F, 0x38, 0x22], 'modrm':None}))
  arch_ext = 4

class pmovsxwd(DispatchInstruction):
  dispatch = (
    (xmm_xmm,   {'opcode':[0x66, 0x0F, 0x38, 0x23], 'modrm':None}),
    (xmm_mem64, {'opcode':[0x66, 0x0F, 0x38, 0x23], 'modrm':None}))
  arch_ext = 4

class pmovsxwq(DispatchInstruction):
  dispatch = (
    (xmm_xmm,   {'opcode':[0x66, 0x0F, 0x38, 0x24], 'modrm':None}),
    (xmm_mem32, {'opcode':[0x66, 0x0F, 0x38, 0x24], 'modrm':None}))
  arch_ext = 4

class pmovsxdq(DispatchInstruction):
  dispatch = (
    (xmm_xmm,   {'opcode':[0x66, 0x0F, 0x38, 0x25], 'modrm':None}),
    (xmm_mem64, {'opcode':[0x66, 0x0F, 0x38, 0x25], 'modrm':None}))
  arch_ext = 4

class pmovzxbw(DispatchInstruction):
  dispatch = (
    (xmm_xmm,   {'opcode':[0x66, 0x0F, 0x38, 0x30], 'modrm':None}),
    (xmm_mem64, {'opcode':[0x66, 0x0F, 0x38, 0x30], 'modrm':None}))
  arch_ext = 4

class pmovzxbd(DispatchInstruction):
  dispatch = (
    (xmm_xmm,   {'opcode':[0x66, 0x0F, 0x38, 0x31], 'modrm':None}),
    (xmm_mem32, {'opcode':[0x66, 0x0F, 0x38, 0x31], 'modrm':None}))
  arch_ext = 4

class pmovzxbq(DispatchInstruction):
  dispatch = (
    (xmm_xmm,   {'opcode':[0x66, 0x0F, 0x38, 0x32], 'modrm':None}),
    (xmm_mem16, {'opcode':[0x66, 0x0F, 0x38, 0x32], 'modrm':None}))
  arch_ext = 4

class pmovzxwd(DispatchInstruction):
  dispatch = (
    (xmm_xmm,   {'opcode':[0x66, 0x0F, 0x38, 0x33], 'modrm':None}),
    (xmm_mem64, {'opcode':[0x66, 0x0F, 0x38, 0x33], 'modrm':None}))
  arch_ext = 4

class pmovzxwq(DispatchInstruction):
  dispatch = (
    (xmm_xmm,   {'opcode':[0x66, 0x0F, 0x38, 0x34], 'modrm':None}),
    (xmm_mem32, {'opcode':[0x66, 0x0F, 0x38, 0x34], 'modrm':None}))
  arch_ext = 4

class pmovzxdq(DispatchInstruction):
  dispatch = (
    (xmm_xmm,   {'opcode':[0x66, 0x0F, 0x38, 0x35], 'modrm':None}),
    (xmm_mem64, {'opcode':[0x66, 0x0F, 0x38, 0x35], 'modrm':None}))
  arch_ext = 4

class pmuldq(DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x28], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x28], 'modrm':None}))
  arch_ext = 4

class pmulhrsw(DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x0B], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x0B], 'modrm':None}),
    (mmx_mmx,    {'opcode':[0x0F, 0x38, 0x0B], 'modrm':None}),
    (mmx_mem64,  {'opcode':[0x0F, 0x38, 0x0B], 'modrm':None}))
  arch_ext = 3

class pmulhuw(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xE4], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xE4], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xE4], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xE4], 'modrm':None}))
  arch_ext = 1

class pmulhw(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xE5], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xE5], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xE5], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xE5], 'modrm':None}))
  arch_ext = 1

class pmulld(DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x40], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x40], 'modrm':None}))
  arch_ext = 4

class pmullw(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xD5], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xD5], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xD5], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xD5], 'modrm':None}))
  arch_ext = 2 # and 0

class pmuludq(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xF4], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xF4], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xF4], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xF4], 'modrm':None}))
  arch_ext = 2

class por(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xEB], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xEB], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xEB], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xEB], 'modrm':None}))
  arch_ext = 2 # and 0

class psadbw(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xF6], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xF6], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xF6], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xF6], 'modrm':None}))
  arch_ext = 1

class psignb(DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x08], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x08], 'modrm':None}),
    (mmx_mmx,    {'opcode':[0x0F, 0x38, 0x08], 'modrm':None}),
    (mmx_mem64,  {'opcode':[0x0F, 0x38, 0x08], 'modrm':None}))
  arch_ext = 3

class psignd(DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x0A], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x0A], 'modrm':None}),
    (mmx_mmx,    {'opcode':[0x0F, 0x38, 0x0A], 'modrm':None}),
    (mmx_mem64,  {'opcode':[0x0F, 0x38, 0x0A], 'modrm':None}))
  arch_ext = 3

class psignw(DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x09], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x09], 'modrm':None}),
    (mmx_mmx,    {'opcode':[0x0F, 0x38, 0x09], 'modrm':None}),
    (mmx_mem64,  {'opcode':[0x0F, 0x38, 0x09], 'modrm':None}))
  arch_ext = 3

class pshufb(DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x38, 0x00], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x38, 0x00], 'modrm':None}),
    (mmx_mmx,    {'opcode':[0x0F, 0x38, 0x00], 'modrm':None}),
    (mmx_mem64,  {'opcode':[0x0F, 0x38, 0x00], 'modrm':None}))
  arch_ext = 3

class pshufd(DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm8,    {'opcode':[0x66, 0x0F, 0x70], 'modrm':None}),
    (xmm_mem128_imm8, {'opcode':[0x66, 0x0F, 0x70], 'modrm':None}))
  arch_ext = 2

class pshufhw(DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm8,    {'opcode':[0xF3, 0x0F, 0x70], 'modrm':None}),
    (xmm_mem128_imm8, {'opcode':[0xF3, 0x0F, 0x70], 'modrm':None}))
  arch_ext = 2

class pshuflw(DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm8,    {'opcode':[0xF2, 0x0F, 0x70], 'modrm':None}),
    (xmm_mem128_imm8, {'opcode':[0xF2, 0x0F, 0x70], 'modrm':None}))
  arch_ext = 2

class pshufw(DispatchInstruction):
  dispatch = (
    (mmx_mmx_imm8,    {'opcode':[0x0F, 0x70], 'modrm':None}),
    (mmx_mem64_imm8,  {'opcode':[0x0F, 0x70], 'modrm':None}))
  arch_ext = 1

class pslld(DispatchInstruction):
  dispatch = (
    (xmm_imm8,   {'opcode':[0x66, 0x0F, 0x72], 'modrm':0x30}),
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0xF2], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0xF2], 'modrm':None}),
    (mmx_imm8,   {'opcode':[0x0F, 0x72], 'modrm':0x30}),
    (mmx_mmx,    {'opcode':[0x0F, 0xF2], 'modrm':None}),
    (mmx_mem128, {'opcode':[0x0F, 0xF2], 'modrm':None}))
  arch_ext = 2 # and 0 and 1

class pslldq(Instruction):
  machine_inst = xmm_imm8
  params = {'opcode':[0x66, 0x0F, 0x73],'modrm':0x38}
  arch_ext = 1

class psllq(DispatchInstruction):
  dispatch = (
    (xmm_imm8,   {'opcode':[0x66, 0x0F, 0x73], 'modrm':0x30}),
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0xF3], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0xF3], 'modrm':None}),
    (mmx_imm8,   {'opcode':[0x0F, 0x73], 'modrm':0x30}),
    (mmx_mmx,    {'opcode':[0x0F, 0xF3], 'modrm':None}),
    (mmx_mem128, {'opcode':[0x0F, 0xF3], 'modrm':None}))
  arch_ext = 2 # and 0 and 1

class psllw(DispatchInstruction):
  dispatch = (
    (xmm_imm8,   {'opcode':[0x66, 0x0F, 0x71], 'modrm':0x30}),
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0xF1], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0xF1], 'modrm':None}),
    (mmx_imm8,   {'opcode':[0x0F, 0x71], 'modrm':0x30}),
    (mmx_mmx,    {'opcode':[0x0F, 0xF1], 'modrm':None}),
    (mmx_mem128, {'opcode':[0x0F, 0xF1], 'modrm':None}))
  arch_ext = 2 # and 0 and 1

class psrad(DispatchInstruction):
  dispatch = (
    (xmm_imm8,   {'opcode':[0x66, 0x0F, 0x72], 'modrm':0x20}),
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0xE2], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0xE2], 'modrm':None}),
    (mmx_imm8,   {'opcode':[0x0F, 0x72], 'modrm':0x20}),
    (mmx_mmx,    {'opcode':[0x0F, 0xE2], 'modrm':None}),
    (mmx_mem128, {'opcode':[0x0F, 0xE2], 'modrm':None}))
  arch_ext = 2 # and 0 and 1

class psraw(DispatchInstruction):
  dispatch = (
    (xmm_imm8,   {'opcode':[0x66, 0x0F, 0x71], 'modrm':0x20}),
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0xE1], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0xE1], 'modrm':None}),
    (mmx_imm8,   {'opcode':[0x0F, 0x71], 'modrm':0x20}),
    (mmx_mmx,    {'opcode':[0x0F, 0xE1], 'modrm':None}),
    (mmx_mem128, {'opcode':[0x0F, 0xE1], 'modrm':None}))
  arch_ext = 2 # and 0 and 1

class psrld(DispatchInstruction):
  dispatch = (
    (xmm_imm8,   {'opcode':[0x66, 0x0F, 0x72], 'modrm':0x10}),
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0xD2], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0xD2], 'modrm':None}),
    (mmx_imm8,   {'opcode':[0x0F, 0x72], 'modrm':0x10}),
    (mmx_mmx,    {'opcode':[0x0F, 0xD2], 'modrm':None}),
    (mmx_mem128, {'opcode':[0x0F, 0xD2], 'modrm':None}))
  arch_ext = 2 # and 0 and 1

class psrldq(Instruction):
  machine_inst = xmm_imm8
  params = {'opcode':[0x66, 0x0F, 0x73],'modrm':0x18}
  arch_ext = 1

class psrlq(DispatchInstruction):
  dispatch = (
    (xmm_imm8,   {'opcode':[0x66, 0x0F, 0x73], 'modrm':0x10}),
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0xD3], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0xD3], 'modrm':None}),
    (mmx_imm8,   {'opcode':[0x0F, 0x73], 'modrm':0x10}),
    (mmx_mmx,    {'opcode':[0x0F, 0xD3], 'modrm':None}),
    (mmx_mem128, {'opcode':[0x0F, 0xD3], 'modrm':None}))
  arch_ext = 2 # and 0 and 1

class psrlw(DispatchInstruction):
  dispatch = (
    (xmm_imm8,   {'opcode':[0x66, 0x0F, 0x71], 'modrm':0x10}),
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0xD1], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0xD1], 'modrm':None}),
    (mmx_imm8,   {'opcode':[0x0F, 0x71], 'modrm':0x10}),
    (mmx_mmx,    {'opcode':[0x0F, 0xD1], 'modrm':None}),
    (mmx_mem128, {'opcode':[0x0F, 0xD1], 'modrm':None}))
  arch_ext = 2 # and 0 and 1

class psubb(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xF8], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xF8], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xF8], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xF8], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class psubd(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xFA], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xFA], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xFA], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xFA], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class psubq(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xFB], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xFB], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xFB], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xFB], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class psubsb(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xE8], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xE8], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xE8], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xE8], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class psubsw(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xE9], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xE9], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xE9], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xE9], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class psubusb(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xD8], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xD8], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xD8], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xD8], 'modrm':None}))
  arch_ext = 0

class psubusw(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xD9], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xD9], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xD9], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xD9], 'modrm':None}))
  arch_ext = 0

class psubw(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xF9], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xF9], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xF9], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xF9], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class punpckhbw(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0x68], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0x68], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0x68], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0x68], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class punpckhdq(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0x6A], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0x6A], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0x6A], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0x6A], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class punpckhqdq(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0x6D], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0x6D], 'modrm':None}))
  arch_ext = 2

class punpckhwd(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0x69], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0x69], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0x69], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0x69], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class punpcklbw(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0x60], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0x60], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0x60], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0x60], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class punpckldq(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0x62], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0x62], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0x62], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0x62], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class punpcklqdq(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0x6C], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0x6C], 'modrm':None}))
  arch_ext = 2

class punpcklwd(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0x61], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0x61], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0x61], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0x61], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class pxor(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x66, 0x0F, 0xEF], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x66, 0x0F, 0xEF], 'modrm':None}),
    (mmx_mmx,        {'opcode':[0x0F, 0xEF], 'modrm':None}),
    (mmx_mem64,      {'opcode':[0x0F, 0xEF], 'modrm':None}))
  arch_ext = 2 # TODO - err, some are 2, some are 0

class rcpps(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x0F, 0x53], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x0F, 0x53], 'modrm':None}))
  arch_ext = 1

class rcpss(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0xF3, 0x0F, 0x53], 'modrm':None}),
    (xmm_mem32,      {'opcode':[0xF3, 0x0F, 0x53], 'modrm':None}))
  arch_ext = 2

class roundpd(DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm8,    {'opcode':[0x66, 0x0F, 0x3A, 0x09], 'modrm':None}),
    (xmm_mem128_imm8, {'opcode':[0x66, 0x0F, 0x3A, 0x09], 'modrm':None}))
  arch_ext = 4

class roundps(DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm8,    {'opcode':[0x66, 0x0F, 0x3A, 0x08], 'modrm':None}),
    (xmm_mem128_imm8, {'opcode':[0x66, 0x0F, 0x3A, 0x08], 'modrm':None}))
  arch_ext = 4

class roundsd(DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm8,    {'opcode':[0x66, 0x0F, 0x3A, 0x0B], 'modrm':None}),
    (xmm_mem64_imm8,  {'opcode':[0x66, 0x0F, 0x3A, 0x0B], 'modrm':None}))
  arch_ext = 4

class roundss(DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm8,    {'opcode':[0x66, 0x0F, 0x3A, 0x0A], 'modrm':None}),
    (xmm_mem128_imm8, {'opcode':[0x66, 0x0F, 0x3A, 0x0A], 'modrm':None}))
  arch_ext = 4

class rsqrtps(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0x0F, 0x52], 'modrm':None}),
    (xmm_mem128,     {'opcode':[0x0F, 0x52], 'modrm':None}))
  arch_ext = 1

class rsqrtss(DispatchInstruction):
  dispatch = (
    (xmm_xmm,        {'opcode':[0xF3, 0x0F, 0x52], 'modrm':None}),
    (xmm_mem32,      {'opcode':[0xF3, 0x0F, 0x52], 'modrm':None}))
  arch_ext = 2

class shufpd(DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm8,    {'opcode':[0x66, 0x0F, 0xC6], 'modrm':None}),
    (xmm_mem128_imm8, {'opcode':[0x66, 0x0F, 0xC6], 'modrm':None}))
  arch_ext = 2

class shufps(DispatchInstruction):
  dispatch = (
    (xmm_xmm_imm8,    {'opcode':[0x0F, 0xC6], 'modrm':None}),
    (xmm_mem128_imm8, {'opcode':[0x0F, 0xC6], 'modrm':None}))
  arch_ext = 1

class sqrtpd(DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x51], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x51], 'modrm':None}))
  arch_ext = 2

class sqrtps(DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x0F, 0x51], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x0F, 0x51], 'modrm':None}))
  arch_ext = 1

class sqrtsd(DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0xF2, 0x0F, 0x51], 'modrm':None}),
    (xmm_mem64,  {'opcode':[0xF2, 0x0F, 0x51], 'modrm':None}))
  arch_ext = 2

class sqrtss(DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0xF3, 0x0F, 0x51], 'modrm':None}),
    (xmm_mem128, {'opcode':[0xF3, 0x0F, 0x51], 'modrm':None}))
  arch_ext = 1

class stmxcsr(Instruction):
  machine_inst = mem32
  params = {'opcode':[0x0F, 0xAE],'modrm':0x18}
  arch_ext = 1

class subpd(DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x5C], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x5C], 'modrm':None}))
  arch_ext = 2

class subps(DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x0F, 0x5C], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x0F, 0x5C], 'modrm':None}))
  arch_ext = 1

class subsd(DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0xF2, 0x0F, 0x5C], 'modrm':None}),
    (xmm_mem64,  {'opcode':[0xF2, 0x0F, 0x5C], 'modrm':None}))
  arch_ext = 2

class subss(DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0xF3, 0x0F, 0x5C], 'modrm':None}),
    (xmm_mem128, {'opcode':[0xF3, 0x0F, 0x5C], 'modrm':None}))
  arch_ext = 1

class ucomisd(DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x2E], 'modrm':None}),
    (xmm_mem64,  {'opcode':[0x66, 0x0F, 0x2E], 'modrm':None}))
  arch_ext = 2

class ucomiss(DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x0F, 0x2E], 'modrm':None}),
    (xmm_mem32,  {'opcode':[0x0F, 0x2E], 'modrm':None}))
  arch_ext = 1

class unpckhpd(DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x15], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x15], 'modrm':None}))
  arch_ext = 1

class unpckhps(DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x0F, 0x15], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x0F, 0x15], 'modrm':None}))
  arch_ext = 1

class unpcklpd(DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x14], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x14], 'modrm':None}))
  arch_ext = 1

class unpcklps(DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x0F, 0x14], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x0F, 0x14], 'modrm':None}))
  arch_ext = 1

class xorpd(DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x66, 0x0F, 0x57], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x66, 0x0F, 0x57], 'modrm':None}))
  arch_ext = 2

class xorps(DispatchInstruction):
  dispatch = (
    (xmm_xmm,    {'opcode':[0x0F, 0x57], 'modrm':None}),
    (xmm_mem128, {'opcode':[0x0F, 0x57], 'modrm':None}))
  arch_ext = 1

