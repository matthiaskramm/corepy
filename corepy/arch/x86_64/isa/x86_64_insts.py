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

from corepy.spre.spe import MachineInstruction
#from corepy.arch.x86_64.lib.memory import MemoryReference
from x86_64_fields import *
import corepy.arch.x86_64.types.registers as regs

# TODO - blah, anything that wraps around a common_memref needs an if-statement

# ------------------------------
# Type Instances
# ------------------------------

# Instances for common types

# Constants
one_t = x86ConstantOperand("one", 1)

# Immediates
imm8_t  = Imm8("imm8",  (-128, 256))
simm8_t = Imm8("simm8", (-128, 128))
imm16_t = Imm16("imm16", (-32768, 65536))
imm32_t = Imm32("imm32", (-2147483648, 4294967296))
simm32_t = Imm32("simm32", (-2147483648, 2147483648))
imm64_t = Imm64("imm64", (-9223372036854775808, 18446744073709551616))

# Memory
mem_t   = x86MemoryOperand("mem", None)
mem8_t  = x86MemoryOperand("mem8", 8)
mem16_t = x86MemoryOperand("mem16", 16)
mem32_t = x86MemoryOperand("mem32", 32)
mem64_t = x86MemoryOperand("mem64", 64)
mem80_t = x86MemoryOperand("mem80", 80)
mem128_t = x86MemoryOperand("mem128", 128)
mem228_t = x86MemoryOperand("mem228", 228)
#mem512_t = x86MemoryOperand("mem512", 512)
mem752_t = x86MemoryOperand("mem752", 752)
mem4096_t = x86MemoryOperand("mem4096", 4096)

# Registers
reg8_t  = x86RegisterOperand("reg8", regs.GPRegister8)
reg16_t = x86RegisterOperand("reg16", regs.GPRegister16)
reg32_t = x86RegisterOperand("reg32", regs.GPRegister32)
reg64_t = x86RegisterOperand("reg64", regs.GPRegister64)
regst_t = x86RegisterOperand("regst", regs.FPRegister)
mmx_t  = x86RegisterOperand("mmx", regs.MMXRegister)
xmm_t  = x86RegisterOperand("xmm", regs.XMMRegister)

# Fixed Registers
al_t = FixedRegisterOperand("al", regs.GPRegister8, 0)
ax_t = FixedRegisterOperand("ax", regs.GPRegister16, 0)
cl_t = FixedRegisterOperand("cl", regs.GPRegister8, 1)
dx_t = FixedRegisterOperand("dx", regs.GPRegister16, 2)
eax_t = FixedRegisterOperand("eax", regs.GPRegister32, 0)
rax_t = FixedRegisterOperand("rax", regs.GPRegister64, 0)
st0_t = FixedRegisterOperand("st0", regs.FPRegister, 0)

# Relative offsets
lbl8off_t = x86LabelOperand("lbl8off", (-128, 128))
#lbl16off_t = x86LabelOperand("lbl16off", (-65536, 65536))
lbl32off_t = x86LabelOperand("lbl32off", (-2147483648, 2147483648))

rel8off_t  = Rel8off("rel8off",  (-128, 128))
#rel16off_t = Rel16off("rel16off", (-32768, 32768))
rel32off_t = Rel32off("rel32off", (-2147483648, 2147483648))

# Prefix bytes
#prefix = []
lock_p = x86PrefixOperand("lock", 0xF0)
addr_p = x86PrefixOperand("addr", 0x67)

# Array of reg8's that require some sort of REX
reg8_rex_list = (regs.sil, regs.dil, regs.bpl, regs.spl)


# ------------------------------
# Utility functions
# ------------------------------

# w8() just makes sure n is an unsigned byte; the others do this implicitly.
# All of these are little endian specific!!

def w8(n):
  return [(256 + n) & 0xFF]
  
def w16(n):
  return [n & 0xFF, (n & 0xFF00) >> 8]

def w32(n):
  return [n & 0xFF, (n & 0xFF00) >> 8, (n & 0xFF0000) >> 16, (n & 0xFF000000l) >> 24]

def w64(n):
  return [n & 0xFF, (n & 0xFF00) >> 8, (n & 0xFF0000) >> 16, (n & 0xFF000000l) >> 24, (n & 0xFF00000000l) >> 32, (n & 0xFF0000000000l) >> 40, (n & 0xFF000000000000l) >> 48, (n & 0xFF00000000000000l) >> 56]


def common_memref_modrm(opcode, ref, modrm, rex, force_rex):
  if ref.disp != None and ref.disp != 0:    # [base + disp]
    if ref.base in (regs.rip, regs.eip):
      rex = [0x40 | rex]
      if rex == [0x40] and not force_rex:
        rex = []
      return rex + opcode + [0x5 | modrm] + w32(ref.disp)
    elif ref.index != None:                 # [base+index*scale+disp]
      sib = ref.scale_sib | (ref.index.reg << 3) | ref.base.reg
      rex = [0x40 | (ref.index.rex << 1) | ref.base.rex | rex]
      if rex == [0x40] and not force_rex:
        rex = []

      if simm8_t.fits(ref.disp):
        return rex + opcode + [0x44 | modrm, sib] + w8(ref.disp)
      elif simm32_t.fits(ref.disp):
        return rex + opcode + [0x84 | modrm, sib] + w32(ref.disp)
    elif ref.index == None:                 # [base + disp]
      rex = [0x40 | ref.base.rex | rex]
      if rex == [0x40] and not force_rex:
        rex = []

      if ref.base in (regs.rsp, regs.r12, regs.esp):
        if simm8_t.fits(ref.disp):           # [rsp + disp], [r12 + disp]
          return rex + opcode + [0x44 | modrm, 0x24] + w8(ref.disp)
        elif simm32_t.fits(ref.disp):
          return rex + opcode + [0x80 | modrm, 0x24] + w32(ref.disp)
      elif simm8_t.fits(ref.disp):
        return rex + opcode + [0x40 | modrm | ref.base.reg] + w8(ref.disp)
      elif simm32_t.fits(ref.disp):
        return rex + opcode + [0x80 | modrm | ref.base.reg] + w32(ref.disp)
  elif ref.index != None:
    sib = ref.scale_sib | (ref.index.reg << 3) | ref.base.reg
    rex = [0x40 | (ref.index.rex << 1) | ref.base.rex | rex]
    if rex == [0x40] and not force_rex:
      rex = []

    if ref.base in (regs.rbp, regs.r13, regs.ebp):
      return rex + opcode + [0x44 | modrm, sib, 0x00] # [rbp, index], [r13, index]
    return rex + opcode + [0x04 | modrm, sib]
  elif ref.index == None:
    rex = [0x40 | ref.base.rex | rex]
    if rex == [0x40] and not force_rex:
      rex = []

    if ref.base in (regs.rbp, regs.r13, regs.ebp):
      return rex + opcode + [0x45 | modrm, 0x00] # [rbp], [r13]
    elif ref.base in (regs.rsp, regs.r12, regs.esp):
      return rex + opcode + [0x04 | modrm, 0x24] # [rsp], [r12]
    return rex + opcode + [modrm | ref.base.reg] # [base]


def common_memref(opcode, ref, modrm, rex = 0, force_rex = False):
  if ref.addr != None:  # Absolute address
    rex = [0x40 | rex]
    if rex == [0x40] and not force_rex:
      rex = []
    return rex + opcode + [0x04 | modrm, 0x25] + w32(ref.addr)
  elif ref.addr_size == 64: # 64bit modRM address, RIP-relative
    return common_memref_modrm(opcode, ref, modrm, rex, force_rex)
  elif ref.addr_size == 32: # 32bit modRM address, EIP-relative
    return [0x67] + common_memref_modrm(opcode, ref, modrm, rex, force_rex)
  return None


# ------------------------------
# x86_64 Machine Instructions
# ------------------------------

class al_dx(MachineInstruction):
  signature = (al_t, dx_t)
  opt_kw = ()
  
  def _render(params, operands):
    return params['opcode']
  render = staticmethod(_render)
  

class al_imm8(MachineInstruction):
  signature = (al_t, imm8_t)
  opt_kw = ()
  
  def _render(params, operands):
    return params['opcode'] + w8(operands['imm8'])
  render = staticmethod(_render)

  
class ax(MachineInstruction):
  signature = (ax_t,)
  opt_kw = ()

  def _render(params, operands):
    return params['opcode']
  render = staticmethod(_render)


class ax_dx(MachineInstruction):
  signature = (ax_t, dx_t)
  opt_kw = ()
  
  def _render(params, operands):
    return params['opcode']
  render = staticmethod(_render)    


class ax_imm16(MachineInstruction):
  signature = (ax_t, imm16_t)
  opt_kw = ()
  
  def _render(params, operands):
    return params['opcode'] + w16(operands['imm16'])
  render = staticmethod(_render)

  
class ax_imm8(MachineInstruction):
  signature = (ax_t, imm8_t)
  opt_kw = ()
  
  def _render(params, operands):
    return params['opcode'] + w8(operands['imm8'])
  render = staticmethod(_render)

  
class ax_reg16(MachineInstruction):
  signature = (ax_t, reg16_t)
  opt_kw = ()
  
  def _render(params, operands):
    opcode = params['opcode']
    reg16 = operands['reg16']
    rex = [0x40 | reg16.rex]
    if rex == [0x40]:
      rex = []

    return [0x66] + rex + opcode[:-1] + [opcode[-1] + reg16.reg]
  render = staticmethod(_render)


class dx_al(MachineInstruction):
  signature = (dx_t, al_t)
  opt_kw = ()
  
  def _render(params, operands):
    return params['opcode']
  render = staticmethod(_render)

  
class dx_ax(MachineInstruction):
  signature = (dx_t, ax_t)
  opt_kw = ()
  
  def _render(params, operands):
    return params['opcode']
  render = staticmethod(_render)

  
class dx_eax(MachineInstruction):
  signature = (dx_t, eax_t)
  opt_kw = ()
  
  def _render(params, operands):
    return params['opcode']
  render = staticmethod(_render)

  
class eax_dx(MachineInstruction):
  signature = (eax_t, dx_t)
  opt_kw = ()
  
  def _render(params, operands):
    return params['opcode']
  render = staticmethod(_render)

  
class eax_imm32(MachineInstruction):
  signature = (eax_t, imm32_t)
  opt_kw = ()
  
  def _render(params, operands):
    return params['opcode'] + w32(operands['imm32'])
  render = staticmethod(_render)

  
class eax_imm8(MachineInstruction):
  signature = (eax_t, imm8_t)
  opt_kw = ()
  
  def _render(params, operands):
    return params['opcode'] + w8(operands['imm8'])
  render = staticmethod(_render)

  
class eax_reg32(MachineInstruction):
  signature = (eax_t, reg32_t)
  opt_kw = ()
  
  def _render(params, operands):
    opcode = params['opcode']
    reg32 = operands['reg32']
    rex = [0x40 | reg32.rex]
    if rex == [0x40]:
      rex = []

    return rex + opcode[:-1] + [opcode[-1] + reg32.reg]
  render = staticmethod(_render)

  
class imm16(MachineInstruction):
  signature = (imm16_t,)
  opt_kw = ()
  
  def _render(params, operands):
    return params['opcode'] + w16(operands['imm16'])
  render = staticmethod(_render)

  
class imm16_imm8(MachineInstruction):
  signature = (imm16_t, imm8_t)
  opt_kw = ()

  # Note -- should not be adding 0x66 here! 
  def _render(params, operands):
    return params['opcode'] + w16(operands['imm16']) + w8(operands['imm8'])
  render = staticmethod(_render)

  
class imm32(MachineInstruction):
  signature = (imm32_t,)
  opt_kw = ()
  
  def _render(params, operands):
    return params['opcode'] + w32(operands['imm32'])
  render = staticmethod(_render)


class imm8(MachineInstruction):
  signature = (imm8_t,)
  opt_kw = ()
  
  def _render(params, operands):
    return params['opcode'] + w8(operands['imm8'])
  render = staticmethod(_render)


class imm8_al(MachineInstruction):
  signature = (imm8_t, al_t)
  opt_kw = ()
  
  def _render(params, operands):
    return params['opcode'] + w8(operands['imm8'])
  render = staticmethod(_render)

  
class imm8_ax(MachineInstruction):
  signature = (imm8_t, ax_t)
  opt_kw = ()
  
  def _render(params, operands):
    return params['opcode'] + w8(operands['imm8'])
  render = staticmethod(_render)

  
class imm8_eax(MachineInstruction):
  signature = (imm8_t, eax_t)
  opt_kw = ()
  
  def _render(params, operands):
    return params['opcode'] + w8(operands['imm8'])
  render = staticmethod(_render)


# Label offset that allows for either an 8 or 32bit offset encoding
class lbl32_8off(MachineInstruction):
  signature = (lbl32off_t,)
  opt_kw = ()
  
  def _render(params, operands):
    lbl = operands['lbl32off']
    # Relative offset is computed from the end of this instruction
    #print "lbl position", lbl.position
    #print "inst position", operands['position']
    offset = lbl.position - operands['position']
    #print "offset", offset

    # Will an 8bit offset do the job?
    off = offset - (len(params['opcode'][0]) + 1)
    #print "off is", off
    if rel8off_t.fits(off):
      #print "encoding 8bit offset", off
      return params['opcode'][0] + w8(off)

    # Fall back to 32bit, or nothing if even that doesn't fit
    off = offset - (len(params['opcode'][1]) + 4)
    #print "off is", off, len(params['opcode'][1]) + 4
    if rel32off_t.fits(off):
      #print "encoding 32bit offset", off
      return params['opcode'][1] + w32(off)
  render = staticmethod(_render)


class lbl32off(MachineInstruction):
  signature = (lbl32off_t,)
  opt_kw = ()
  
  def _render(params, operands):
    lbl = operands['lbl32off']
    # Relative offset is computed from the end of this instruction
    offset = lbl.position - operands['position']

    off = offset - (len(params['opcode']) + 4)
    if rel32off_t.fits(off):
      return params['opcode'] + w32(off)
  render = staticmethod(_render)


class lbl8off(MachineInstruction):
  signature = (lbl8off_t,)
  opt_kw = ()
  
  def _render(params, operands):
    lbl = operands['lbl8off']
    offset = lbl.position - operands['position']
    #print "offset", offset

    off = offset - (len(params['opcode']) + 1)
    if rel8off_t.fits(off):
      #print "encoding 8bit offset", off
      return params['opcode'] + w8(off)
  render = staticmethod(_render)


class mem128(MachineInstruction):
  signature = (mem128_t,)
  opt_kw = (lock_p,)

  # Only cmpxchg16b uses this -- if something else does, it might not be correct.
  def _render(params, operands):
    ret = common_memref(params['opcode'], operands['mem128'], params['modrm'], 0x08)
    if operands.has_key('lock') and operands['lock'] == True and ret is not None:
      return [lock_p.value] + ret
    return ret
  render = staticmethod(_render)


class mem128_xmm(MachineInstruction):
  signature = (mem128_t, xmm_t)
  opt_kw = ()
  
  def _render(params, operands):
    xmm = operands['xmm']
    ret = common_memref(params['opcode'], operands['mem128'], xmm.reg << 3, xmm.rex << 2)
    if ret != None:
      return params['prefix'] + ret
  render = staticmethod(_render)


class mem16(MachineInstruction):
  signature = (mem16_t,)
  opt_kw = ()
  
  def _render(params, operands):
    ret = common_memref(params['opcode'], operands['mem16'], params['modrm'])
    if ret != None:
      return ret
  render = staticmethod(_render)  


class mem16_1(MachineInstruction):
  signature = (mem16_t, one_t)
  opt_kw = ()
  
  def _render(params, operands):
    ret = common_memref(params['opcode'], operands['mem16'], params['modrm'])
    if ret != None:
      return [0x66] + ret
  render = staticmethod(_render)


class mem16_cl(MachineInstruction):
  signature = (mem16_t, cl_t)
  opt_kw = ()
  
  def _render(params, operands):
    ret = common_memref(params['opcode'], operands['mem16'], params['modrm'])
    if ret != None:
      return [0x66] + ret
  render = staticmethod(_render)


class mem16_imm16(MachineInstruction):
  signature = (mem16_t, imm16_t)
  opt_kw = ()
  
  def _render(params, operands):
    ret = common_memref(params['opcode'], operands['mem16'], params['modrm'])
    if ret != None:
      return [0x66] + ret + w16(operands['imm16'])
  render = staticmethod(_render)

 
class mem16_imm8(MachineInstruction):
  signature = (mem16_t, imm8_t)
  opt_kw = ()
  
  def _render(params, operands):
    ret = common_memref(params['opcode'], operands['mem16'], params['modrm'])
    if ret != None:
      return [0x66] + ret + w8(operands['imm8'])
  render = staticmethod(_render)


class mem16_reg16(MachineInstruction):
  signature = (mem16_t, reg16_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg16 = operands['reg16']
    ret = common_memref(params['opcode'], operands['mem16'], reg16.reg << 3, reg16.rex << 2)
    if ret != None:
      return [0x66] + ret
  render = staticmethod(_render)

 
class mem16_reg16_cl(MachineInstruction):
  signature = (mem16_t, reg16_t, cl_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg16 = operands['reg16']
    ret = common_memref(params['opcode'], operands['mem16'], reg16.reg << 3, reg16.rex << 2)
    if ret != None:
      return [0x66] + ret
  render = staticmethod(_render)


class mem16_reg16_imm8(MachineInstruction):
  signature = (mem16_t, reg16_t, imm8_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg16 = operands['reg16']
    ret = common_memref(params['opcode'], operands['mem16'], reg16.reg << 3, reg16.rex << 2)
    if ret != None:
      return [0x66] + ret + w8(operands['imm8'])
  render = staticmethod(_render)


class mem16_xmm_imm8(MachineInstruction):
  signature = (mem16_t, xmm_t, imm8_t)
  opt_kw = ()
  
  def _render(params, operands):
    xmm = operands['xmm']
    ret = common_memref(params['opcode'], operands['mem16'], xmm.reg << 3, xmm.rex << 2)
    if ret != None:
      return params['prefix'] + ret + w8(operands['imm8'])
  render = staticmethod(_render)


class mem16_simm8(MachineInstruction):
  signature = (mem16_t, simm8_t)
  opt_kw = ()
  
  def _render(params, operands):
    ret = common_memref(params['opcode'], operands['mem16'], params['modrm'])
    if ret != None:
      return [0x66] + ret + w8(operands['simm8'])
  render = staticmethod(_render)


class mem228(MachineInstruction):
  signature = (mem228_t,)
  opt_kw = ()
  
  def _render(params, operands):
    return common_memref(params['opcode'], operands['mem228'], params['modrm'])
  render = staticmethod(_render)


class mem32(MachineInstruction):
  signature = (mem32_t,)
  opt_kw = ()
  
  def _render(params, operands):
    return common_memref(params['opcode'], operands['mem32'], params['modrm'])
  render = staticmethod(_render)

  
class mem32_1(MachineInstruction):
  signature = (mem32_t, one_t)
  opt_kw = ()
  
  def _render(params, operands):
    return common_memref(params['opcode'], operands['mem32'], params['modrm'])
  render = staticmethod(_render)


class mem32_cl(MachineInstruction):
  signature = (mem32_t, cl_t)
  opt_kw = ()
  
  def _render(params, operands):
    return common_memref(params['opcode'], operands['mem32'], params['modrm'])
  render = staticmethod(_render)


class mem32_imm32(MachineInstruction):
  signature = (mem32_t, imm32_t)
  opt_kw = (lock_p,)
  
  def _render(params, operands):
    ret = common_memref(params['opcode'], operands['mem32'], params['modrm'])
    if ret != None:
      #if operands.has_key('lock') and operands['lock'] == True and ret is not None:
      #  return [lock_p.value] + ret + w32(operands['imm32'])
      return ret + w32(operands['imm32'])
  render = staticmethod(_render)


class mem32_imm8(MachineInstruction):
  signature = (mem32_t, imm8_t)
  opt_kw = ()
  
  def _render(params, operands):
    ret = common_memref(params['opcode'], operands['mem32'], params['modrm'])
    if ret != None:
      return ret + w8(operands['imm8'])
  render = staticmethod(_render)


class mem32_mmx(MachineInstruction):
  signature = (mem32_t, mmx_t)
  opt_kw = ()
  
  def _render(params, operands):
    return common_memref(params['opcode'], operands['mem32'], operands['mmx'].reg << 3)
  render = staticmethod(_render)


class mem32_reg32(MachineInstruction):
  signature = (mem32_t, reg32_t)
  opt_kw = (lock_p,)
  
  def _render(params, operands):
    reg32 = operands['reg32']
    ret = common_memref(params['opcode'], operands['mem32'], reg32.reg << 3, reg32.rex << 2)

    # TODO - only let instructions that support LOCK use it.
    # idea - validate_operands should pass self (the instruction) down to all
    # the check routines.  Then, x86PrefixOperand can check to make sure that
    # instruction is supported by the (lock) prefix.

    # TODO - is there a better way to do this as well?  Will need to duplicate
    # it in every operand combination function that supports lock.  Same thing
    # for all the other prefixes..
    # Should it go into common_memref?
    if operands.has_key('lock') and operands['lock'] == True and ret is not None:
      return [lock_p.value] + ret
    return ret
    #return common_memref32(params['opcode'], operands['mem32'], operands['reg32'].reg << 3)
  render = staticmethod(_render)


class mem32_reg32_cl(MachineInstruction):
  signature = (mem32_t, reg32_t, cl_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg32 = operands['reg32']
    return common_memref(params['opcode'], operands['mem32'], reg32.reg << 3, reg32.rex << 2)
  render = staticmethod(_render)


class mem32_reg32_imm32(MachineInstruction):
  signature = (mem32_t, reg32_t, imm32_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg32 = operands['reg32']
    ret = common_memref(params['opcode'], operands['mem32'], reg32.reg << 3, reg32.rex << 2)
    if ret != None:
      return ret + w32(operands['imm32'])
  render = staticmethod(_render)


class mem32_reg32_imm8(MachineInstruction):
  signature = (mem32_t, reg32_t, imm8_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg32 = operands['reg32']
    ret = common_memref(params['opcode'], operands['mem32'], reg32.reg << 3, reg32.rex << 2)
    if ret is not None:
      return ret + w8(operands['imm8'])
  render = staticmethod(_render)


class mem32_simm8(MachineInstruction):
  signature = (mem32_t, simm8_t)
  opt_kw = (lock_p,)
  
  def _render(params, operands):
    ret = common_memref(params['opcode'], operands['mem32'], params['modrm'])
    if ret is not None:
      if operands.has_key('lock') and operands['lock'] == True:
        return [lock_p.value] + ret + w8(operands['simm8'])
      return ret + w8(operands['simm8'])
  render = staticmethod(_render)


class mem32_xmm(MachineInstruction):
  signature = (mem32_t, xmm_t)
  opt_kw = ()
  
  def _render(params, operands):
    xmm = operands['xmm']
    ret = common_memref(params['opcode'], operands['mem32'], xmm.reg << 3, xmm.rex << 2)
    if ret != None:
      return params['prefix'] + ret
  render = staticmethod(_render)


class mem32_xmm_imm8(MachineInstruction):
  signature = (mem32_t, xmm_t, imm8_t)
  opt_kw = ()
  
  def _render(params, operands):
    xmm = operands['xmm']
    ret = common_memref(params['opcode'], operands['mem32'], xmm.reg << 3, xmm.rex << 2)
    if ret != None:
      return params['prefix'] + ret + w8(operands['imm8'])
  render = staticmethod(_render)


class mem4096(MachineInstruction):
  signature = (mem4096_t,)
  opt_kw = ()
  
  def _render(params, operands):
    return common_memref(params['opcode'], operands['mem4096'], params['modrm'])
  render = staticmethod(_render)


#class mem512(MachineInstruction):
#  signature = (mem512_t,)
#  opt_kw = ()
#  
#  def _render(params, operands):
#    return common_memref(params['opcode'], operands['mem512'], params['modrm'])
#  render = staticmethod(_render)


class mem64(MachineInstruction):
  signature = (mem64_t,)
  opt_kw = (lock_p,)
  
  def _render(params, operands):
    # TODO - there are a number of 64-bit default instructions that don't need
    # the 0x08 in the REX -- it doesn't hurt them to have it, but causes test
    # failures.
    #return common_memref(params['opcode'], operands['mem64'], params['modrm'])
    ret = common_memref(params['opcode'], operands['mem64'], params['modrm'], 0x08)
    if operands.has_key('lock') and operands['lock'] == True and ret is not None:
      return [lock_p.value] + ret
    return ret
  render = staticmethod(_render)


class mem64_1(MachineInstruction):
  signature = (mem64_t, one_t)
  opt_kw = ()
  
  def _render(params, operands):
    return common_memref(params['opcode'], operands['mem64'], params['modrm'], 0x08)
  render = staticmethod(_render)


# Note - this is a special version of mem64 for 32bit instructions (cmpxchg8b,
# call).
class mem64_32(MachineInstruction):
  signature = (mem64_t,)
  opt_kw = ()
 
  def _render(params, operands):
    return common_memref(params['opcode'], operands['mem64'], params['modrm'])
  render = staticmethod(_render)


class mem64_cl(MachineInstruction):
  signature = (mem64_t, cl_t)
  opt_kw = ()
  
  def _render(params, operands):
    return common_memref(params['opcode'], operands['mem64'], params['modrm'], 0x08)
  render = staticmethod(_render)


class mem64_imm32(MachineInstruction):
  signature = (mem64_t, imm32_t)
  opt_kw = ()
  
  def _render(params, operands):
    ret = common_memref(params['opcode'], operands['mem64'], params['modrm'], 0x08)
    if ret != None:
      return ret + w32(operands['imm32'])
  render = staticmethod(_render)


class mem64_imm8(MachineInstruction):
  signature = (mem64_t, imm8_t)
  opt_kw = ()
  
  def _render(params, operands):
    ret = common_memref(params['opcode'], operands['mem64'], params['modrm'], 0x08)
    if ret != None:
      return ret + w8(operands['imm8'])
  render = staticmethod(_render)


class mem64_reg64(MachineInstruction):
  signature = (mem64_t, reg64_t)
  opt_kw = (lock_p,)
  
  def _render(params, operands):
    reg64 = operands['reg64']
    ret = common_memref(params['opcode'], operands['mem64'], reg64.reg << 3, 0x08 | (reg64.rex << 2))
    if operands.has_key('lock') and operands['lock'] == True and ret is not None:
      return [lock_p.value] + ret
    return ret
  render = staticmethod(_render)


class mem64_reg64_cl(MachineInstruction):
  signature = (mem64_t, reg64_t, cl_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg64 = operands['reg64']
    return common_memref(params['opcode'], operands['mem64'], reg64.reg << 3, 0x08 | (reg64.rex << 2))
  render = staticmethod(_render)


class mem64_reg64_imm8(MachineInstruction):
  signature = (mem64_t, reg64_t, imm8_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg64 = operands['reg64']
    ret = common_memref(params['opcode'], operands['mem64'], reg64.reg << 3, 0x08 | (reg64.rex << 2))
    if ret != None:
      return ret + w8(operands['imm8'])
  render = staticmethod(_render)


class mem64_mmx(MachineInstruction):
  signature = (mem64_t, mmx_t)
  opt_kw = ()
  
  def _render(params, operands):
    return common_memref(params['opcode'], operands['mem64'], operands['mmx'].reg << 3)
  render = staticmethod(_render)


class mem64_simm8(MachineInstruction):
  signature = (mem64_t, simm8_t)
  opt_kw = ()
  
  def _render(params, operands):
    ret = common_memref(params['opcode'], operands['mem64'], params['modrm'], 0x08)
    if ret != None:
      return ret + w8(operands['simm8'])
  render = staticmethod(_render)


class mem64_xmm(MachineInstruction):
  signature = (mem64_t, xmm_t)
  opt_kw = ()
  
  def _render(params, operands):
    xmm = operands['xmm']
    ret = common_memref(params['opcode'], operands['mem64'], xmm.reg << 3, xmm.rex << 2)
    if ret != None:
      return params['prefix'] + ret
  render = staticmethod(_render)


class mem64_xmm_imm8(MachineInstruction):
  signature = (mem64_t, xmm_t, imm8_t)
  opt_kw = ()
  
  def _render(params, operands):
    xmm = operands['xmm']
    ret = common_memref(params['opcode'], operands['mem64'], xmm.reg << 3, xmm.rex << 2)
    if ret != None:
      return params['prefix'] + ret + w8(operands['imm8'])
  render = staticmethod(_render)


class mem752(MachineInstruction):
  signature = (mem752_t,)
  opt_kw = ()
  
  def _render(params, operands):
    return common_memref(params['opcode'], operands['mem752'], params['modrm'])
  render = staticmethod(_render)


class mem8(MachineInstruction):
  signature = (mem8_t,)
  opt_kw = ()
  
  def _render(params, operands):
    return common_memref(params['opcode'], operands['mem8'], params['modrm'])
  render = staticmethod(_render)
  

class mem8_1(MachineInstruction):
  signature = (mem8_t, one_t)
  opt_kw = ()
  
  def _render(params, operands):
    return common_memref(params['opcode'], operands['mem8'], params['modrm'])
  render = staticmethod(_render)


class mem8_cl(MachineInstruction):
  signature = (mem8_t, cl_t)
  opt_kw = ()
  
  def _render(params, operands):
    return common_memref(params['opcode'], operands['mem8'], params['modrm'])
  render = staticmethod(_render)


class mem8_imm8(MachineInstruction):
  signature = (mem8_t, imm8_t)
  opt_kw = ()
  
  def _render(params, operands):
    ret = common_memref(params['opcode'], operands['mem8'], params['modrm'])
    if ret != None:
      return ret + w8(operands['imm8'])
  render = staticmethod(_render)


class mem8_reg8(MachineInstruction):
  signature = (mem8_t, reg8_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg8 = operands['reg8']

    return common_memref(params['opcode'], operands['mem8'], reg8.reg << 3, reg8.rex << 2, (reg8 in reg8_rex_list))
  render = staticmethod(_render)


class mem8_xmm_imm8(MachineInstruction):
  signature = (mem8_t, xmm_t, imm8_t)
  opt_kw = ()
  
  def _render(params, operands):
    xmm = operands['xmm']
    ret = common_memref(params['opcode'], operands['mem8'], xmm.reg << 3, xmm.rex << 2)
    if ret != None:
      return params['prefix'] + ret + w8(operands['imm8'])
  render = staticmethod(_render)


class mem80(MachineInstruction):
  signature = (mem80_t,)
  opt_kw = ()
  
  def _render(params, operands):
    return common_memref(params['opcode'], operands['mem80'], params['modrm'])
  render = staticmethod(_render)


class mmx_imm8(MachineInstruction):
  signature = (mmx_t, imm8_t)
  opt_kw = ()
  
  def _render(params, operands):
    return params['opcode'] + [0xC0 | params['modrm'] | operands['mmx'].reg] + w8(operands['imm8'])
  render = staticmethod(_render)


class mmx_mem128(MachineInstruction):
  signature = (mmx_t, mem128_t)
  opt_kw = ()
  
  def _render(params, operands):
    ret = common_memref(params['opcode'], operands['mem128'], operands['mmx'].reg << 3)
    if ret != None and params.has_key('prefix'):
      return params['prefix'] + ret
    return ret
  render = staticmethod(_render)


class mmx_mem16_imm8(MachineInstruction):
  signature = (mmx_t, mem16_t, imm8_t)
  opt_kw = ()
  
  def _render(params, operands):
    ret =  common_memref(params['opcode'], operands['mem16'], operands['mmx'].reg << 3)
    if ret != None:
      return ret + [operands['imm8']]
  render = staticmethod(_render)


class mmx_mem32(MachineInstruction):
  signature = (mmx_t, mem32_t)
  opt_kw = ()
  
  def _render(params, operands):
    return common_memref(params['opcode'], operands['mem32'], operands['mmx'].reg << 3)
  render = staticmethod(_render)


class mmx_mem64(MachineInstruction):
  signature = (mmx_t, mem64_t)
  opt_kw = ()
  
  def _render(params, operands):
    ret = common_memref(params['opcode'], operands['mem64'], operands['mmx'].reg << 3)
    if ret != None and params.has_key('prefix'):
      return params['prefix'] + ret
    return ret
  render = staticmethod(_render)


class mmx_mem64_imm8(MachineInstruction):
  signature = (mmx_t, mem64_t, imm8_t)
  opt_kw = ()
  
  def _render(params, operands):
    ret = common_memref(params['opcode'], operands['mem64'], operands['mmx'].reg << 3)
    if ret != None:
      return ret + w8(operands['imm8'])
  render = staticmethod(_render)


class mmx_mmx(MachineInstruction):
  signature = (mmx_t('rd'), mmx_t('ra'))
  opt_kw = ()
  
  def _render(params, operands):
    return params['opcode'] + [0xC0 | (operands['rd'].reg << 3) | operands['ra'].reg]
  render = staticmethod(_render)


class mmx_mmx_imm8(MachineInstruction):
  signature = (mmx_t('rd'), mmx_t('ra'), imm8_t)
  opt_kw = ()
  
  def _render(params, operands):
    return params['opcode'] + [0xC0 | (operands['rd'].reg << 3) | operands['ra'].reg] + w8(operands['imm8'])
  render = staticmethod(_render)


class mmx_reg32(MachineInstruction):
  signature = (mmx_t, reg32_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg32 = operands['reg32']
    rex = [0x40 | reg32.rex]
    if rex == [0x40]:
      rex = []

    return rex + params['opcode'] + [0xC0 | (operands['mmx'].reg << 3) | reg32.reg]
  render = staticmethod(_render)


class mmx_reg32_imm8(MachineInstruction):
  signature = (mmx_t, reg32_t, imm8_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg32 = operands['reg32']
    rex = [0x40 | reg32.rex]
    if rex == [0x40]:
      rex = []

    return rex + params['opcode'] + [0xC0 | (operands['mmx'].reg << 3) | reg32.reg] + w8(operands['imm8'])
  render = staticmethod(_render)


class mmx_reg64(MachineInstruction):
  signature = (mmx_t, reg64_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg64 = operands['reg64']
    return [0x48 | reg64.rex] + params['opcode'] + [0xC0 | (operands['mmx'].reg << 3) | reg64.reg]
  render = staticmethod(_render)


class mmx_xmm(MachineInstruction):
  signature = (mmx_t, xmm_t)
  opt_kw = ()
  
  def _render(params, operands):
    xmm = operands['xmm']
    rex = [0x40 | xmm.rex]
    if rex == [0x40]:
      rex = []

    return params['prefix'] + rex + params['opcode'] + [0xC0 | (operands['mmx'].reg << 3) | operands['xmm'].reg]
  render = staticmethod(_render)


class no_op(MachineInstruction):
  signature = ()
  opt_kw = ()
  
  def _render(params, operands):
    return params['opcode']
  render = staticmethod(_render)


class rax_imm32(MachineInstruction):
  signature = (rax_t, imm32_t)
  opt_kw = ()
 
  # TODO - REX prefix is always 0x48, move it into the opcode? 
  def _render(params, operands):
    return [0x48] + params['opcode'] + w32(operands['imm32'])
  render = staticmethod(_render)

  
class rax_reg64(MachineInstruction):
  signature = (rax_t, reg64_t)
  opt_kw = ()
  
  def _render(params, operands):
    opcode = params['opcode']
    reg64 = operands['reg64']
    return [0x48 | reg64.rex] + opcode[:-1] + [opcode[-1] + reg64.reg]
  render = staticmethod(_render)

  
class reg16(MachineInstruction):
  signature = (reg16_t,)
  opt_kw = ()
  
  def _render(params, operands):
    opcode = params['opcode']
    modrm = params['modrm']
    reg16 = operands['reg16']
    rex = [0x40 | reg16.rex]
    if rex == [0x40]:
      rex = []

    if modrm != None:
      return [0x66] + rex + opcode + [0xC0 | modrm | reg16.reg]
    else:
      return [0x66] + rex + opcode[:-1] + [opcode[-1] + reg16.reg]
  render = staticmethod(_render)    


class reg16_1(MachineInstruction):
  signature = (reg16_t, one_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg16 = operands['reg16']
    rex = [0x40 | reg16.rex]
    if rex == [0x40]:
      rex = []

    return [0x66] + rex + params['opcode'] + [0xC0 | params['modrm'] | reg16.reg]
  render = staticmethod(_render)

  
class reg16_ax(MachineInstruction):
  signature = (reg16_t, ax_t)
  opt_kw = ()
  
  def _render(params, operands):
    opcode = params['opcode']
    reg16 = operands['reg16']
    rex = [0x40 | reg16.rex]
    if rex == [0x40]:
      rex = []

    return [0x66] + rex + opcode[:-1] + [opcode[-1] + reg16.reg]
  render = staticmethod(_render)

  
class reg16_cl(MachineInstruction):
  signature = (reg16_t, cl_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg16 = operands['reg16']
    rex = [0x40 | reg16.rex]
    if rex == [0x40]:
      rex = []

    return [0x66] + rex + params['opcode'] + [0xC0 | params['modrm'] | reg16.reg]
  render = staticmethod(_render)

  
class reg16_imm16(MachineInstruction):
  signature = (reg16_t, imm16_t)
  opt_kw = ()
  
  def _render(params, operands):
    opcode = params['opcode']
    modrm = params['modrm']
    reg16 = operands['reg16']
    rex = [0x40 | reg16.rex]
    if rex == [0x40]:
      rex = []

    if modrm != None:
      return [0x66] + rex + opcode + [0xC0 | modrm | reg16.reg] + w16(operands['imm16'])
    else:
      return [0x66] + rex + opcode[:-1] + [opcode[-1] + reg16.reg] + w16(operands['imm16'])
  render = staticmethod(_render)

  
class reg16_imm8(MachineInstruction):
  signature = (reg16_t, imm8_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg16 = operands['reg16']
    rex = [0x40 | reg16.rex]
    if rex == [0x40]:
      rex = []

    return [0x66] + rex + params['opcode'] + [0xC0 | params['modrm'] | reg16.reg] + w8(operands['imm8'])
  render = staticmethod(_render)

  
class reg16_mem(MachineInstruction):
  signature = (reg16_t, mem_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg16 = operands['reg16']
    ret = common_memref(params['opcode'], operands['mem'], params['modrm'] | (reg16.reg << 3), reg16.rex << 2)
    if ret != None:
      return [0x66] + ret
  render = staticmethod(_render)

  
class reg16_mem8(MachineInstruction):
  signature = (reg16_t, mem8_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg16 = operands['reg16']
    ret = common_memref(params['opcode'], operands['mem8'], reg16.reg << 3, reg16.rex << 2)
    if ret != None:
      return [0x66] + ret
  render = staticmethod(_render)

class reg16_mem16(MachineInstruction):
  signature = (reg16_t, mem16_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg16 = operands['reg16']
    ret = common_memref(params['opcode'], operands['mem16'], reg16.reg << 3, reg16.rex << 2)
    if ret != None:
      if params.has_key('prefix'):
        return [0x66] + params['prefix'] + ret
      return [0x66] + ret
  render = staticmethod(_render)


class reg16_mem16_imm16(MachineInstruction):
  signature = (reg16_t, mem16_t, imm16_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg16 = operands['reg16']
    ret = common_memref(params['opcode'], operands['mem16'], reg16.reg << 3, reg16.rex << 2)
    if ret != None:
      return [0x66] + ret + w16(operands['imm16'])
  render = staticmethod(_render)


class reg16_mem16_simm8(MachineInstruction):
  signature = (reg16_t, mem16_t, simm8_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg16 = operands['reg16']
    ret = common_memref(params['opcode'], operands['mem16'], reg16.reg << 3, reg16.rex << 2)
    if ret != None:
      return [0x66] + ret + w8(operands['simm8'])
  render = staticmethod(_render)


class reg16_reg16(MachineInstruction):
  signature = (reg16_t('rd'), reg16_t('ra'))
  opt_kw = ()
  
  def _render(params, operands):
    rd = operands['rd']
    ra = operands['ra']
    rex = [0x40 | (ra.rex << 2) | rd.rex]
    if rex == [0x40]:
      rex = []

    ret = rex + params['opcode'] + [0xC0 | (ra.reg << 3) | rd.reg]
    if ret != None and params.has_key('prefix'):
      return [0x66] + params['prefix'] + ret
    return [0x66] + ret
  render = staticmethod(_render)

  
class reg16_reg16_cl(MachineInstruction):
  signature = (reg16_t('rd'), reg16_t('ra'), cl_t)
  opt_kw = ()
  
  def _render(params, operands):
    rd = operands['rd']
    ra = operands['ra']
    rex = [0x40 | (ra.rex << 2) | rd.rex]
    if rex == [0x40]:
      rex = []

    return [0x66] + rex + params['opcode'] + [0xC0 | (ra.reg << 3) | rd.reg]
  render = staticmethod(_render)

  
class reg16_reg16_imm16(MachineInstruction):
  signature = (reg16_t('rd'), reg16_t('ra'), imm16_t)
  opt_kw = ()
  
  def _render(params, operands):
    rd = operands['rd']
    ra = operands['ra']
    rex = [0x40 | (rd.rex << 2) | ra.rex]
    if rex == [0x40]:
      rex = []

    return [0x66] + rex + params['opcode'] + [0xC0 | (rd.reg << 3) | ra.reg] + w16(operands['imm16'])
  render = staticmethod(_render)

  
class reg16_reg16_imm8(MachineInstruction):
  signature = (reg16_t('rd'), reg16_t('ra'), imm8_t)
  opt_kw = ()
  
  def _render(params, operands):
    rd = operands['rd']
    ra = operands['ra']
    rex = [0x40 | (ra.rex << 2) | rd.rex]
    if rex == [0x40]:
      rex = []

    return [0x66] + rex + params['opcode'] + [0xC0 | (ra.reg << 3) | rd.reg] + w8(operands['imm8'])
  render = staticmethod(_render)

  
class reg16_reg16_simm8_rev(MachineInstruction):
  signature = (reg16_t('rd'), reg16_t('ra'), simm8_t)
  opt_kw = ()
  
  def _render(params, operands):
    rd = operands['rd']
    ra = operands['ra']
    rex = [0x40 | (rd.rex << 2) | ra.rex]
    if rex == [0x40]:
      rex = []

    return [0x66] + rex + params['opcode'] + [0xC0 | (rd.reg << 3) | ra.reg] + w8(operands['simm8'])
  render = staticmethod(_render)

  
# Reverse the operands, for CMOVcc
class reg16_reg16_rev(MachineInstruction):
  signature = (reg16_t('rd'), reg16_t('ra'))
  opt_kw = ()
  
  def _render(params, operands):
    rd = operands['rd']
    ra = operands['ra']
    rex = [0x40 | (rd.rex << 2) | ra.rex]
    if rex == [0x40]:
      rex = []

    return [0x66] + rex + params['opcode'] + [0xC0 | (rd.reg << 3) | ra.reg]
  render = staticmethod(_render)

  
class reg16_reg8(MachineInstruction):
  signature = (reg16_t, reg8_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg16 = operands['reg16']
    reg8 = operands['reg8']
    rex = [0x40 | (reg16.rex << 2) | reg8.rex]
    if rex == [0x40] and not reg8 in reg8_rex_list:
      rex = []

    return [0x66] + rex + params['opcode'] + [0xC0 | (reg16.reg << 3) | reg8.reg]
  render = staticmethod(_render)

  
class reg16_simm8(MachineInstruction):
  signature = (reg16_t, simm8_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg16 = operands['reg16']
    rex = [0x40 | reg16.rex]
    if rex == [0x40]:
      rex = []

    return [0x66] + rex + params['opcode'] + [0xC0 | params['modrm'] | reg16.reg] + w8(operands['simm8'])
  render = staticmethod(_render)

  
class reg32(MachineInstruction):
  signature = (reg32_t,)
  opt_kw = ()
  
  def _render(params, operands):
    opcode = params['opcode']
    modrm = params['modrm']
    reg32 = operands['reg32']
    rex = [0x40 | reg32.rex]
    if rex == [0x40]:
      rex = []

    if modrm != None:
      return rex + opcode + [0xC0 | modrm | operands['reg32'].reg]
    else:
      return rex + opcode[:-1] + [opcode[-1] + operands['reg32'].reg]
  render = staticmethod(_render)

  
class reg32_1(MachineInstruction):
  signature = (reg32_t, one_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg32 = operands['reg32']
    rex = [0x40 | reg32.rex]
    if rex == [0x40]:
      rex = []

    return rex + params['opcode'] + [0xC0 | params['modrm'] | reg32.reg]
  render = staticmethod(_render)

  
class reg32_eax(MachineInstruction):
  signature = (reg32_t, eax_t)
  opt_kw = ()
  
  def _render(params, operands):
    opcode = params['opcode']
    reg32 = operands['reg32']
    rex = [0x40 | reg32.rex]
    if rex == [0x40]:
      rex = []

    return rex + opcode[:-1] + [opcode[-1] + reg32.reg]
  render = staticmethod(_render)

  
class reg32_cl(MachineInstruction):
  signature = (reg32_t, cl_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg32 = operands['reg32']
    rex = [0x40 | reg32.rex]
    if rex == [0x40]:
      rex = []

    return rex + params['opcode'] + [0xC0 | params['modrm'] | reg32.reg]
  render = staticmethod(_render)


class reg32_imm32(MachineInstruction):
  signature = (reg32_t, imm32_t)
  opt_kw = ()
  
  def _render(params, operands):
    opcode = params['opcode']
    modrm = params['modrm']
    reg32 = operands['reg32']
    rex = [0x40 | reg32.rex]
    if rex == [0x40]:
      rex = []

    if modrm != None:
      return rex + opcode + [0xC0 | modrm | operands['reg32'].reg] + w32(operands['imm32'])
    else:
      return rex + opcode[:-1] + [opcode[-1] + operands['reg32'].reg] + w32(operands['imm32'])
  render = staticmethod(_render)

  
class reg32_imm8(MachineInstruction):
  signature = (reg32_t, imm8_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg32 = operands['reg32']
    rex = [0x40 | reg32.rex]
    if rex == [0x40]:
      rex = []

    return rex + params['opcode'] + [0xC0 | params['modrm'] | operands['reg32'].reg] + w8(operands['imm8'])
  render = staticmethod(_render)

  
class reg32_mem(MachineInstruction):
  signature = (reg32_t, mem_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg32 = operands['reg32']
    return common_memref(params['opcode'], operands['mem'], params['modrm'] | (reg32.reg << 3), reg32.rex << 2)
  render = staticmethod(_render)

  
class reg32_mem16(MachineInstruction):
  signature = (reg32_t, mem16_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg32 = operands['reg32']
    return params['prefix'] + common_memref(params['opcode'], operands['mem16'], reg32.reg << 3, reg32.rex << 2)
  render = staticmethod(_render)


class reg32_mem32(MachineInstruction):
  signature = (reg32_t, mem32_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg32 = operands['reg32']
    ret = common_memref(params['opcode'], operands['mem32'], reg32.reg << 3, reg32.rex << 2)
    if ret != None and params.has_key('prefix'):
      return params['prefix'] + ret
    return ret
  render = staticmethod(_render)


class reg32_mem32_imm32(MachineInstruction):
  signature = (reg32_t, mem32_t, imm32_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg32 = operands['reg32']
    ret = common_memref(params['opcode'], operands['mem32'], reg32.reg << 3, reg32.rex << 2)
    if ret != None:
      return ret + w32(operands['imm32'])
  render = staticmethod(_render)

 
class reg32_mem32_simm8(MachineInstruction):
  signature = (reg32_t, mem32_t, simm8_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg32 = operands['reg32']
    ret = common_memref(params['opcode'], operands['mem32'], reg32.reg << 3, reg32.rex << 2)
    if ret != None:
      return ret + w8(operands['simm8'])
  render = staticmethod(_render)

 
class reg32_mem64(MachineInstruction):
  signature = (reg32_t, mem64_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg32 = operands['reg32']
    ret = common_memref(params['opcode'], operands['mem64'], reg32.reg << 3, reg32.rex << 2)
    if ret != None and params.has_key('prefix'):
      return params['prefix'] + ret
    return ret
  render = staticmethod(_render)


class reg32_mem8(MachineInstruction):
  signature = (reg32_t, mem8_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg32 = operands['reg32']
    return params['prefix'] + common_memref(params['opcode'], operands['mem8'], reg32.reg << 3, reg32.rex << 2)
  render = staticmethod(_render)      


class reg32_mmx(MachineInstruction):
  signature = (reg32_t, mmx_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg32 = operands['reg32']
    rex = [0x40 | reg32.rex << 2]
    if rex == [0x40]:
      rex = []

    return rex + params['opcode'] + [0xC0 | (reg32.reg << 3) | operands['mmx'].reg]
  render = staticmethod(_render)


# Reversed operands for movd
class reg32_mmx_rev(MachineInstruction):
  signature = (reg32_t, mmx_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg32 = operands['reg32']
    rex = [0x40 | reg32.rex]
    if rex == [0x40]:
      rex = []

    return rex + params['opcode'] + [0xC0 | reg32.reg | (operands['mmx'].reg << 3)]
  render = staticmethod(_render)


class reg32_mmx_imm8(MachineInstruction):
  signature = (reg32_t, mmx_t, imm8_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg32 = operands['reg32']
    rex = [0x40 | reg32.rex << 2]
    if rex == [0x40]:
      rex = []

    return rex + params['opcode'] + [0xC0 | (reg32.reg << 3) | operands['mmx'].reg] + w8(operands['imm8'])
  render = staticmethod(_render)


class reg32_reg16(MachineInstruction):
  signature = (reg32_t, reg16_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg32 = operands['reg32']
    reg16 = operands['reg16']
    rex = [0x40 | reg32.rex << 2 | reg16.rex]
    if rex == [0x40]:
      rex = []

    return params['prefix'] + rex + params['opcode'] + [0xC0 | (reg32.reg << 3) | reg16.reg]
  render = staticmethod(_render)

  
class reg32_reg32(MachineInstruction):
  signature = (reg32_t('rd'), reg32_t('ra'))
  opt_kw = ()
  
  def _render(params, operands):
    rd = operands['rd']
    ra = operands['ra']
    rex = [0x40 | (ra.rex << 2) | rd.rex]
    if rex == [0x40]:
      rex = []

    ret = rex + params['opcode'] + [0xC0 | (ra.reg << 3) | rd.reg]
    if ret != None and params.has_key('prefix'):
      return params['prefix'] + ret
    return ret
  render = staticmethod(_render)

  
class reg32_reg32_cl(MachineInstruction):
  signature = (reg32_t('rd'), reg32_t('ra'), cl_t)
  opt_kw = ()
  
  def _render(params, operands):
    rd = operands['rd']
    ra = operands['ra']
    rex = [0x40 | (ra.rex << 2) | rd.rex]
    if rex == [0x40]:
      rex = []

    return rex + params['opcode'] + [0xC0 | (ra.reg << 3) | rd.reg]
  render = staticmethod(_render)

  
class reg32_reg32_imm32(MachineInstruction):
  signature = (reg32_t('rd'), reg32_t('ra'), imm32_t)
  opt_kw = ()
  
  def _render(params, operands):
    rd = operands['rd']
    ra = operands['ra']
    rex = [0x40 | (rd.rex << 2) | ra.rex]
    if rex == [0x40]:
      rex = []

    return rex + params['opcode'] + [0xC0 | (rd.reg << 3) | ra.reg] + w32(operands['imm32'])
  render = staticmethod(_render)

  
class reg32_reg32_simm8_rev(MachineInstruction):
  signature = (reg32_t('rd'), reg32_t('ra'), simm8_t)
  opt_kw = ()
  
  def _render(params, operands):
    rd = operands['rd']
    ra = operands['ra']
    rex = [0x40 | (ra.rex << 2) | rd.rex]
    if rex == [0x40]:
      rex = []

    return rex + params['opcode'] + [0xC0 | (ra.reg << 3) | rd.reg] + w8(operands['simm8'])
  render = staticmethod(_render)

  
class reg32_reg32_imm8(MachineInstruction):
  signature = (reg32_t('rd'), reg32_t('ra'), imm8_t)
  opt_kw = ()
  
  def _render(params, operands):
    rd = operands['rd']
    ra = operands['ra']
    rex = [0x40 | (ra.rex << 2) | rd.rex]
    if rex == [0x40]:
      rex = []

    return rex + params['opcode'] + [0xC0 | (ra.reg << 3) | rd.reg] + w8(operands['imm8'])
  render = staticmethod(_render)


# Reversed operands for CMOVcc
class reg32_reg32_rev(MachineInstruction):
  signature = (reg32_t('rd'), reg32_t('ra'))
  opt_kw = ()
  
  def _render(params, operands):
    rd = operands['rd']
    ra = operands['ra']
    rex = [0x40 | (rd.rex << 2) | ra.rex]
    if rex == [0x40]:
      rex = []

    return rex + params['opcode'] + [0xC0 | (rd.reg << 3) | ra.reg]
  render = staticmethod(_render)

  
class reg32_reg8(MachineInstruction):
  signature = (reg32_t, reg8_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg32 = operands['reg32']
    reg8 = operands['reg8']
    rex = [0x40 | (reg32.rex << 2) | reg8.rex]
    if rex == [0x40] and not reg8 in reg8_rex_list:
      rex = []

    return params['prefix'] + rex + params['opcode'] + [0xC0 | (reg32.reg << 3) | reg8.reg]
  render = staticmethod(_render)      


class reg32_simm8(MachineInstruction):
  signature = (reg32_t, simm8_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg32 = operands['reg32']
    rex = [0x40 | reg32.rex]
    if rex == [0x40]:
      rex = []

    return rex + params['opcode'] + [0xC0 | params['modrm'] | operands['reg32'].reg] + w8(operands['simm8'])
  render = staticmethod(_render)

  
class reg32_xmm(MachineInstruction):
  signature = (reg32_t, xmm_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg32 = operands['reg32']
    xmm = operands['xmm']
    #rex = [0x40 | xmm.rex << 2 | reg32.rex]
    rex = [0x40 | xmm.rex | (reg32.rex << 2)]
    if rex == [0x40]:
      rex = []

    # great.. more reversed operands
    return params['prefix'] + rex + params['opcode'] + [0xC0 | xmm.reg | (reg32.reg << 3)]
    #return params['prefix'] + rex + params['opcode'] + [0xC0 | (xmm.reg << 3) | reg32.reg]
  render = staticmethod(_render)


class reg32_xmm_imm8(MachineInstruction):
  signature = (reg32_t, xmm_t, imm8_t)
  opt_kw = ()

  def _render(params, operands):
    reg32 = operands['reg32']
    xmm = operands['xmm']
    rex = [0x40 | xmm.rex | (reg32.rex << 2)]
    if rex == [0x40]:
      rex = []

    return params['prefix'] + rex + params['opcode'] + [0xC0 | (reg32.reg << 3) | xmm.reg] + w8(operands['imm8'])
  render = staticmethod(_render)


class reg32_xmm_imm8_rev(MachineInstruction):
  signature = (reg32_t, xmm_t, imm8_t)
  opt_kw = ()

  def _render(params, operands):
    reg32 = operands['reg32']
    xmm = operands['xmm']
    rex = [0x40 | xmm.rex << 2 | reg32.rex]
    if rex == [0x40]:
      rex = []

    return params['prefix'] + rex + params['opcode'] + [0xC0 | reg32.reg | (xmm.reg << 3)] + w8(operands['imm8'])
  render = staticmethod(_render)


# Reversed operands for movd
class reg32_xmm_rev(MachineInstruction):
  signature = (reg32_t, xmm_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg32 = operands['reg32']
    xmm = operands['xmm']
    rex = [0x40 | (xmm.rex << 2) | reg32.rex]
    #rex = [0x40 | xmm.rex | reg32.rex << 2]
    if rex == [0x40]:
      rex = []

    # great.. more reversed operands
    #return params['prefix'] + rex + params['opcode'] + [0xC0 | xmm.reg | (reg32.reg << 3)]
    return params['prefix'] + rex + params['opcode'] + [0xC0 | (xmm.reg << 3) | reg32.reg]
  render = staticmethod(_render)


class reg64(MachineInstruction):
  signature = (reg64_t,)
  opt_kw = ()
  
  def _render(params, operands):
    opcode = params['opcode']
    modrm = params['modrm']
    reg64 = operands['reg64']

    if modrm != None:
      if reg64.rex != None: # and params['rex'] == True:
        return [0x48 | reg64.rex] + opcode + [0xC0 | modrm | reg64.reg]
      return opcode + [0xC0 | modrm | reg64.reg]
    else:
      if reg64.rex != None: # and params['rex'] == True:
        return [0x48 | reg64.rex] + opcode[:-1] + [opcode[-1] + reg64.reg]
      return opcode[:-1] + [opcode[-1] + reg64.reg]
  render = staticmethod(_render)

  
class reg64_1(MachineInstruction):
  signature = (reg64_t, one_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg64 = operands['reg64']
    return [0x48 | reg64.rex] + params['opcode'] + [0xC0 | params['modrm'] | reg64.reg]
  render = staticmethod(_render)

  
class reg64_cl(MachineInstruction):
  signature = (reg64_t, cl_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg64 = operands['reg64']
    return [0x48 | reg64.rex] + params['opcode'] + [0xC0 | params['modrm'] | reg64.reg]
  render = staticmethod(_render)


class reg64_imm32(MachineInstruction):
  signature = (reg64_t, imm32_t)
  opt_kw = ()
 
  def _render(params, operands):
    opcode = params['opcode']
    modrm = params['modrm']
    reg64 = operands['reg64']

    if modrm != None:
      return [0x48 | reg64.rex] + opcode + [0xC0 | modrm | reg64.reg] + w32(operands['imm32'])
    else:
      return [0x48 | reg64.rex] + opcode[:-1] + [opcode[-1] + reg64.reg] + w32(operands['imm32'])
  render = staticmethod(_render)

  
class reg64_imm64(MachineInstruction):
  signature = (reg64_t, imm64_t)
  opt_kw = ()

  def _render(params, operands):
    opcode = params['opcode']
    reg64 = operands['reg64']

    return [0x48 | reg64.rex] + opcode[:-1] + [opcode[-1] + reg64.reg] + w64(operands['imm64'])
  render = staticmethod(_render)

  
class reg64_imm8(MachineInstruction):
  signature = (reg64_t, imm8_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg64 = operands['reg64']
    return [0x48 | reg64.rex] + params['opcode'] + [0xC0 | params['modrm'] | reg64.reg] + w8(operands['imm8'])
  render = staticmethod(_render)

  
class reg64_mem(MachineInstruction):
  signature = (reg64_t, mem_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg64 = operands['reg64']
    return common_memref(params['opcode'], operands['mem'], params['modrm'] | (reg64.reg << 3), 0x08 | (reg64.rex << 2))
  render = staticmethod(_render)

  
class reg64_mem16(MachineInstruction):
  signature = (reg64_t, mem16_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg64 = operands['reg64']
    return common_memref(params['opcode'], operands['mem16'], reg64.reg << 3, 0x08 | (reg64.rex << 2))
  render = staticmethod(_render)      


class reg64_mem32(MachineInstruction):
  signature = (reg64_t, mem32_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg64 = operands['reg64']
    return common_memref(params['opcode'], operands['mem32'], reg64.reg << 3, 0x08 | (reg64.rex << 2))
  render = staticmethod(_render)      


class reg64_mem64(MachineInstruction):
  signature = (reg64_t, mem64_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg64 = operands['reg64']

    # Some SSE instructions have/need a prefix byte, but general instructions don't.
    ret = common_memref(params['opcode'], operands['mem64'], reg64.reg << 3, 0x08 | reg64.rex << 2)
    if ret != None and params.has_key('prefix'):
      return params['prefix'] + ret
    return ret
  render = staticmethod(_render)


class reg64_mem64_imm32(MachineInstruction):
  signature = (reg64_t, mem64_t, imm32_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg64 = operands['reg64']
    ret = common_memref(params['opcode'], operands['mem64'], reg64.reg << 3, 0x08 | (reg64.rex << 2))
    if ret != None:
      return ret + w32(operands['imm32'])
  render = staticmethod(_render)

 
class reg64_mem64_simm8(MachineInstruction):
  signature = (reg64_t, mem64_t, simm8_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg64 = operands['reg64']
    ret = common_memref(params['opcode'], operands['mem64'], reg64.reg << 3, 0x08 | (reg64.rex << 2))
    if ret != None:
      return ret + w8(operands['simm8'])
  render = staticmethod(_render)

 
class reg64_mem8(MachineInstruction):
  signature = (reg64_t, mem8_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg64 = operands['reg64']
    return params['prefix'] + common_memref(params['opcode'], operands['mem8'], reg64.reg << 3, 0x08 | (reg64.rex << 2))
  render = staticmethod(_render)      


class reg64_mmx(MachineInstruction):
  signature = (reg64_t, mmx_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg64 = operands['reg64']
    return [0x48 | reg64.rex << 2] + params['opcode'] + [0xC0 | (reg64.reg << 3) | operands['mmx'].reg]
  render = staticmethod(_render)


class reg64_mmx_imm8(MachineInstruction):
  signature = (reg64_t, mmx_t, imm8_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg64 = operands['reg64']
    return [0x48 | reg64.rex << 2] + params['opcode'] + [0xC0 | (reg64.reg << 3) | operands['mmx'].reg] + w8(operands['imm8'])
  render = staticmethod(_render)


class reg64_mmx_rev(MachineInstruction):
  signature = (reg64_t, mmx_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg64 = operands['reg64']
    return [0x48 | reg64.rex] + params['opcode'] + [0xC0 | reg64.reg | (operands['mmx'].reg << 3)]
  render = staticmethod(_render)


class reg64_rax(MachineInstruction):
  signature = (reg64_t, rax_t)
  opt_kw = ()
  
  def _render(params, operands):
    opcode = params['opcode']
    reg64 = operands['reg64']
    return [0x48 | reg64.rex] + opcode[:-1] + [opcode[-1] + reg64.reg]
  render = staticmethod(_render)

  
class reg64_reg16(MachineInstruction):
  signature = (reg64_t, reg16_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg64 = operands['reg64']
    reg16 = operands['reg16']
    return [0x48 | reg64.rex << 2 | reg16.rex] + params['opcode'] + [0xC0 | (reg64.reg << 3) | reg16.reg]
  render = staticmethod(_render)      


class reg64_reg32(MachineInstruction):
  signature = (reg64_t, reg32_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg64 = operands['reg64']
    reg32 = operands['reg32']
    return [0x48 | reg64.rex << 2 | reg32.rex] + params['opcode'] + [0xC0 | (reg64.reg << 3) | reg32.reg]
  render = staticmethod(_render)      


class reg64_reg64(MachineInstruction):
  signature = (reg64_t('rd'), reg64_t('ra'))
  opt_kw = ()
  
  def _render(params, operands):
    ra = operands['ra']
    rd = operands['rd']

    ret = [0x48 | (ra.rex << 2) | rd.rex] + params['opcode'] + [0xC0 | (ra.reg << 3) | rd.reg]
    if ret != None and params.has_key('prefix'):
      return params['prefix'] + ret
    return ret
  render = staticmethod(_render)


class reg64_reg64_cl(MachineInstruction):
  signature = (reg64_t('rd'), reg64_t('ra'), cl_t)
  opt_kw = ()
  
  def _render(params, operands):
    ra = operands['ra']
    rd = operands['rd']

    return [0x48 | (ra.rex << 2) | rd.rex] + params['opcode'] + [0xC0 | (ra.reg << 3) | rd.reg]
    #return params['opcode'] + [0xC0 | (operands['rd'].reg << 3) | operands['ra'].reg]
  render = staticmethod(_render)

  
class reg64_reg64_imm32(MachineInstruction):
  signature = (reg64_t('rd'), reg64_t('ra'), imm32_t)
  opt_kw = ()
  
  def _render(params, operands):
    ra = operands['ra']
    rd = operands['rd']

    return [0x48 | (rd.rex << 2) | ra.rex] + params['opcode'] + [0xC0 | (rd.reg << 3) | ra.reg] + w32(operands['imm32'])
  render = staticmethod(_render)

  
class reg64_reg64_imm8(MachineInstruction):
  signature = (reg64_t('rd'), reg64_t('ra'), imm8_t)
  opt_kw = ()
  
  def _render(params, operands):
    ra = operands['ra']
    rd = operands['rd']
    
    return [0x48 | (ra.rex << 2) | rd.rex] + params['opcode'] + [0xC0 | (ra.reg << 3) | rd.reg] + w8(operands['imm8'])
  render = staticmethod(_render)


# Reverse the two register operands, for imul 
class reg64_reg64_simm8_rev(MachineInstruction):
  signature = (reg64_t('rd'), reg64_t('ra'), simm8_t)
  opt_kw = ()
  
  def _render(params, operands):
    ra = operands['ra']
    rd = operands['rd']
    
    return [0x48 | (rd.rex << 2) | ra.rex] + params['opcode'] + [0xC0 | (rd.reg << 3) | ra.reg] + w8(operands['simm8'])
  render = staticmethod(_render)

  
# Reverse the operands, for CMOVcc
class reg64_reg64_rev(MachineInstruction):
  signature = (reg64_t('rd'), reg64_t('ra'))
  opt_kw = ()
  
  def _render(params, operands):
    ra = operands['ra']
    rd = operands['rd']

    return [0x48 | (rd.rex << 2) | ra.rex] + params['opcode'] + [0xC0 | (rd.reg << 3) | ra.reg]
  render = staticmethod(_render)


class reg64_reg8(MachineInstruction):
  signature = (reg64_t, reg8_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg64 = operands['reg64']
    reg8 = operands['reg8']
    return params['prefix'] + [0x48 | reg64.rex << 2 | reg8.rex] + params['opcode'] + [0xC0 | (reg64.reg << 3) | reg8.reg]
  render = staticmethod(_render)      


class reg64_simm8(MachineInstruction):
  signature = (reg64_t, simm8_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg64 = operands['reg64']
    return [0x48 | reg64.rex] + params['opcode'] + [0xC0 | params['modrm'] | reg64.reg] + w8(operands['simm8'])
  render = staticmethod(_render)

  
class reg64_xmm(MachineInstruction):
  signature = (reg64_t, xmm_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg64 = operands['reg64']
    xmm = operands['xmm']
    return params['prefix'] + [0x48 | (reg64.rex << 2) | xmm.rex] + params['opcode'] + [0xC0 | xmm.reg | (reg64.reg << 3)]
  render = staticmethod(_render)


class reg64_xmm_imm8(MachineInstruction):
  signature = (reg64_t, xmm_t, imm8_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg64 = operands['reg64']
    xmm = operands['xmm']
    return params['prefix'] + [0x48 | (reg64.rex << 2) | xmm.rex] + params['opcode'] + [0xC0 | xmm.reg | (reg64.reg << 3)] + w8(operands['imm8'])
  render = staticmethod(_render)


# Reversed operands for extractps
class reg64_xmm_imm8_rev(MachineInstruction):
  signature = (reg64_t, xmm_t, imm8_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg64 = operands['reg64']
    xmm = operands['xmm']
    return params['prefix'] + [0x48 | reg64.rex | (xmm.rex << 2)] + params['opcode'] + [0xC0 | (xmm.reg << 3) | reg64.reg] + w8(operands['imm8'])
  render = staticmethod(_render)


# Reversed operands for movd
class reg64_xmm_rev(MachineInstruction):
  signature = (reg64_t, xmm_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg64 = operands['reg64']
    xmm = operands['xmm']
    return params['prefix'] + [0x48 | reg64.rex | (xmm.rex << 2)] + params['opcode'] + [0xC0 | (xmm.reg << 3) | reg64.reg]
  render = staticmethod(_render)


class reg8(MachineInstruction):
  signature = (reg8_t,)
  opt_kw = ()

  def _render(params, operands):
    reg8 = operands['reg8']
    rex = [0x40 | reg8.rex]
    if rex == [0x40] and not reg8 in reg8_rex_list:
      rex = []

    return rex + params['opcode'] + [0xC0 | params['modrm'] | reg8.reg]
  render = staticmethod(_render)

  
class reg8_1(MachineInstruction):
  signature = (reg8_t, one_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg8 = operands['reg8']
    rex = [0x40 | reg8.rex]
    if rex == [0x40] and not reg8 in reg8_rex_list:
      rex = []

    return rex + params['opcode'] + [0xC0 | params['modrm'] | reg8.reg]
  render = staticmethod(_render)

  
class reg8_imm8(MachineInstruction):
  signature = (reg8_t, imm8_t)
  opt_kw = ()
  
  def _render(params, operands):
    opcode = params['opcode']
    modrm = params['modrm']
    reg8 = operands['reg8']
    rex = [0x40 | reg8.rex]
    if rex == [0x40] and not reg8 in reg8_rex_list:
      rex = []

    if modrm != None:
      return rex + opcode + [0xC0 | modrm | reg8.reg] + w8(operands['imm8'])
    else:
      return rex + opcode[:-1] + [opcode[-1] + reg8.reg] + w8(operands['imm8'])
  render = staticmethod(_render)
  

class reg8_cl(MachineInstruction):
  signature = (reg8_t, cl_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg8 = operands['reg8']
    rex = [0x40 | reg8.rex]
    if rex == [0x40] and not reg8 in reg8_rex_list:
      rex = []

    return rex + params['opcode'] + [0xC0 | params['modrm'] | reg8.reg]
  render = staticmethod(_render)

  
class reg8_mem8(MachineInstruction):
  signature = (reg8_t, mem8_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg8 = operands['reg8']
    force_rex = False
    return common_memref(params['opcode'], operands['mem8'], reg8.reg << 3, reg8.rex << 2, (reg8 in reg8_rex_list))
  render = staticmethod(_render)


class reg8_reg8(MachineInstruction):
  signature = (reg8_t('rd'), reg8_t('ra'))
  opt_kw = ()
  
  def _render(params, operands):
    rd = operands['rd']
    ra = operands['ra']
    rex = [0x40 | (ra.rex << 2) | rd.rex]
    if rex == [0x40] and not rd in reg8_rex_list and not ra in reg8_rex_list:
      rex = []

    return rex + params['opcode'] + [0xC0 | (ra.reg << 3) | rd.reg]
  render = staticmethod(_render)


#class rel16off(MachineInstruction):
#  signature = (rel16off_t,)
#  opt_kw = ()
  
#  def _render(params, operands):
#    return params['opcode'] + w16(operands['rel16off'])
#  render = staticmethod(_render)


# Label offset that allows for either an 8 or 32bit offset encoding
class rel32_8off(MachineInstruction):
  signature = (rel32off_t,)
  opt_kw = ()
  
  def _render(params, operands):
    rel = operands['rel32off']
    # Relative offset is computed from the end of this instruction
    #print "lbl position", lbl.position
    #print "inst position", operands['position']
    #offset = rel - operands['position']
    #print "offset", offset

    # Will an 8bit offset do the job?
    #off = offset - (len(params['opcode'][0]) + 1)
    off = rel - (operands['position'] + len(params['opcode'][0]) + 1)
    #print "off is", off
    if rel8off_t.fits(off):
      #print "encoding 8bit offset", off
      return params['opcode'][0] + w8(off)

    # Fall back to 32bit, or nothing if even that doesn't fit
    #off = offset - (len(params['opcode'][1]) + 4)
    off = rel - (operands['position'] + len(params['opcode'][1]) + 4)
    #print "off is", off, len(params['opcode'][1]) + 4
    if rel32off_t.fits(off):
      #print "encoding 32bit offset", off
      return params['opcode'][1] + w32(off)
  render = staticmethod(_render)


class rel32off(MachineInstruction):
  signature = (rel32off_t,)
  opt_kw = ()
  
  def _render(params, operands):
    rel32off = operands['rel32off']
    offset = rel32off - (operands['position'] + len(params['opcode']) + 4)
    return params['opcode'] + w32(offset)

  render = staticmethod(_render)


class rel8off(MachineInstruction):
  signature = (rel8off_t,)
  opt_kw = ()
  
  def _render(params, operands):
    rel8off = operands['rel8off']
    offset = rel8off - (operands['position'] + len(params['opcode']) + 1)
    return params['opcode'] + w8(offset)
  render = staticmethod(_render)

  
class st0_sti(MachineInstruction):
  signature = (st0_t, regst_t)
  opt_kw = ()

  def _render(params, operands):
    opcode = params['opcode']
    return opcode[:-1] + [opcode[-1] + operands['regst'].reg]
  render = staticmethod(_render)


class sti(MachineInstruction):
  signature = (regst_t,)
  opt_kw = ()

  def _render(params, operands):
    opcode = params['opcode']
    return opcode[:-1] + [opcode[-1] + operands['regst'].reg]
  render = staticmethod(_render)


class sti_st0(MachineInstruction):
  signature = (regst_t, st0_t)
  opt_kw = ()

  def _render(params, operands):
    opcode = params['opcode']
    return opcode[:-1] + [opcode[-1] + operands['regst'].reg]
  render = staticmethod(_render)


class xmm_imm8(MachineInstruction):
  signature = (xmm_t, imm8_t)
  opt_kw = ()
  
  def _render(params, operands):
    xmm = operands['xmm']
    rex = [0x40 | xmm.rex]
    if rex == [0x40]:
      rex = []

    return params['prefix'] + rex + params['opcode'] + [0xC0 | params['modrm'] | xmm.reg] + w8(operands['imm8'])
  render = staticmethod(_render)


class xmm_imm8_imm8(MachineInstruction):
  signature = (xmm_t, imm8_t('ia'), imm8_t('ib'))
  opt_kw = ()
  
  def _render(params, operands):
    xmm = operands['xmm']
    rex = [0x40 | xmm.rex]
    if rex == [0x40]:
      rex = []

    return params['prefix'] + rex + params['opcode'] + [0xC0 | params['modrm'] | xmm.reg] + w8(operands['ia']) + w8(operands['ib'])
  render = staticmethod(_render)


class xmm_mem128(MachineInstruction):
  signature = (xmm_t, mem128_t)
  opt_kw = ()
  
  def _render(params, operands):
    xmm = operands['xmm']
    ret = common_memref(params['opcode'], operands['mem128'], xmm.reg << 3, xmm.rex << 2)
    if ret != None:
      return params['prefix'] + ret
  render = staticmethod(_render)


class xmm_mem128_imm(MachineInstruction):
  signature = (xmm_t, mem128_t)
  opt_kw = ()
  
  def _render(params, operands):
    xmm = operands['xmm']
    ret = common_memref(params['opcode'], operands['mem128'], xmm.reg << 3, xmm.rex << 2)
    if ret != None:
      return params['prefix'] + ret + w8(params['imm'])
  render = staticmethod(_render)


class xmm_mem128_imm8(MachineInstruction):
  signature = (xmm_t, mem128_t, imm8_t)
  opt_kw = ()
  
  def _render(params, operands):
    xmm = operands['xmm']
    ret = common_memref(params['opcode'], operands['mem128'], xmm.reg << 3, xmm.rex << 2)
    if ret != None:
      return params['prefix'] + ret + w8(operands['imm8'])
  render = staticmethod(_render)


class xmm_mem16(MachineInstruction):
  signature = (xmm_t, mem16_t)
  opt_kw = ()
  
  def _render(params, operands):
    xmm = operands['xmm']
    ret = common_memref(params['opcode'], operands['mem16'], xmm.reg << 3, xmm.rex << 2)
    if ret != None:
      return params['prefix'] + ret
  render = staticmethod(_render)


class xmm_mem16_imm8(MachineInstruction):
  signature = (xmm_t, mem16_t, imm8_t)
  opt_kw = ()
  
  def _render(params, operands):
    xmm = operands['xmm']
    ret = common_memref(params['opcode'], operands['mem16'], xmm.reg << 3, xmm.rex << 2)
    if ret != None:
      return params['prefix'] + ret + w8(operands['imm8'])
  render = staticmethod(_render)


class xmm_mem32(MachineInstruction):
  signature = (xmm_t, mem32_t)
  opt_kw = ()
  
  def _render(params, operands):
    xmm = operands['xmm']
    ret = common_memref(params['opcode'], operands['mem32'], xmm.reg << 3, xmm.rex << 2)
    if ret != None:
      return params['prefix'] + ret
  render = staticmethod(_render)


class xmm_mem32_imm(MachineInstruction):
  signature = (xmm_t, mem32_t)
  opt_kw = ()
  
  def _render(params, operands):
    xmm = operands['xmm']
    ret = common_memref(params['opcode'], operands['mem32'], xmm.reg << 3, xmm.rex << 2)
    if ret != None:
      return params['prefix'] + ret + w8(params['imm'])
  render = staticmethod(_render)


class xmm_mem32_imm8(MachineInstruction):
  signature = (xmm_t, mem32_t, imm8_t)
  opt_kw = ()
  
  def _render(params, operands):
    xmm = operands['xmm']
    ret = common_memref(params['opcode'], operands['mem32'], xmm.reg << 3, xmm.rex << 2) 
    if ret != None:
      return params['prefix'] + ret + w8(operands['imm8'])
  render = staticmethod(_render)


class xmm_mem64(MachineInstruction):
  signature = (xmm_t, mem64_t)
  opt_kw = ()
  
  def _render(params, operands):
    xmm = operands['xmm']
    ret = common_memref(params['opcode'], operands['mem64'], operands['xmm'].reg << 3, xmm.rex << 2)
    if ret != None:
      return params['prefix'] + ret
  render = staticmethod(_render)


class xmm_mem64_imm(MachineInstruction):
  signature = (xmm_t, mem64_t)
  opt_kw = ()
  
  def _render(params, operands):
    xmm = operands['xmm']
    ret = common_memref(params['opcode'], operands['mem64'], operands['xmm'].reg << 3, xmm.rex << 2)
    if ret != None:
      return params['prefix'] + ret + w8(params['imm'])
  render = staticmethod(_render)


class xmm_mem64_imm8(MachineInstruction):
  signature = (xmm_t, mem64_t, imm8_t)
  opt_kw = ()
  
  def _render(params, operands):
    xmm = operands['xmm']
    ret = common_memref(params['opcode'], operands['mem64'], operands['xmm'].reg << 3, xmm.rex << 2)
    if ret != None:
      return params['prefix'] + ret + w8(operands['imm8'])
  render = staticmethod(_render)


class xmm_mem8_imm8(MachineInstruction):
  signature = (xmm_t, mem8_t, imm8_t)
  opt_kw = ()
  
  def _render(params, operands):
    xmm = operands['xmm']
    ret = common_memref(params['opcode'], operands['mem8'], xmm.reg << 3, xmm.rex << 2)
    if ret != None:
      return params['prefix'] + ret + w8(operands['imm8'])
  render = staticmethod(_render)


class xmm_mmx(MachineInstruction):
  signature = (xmm_t, mmx_t)
  opt_kw = ()
 
  # TODO - fix me!! xmm cold need an REX 
  def _render(params, operands):
    xmm = operands['xmm']
    rex = [0x40 | xmm.rex << 2]
    if rex == [0x40]:
      rex = []

    return params['prefix'] + rex + params['opcode'] + [0xC0 | (xmm.reg << 3) | operands['mmx'].reg]
  render = staticmethod(_render)


class xmm_reg32(MachineInstruction):
  signature = (xmm_t, reg32_t)
  opt_kw = ()
  
  def _render(params, operands):
    xmm = operands['xmm']
    rex = [0x40 | xmm.rex << 2]
    if rex == [0x40]:
      rex = []

    return params['prefix'] + rex + params['opcode'] + [0xC0 | (xmm.reg << 3) | operands['reg32'].reg]
  render = staticmethod(_render)


class xmm_reg32_imm8(MachineInstruction):
  signature = (xmm_t, reg32_t, imm8_t)
  opt_kw = ()
  
  def _render(params, operands):
    xmm = operands['xmm']
    rex = [0x40 | xmm.rex << 2]
    if rex == [0x40]:
      rex = []

    return params['prefix'] + rex + params['opcode'] + [0xC0 | (operands['xmm'].reg << 3) | operands['reg32'].reg] + w8(operands['imm8'])
  render = staticmethod(_render)


class xmm_reg64(MachineInstruction):
  signature = (xmm_t, reg64_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg64 = operands['reg64']
    xmm = operands['xmm']
    return params['prefix'] + [0x48 | reg64.rex | (xmm.rex << 2)] + params['opcode'] + [0xC0 | (xmm.reg << 3) | reg64.reg]
  render = staticmethod(_render)


class xmm_reg64_imm8(MachineInstruction):
  signature = (xmm_t, reg64_t, imm8_t)
  opt_kw = ()
  
  def _render(params, operands):
    reg64 = operands['reg64']
    xmm = operands['xmm']
    return params['prefix'] + [0x48 | reg64.rex | (xmm.rex << 2)] + params['opcode'] + [0xC0 | (xmm.reg << 3) | reg64.reg] + w8(operands['imm8'])
  render = staticmethod(_render)


class xmm_xmm(MachineInstruction):
  signature = (xmm_t('rd'), xmm_t('ra'))
  opt_kw = ()
  
  def _render(params, operands):
    rd = operands['rd']
    ra = operands['ra']
    rex = [0x40 | (rd.rex << 2) | ra.rex]
    if rex == [0x40]:
      rex = []

    return params['prefix'] + rex + params['opcode'] + [0xC0 | (rd.reg << 3) | ra.reg]
  render = staticmethod(_render)


class xmm_xmm_imm(MachineInstruction):
  signature = (xmm_t('rd'), xmm_t('ra'))
  opt_kw = ()
  
  def _render(params, operands):
    rd = operands['rd']
    ra = operands['ra']
    rex = [0x40 | (rd.rex << 2) | ra.rex]
    if rex == [0x40]:
      rex = []

    return params['prefix'] + rex + params['opcode'] + [0xC0 | (rd.reg << 3) | ra.reg] + w8(params['imm'])
  render = staticmethod(_render)


class xmm_xmm_imm8(MachineInstruction):
  signature = (xmm_t('rd'), xmm_t('ra'), imm8_t)
  opt_kw = ()
  
  def _render(params, operands):
    rd = operands['rd']
    ra = operands['ra']
    rex = [0x40 | (rd.rex << 2) | ra.rex]
    if rex == [0x40]:
      rex = []

    return params['prefix'] + rex + params['opcode'] + [0xC0 | (rd.reg << 3) | ra.reg] + w8(operands['imm8'])
  render = staticmethod(_render)


class xmm_xmm_imm8_imm8(MachineInstruction):
  signature = (xmm_t('rd'), xmm_t('ra'), imm8_t('ia'), imm8_t('ib'))
  opt_kw = ()
  
  def _render(params, operands):
    rd = operands['rd']
    ra = operands['ra']
    rex = [0x40 | (rd.rex << 2) | ra.rex]
    if rex == [0x40]:
      rex = []

    return params['prefix'] + rex + params['opcode'] + [0xC0 | (operands['rd'].reg << 3) | operands['ra'].reg] + w8(operands['ia']) + w8(operands['ib'])
  render = staticmethod(_render)


