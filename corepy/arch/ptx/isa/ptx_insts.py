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
from ptx_fields import *
import corepy.arch.ptx.types.registers as regs

# TODO - blah, anything that wraps around a common_memref needs an if-statement

# ------------------------------
# Type Instances
# ------------------------------

# # Instances for common types

# # Constants
# one_t = x86ConstantOperand("one", 1)

# # Immediates
# imm8_t  = Imm8("imm8",  (-128, 256))
# simm8_t = Imm8("simm8", (-128, 128))
# imm16_t = Imm16("imm16", (-32768, 65536))
# imm32_t = Imm32("imm32", (-2147483648, 4294967296))
# simm32_t = Imm32("simm32", (-2147483648, 2147483648))
# imm64_t = Imm64("imm64", (-9223372036854775808, 18446744073709551616))

# # Memory
# mem_t   = x86MemoryOperand("mem", None)
# mem8_t  = x86MemoryOperand("mem8", 8)
# mem16_t = x86MemoryOperand("mem16", 16)
# mem32_t = x86MemoryOperand("mem32", 32)
# mem64_t = x86MemoryOperand("mem64", 64)
# mem80_t = x86MemoryOperand("mem80", 80)
# mem128_t = x86MemoryOperand("mem128", 128)
# mem228_t = x86MemoryOperand("mem228", 228)
# mem512_t = x86MemoryOperand("mem512", 512)
# mem752_t = x86MemoryOperand("mem752", 752)

# # Registers
# reg8_t  = x86RegisterOperand("reg8", regs.GPRegister8)
# reg16_t = x86RegisterOperand("reg16", regs.GPRegister16)
# reg32_t = x86RegisterOperand("reg32", regs.GPRegister32)
# reg64_t = x86RegisterOperand("reg64", regs.GPRegister64)
# regst_t = x86RegisterOperand("regst", regs.FPRegister)
# mmx_t  = x86RegisterOperand("mmx", regs.MMXRegister)
# xmm_t  = x86RegisterOperand("xmm", regs.XMMRegister)

# # Fixed Registers
# al_t = FixedRegisterOperand("al", regs.GPRegister8, 0)
# ax_t = FixedRegisterOperand("ax", regs.GPRegister16, 0)
# cl_t = FixedRegisterOperand("cl", regs.GPRegister8, 1)
# dx_t = FixedRegisterOperand("dx", regs.GPRegister16, 2)
# eax_t = FixedRegisterOperand("eax", regs.GPRegister32, 0)
# rax_t = FixedRegisterOperand("rax", regs.GPRegister64, 0)
# st0_t = FixedRegisterOperand("st0", regs.FPRegister, 0)

# # Relative offsets
# lbl8off_t = x86LabelOperand("lbl8off", (-128, 128))
# #lbl16off_t = x86LabelOperand("lbl16off", (-65536, 65536))
# lbl32off_t = x86LabelOperand("lbl32off", (-2147483648, 2147483648))

# rel8off_t  = Rel8off("rel8off",  (-128, 128))
# #rel16off_t = Rel16off("rel16off", (-32768, 32768))
# rel32off_t = Rel32off("rel32off", (-2147483648, 2147483648))

# # Prefix bytes
# #prefix = []
# lock_p = x86PrefixOperand("lock", 0xF0)
# addr_p = x86PrefixOperand("addr", 0x67)

# # Array of reg8's that require some sort of REX
# reg8_rex_list = (regs.sil, regs.dil, regs.bpl, regs.spl)


# # ------------------------------
# # Utility functions
# # ------------------------------

# # w8() just makes sure n is an unsigned byte; the others do this implicitly.
# # All of these are little endian specific!!

# def w8(n):
#   return [(256 + n) & 0xFF]
  
# def w16(n):
#   return [n & 0xFF, (n & 0xFF00) >> 8]

# def w32(n):
#   return [n & 0xFF, (n & 0xFF00) >> 8, (n & 0xFF0000) >> 16, (n & 0xFF000000l) >> 24]

# def w64(n):
#   return [n & 0xFF, (n & 0xFF00) >> 8, (n & 0xFF0000) >> 16, (n & 0xFF000000l) >> 24, (n & 0xFF00000000l) >> 32, (n & 0xFF0000000000l) >> 40, (n & 0xFF000000000000l) >> 48, (n & 0xFF00000000000000l) >> 56]


# def common_memref_modrm(opcode, ref, modrm, rex, force_rex):
#   if ref.disp != None and ref.disp != 0:    # [base + disp]
#     if ref.base in (regs.rip, regs.eip):
#       rex = [0x40 | rex]
#       if rex == [0x40] and not force_rex:
#         rex = []
#       return rex + opcode + [0x5 | modrm] + w32(ref.disp)
#     elif ref.index != None:                 # [base+index*scale+disp]
#       sib = ref.scale_sib | (ref.index.reg << 3) | ref.base.reg
#       rex = [0x40 | (ref.index.rex << 1) | ref.base.rex | rex]
#       if rex == [0x40] and not force_rex:
#         rex = []

#       if simm8_t.fits(ref.disp):
#         return rex + opcode + [0x44 | modrm, sib] + w8(ref.disp)
#       elif simm32_t.fits(ref.disp):
#         return rex + opcode + [0x84 | modrm, sib] + w32(ref.disp)
#     elif ref.index == None:                 # [base + disp]
#       rex = [0x40 | ref.base.rex | rex]
#       if rex == [0x40] and not force_rex:
#         rex = []

#       if ref.base in (regs.rsp, regs.r12, regs.esp):
#         if simm8_t.fits(ref.disp):           # [rsp + disp], [r12 + disp]
#           return rex + opcode + [0x44 | modrm, 0x24] + w8(ref.disp)
#         elif simm32_t.fits(ref.disp):
#           return rex + opcode + [0x80 | modrm, 0x24] + w32(ref.disp)
#       elif simm8_t.fits(ref.disp):
#         return rex + opcode + [0x40 | modrm | ref.base.reg] + w8(ref.disp)
#       elif simm32_t.fits(ref.disp):
#         return rex + opcode + [0x80 | modrm | ref.base.reg] + w32(ref.disp)
#   elif ref.index != None:
#     sib = ref.scale_sib | (ref.index.reg << 3) | ref.base.reg
#     rex = [0x40 | (ref.index.rex << 1) | ref.base.rex | rex]
#     if rex == [0x40] and not force_rex:
#       rex = []

#     if ref.base in (regs.rbp, regs.r13, regs.ebp):
#       return rex + opcode + [0x44 | modrm, sib, 0x00] # [rbp, index], [r13, index]
#     return rex + opcode + [0x04 | modrm, sib]
#   elif ref.index == None:
#     rex = [0x40 | ref.base.rex | rex]
#     if rex == [0x40] and not force_rex:
#       rex = []

#     if ref.base in (regs.rbp, regs.r13, regs.ebp):
#       return rex + opcode + [0x45 | modrm, 0x00] # [rbp], [r13]
#     elif ref.base in (regs.rsp, regs.r12, regs.esp):
#       return rex + opcode + [0x04 | modrm, 0x24] # [rsp], [r12]
#     return rex + opcode + [modrm | ref.base.reg] # [base]


# def common_memref(opcode, ref, modrm, rex = 0, force_rex = False):
#   if ref.addr != None:  # Absolute address
#     rex = [0x40 | rex]
#     if rex == [0x40] and not force_rex:
#       rex = []
#     return rex + opcode + [0x04 | modrm, 0x25] + w32(ref.addr)
#   elif ref.addr_size == 64: # 64bit modRM address, RIP-relative
#     return common_memref_modrm(opcode, ref, modrm, rex, force_rex)
#   elif ref.addr_size == 32: # 32bit modRM address, EIP-relative
#     return [0x67] + common_memref_modrm(opcode, ref, modrm, rex, force_rex)
#   return None

# -----------------------------
# PTX helper functions
# -----------------------------

def pred_str(operands):
  s = ''
  if 'npred' in operands and operands['npred'] != None:
    s =  '@!' + pred.render(operands['npred'])
  elif 'pred' in operands and operands['pred'] != None:
    s = '@' + pred.render(operands['pred'])
  return s

# ------------------------------
# PTX Machine Instructions
# ------------------------------

# EVERY SINGLE ONE of these has to have pred and npred in opt_kw . Sorry, no way around it...
# It's for predicates and so far we don't have any better way to set those. - BDM

class x3(MachineInstruction):
  signature = (d, s0, s1)
  opt_kw = (pred, npred, sat)
  
  def _render(params, operands):
    #print operands
    pstr = pred_str(operands)
    suffix = '.' + params['type']
    return pstr + '\t' + params['opcode'] + suffix + ' ' + \
           d.render(operands['d']) + ', ' + \
           s0.render(operands['s0']) + ', ' + \
           s1.render(operands['s1']) + ';'
  render = staticmethod(_render)

class cc_x3(MachineInstruction):
  signature = (d, s0, s1)
  opt_kw = (pred, npred, cc)
  
  def _render(params, operands):
    #print operands
    pstr = pred_str(operands)
    suffix = '.' + params['type']
    return pstr + '\t' + params['opcode'] + cc.render(operands['cc']) + suffix + ' ' + \
           d.render(operands['d']) + ', ' + \
           s0.render(operands['s0']) + ', ' + \
           s1.render(operands['s1']) + ';'
  render = staticmethod(_render)

class s_x3(MachineInstruction):
  signature = (d, s0, s1)
  opt_kw = (pred, npred, sat)
  
  def _render(params, operands):
    #print operands
    pstr = pred_str(operands)
    suffix = '.' + params['type']
    return pstr + '\t' + params['opcode'] + sat.render(operands['sat']) + suffix + ' ' + \
           d.render(operands['d']) + ', ' + \
           s0.render(operands['s0']) + ', ' + \
           s1.render(operands['s1']) + ';'
  render = staticmethod(_render)

class r_s_x3(MachineInstruction):
  signature = (d, s0, s1)
  opt_kw = (pred, npred, rnd, sat)
  
  def _render(params, operands):
    #print operands
    pstr = pred_str(operands)
    suffix = '.' + params['type']
    return pstr + '\t' + params['opcode'] + rnd.render(operands['rnd']) + sat.render(operands['sat']) + suffix + ' ' + \
           d.render(operands['d']) + ', ' + \
           s0.render(operands['s0']) + ', ' + \
           s1.render(operands['s1']) + ';'
  render = staticmethod(_render)

class hlw_x3(MachineInstruction):
  signature = (d, s0, s1)
  opt_kw = (pred, npred, hlw)
  
  def _render(params, operands):
    pstr = pred_str(operands)
    suffix = '.' + params['type']
    return pstr + '\t' + params['opcode'] + hlw.render(operands['hlw']) + suffix + ' ' + \
           d.render(operands['d']) + ', ' + \
           s0.render(operands['s0']) + ', ' + \
           s1.render(operands['s1']) + ';'
  render = staticmethod(_render)

class hlws_x4(MachineInstruction):
  signature = (d, s0, s1, s2)
  opt_kw = (pred, npred, hlw, sat)
  
  def _render(params, operands):
    pstr = pred_str(operands)
    suffix = '.' + params['type']
    return pstr + '\t' + params['opcode'] + hlw.render(operands['hlw']) + sat.render(operand['sat']) + suffix + ' ' + \
           d.render(operands['d']) + ', ' + \
           s0.render(operands['s0']) + ', ' + \
           s1.render(operands['s1']) + ', ' + \
           s2.render(operands['s2']) + ';'
  render = staticmethod(_render)

class r_s_x4(MachineInstruction):
  signature = (d, s0, s1, s2)
  opt_kw = (pred, npred, rnd, sat)
  
  def _render(params, operands):
    pstr = pred_str(operands)
    suffix = '.' + params['type']
    return pstr + '\t' + params['opcode'] + rnd.render(operands['rnd']) + sat.render(operands['sat']) + suffix + ' ' + \
           d.render(operands['d']) + ', ' + \
           s0.render(operands['s0']) + ', ' + \
           s1.render(operands['s1']) + ', ' + \
           s2.render(operands['s2']) + ';'
  render = staticmethod(_render)

class d_sora(MachineInstruction):
  signature = (d, s_or_a)
  opt_kw = (pred, npred)
  
  def _render(params, operands):
    pstr = pred_str(operands)
    suffix = '.' + params['type']
    return pstr + '\t' + params['opcode'] + suffix + ' ' + \
           d.render(operands['d']) + ', ' + \
           s_or_a.render(operands['s_or_a']) + ';'
  render = staticmethod(_render)

class space_d_a(MachineInstruction):
  signature = (space, d, a)
  opt_kw = (pred, npred, volatile)
  
  def _render(params, operands):
    pstr = pred_str(operands)
    suffix = '.' + params['type']
    return pstr + '\t' + params['opcode'] + space.render(operands['space']) + suffix + ' ' + \
           d.render(operands['d']) + ', ' + \
           a.render(operands['a']) + ';'
  render = staticmethod(_render)

class space_d_r(MachineInstruction):
  signature = (space, a, s0)
  opt_kw = (pred, npred, volatile)
  
  def _render(params, operands):
    pstr = pred_str(operands)
    suffix = '.' + params['type']
    return pstr + '\t' + params['opcode'] + space.render(operands['space']) + suffix + ' ' + \
           a.render(operands['a']) + ', ' + \
           s0.render(operands['s0']) + ';'
  render = staticmethod(_render)

class r_s_d_a(MachineInstruction):
  signature = (d, s0)
  opt_kw = (pred, npred, rnd, sat)
  
  def _render(params, operands):
    pstr = pred_str(operands)
    suffix = '.' + params['dtype'] + '.' + params['atype']
    return pstr + '\t' + params['opcode'] + rnd.render(operands['rnd']) + sat.render(operands['sat']) + suffix + ' ' + \
           d.render(operands['d']) + ', ' + \
           s0.render(operands['s0']) + ';'
  render = staticmethod(_render)


class uni_lorr(MachineInstruction):
  signature = (r_or_l,)
  opt_kw = (pred, npred, uni)

  def _render(params, operands):
    pstr = pred_str(operands)
    return pstr + '\t' + params['opcode'] + uni.render(operands['uni']) + ' ' + \
           r_or_l.render(operands['r_or_l']) + ';'    
  render = staticmethod(_render)

class x0_uni(MachineInstruction):
  signature = ()
  opt_kw = (pred, npred, uni)

  def _render(params, operands):
    pstr = pred_str(operands)
    return pstr + '\t' + params['opcode'] + uni.render(operands['uni']) + ';'
  render = staticmethod(_render)

class x0(MachineInstruction):
  signature = ()
  opt_kw = (pred, npred)

  def _render(params, operands):
    pstr = pred_str(operands)
    return pstr + '\t' + params['opcode'] + ';'
  render = staticmethod(_render)

class bar(MachineInstruction):
  signature = (imm,)
  opt_kw = (pred, npred)

  def _render(params, operands):
    pstr = pred_str(operands)
    return pstr + '\t' + params['opcode'] + imm.render(operands['imm']) + ';'    
  render = staticmethod(_render)

class compop_p_a_b(MachineInstruction):
  signature = (compop, d, s0, s1)
  opt_kw = (pred, npred, neg)
  
  def _render(params, operands):
    pstr = pred_str(operands)
    suffix = '.' + params['type']
    return pstr + '\t' + params['opcode'] + compop.render(operands['compop']) + suffix + ' ' + \
           d.render(operands['d']) + ', ' + \
           s0.render(operands['s0']) + ', ' + \
           s1.render(operands['s1']) + ';'
  render = staticmethod(_render)

# class compop_p_q_a_b(MachineInstruction):
#   signature = (compop, d, q, s0, s1)
#   opt_kw = (pred, npred)
  
#   def _render(params, operands):
#     pstr = pred_str(operands)
#     suffix = '.' + params['type']
#     return pstr + '\t' + params['opcode'] + compop.render(operands['compop']) + suffix + ' ' + \
#            d.render(operands['d']) + ', ' + \
#            q.render(operands['q']) + ', ' + \
#            s0.render(operands['s0']) + ', ' + \
#            s1.render(operands['s1']) + ';'
#   render = staticmethod(_render)

class compop_boolop_p_a_b(MachineInstruction):
  signature = (compop, boolop, d, s0, s1, s2)
  opt_kw = (pred, npred, neg)
  
  def _render(params, operands):
    pstr = pred_str(operands)
    suffix = '.' + params['type']
    return pstr + '\t' + params['opcode'] + compop.render(operands['compop']) + \
           boolop.render(operands['boolop']) + suffix + ' ' + \
           d.render(operands['d']) + ', ' + \
           s0.render(operands['s0']) + ', ' + \
           s1.render(operands['s1']) + ',' + \
           neg.render(operands['neg']) + \
           s2.render(operands['s2']) + ';'
  render = staticmethod(_render)

class compop_d_a_b(MachineInstruction):
  signature = (compop, d, s0, s1)
  opt_kw = (pred, npred)
  
  def _render(params, operands):
    pstr = pred_str(operands)
    suffix = '.' + params['dtype'] + '.' + params['stype']
    return pstr + '\t' + params['opcode'] + compop.render(operands['compop']) + suffix + ' ' + \
           d.render(operands['d']) + ', ' + \
           s0.render(operands['s0']) + ', ' + \
           s1.render(operands['s1']) + ';'
  render = staticmethod(_render)

class compop_boolop_d_a_b(MachineInstruction):
  signature = (compop, boolop, d, s0, s1, s2)
  opt_kw = (pred, npred)
  
  def _render(params, operands):
    pstr = pred_str(operands)
    suffix = '.' + params['dtype'] + '.' + params['stype']
    return pstr + '\t' + params['opcode'] + compop.render(operands['compop']) + \
           boolop.render(operands['boolop']) + suffix + ' ' + \
           d.render(operands['d']) + ', ' + \
           s0.render(operands['s0']) + ', ' + \
           s1.render(operands['s1']) + ',' + \
           s2.render(operands['s2']) + ';'
  render = staticmethod(_render)

class x4(MachineInstruction):
  signature = (d, s0, s1, s2)
  opt_kw = (pred, npred)
  
  def _render(params, operands):
    pstr = pred_str(operands)
    suffix = '.' + params['type']
    return pstr + '\t' + params['opcode'] + suffix + ' ' + \
           d.render(operands['d']) + ', ' + \
           s0.render(operands['s0']) + ', ' + \
           s1.render(operands['s1']) + ', ' + \
           s2.render(operands['s2']) + ';'
  render = staticmethod(_render)

class x3_x1(MachineInstruction):
  signature = (d, s0, s1, s2)
  opt_kw = (pred, npred)
  
  def _render(params, operands):
    pstr = pred_str(operands)
    suffix = '.' + params['dtype'] + '.' + params['ctype']
    return pstr + '\t' + params['opcode'] + suffix + ' ' + \
           d.render(operands['d']) + ', ' + \
           s0.render(operands['s0']) + ', ' + \
           s1.render(operands['s1']) + ', ' + \
           s2.render(operands['s2']) + ';'
  render = staticmethod(_render)

class space_op_a_b(MachineInstruction):
  signature = (space, redop, a, s0)
  opt_kw = (pred, npred)
  
  def _render(params, operands):
    pstr = pred_str(operands)
    suffix = '.' + params['type']
    return pstr + '\t' + params['opcode'] + space.render(operands['space']) + \
           redop.render(operands['op']) + suffix + ' ' + \
           a.render(operands['a']) + ', ' + \
           s0.render(operands['s0']) + ';'
  render = staticmethod(_render)

class space_op_d_a_b(MachineInstruction):
  signature = (space, redop, d, a, s0)
  opt_kw = (pred, npred)
  
  def _render(params, operands):
    pstr = pred_str(operands)
    suffix = '.' + params['type']
    return pstr + '\t' + params['opcode'] + space.render(operands['space']) + \
           redop.render(operands['op']) + suffix + ' ' + \
           d.render(operands['d']) + ', ' + \
           a.render(operands['a']) + ', ' + \
           s0.render(operands['s0']) + ';'
  render = staticmethod(_render)

class space_op_d_a_b_c(MachineInstruction):
  signature = (space, redop, d, a, s0, s1)
  opt_kw = (pred, npred)
  
  def _render(params, operands):
    pstr = pred_str(operands)
    suffix = '.' + params['type']
    return pstr + '\t' + params['opcode'] + space.render(operands['space']) + \
           redop.render(operands['op']) + suffix + ' ' + \
           d.render(operands['d']) + ', ' + \
           a.render(operands['a']) + ', ' + \
           s0.render(operands['s0']) + ', ' + \
           s1.render(operands['s1']) + ';'
  render = staticmethod(_render)

class mode_x2(MachineInstruction):
  signature = (mode, d, s0)
  opt_kw = (pred, npred, neg)
  
  def _render(params, operands):
    pstr = pred_str(operands)
    suffix = '.pred'
    return pstr + '\t' + params['opcode'] + mode.render(operands['mode']) + suffix + ' ' + \
           d.render(operands['d']) + ', ' + \
           neg.render(operands['neg']) + \
           s0.render(operands['s0']) + ';'
  render = staticmethod(_render)
