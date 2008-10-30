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

from corepy.spre.spe import Instruction, DispatchInstruction, Register
from spu_insts import *

__doc__="""
ISA for the Cell Broadband Engine's SPU.
"""

# Only used for branch instructions
def spu_type(op):
  if isinstance(op, (int, long)):
    if I7.fits(op):
      return I7
    if I8.fits(op):
      return I8
    if I9.fits(op):
      return I9
    if I10.fits(op):
      return I10
    if I16.fits(op):
      return I16
    if I18.fits(op):
      return I18
  elif isinstance(op, Label):
    return LBL16
  elif isinstance(op, Register):
    return T3
  return

class SPUInstruction(Instruction): pass
class SPUDispatchInstruction(DispatchInstruction):
  type_id = [spu_type]


class lqx(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':452}
  cycles = (1, 6, 0)


class stqx(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':324}
  cycles = (1, 6, 0)


class cbx(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':468}
  cycles = (1, 4, 0)


class chx(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':469}
  cycles = (1, 4, 0)


class cwx(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':470}
  cycles = (1, 4, 0)


class cdx(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':471}
  cycles = (1, 4, 0)


class ah(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':200}
  cycles = (0, 2, 0)


class a(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':192}
  cycles = (0, 2, 0)


class sfh(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':72}
  cycles = (0, 2, 0)


class sf(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':64}
  cycles = (0, 2, 0)


class addx(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':832}
  cycles = (0, 2, 0)


class cg(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':194}
  cycles = (0, 2, 0)


class cgx(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':834}
  cycles = (0, 2, 0)


class sfx(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':833}
  cycles = (0, 2, 0)


class bg(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':66}
  cycles = (0, 2, 0)


class bgx(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':835}
  cycles = (0, 2, 0)


class mpy(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':964}
  cycles = (0, 7, 0)


class mpyu(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':972}
  cycles = (0, 7, 0)


class mpyh(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':965}
  cycles = (0, 7, 0)


class mpys(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':967}
  cycles = (0, 7, 0)


class mpyhh(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':966}
  cycles = (0, 7, 0)


class mpyhha(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':838}
  cycles = (0, 7, 0)


class mpyhhu(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':974}
  cycles = (0, 7, 0)


class mpyhhau(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':846}
  cycles = (0, 7, 0)


class clz(SPUInstruction):
  machine_inst = OPCD_A_T
  params = {'OPCD':677}
  cycles = (0, 2, 0)


class cntb(SPUInstruction):
  machine_inst = OPCD_A_T
  params = {'OPCD':692}
  cycles = (0, 4, 0)


class fsmb(SPUInstruction):
  machine_inst = OPCD_A_T
  params = {'OPCD':438}
  cycles = (1, 4, 0)


class fsmh(SPUInstruction):
  machine_inst = OPCD_A_T
  params = {'OPCD':437}
  cycles = (1, 4, 0)


class fsm(SPUInstruction):
  machine_inst = OPCD_A_T
  params = {'OPCD':436}
  cycles = (1, 4, 0)


class gbb(SPUInstruction):
  machine_inst = OPCD_A_T
  params = {'OPCD':434}
  cycles = (1, 4, 0)


class gbh(SPUInstruction):
  machine_inst = OPCD_A_T
  params = {'OPCD':433}
  cycles = (1, 4, 0)


class gb(SPUInstruction):
  machine_inst = OPCD_A_T
  params = {'OPCD':432}
  cycles = (1, 4, 0)


class avgb(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':211}
  cycles = (0, 4, 0)


class absdb(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':83}
  cycles = (0, 4, 0)


class sumb(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':595}
  cycles = (0, 4, 0)


class xsbh(SPUInstruction):
  machine_inst = OPCD_A_T
  params = {'OPCD':694}
  cycles = (0, 2, 0)


class xshw(SPUInstruction):
  machine_inst = OPCD_A_T
  params = {'OPCD':686}
  cycles = (0, 2, 0)


class xswd(SPUInstruction):
  machine_inst = OPCD_A_T
  params = {'OPCD':678}
  cycles = (0, 2, 0)


class and_(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':193}
  cycles = (0, 2, 0)


class andc(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':705}
  cycles = (0, 2, 0)


class or_(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':65}
  cycles = (0, 2, 0)


class orc(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':713}
  cycles = (0, 2, 0)


class orx(SPUInstruction):
  machine_inst = OPCD_A_T
  params = {'OPCD':496}
  cycles = (1, 4, 0)


class xor(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':577}
  cycles = (0, 2, 0)


class nand(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':201}
  cycles = (0, 2, 0)


class nor(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':73}
  cycles = (0, 2, 0)


class eqv(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':585}
  cycles = (0, 2, 0)


class shlh(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':95}
  cycles = (0, 4, 0)


class shl(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':91}
  cycles = (0, 4, 0)


class shlqbi(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':475}
  cycles = (1, 4, 0)


class shlqby(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':479}
  cycles = (1, 4, 0)


class shlqbybi(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':463}
  cycles = (1, 4, 0)


class roth(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':92}
  cycles = (0, 4, 0)


class rot(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':88}
  cycles = (0, 4, 0)


class rotqby(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':476}
  cycles = (1, 4, 0)


class rotqbybi(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':460}
  cycles = (1, 4, 0)


class rotqbi(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':472}
  cycles = (1, 4, 0)


class rothm(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':93}
  cycles = (0, 4, 0)


class rotm(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':89}
  cycles = (0, 4, 0)


class rotqmby(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':477}
  cycles = (1, 4, 0)


class rotqmbybi(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':461}
  cycles = (1, 4, 0)


class rotqmbi(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':473}
  cycles = (1, 4, 0)


class rotmah(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':94}
  cycles = (0, 4, 0)


class rotma(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':90}
  cycles = (0, 4, 0)


class heq(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':984}
  cycles = (0, 2, 0)


class hgt(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':600}
  cycles = (0, 2, 0)


class hlgt(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':728}
  cycles = (0, 2, 0)


class ceqb(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':976}
  cycles = (0, 2, 0)


class ceqh(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':968}
  cycles = (0, 2, 0)


class ceq(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':960}
  cycles = (0, 2, 0)


class cgtb(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':592}
  cycles = (0, 2, 0)


class cgth(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':584}
  cycles = (0, 2, 0)


class cgt(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':576}
  cycles = (0, 2, 0)


class clgtb(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':720}
  cycles = (0, 2, 0)


class clgth(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':712}
  cycles = (0, 2, 0)


class clgt(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':704}
  cycles = (0, 2, 0)


class bi(SPUInstruction):
  machine_inst = OPCD_A_D_E
  params = {'OPCD':424}
  cycles = (1, 4, 0)


class iret(SPUInstruction):
  machine_inst = OPCD_A_D_E
  params = {'OPCD':426}
  cycles = (1, 4, 0)


class bisled(SPUInstruction):
  machine_inst = OPCD_A_T_D_E
  params = {'OPCD':427}
  cycles = (1, 4, 0)


class bisl(SPUInstruction):
  machine_inst = OPCD_A_T_D_E
  params = {'OPCD':425}
  cycles = (1, 4, 0)


class biz(SPUInstruction):
  machine_inst = OPCD_A_T_D_E
  params = {'OPCD':296}
  cycles = (1, 4, 0)


class binz(SPUInstruction):
  machine_inst = OPCD_A_T_D_E
  params = {'OPCD':297}
  cycles = (1, 4, 0)


class bihz(SPUInstruction):
  machine_inst = OPCD_A_T_D_E
  params = {'OPCD':294}
  cycles = (1, 4, 0)


class bihnz(SPUInstruction):
  machine_inst = OPCD_A_T_D_E
  params = {'OPCD':299}
  cycles = (1, 4, 0)


# TODO - can we check that if P is set then RO is zero as required?
class hbr(SPUDispatchInstruction):
  cycles = (1, 15, 0)
  dispatch = (
    (OPCD_RO_A_P,   {'OPCD':428}),
    (OPCD_LBL9_A_P, {'OPCD':428}))


class fa(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':708}
  cycles = (0, 6, 0)


class dfa(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':716}
  cycles = (0, 13, 6)


class fs(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':709}
  cycles = (0, 6, 0)


class dfs(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':717}
  cycles = (0, 13, 6)


class fm(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':710}
  cycles = (0, 6, 0)


class dfm(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':718}
  cycles = (0, 13, 6)


class dfma(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':860}
  cycles = (0, 13, 6)


class dfnms(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':862}
  cycles = (0, 13, 6)


class dfms(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':861}
  cycles = (0, 13, 6)


class dfnma(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':863}
  cycles = (0, 13, 6)


class frest(SPUInstruction):
  machine_inst = OPCD_A_T
  params = {'OPCD':440}
  cycles = (1, 4, 0)


class frsqest(SPUInstruction):
  machine_inst = OPCD_A_T
  params = {'OPCD':441}
  cycles = (1, 4, 0)


class fi(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':980}
  cycles = (0, 7, 0)


class frds(SPUInstruction):
  machine_inst = OPCD_A_T
  params = {'OPCD':953}
  cycles = (0, 13, 6)


class fesd(SPUInstruction):
  machine_inst = OPCD_A_T
  params = {'OPCD':952}
  cycles = (0, 13, 6)


class fceq(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':962}
  cycles = (0, 2, 0)


class fcmeq(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':970}
  cycles = (0, 2, 0)


class fcgt(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':706}
  cycles = (0, 2, 0)


class fcmgt(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':714}
  cycles = (0, 2, 0)


class fscrwr(SPUInstruction):
  machine_inst = OPCD_A_T
  params = {'OPCD':954}
  cycles = (0, 7, 0)


class fscrrd(SPUInstruction):
  machine_inst = OPCD_T
  params = {'OPCD':920}
  cycles = (0, 13, 6)


class stop(SPUInstruction):
  machine_inst = OPCD_STOP_SIG
  params = {'OPCD':0}
  cycles = (1, 4, 0)


class stopd(SPUInstruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':320}
  cycles = (1, 4, 0)


class lnop(SPUInstruction):
  machine_inst = OPCD
  params = {'OPCD':1}
  cycles = (1, 0, 0)


class nop(SPUInstruction):
  machine_inst = OPCD_T
  params = {'OPCD':513}
  cycles = (0, 0, 0)


class sync(SPUInstruction):
  machine_inst = OPCD_CF
  params = {'OPCD':2}
  cycles = (1, 4, 0)


class dsync(SPUInstruction):
  machine_inst = OPCD
  params = {'OPCD':3}
  cycles = (1, 4, 0)


class mfspr(SPUInstruction):
  machine_inst = OPCD_SA_T
  params = {'OPCD':12}
  cycles = (1, 6, 0)


class mtspr(SPUInstruction):
  machine_inst = OPCD_SA_T
  params = {'OPCD':268}
  cycles = (1, 6, 0)


class rdch(SPUInstruction):
  machine_inst = OPCD_A_T
  params = {'OPCD':13}
  cycles = (1, 6, 0)


class rchcnt(SPUInstruction):
  machine_inst = OPCD_A_T
  params = {'OPCD':15}
  cycles = (1, 6, 0)


class wrch(SPUInstruction):
  machine_inst = OPCD_A_T
  params = {'OPCD':269}
  cycles = (1, 6, 0)


class mpya(SPUInstruction):
  machine_inst = OPCD_T_B_A_C
  params = {'OPCD':12}
  cycles = (0, 7, 0)


class selb(SPUInstruction):
  machine_inst = OPCD_T_B_A_C
  params = {'OPCD':8}
  cycles = (0, 2, 0)


class shufb(SPUInstruction):
  machine_inst = OPCD_T_B_A_C
  params = {'OPCD':11}
  cycles = (1, 4, 0)


class fma(SPUInstruction):
  machine_inst = OPCD_T_B_A_C
  params = {'OPCD':14}
  cycles = (0, 6, 0)


class fnms(SPUInstruction):
  machine_inst = OPCD_T_B_A_C
  params = {'OPCD':13}
  cycles = (0, 6, 0)


class fms(SPUInstruction):
  machine_inst = OPCD_T_B_A_C
  params = {'OPCD':15}
  cycles = (0, 6, 0)


class cbd(SPUInstruction):
  machine_inst = OPCD_I7_A_T
  params = {'OPCD':500}
  cycles = (1, 4, 0)


class chd(SPUInstruction):
  machine_inst = OPCD_I7_A_T
  params = {'OPCD':501}
  cycles = (1, 4, 0)


class cwd(SPUInstruction):
  machine_inst = OPCD_I7_A_T
  params = {'OPCD':502}
  cycles = (1, 4, 0)


class cdd(SPUInstruction):
  machine_inst = OPCD_I7_A_T
  params = {'OPCD':503}
  cycles = (1, 4, 0)


class shlhi(SPUInstruction):
  machine_inst = OPCD_I7_A_T
  params = {'OPCD':127}
  cycles = (0, 4, 0)


class shli(SPUInstruction):
  machine_inst = OPCD_I7_A_T
  params = {'OPCD':123}
  cycles = (0, 4, 0)


class shlqbii(SPUInstruction):
  machine_inst = OPCD_I7_A_T
  params = {'OPCD':507}
  cycles = (1, 4, 0)


class shlqbyi(SPUInstruction):
  machine_inst = OPCD_I7_A_T
  params = {'OPCD':511}
  cycles = (1, 4, 0)


class rothi(SPUInstruction):
  machine_inst = OPCD_I7_A_T
  params = {'OPCD':124}
  cycles = (0, 4, 0)


class roti(SPUInstruction):
  machine_inst = OPCD_I7_A_T
  params = {'OPCD':120}
  cycles = (0, 4, 0)


class rotqbyi(SPUInstruction):
  machine_inst = OPCD_I7_A_T
  params = {'OPCD':508}
  cycles = (1, 4, 0)


class rotqbii(SPUInstruction):
  machine_inst = OPCD_I7_A_T
  params = {'OPCD':504}
  cycles = (1, 4, 0)


class rothmi(SPUInstruction):
  machine_inst = OPCD_I7_A_T
  params = {'OPCD':125}
  cycles = (0, 4, 0)


class rotmi(SPUInstruction):
  machine_inst = OPCD_I7_A_T
  params = {'OPCD':121}
  cycles = (0, 4, 0)


class rotqmbyi(SPUInstruction):
  machine_inst = OPCD_I7_A_T
  params = {'OPCD':509}
  cycles = (1, 4, 0)


class rotqmbii(SPUInstruction):
  machine_inst = OPCD_I7_A_T
  params = {'OPCD':505}
  cycles = (1, 4, 0)


class rotmahi(SPUInstruction):
  machine_inst = OPCD_I7_A_T
  params = {'OPCD':126}
  cycles = (0, 4, 0)


class rotmai(SPUInstruction):
  machine_inst = OPCD_I7_A_T
  params = {'OPCD':122}
  cycles = (0, 4, 0)


class csflt(SPUInstruction):
  machine_inst = OPCD_I8_A_T
  params = {'OPCD':474}
  cycles = (0, 7, 0)


class cflts(SPUInstruction):
  machine_inst = OPCD_I8_A_T
  params = {'OPCD':472}
  cycles = (0, 7, 0)


class cuflt(SPUInstruction):
  machine_inst = OPCD_I8_A_T
  params = {'OPCD':475}
  cycles = (0, 7, 0)


class cfltu(SPUInstruction):
  machine_inst = OPCD_I8_A_T
  params = {'OPCD':473}
  cycles = (0, 7, 0)


class lqd(SPUInstruction):
  machine_inst = OPCD_I10_A_T
  params = {'OPCD':52}
  cycles = (1, 6, 0)


class stqd(SPUInstruction):
  machine_inst = OPCD_I10_A_T
  params = {'OPCD':36}
  cycles = (1, 6, 0)


class ahi(SPUInstruction):
  machine_inst = OPCD_I10_A_T
  params = {'OPCD':29}
  cycles = (0, 2, 0)


class ai(SPUInstruction):
  machine_inst = OPCD_I10_A_T
  params = {'OPCD':28}
  cycles = (0, 2, 0)


class sfhi(SPUInstruction):
  machine_inst = OPCD_I10_A_T
  params = {'OPCD':13}
  cycles = (0, 2, 0)


class sfi(SPUInstruction):
  machine_inst = OPCD_I10_A_T
  params = {'OPCD':12}
  cycles = (0, 2, 0)


class mpyi(SPUInstruction):
  machine_inst = OPCD_I10_A_T
  params = {'OPCD':116}
  cycles = (0, 7, 0)


class mpyui(SPUInstruction):
  machine_inst = OPCD_I10_A_T
  params = {'OPCD':117}
  cycles = (0, 7, 0)


class andbi(SPUInstruction):
  machine_inst = OPCD_I10_A_T
  params = {'OPCD':22}
  cycles = (0, 2, 0)


class andhi(SPUInstruction):
  machine_inst = OPCD_I10_A_T
  params = {'OPCD':21}
  cycles = (0, 2, 0)


class andi(SPUInstruction):
  machine_inst = OPCD_I10_A_T
  params = {'OPCD':20}
  cycles = (0, 2, 0)


class orbi(SPUInstruction):
  machine_inst = OPCD_I10_A_T
  params = {'OPCD':6}
  cycles = (0, 2, 0)


class orhi(SPUInstruction):
  machine_inst = OPCD_I10_A_T
  params = {'OPCD':5}
  cycles = (0, 2, 0)


class ori(SPUInstruction):
  machine_inst = OPCD_I10_A_T
  params = {'OPCD':4}
  cycles = (0, 2, 0)


class xorbi(SPUInstruction):
  machine_inst = OPCD_I10_A_T
  params = {'OPCD':70}
  cycles = (0, 2, 0)


class xorhi(SPUInstruction):
  machine_inst = OPCD_I10_A_T
  params = {'OPCD':69}
  cycles = (0, 2, 0)


class xori(SPUInstruction):
  machine_inst = OPCD_I10_A_T
  params = {'OPCD':68}
  cycles = (0, 2, 0)


class heqi(SPUInstruction):
  machine_inst = OPCD_I10_A_T
  params = {'OPCD':127}
  cycles = (0, 2, 0)


class hgti(SPUInstruction):
  machine_inst = OPCD_I10_A_T
  params = {'OPCD':79}
  cycles = (0, 2, 0)


class hlgti(SPUInstruction):
  machine_inst = OPCD_I10_A_T
  params = {'OPCD':95}
  cycles = (0, 2, 0)


class ceqbi(SPUInstruction):
  machine_inst = OPCD_I10_A_T
  params = {'OPCD':126}
  cycles = (0, 2, 0)


class ceqhi(SPUInstruction):
  machine_inst = OPCD_I10_A_T
  params = {'OPCD':125}
  cycles = (0, 2, 0)


class ceqi(SPUInstruction):
  machine_inst = OPCD_I10_A_T
  params = {'OPCD':124}
  cycles = (0, 2, 0)


class cgtbi(SPUInstruction):
  machine_inst = OPCD_I10_A_T
  params = {'OPCD':78}
  cycles = (0, 2, 0)


class cgthi(SPUInstruction):
  machine_inst = OPCD_I10_A_T
  params = {'OPCD':77}
  cycles = (0, 2, 0)


class cgti(SPUInstruction):
  machine_inst = OPCD_I10_A_T
  params = {'OPCD':76}
  cycles = (0, 2, 0)


class clgtbi(SPUInstruction):
  machine_inst = OPCD_I10_A_T
  params = {'OPCD':94}
  cycles = (0, 2, 0)


class clgthi(SPUInstruction):
  machine_inst = OPCD_I10_A_T
  params = {'OPCD':93}
  cycles = (0, 2, 0)


class clgti(SPUInstruction):
  machine_inst = OPCD_I10_A_T
  params = {'OPCD':92}
  cycles = (0, 2, 0)


class lqa(SPUInstruction):
  machine_inst = OPCD_I16_T
  params = {'OPCD':97}
  cycles = (1, 6, 0)


class lqr(SPUInstruction):
  machine_inst = OPCD_I16_T
  params = {'OPCD':103}
  cycles = (1, 6, 0)


class stqa(SPUInstruction):
  machine_inst = OPCD_I16_T
  params = {'OPCD':65}
  cycles = (1, 6, 0)


class stqr(SPUInstruction):
  machine_inst = OPCD_I16_T
  params = {'OPCD':71}
  cycles = (1, 6, 0)


class ilh(SPUInstruction):
  machine_inst = OPCD_I16_T
  params = {'OPCD':131}
  cycles = (0, 2, 0)


class ilhu(SPUInstruction):
  machine_inst = OPCD_I16_T
  params = {'OPCD':130}
  cycles = (0, 2, 0)


class il(SPUInstruction):
  machine_inst = OPCD_I16_T
  params = {'OPCD':129}
  cycles = (0, 2, 0)


class iohl(SPUInstruction):
  machine_inst = OPCD_I16_T
  params = {'OPCD':193}
  cycles = (0, 2, 0)


class fsmbi(SPUInstruction):
  machine_inst = OPCD_I16_T
  params = {'OPCD':101}
  cycles = (1, 4, 0)


class br(SPUDispatchInstruction):
  cycles = (1, 4, 0)
  dispatch = (
    (OPCD_I16,    {'OPCD':100}),
    (OPCD_LBL16,  {'OPCD':100}))


# TODO - how can I do absolute branches?
class bra(SPUInstruction):
  machine_inst = OPCD_I16
  params = {'OPCD':96}
  cycles = (1, 4, 0)


class brsl(SPUDispatchInstruction):
  cycles = (1, 4, 0)
  dispatch = (
    (OPCD_I16,    {'OPCD':102}),
    (OPCD_LBL16,  {'OPCD':102}))


class brasl(SPUInstruction):
  machine_inst = OPCD_I16_T
  params = {'OPCD':98}
  cycles = (1, 4, 0)


class brnz(SPUDispatchInstruction):
  cycles = (1, 4, 0)
  dispatch = (
    (OPCD_I16_T,    {'OPCD':66}),
    (OPCD_LBL16_T,  {'OPCD':66}))


class brz(SPUDispatchInstruction):
  cycles = (1, 4, 0)
  dispatch = (
    (OPCD_I16_T,    {'OPCD':64}),
    (OPCD_LBL16_T,  {'OPCD':64}))


class brhnz(SPUDispatchInstruction):
  cycles = (1, 4, 0)
  dispatch = (
    (OPCD_I16,    {'OPCD':70}),
    (OPCD_LBL16,  {'OPCD':70}))


class brhz(SPUDispatchInstruction):
  cycles = (1, 4, 0)
  dispatch = (
    (OPCD_I16,    {'OPCD':68}),
    (OPCD_LBL16,  {'OPCD':68}))


class hbra(SPUInstruction):
  machine_inst = OPCD_ROA_I16
  params = {'OPCD':8}
  cycles = (1, 15, 0)


class hbrr(SPUDispatchInstruction):
  cycles = (1, 15, 0)
  dispatch = (
    (OPCD_ROA_I16,     {'OPCD':9}),
    (OPCD_LBL9_LBL16,  {'OPCD':9}))


class ila(SPUInstruction):
  machine_inst = OPCD_I18_T
  params = {'OPCD':33}
  cycles = (0, 2, 0)


