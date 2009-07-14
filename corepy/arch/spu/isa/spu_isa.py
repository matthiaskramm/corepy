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

from corepy.spre.spe import Instruction, DispatchInstruction, Register
from spu_insts import *

__doc__="""
ISA for the Cell Broadband Engine's SPU.
"""

class lqx(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':452}
  cycles = (1, 6, 0)


class stqx(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':324}
  cycles = (1, 6, 0)


class cbx(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':468}
  cycles = (1, 4, 0)


class chx(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':469}
  cycles = (1, 4, 0)


class cwx(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':470}
  cycles = (1, 4, 0)


class cdx(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':471}
  cycles = (1, 4, 0)


class ah(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':200}
  cycles = (0, 2, 0)


class a(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':192}
  cycles = (0, 2, 0)


class sfh(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':72}
  cycles = (0, 2, 0)


class sf(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':64}
  cycles = (0, 2, 0)


class addx(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':832}
  cycles = (0, 2, 0)


class cg(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':194}
  cycles = (0, 2, 0)


class cgx(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':834}
  cycles = (0, 2, 0)


class sfx(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':833}
  cycles = (0, 2, 0)


class bg(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':66}
  cycles = (0, 2, 0)


class bgx(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':835}
  cycles = (0, 2, 0)


class mpy(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':964}
  cycles = (0, 7, 0)


class mpyu(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':972}
  cycles = (0, 7, 0)


class mpyh(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':965}
  cycles = (0, 7, 0)


class mpys(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':967}
  cycles = (0, 7, 0)


class mpyhh(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':966}
  cycles = (0, 7, 0)


class mpyhha(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':838}
  cycles = (0, 7, 0)


class mpyhhu(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':974}
  cycles = (0, 7, 0)


class mpyhhau(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':846}
  cycles = (0, 7, 0)


class clz(Instruction):
  machine_inst = OPCD_A_T
  params = {'OPCD':677}
  cycles = (0, 2, 0)


class cntb(Instruction):
  machine_inst = OPCD_A_T
  params = {'OPCD':692}
  cycles = (0, 4, 0)


class fsmb(Instruction):
  machine_inst = OPCD_A_T
  params = {'OPCD':438}
  cycles = (1, 4, 0)


class fsmh(Instruction):
  machine_inst = OPCD_A_T
  params = {'OPCD':437}
  cycles = (1, 4, 0)


class fsm(Instruction):
  machine_inst = OPCD_A_T
  params = {'OPCD':436}
  cycles = (1, 4, 0)


class gbb(Instruction):
  machine_inst = OPCD_A_T
  params = {'OPCD':434}
  cycles = (1, 4, 0)


class gbh(Instruction):
  machine_inst = OPCD_A_T
  params = {'OPCD':433}
  cycles = (1, 4, 0)


class gb(Instruction):
  machine_inst = OPCD_A_T
  params = {'OPCD':432}
  cycles = (1, 4, 0)


class avgb(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':211}
  cycles = (0, 4, 0)


class absdb(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':83}
  cycles = (0, 4, 0)


class sumb(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':595}
  cycles = (0, 4, 0)


class xsbh(Instruction):
  machine_inst = OPCD_A_T
  params = {'OPCD':694}
  cycles = (0, 2, 0)


class xshw(Instruction):
  machine_inst = OPCD_A_T
  params = {'OPCD':686}
  cycles = (0, 2, 0)


class xswd(Instruction):
  machine_inst = OPCD_A_T
  params = {'OPCD':678}
  cycles = (0, 2, 0)


class and_(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':193}
  cycles = (0, 2, 0)


class andc(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':705}
  cycles = (0, 2, 0)


class or_(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':65}
  cycles = (0, 2, 0)


class orc(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':713}
  cycles = (0, 2, 0)


class orx(Instruction):
  machine_inst = OPCD_A_T
  params = {'OPCD':496}
  cycles = (1, 4, 0)


class xor(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':577}
  cycles = (0, 2, 0)


class nand(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':201}
  cycles = (0, 2, 0)


class nor(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':73}
  cycles = (0, 2, 0)


class eqv(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':585}
  cycles = (0, 2, 0)


class shlh(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':95}
  cycles = (0, 4, 0)


class shl(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':91}
  cycles = (0, 4, 0)


class shlqbi(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':475}
  cycles = (1, 4, 0)


class shlqby(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':479}
  cycles = (1, 4, 0)


class shlqbybi(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':463}
  cycles = (1, 4, 0)


class roth(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':92}
  cycles = (0, 4, 0)


class rot(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':88}
  cycles = (0, 4, 0)


class rotqby(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':476}
  cycles = (1, 4, 0)


class rotqbybi(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':460}
  cycles = (1, 4, 0)


class rotqbi(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':472}
  cycles = (1, 4, 0)


class rothm(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':93}
  cycles = (0, 4, 0)


class rotm(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':89}
  cycles = (0, 4, 0)


class rotqmby(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':477}
  cycles = (1, 4, 0)


class rotqmbybi(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':461}
  cycles = (1, 4, 0)


class rotqmbi(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':473}
  cycles = (1, 4, 0)


class rotmah(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':94}
  cycles = (0, 4, 0)


class rotma(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':90}
  cycles = (0, 4, 0)


class heq(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':984}
  cycles = (0, 2, 0)


class hgt(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':600}
  cycles = (0, 2, 0)


class hlgt(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':728}
  cycles = (0, 2, 0)


class ceqb(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':976}
  cycles = (0, 2, 0)


class ceqh(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':968}
  cycles = (0, 2, 0)


class ceq(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':960}
  cycles = (0, 2, 0)


class cgtb(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':592}
  cycles = (0, 2, 0)


class cgth(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':584}
  cycles = (0, 2, 0)


class cgt(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':576}
  cycles = (0, 2, 0)


class clgtb(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':720}
  cycles = (0, 2, 0)


class clgth(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':712}
  cycles = (0, 2, 0)


class clgt(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':704}
  cycles = (0, 2, 0)


class bi(Instruction):
  machine_inst = OPCD_A_D_E
  params = {'OPCD':424}
  cycles = (1, 4, 0)


class iret(Instruction):
  machine_inst = OPCD_A_D_E
  params = {'OPCD':426}
  cycles = (1, 4, 0)


class bisled(Instruction):
  machine_inst = OPCD_A_T_D_E
  params = {'OPCD':427}
  cycles = (1, 4, 0)


class bisl(Instruction):
  machine_inst = OPCD_A_T_D_E
  params = {'OPCD':425}
  cycles = (1, 4, 0)


class biz(Instruction):
  machine_inst = OPCD_A_T_D_E
  params = {'OPCD':296}
  cycles = (1, 4, 0)


class binz(Instruction):
  machine_inst = OPCD_A_T_D_E
  params = {'OPCD':297}
  cycles = (1, 4, 0)


class bihz(Instruction):
  machine_inst = OPCD_A_T_D_E
  params = {'OPCD':294}
  cycles = (1, 4, 0)


class bihnz(Instruction):
  machine_inst = OPCD_A_T_D_E
  params = {'OPCD':299}
  cycles = (1, 4, 0)


# TODO - can we check that if P is set then RO is zero as required?
class hbr(DispatchInstruction):
  cycles = (1, 15, 0)
  dispatch = (
    (OPCD_RO_A_P,   {'OPCD':428}),
    (OPCD_LBL9_A_P, {'OPCD':428}))


class fa(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':708}
  cycles = (0, 6, 0)


class dfa(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':716}
  cycles = (0, 13, 6)


class fs(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':709}
  cycles = (0, 6, 0)


class dfs(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':717}
  cycles = (0, 13, 6)


class fm(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':710}
  cycles = (0, 6, 0)


class dfm(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':718}
  cycles = (0, 13, 6)


class dfma(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':860}
  cycles = (0, 13, 6)


class dfnms(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':862}
  cycles = (0, 13, 6)


class dfms(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':861}
  cycles = (0, 13, 6)


class dfnma(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':863}
  cycles = (0, 13, 6)


class frest(Instruction):
  machine_inst = OPCD_A_T
  params = {'OPCD':440}
  cycles = (1, 4, 0)


class frsqest(Instruction):
  machine_inst = OPCD_A_T
  params = {'OPCD':441}
  cycles = (1, 4, 0)


class fi(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':980}
  cycles = (0, 7, 0)


class frds(Instruction):
  machine_inst = OPCD_A_T
  params = {'OPCD':953}
  cycles = (0, 13, 6)


class fesd(Instruction):
  machine_inst = OPCD_A_T
  params = {'OPCD':952}
  cycles = (0, 13, 6)


class fceq(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':962}
  cycles = (0, 2, 0)


class fcmeq(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':970}
  cycles = (0, 2, 0)


class fcgt(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':706}
  cycles = (0, 2, 0)


class fcmgt(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':714}
  cycles = (0, 2, 0)


class fscrwr(Instruction):
  machine_inst = OPCD_A_T
  params = {'OPCD':954}
  cycles = (0, 7, 0)


class fscrrd(Instruction):
  machine_inst = OPCD_T
  params = {'OPCD':920}
  cycles = (0, 13, 6)


class stop(Instruction):
  machine_inst = OPCD_STOP_SIG
  params = {'OPCD':0}
  cycles = (1, 4, 0)


class stopd(Instruction):
  machine_inst = OPCD_B_A_T
  params = {'OPCD':320}
  cycles = (1, 4, 0)


class lnop(Instruction):
  machine_inst = OPCD
  params = {'OPCD':1}
  cycles = (1, 0, 0)


class nop(Instruction):
  machine_inst = OPCD_T
  params = {'OPCD':513}
  cycles = (0, 0, 0)


class sync(Instruction):
  machine_inst = OPCD_CF
  params = {'OPCD':2}
  cycles = (1, 4, 0)


class dsync(Instruction):
  machine_inst = OPCD
  params = {'OPCD':3}
  cycles = (1, 4, 0)


class mfspr(Instruction):
  machine_inst = OPCD_SA_T
  params = {'OPCD':12}
  cycles = (1, 6, 0)


class mtspr(Instruction):
  machine_inst = OPCD_SA_T
  params = {'OPCD':268}
  cycles = (1, 6, 0)


class rdch(Instruction):
  machine_inst = OPCD_A_T
  params = {'OPCD':13}
  cycles = (1, 6, 0)


class rchcnt(Instruction):
  machine_inst = OPCD_A_T
  params = {'OPCD':15}
  cycles = (1, 6, 0)


class wrch(Instruction):
  machine_inst = OPCD_A_T
  params = {'OPCD':269}
  cycles = (1, 6, 0)


class mpya(Instruction):
  machine_inst = OPCD_T_B_A_C
  params = {'OPCD':12}
  cycles = (0, 7, 0)


class selb(Instruction):
  machine_inst = OPCD_T_B_A_C
  params = {'OPCD':8}
  cycles = (0, 2, 0)


class shufb(Instruction):
  machine_inst = OPCD_T_B_A_C
  params = {'OPCD':11}
  cycles = (1, 4, 0)


class fma(Instruction):
  machine_inst = OPCD_T_B_A_C
  params = {'OPCD':14}
  cycles = (0, 6, 0)


class fnms(Instruction):
  machine_inst = OPCD_T_B_A_C
  params = {'OPCD':13}
  cycles = (0, 6, 0)


class fms(Instruction):
  machine_inst = OPCD_T_B_A_C
  params = {'OPCD':15}
  cycles = (0, 6, 0)


class cbd(Instruction):
  machine_inst = OPCD_I7_A_T
  params = {'OPCD':500}
  cycles = (1, 4, 0)


class chd(Instruction):
  machine_inst = OPCD_I7_A_T
  params = {'OPCD':501}
  cycles = (1, 4, 0)


class cwd(Instruction):
  machine_inst = OPCD_I7_A_T
  params = {'OPCD':502}
  cycles = (1, 4, 0)


class cdd(Instruction):
  machine_inst = OPCD_I7_A_T
  params = {'OPCD':503}
  cycles = (1, 4, 0)


class shlhi(Instruction):
  machine_inst = OPCD_I7_A_T
  params = {'OPCD':127}
  cycles = (0, 4, 0)


class shli(Instruction):
  machine_inst = OPCD_I7_A_T
  params = {'OPCD':123}
  cycles = (0, 4, 0)


class shlqbii(Instruction):
  machine_inst = OPCD_I7_A_T
  params = {'OPCD':507}
  cycles = (1, 4, 0)


class shlqbyi(Instruction):
  machine_inst = OPCD_I7_A_T
  params = {'OPCD':511}
  cycles = (1, 4, 0)


class rothi(Instruction):
  machine_inst = OPCD_I7_A_T
  params = {'OPCD':124}
  cycles = (0, 4, 0)


class roti(Instruction):
  machine_inst = OPCD_I7_A_T
  params = {'OPCD':120}
  cycles = (0, 4, 0)


class rotqbyi(Instruction):
  machine_inst = OPCD_I7_A_T
  params = {'OPCD':508}
  cycles = (1, 4, 0)


class rotqbii(Instruction):
  machine_inst = OPCD_I7_A_T
  params = {'OPCD':504}
  cycles = (1, 4, 0)


class rothmi(Instruction):
  machine_inst = OPCD_I7_A_T
  params = {'OPCD':125}
  cycles = (0, 4, 0)


class rotmi(Instruction):
  machine_inst = OPCD_I7_A_T
  params = {'OPCD':121}
  cycles = (0, 4, 0)


class rotqmbyi(Instruction):
  machine_inst = OPCD_I7_A_T
  params = {'OPCD':509}
  cycles = (1, 4, 0)


class rotqmbii(Instruction):
  machine_inst = OPCD_I7_A_T
  params = {'OPCD':505}
  cycles = (1, 4, 0)


class rotmahi(Instruction):
  machine_inst = OPCD_I7_A_T
  params = {'OPCD':126}
  cycles = (0, 4, 0)


class rotmai(Instruction):
  machine_inst = OPCD_I7_A_T
  params = {'OPCD':122}
  cycles = (0, 4, 0)


class csflt(Instruction):
  machine_inst = OPCD_I8_A_T
  params = {'OPCD':474}
  cycles = (0, 7, 0)


class cflts(Instruction):
  machine_inst = OPCD_I8_A_T
  params = {'OPCD':472}
  cycles = (0, 7, 0)


class cuflt(Instruction):
  machine_inst = OPCD_I8_A_T
  params = {'OPCD':475}
  cycles = (0, 7, 0)


class cfltu(Instruction):
  machine_inst = OPCD_I8_A_T
  params = {'OPCD':473}
  cycles = (0, 7, 0)


class lqd(Instruction):
  machine_inst = OPCD_I10_A_T
  params = {'OPCD':52}
  cycles = (1, 6, 0)


class stqd(Instruction):
  machine_inst = OPCD_I10_A_T
  params = {'OPCD':36}
  cycles = (1, 6, 0)


class ahi(Instruction):
  machine_inst = OPCD_I10_A_T
  params = {'OPCD':29}
  cycles = (0, 2, 0)


class ai(Instruction):
  machine_inst = OPCD_I10_A_T
  params = {'OPCD':28}
  cycles = (0, 2, 0)


class sfhi(Instruction):
  machine_inst = OPCD_I10_A_T
  params = {'OPCD':13}
  cycles = (0, 2, 0)


class sfi(Instruction):
  machine_inst = OPCD_I10_A_T
  params = {'OPCD':12}
  cycles = (0, 2, 0)


class mpyi(Instruction):
  machine_inst = OPCD_I10_A_T
  params = {'OPCD':116}
  cycles = (0, 7, 0)


class mpyui(Instruction):
  machine_inst = OPCD_I10_A_T
  params = {'OPCD':117}
  cycles = (0, 7, 0)


class andbi(Instruction):
  machine_inst = OPCD_I10_A_T
  params = {'OPCD':22}
  cycles = (0, 2, 0)


class andhi(Instruction):
  machine_inst = OPCD_I10_A_T
  params = {'OPCD':21}
  cycles = (0, 2, 0)


class andi(Instruction):
  machine_inst = OPCD_I10_A_T
  params = {'OPCD':20}
  cycles = (0, 2, 0)


class orbi(Instruction):
  machine_inst = OPCD_I10_A_T
  params = {'OPCD':6}
  cycles = (0, 2, 0)


class orhi(Instruction):
  machine_inst = OPCD_I10_A_T
  params = {'OPCD':5}
  cycles = (0, 2, 0)


class ori(Instruction):
  machine_inst = OPCD_I10_A_T
  params = {'OPCD':4}
  cycles = (0, 2, 0)


class xorbi(Instruction):
  machine_inst = OPCD_I10_A_T
  params = {'OPCD':70}
  cycles = (0, 2, 0)


class xorhi(Instruction):
  machine_inst = OPCD_I10_A_T
  params = {'OPCD':69}
  cycles = (0, 2, 0)


class xori(Instruction):
  machine_inst = OPCD_I10_A_T
  params = {'OPCD':68}
  cycles = (0, 2, 0)


class heqi(Instruction):
  machine_inst = OPCD_I10_A_T
  params = {'OPCD':127}
  cycles = (0, 2, 0)


class hgti(Instruction):
  machine_inst = OPCD_I10_A_T
  params = {'OPCD':79}
  cycles = (0, 2, 0)


class hlgti(Instruction):
  machine_inst = OPCD_I10_A_T
  params = {'OPCD':95}
  cycles = (0, 2, 0)


class ceqbi(Instruction):
  machine_inst = OPCD_I10_A_T
  params = {'OPCD':126}
  cycles = (0, 2, 0)


class ceqhi(Instruction):
  machine_inst = OPCD_I10_A_T
  params = {'OPCD':125}
  cycles = (0, 2, 0)


class ceqi(Instruction):
  machine_inst = OPCD_I10_A_T
  params = {'OPCD':124}
  cycles = (0, 2, 0)


class cgtbi(Instruction):
  machine_inst = OPCD_I10_A_T
  params = {'OPCD':78}
  cycles = (0, 2, 0)


class cgthi(Instruction):
  machine_inst = OPCD_I10_A_T
  params = {'OPCD':77}
  cycles = (0, 2, 0)


class cgti(Instruction):
  machine_inst = OPCD_I10_A_T
  params = {'OPCD':76}
  cycles = (0, 2, 0)


class clgtbi(Instruction):
  machine_inst = OPCD_I10_A_T
  params = {'OPCD':94}
  cycles = (0, 2, 0)


class clgthi(Instruction):
  machine_inst = OPCD_I10_A_T
  params = {'OPCD':93}
  cycles = (0, 2, 0)


class clgti(Instruction):
  machine_inst = OPCD_I10_A_T
  params = {'OPCD':92}
  cycles = (0, 2, 0)


class lqa(Instruction):
  machine_inst = OPCD_I16_T
  params = {'OPCD':97}
  cycles = (1, 6, 0)


class lqr(Instruction):
  machine_inst = OPCD_I16_T
  params = {'OPCD':103}
  cycles = (1, 6, 0)


class stqa(Instruction):
  machine_inst = OPCD_I16_T
  params = {'OPCD':65}
  cycles = (1, 6, 0)


class stqr(Instruction):
  machine_inst = OPCD_I16_T
  params = {'OPCD':71}
  cycles = (1, 6, 0)


class ilh(Instruction):
  machine_inst = OPCD_I16_T
  params = {'OPCD':131}
  cycles = (0, 2, 0)


class ilhu(Instruction):
  machine_inst = OPCD_I16_T
  params = {'OPCD':130}
  cycles = (0, 2, 0)


class il(Instruction):
  machine_inst = OPCD_I16_T
  params = {'OPCD':129}
  cycles = (0, 2, 0)


class iohl(Instruction):
  machine_inst = OPCD_I16_T
  params = {'OPCD':193}
  cycles = (0, 2, 0)


class fsmbi(Instruction):
  machine_inst = OPCD_I16_T
  params = {'OPCD':101}
  cycles = (1, 4, 0)


class br(DispatchInstruction):
  cycles = (1, 4, 0)
  dispatch = (
    (OPCD_I16,    {'OPCD':100}),
    (OPCD_LBL16,  {'OPCD':100}))


# TODO - how can I do absolute branches?
class bra(Instruction):
  machine_inst = OPCD_I16
  params = {'OPCD':96}
  cycles = (1, 4, 0)


# TODO - I16 has two zero bits appended, do I handle this correctly?
# What is the correct way, anyway?
class brsl(DispatchInstruction):
  cycles = (1, 4, 0)
  dispatch = (
    (OPCD_I16_T,    {'OPCD':102}),
    (OPCD_LBL16_T,  {'OPCD':102}))


class brasl(Instruction):
  machine_inst = OPCD_I16_T
  params = {'OPCD':98}
  cycles = (1, 4, 0)


class brnz(DispatchInstruction):
  cycles = (1, 4, 0)
  dispatch = (
    (OPCD_I16_T,    {'OPCD':66}),
    (OPCD_LBL16_T,  {'OPCD':66}))


class brz(DispatchInstruction):
  cycles = (1, 4, 0)
  dispatch = (
    (OPCD_I16_T,    {'OPCD':64}),
    (OPCD_LBL16_T,  {'OPCD':64}))


class brhnz(DispatchInstruction):
  cycles = (1, 4, 0)
  dispatch = (
    (OPCD_I16,    {'OPCD':70}),
    (OPCD_LBL16,  {'OPCD':70}))


class brhz(DispatchInstruction):
  cycles = (1, 4, 0)
  dispatch = (
    (OPCD_I16,    {'OPCD':68}),
    (OPCD_LBL16,  {'OPCD':68}))


class hbra(Instruction):
  machine_inst = OPCD_LBL9_I16
  params = {'OPCD':8}
  cycles = (1, 15, 0)


class hbrr(DispatchInstruction):
  cycles = (1, 15, 0)
  dispatch = (
    (OPCD_ROA_I16,     {'OPCD':9}),
    (OPCD_LBL9_LBL16,  {'OPCD':9}))


class ila(Instruction):
  machine_inst = OPCD_I18_T
  params = {'OPCD':33}
  cycles = (0, 2, 0)


