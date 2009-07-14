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
from ppc_insts import *

__doc__="""
PowerPC User Model Instruction Set Architectre (ISA) and User
Model Virtual Environment Architecture (VEA).

To use, import this module and call the Instructions as Python
functions to generate a properly coded version.  For example, to
create an addx instruction:

import ppc_isa as ppc

inst = ppc.addx(3, 4, 5) # add r4 to r5 and place the result in r3

Operands are in the same order as presented in the Programming
Environments manual.

For a complete reference and details for all instructions, please
referer to: 

'PowerPC Microprocessor Family: The Programming Environments for
 32-Bit Microprocessors' from IBM.

URL (valid as of June 1, 2006):
http://www-306.ibm.com/chips/techlib/techlib.nsf/techdocs/852569B20050FF778525699600719DF2
"""


class addx(Instruction):
  machine_inst = OPCD_D_A_B_OE_XO_2_Rc
  params = {'OPCD':31, 'XO':266}

class addcx(Instruction):
  machine_inst = OPCD_D_A_B_OE_XO_2_Rc
  params = {'OPCD':31, 'XO':10}

class addex(Instruction):
  machine_inst = OPCD_D_A_B_OE_XO_2_Rc
  params = {'OPCD':31, 'XO':138}

class addi(Instruction):
  machine_inst = OPCD_D_A_SIMM
  params = {'OPCD':14}

class addic(Instruction):
  machine_inst = OPCD_D_A_SIMM
  params = {'OPCD':12}

class addic_(Instruction):
  machine_inst = OPCD_D_A_SIMM
  params = {'OPCD':13}

class addis(Instruction):
  machine_inst = OPCD_D_A_SIMM
  params = {'OPCD':15}

class addmex(Instruction):
  machine_inst = OPCD_D_A_OE_XO_2_Rc
  params = {'OPCD':31, 'XO':234}

class addzex(Instruction):
  machine_inst = OPCD_D_A_OE_XO_2_Rc
  params = {'OPCD':31, 'XO':202}

class andx(Instruction):
  machine_inst = OPCD_S_A_B_XO_1_Rc
  params = {'OPCD':31, 'XO':28}

class andcx(Instruction):
  machine_inst = OPCD_S_A_B_XO_1_Rc
  params = {'OPCD':31, 'XO':60}

class andi(Instruction):
  machine_inst = OPCD_S_A_UIMM
  params = {'OPCD':28}

class andis(Instruction):
  machine_inst = OPCD_S_A_UIMM
  params = {'OPCD':29}

class bx(DispatchInstruction):
  dispatch = (
    (OPCD_LI_AA_LK,     {'OPCD':18}),
    (OPCD_LILBL_AA_LK,  {'OPCD':18}))

class bcx(DispatchInstruction):
  dispatch = (
    (OPCD_BO_BI_BD_AA_LK,    {'OPCD':16}),
    (OPCD_BO_BI_BDLBL_AA_LK, {'OPCD':16}))

class bcctrx(Instruction):
  machine_inst = OPCD_BO_BI_XO_1_LK
  params = {'OPCD':19, 'XO':528}

class bclrx(Instruction):
  machine_inst = OPCD_BO_BI_XO_1_LK
  params = {'OPCD':19, 'XO':16}

class cmp_(Instruction):
  machine_inst = OPCD_crfD_L_A_B
  params = {'OPCD':31}

class cmpi(Instruction):
  machine_inst = OPCD_crfD_L_A_SIMM
  params = {'OPCD':11}

class cmpl(Instruction):
  machine_inst = OPCD_crfD_L_A_B_XO_1
  params = {'OPCD':31, 'XO':32}

class cmpli(Instruction):
  machine_inst = OPCD_crfD_L_A_UIMM
  params = {'OPCD':10}

class cntlzwx(Instruction):
  machine_inst = OPCD_S_A_XO_1_Rc
  params = {'OPCD':31, 'XO':26}

class crand(Instruction):
  machine_inst = OPCD_crbD_crbA_crbB_XO_1
  params = {'OPCD':19, 'XO':257}

class crandc(Instruction):
  machine_inst = OPCD_crbD_crbA_crbB_XO_1
  params = {'OPCD':19, 'XO':129}

class creqv(Instruction):
  machine_inst = OPCD_crbD_crbA_crbB_XO_1
  params = {'OPCD':19, 'XO':289}

class crnand(Instruction):
  machine_inst = OPCD_crbD_crbA_crbB_XO_1
  params = {'OPCD':19, 'XO':225}

class crnor(Instruction):
  machine_inst = OPCD_crbD_crbA_crbB_XO_1
  params = {'OPCD':19, 'XO':33}

class cror(Instruction):
  machine_inst = OPCD_crbD_crbA_crbB_XO_1
  params = {'OPCD':19, 'XO':449}

class crorc(Instruction):
  machine_inst = OPCD_crbD_crbA_crbB_XO_1
  params = {'OPCD':19, 'XO':417}

class crxor(Instruction):
  machine_inst = OPCD_crbD_crbA_crbB_XO_1
  params = {'OPCD':19, 'XO':193}

class dcba(Instruction):
  machine_inst = OPCD_A_B_XO_1
  params = {'OPCD':31, 'XO':758}

class dcbf(Instruction):
  machine_inst = OPCD_A_B_XO_1
  params = {'OPCD':31, 'XO':86}

class dcbi(Instruction):
  machine_inst = OPCD_A_B_XO_1
  params = {'OPCD':31, 'XO':470}

class dcbst(Instruction):
  machine_inst = OPCD_A_B_XO_1
  params = {'OPCD':31, 'XO':54}

class dcbt(Instruction):
  machine_inst = OPCD_A_B_XO_1
  params = {'OPCD':31, 'XO':278}

class dcbtst(Instruction):
  machine_inst = OPCD_A_B_XO_1
  params = {'OPCD':31, 'XO':246}

class dcbz(Instruction):
  machine_inst = OPCD_A_B_XO_1
  params = {'OPCD':31, 'XO':1014}

class divwx(Instruction):
  machine_inst = OPCD_D_A_B_OE_XO_2_Rc
  params = {'OPCD':31, 'XO':491}

class divwux(Instruction):
  machine_inst = OPCD_D_A_B_OE_XO_2_Rc
  params = {'OPCD':31, 'XO':459}

class eciwx(Instruction):
  machine_inst = OPCD_D_A_B_XO_1
  params = {'OPCD':31, 'XO':310}

class ecowx(Instruction):
  machine_inst = OPCD_S_A_B_XO_1
  params = {'OPCD':31, 'XO':438}

class eieio(Instruction):
  machine_inst = OPCD_XO_1
  params = {'OPCD':31, 'XO':854}

class eqvx(Instruction):
  machine_inst = OPCD_S_A_B_XO_1_Rc
  params = {'OPCD':31, 'XO':284}

class extsbx(Instruction):
  machine_inst = OPCD_S_A_XO_1_Rc
  params = {'OPCD':31, 'XO':954}

class extshx(Instruction):
  machine_inst = OPCD_S_A_XO_1_Rc
  params = {'OPCD':31, 'XO':922}

class fabsx(Instruction):
  machine_inst = OPCD_D_B_XO_1_Rc
  params = {'OPCD':63, 'XO':264}

class faddx(Instruction):
  machine_inst = OPCD_D_A_B_XO_3_Rc
  params = {'OPCD':63, 'XO':21}

class faddsx(Instruction):
  machine_inst = OPCD_D_A_B_XO_3_Rc
  params = {'OPCD':59, 'XO':21}

class fcmpo(Instruction):
  machine_inst = OPCD_crfD_A_B_XO_1
  params = {'OPCD':63, 'XO':32}

class fcmpu(Instruction):
  machine_inst = OPCD_crfD_A_B
  params = {'OPCD':63}

class fctiwx(Instruction):
  machine_inst = OPCD_D_B_XO_1_Rc
  params = {'OPCD':63, 'XO':14}

class fctiwzx(Instruction):
  machine_inst = OPCD_D_B_XO_1_Rc
  params = {'OPCD':63, 'XO':15}

class fdivx(Instruction):
  machine_inst = OPCD_D_A_B_XO_3_Rc
  params = {'OPCD':63, 'XO':18}

class fdivsx(Instruction):
  machine_inst = OPCD_D_A_B_XO_3_Rc
  params = {'OPCD':59, 'XO':18}

class fmaddx(Instruction):
  machine_inst = OPCD_D_A_B_C_XO_3_Rc
  params = {'OPCD':63, 'XO':29}

class fmaddsx(Instruction):
  machine_inst = OPCD_D_A_B_C_XO_3_Rc
  params = {'OPCD':59, 'XO':29}

class fmrx(Instruction):
  machine_inst = OPCD_D_B_XO_1_Rc
  params = {'OPCD':63, 'XO':72}

class fmsubx(Instruction):
  machine_inst = OPCD_D_A_B_C_XO_3_Rc
  params = {'OPCD':63, 'XO':28}

class fmsubsx(Instruction):
  machine_inst = OPCD_D_A_B_C_XO_3_Rc
  params = {'OPCD':59, 'XO':28}

class fmulx(Instruction):
  machine_inst = OPCD_D_A_C_XO_3_Rc
  params = {'OPCD':63, 'XO':25}

class fmulsx(Instruction):
  machine_inst = OPCD_D_A_C_XO_3_Rc
  params = {'OPCD':59, 'XO':25}

class fnabsx(Instruction):
  machine_inst = OPCD_D_B_XO_1_Rc
  params = {'OPCD':63, 'XO':136}

class fnegx(Instruction):
  machine_inst = OPCD_D_B_XO_1_Rc
  params = {'OPCD':63, 'XO':40}

class fnmaddx(Instruction):
  machine_inst = OPCD_D_A_B_C_XO_3_Rc
  params = {'OPCD':63, 'XO':31}

class fnmaddsx(Instruction):
  machine_inst = OPCD_D_A_B_C_XO_3_Rc
  params = {'OPCD':59, 'XO':31}

class fnmsubx(Instruction):
  machine_inst = OPCD_D_A_B_C_XO_3_Rc
  params = {'OPCD':63, 'XO':30}

class fnmsubsx(Instruction):
  machine_inst = OPCD_D_A_B_C_XO_3_Rc
  params = {'OPCD':59, 'XO':30}

class fresx(Instruction):
  machine_inst = OPCD_D_B_XO_3_Rc
  params = {'OPCD':59, 'XO':24}

class frspx(Instruction):
  machine_inst = OPCD_D_B_XO_1_Rc
  params = {'OPCD':63, 'XO':12}

class frsqrtex(Instruction):
  machine_inst = OPCD_D_B_XO_3_Rc
  params = {'OPCD':63, 'XO':26}

class fselx(Instruction):
  machine_inst = OPCD_D_A_B_C_XO_3_Rc
  params = {'OPCD':63, 'XO':23}

class fsqrtx(Instruction):
  machine_inst = OPCD_D_B_XO_3_Rc
  params = {'OPCD':63, 'XO':22}

class fsqrtsx(Instruction):
  machine_inst = OPCD_D_B_XO_3_Rc
  params = {'OPCD':59, 'XO':22}

class fsubx(Instruction):
  machine_inst = OPCD_D_A_B_XO_3_Rc
  params = {'OPCD':63, 'XO':20}

class fsubsx(Instruction):
  machine_inst = OPCD_D_A_B_XO_3_Rc
  params = {'OPCD':59, 'XO':20}

class icbi(Instruction):
  machine_inst = OPCD_A_B_XO_1
  params = {'OPCD':31, 'XO':982}

class isync(Instruction):
  machine_inst = OPCD_XO_1
  params = {'OPCD':19, 'XO':150}

class lbz(Instruction):
  machine_inst = OPCD_D_A_d
  params = {'OPCD':34}

class lbzu(Instruction):
  machine_inst = OPCD_D_A_d
  params = {'OPCD':35}

class lbzux(Instruction):
  machine_inst = OPCD_D_A_B_XO_1
  params = {'OPCD':31, 'XO':119}

class lbzx(Instruction):
  machine_inst = OPCD_D_A_B_XO_1
  params = {'OPCD':31, 'XO':87}

class lfd(Instruction):
  machine_inst = OPCD_D_A_d
  params = {'OPCD':50}

class lfdu(Instruction):
  machine_inst = OPCD_D_A_d
  params = {'OPCD':51}

class lfdux(Instruction):
  machine_inst = OPCD_D_A_B_XO_1
  params = {'OPCD':31, 'XO':631}

class lfdx(Instruction):
  machine_inst = OPCD_D_A_B_XO_1
  params = {'OPCD':31, 'XO':599}

class lfs(Instruction):
  machine_inst = OPCD_D_A_d
  params = {'OPCD':48}

class lfsu(Instruction):
  machine_inst = OPCD_D_A_d
  params = {'OPCD':49}

class lfsux(Instruction):
  machine_inst = OPCD_D_A_B_XO_1
  params = {'OPCD':31, 'XO':567}

class lfsx(Instruction):
  machine_inst = OPCD_D_A_B_XO_1
  params = {'OPCD':31, 'XO':535}

class lha(Instruction):
  machine_inst = OPCD_D_A_d
  params = {'OPCD':42}

class lhau(Instruction):
  machine_inst = OPCD_D_A_d
  params = {'OPCD':43}

class lhaux(Instruction):
  machine_inst = OPCD_D_A_B_XO_1
  params = {'OPCD':31, 'XO':375}

class lhax(Instruction):
  machine_inst = OPCD_D_A_B_XO_1
  params = {'OPCD':31, 'XO':343}

class lhbrx(Instruction):
  machine_inst = OPCD_D_A_B_XO_1
  params = {'OPCD':31, 'XO':790}

class lhz(Instruction):
  machine_inst = OPCD_D_A_d
  params = {'OPCD':40}

class lhzu(Instruction):
  machine_inst = OPCD_D_A_d
  params = {'OPCD':41}

class lhzux(Instruction):
  machine_inst = OPCD_D_A_B_XO_1
  params = {'OPCD':31, 'XO':311}

class lhzx(Instruction):
  machine_inst = OPCD_D_A_B_XO_1
  params = {'OPCD':31, 'XO':279}

class lmw(Instruction):
  machine_inst = OPCD_D_A_d
  params = {'OPCD':46}

class lswi(Instruction):
  machine_inst = OPCD_D_A_NB_XO_1
  params = {'OPCD':31, 'XO':597}

class lswx(Instruction):
  machine_inst = OPCD_D_A_B_XO_1
  params = {'OPCD':31, 'XO':533}

class lwarx(Instruction):
  machine_inst = OPCD_D_A_B_XO_1
  params = {'OPCD':31, 'XO':20}

class lwbrx(Instruction):
  machine_inst = OPCD_D_A_B_XO_1
  params = {'OPCD':31, 'XO':534}

class lwz(Instruction):
  machine_inst = OPCD_D_A_d
  params = {'OPCD':32}

class lwzu(Instruction):
  machine_inst = OPCD_D_A_d
  params = {'OPCD':33}

class lwzux(Instruction):
  machine_inst = OPCD_D_A_B_XO_1
  params = {'OPCD':31, 'XO':55}

class lwzx(Instruction):
  machine_inst = OPCD_D_A_B_XO_1
  params = {'OPCD':31, 'XO':23}

class mcrf(Instruction):
  machine_inst = OPCD_crfD_crfS
  params = {'OPCD':19}

class mcrfs(Instruction):
  machine_inst = OPCD_crfD_crfS_XO_1
  params = {'OPCD':63, 'XO':64}

class mcrxr(Instruction):
  machine_inst = OPCD_crfD_XO_1
  params = {'OPCD':31, 'XO':512}

class mfcr(Instruction):
  machine_inst = OPCD_D_XO_1
  params = {'OPCD':31, 'XO':19}

class mffsx(Instruction):
  machine_inst = OPCD_D_XO_1_Rc
  params = {'OPCD':63, 'XO':583}

class mfmsr(Instruction):
  machine_inst = OPCD_D_XO_1
  params = {'OPCD':31, 'XO':83}

class mfspr(Instruction):
  machine_inst = OPCD_D_spr_XO_1
  params = {'OPCD':31, 'XO':339}

class mfsr(Instruction):
  machine_inst = OPCD_D_SR_XO_1
  params = {'OPCD':31, 'XO':595}

class mfsrin(Instruction):
  machine_inst = OPCD_D_B_XO_1
  params = {'OPCD':31, 'XO':659}

class mftb(Instruction):
  machine_inst = OPCD_D_tbr_XO_1
  params = {'OPCD':31, 'XO':371}

class mtcrf(Instruction):
  machine_inst = OPCD_S_CRM_XO_1
  params = {'OPCD':31, 'XO':144}

class mtfsb0x(Instruction):
  machine_inst = OPCD_crbD_XO_1_Rc
  params = {'OPCD':63, 'XO':70}

class mtfsb1x(Instruction):
  machine_inst = OPCD_crbD_XO_1_Rc
  params = {'OPCD':63, 'XO':38}

class mtfsfx(Instruction):
  machine_inst = OPCD_FM_B_XO_1_Rc
  params = {'OPCD':63, 'XO':711}

class mtfsfix(Instruction):
  machine_inst = OPCD_crfD_IMM_XO_1_Rc
  params = {'OPCD':63, 'XO':134}

class mtmsr(Instruction):
  machine_inst = OPCD_S_XO_1
  params = {'OPCD':31, 'XO':146}

class mtspr(Instruction):
  machine_inst = OPCD_S_spr_XO_1
  params = {'OPCD':31, 'XO':467}

class mtsr(Instruction):
  machine_inst = OPCD_S_SR_XO_1
  params = {'OPCD':31, 'XO':210}

class mtsrin(Instruction):
  machine_inst = OPCD_S_B_XO_1
  params = {'OPCD':31, 'XO':242}

class mulhwx(Instruction):
  machine_inst = OPCD_D_A_B_XO_2_Rc
  params = {'OPCD':31, 'XO':75}

class mulhwux(Instruction):
  machine_inst = OPCD_D_A_B_XO_2_Rc
  params = {'OPCD':31, 'XO':11}

class mulli(Instruction):
  machine_inst = OPCD_D_A_SIMM
  params = {'OPCD':7}

class mullwx(Instruction):
  machine_inst = OPCD_D_A_B_OE_XO_2_Rc
  params = {'OPCD':31, 'XO':235}

class nandx(Instruction):
  machine_inst = OPCD_S_A_B_XO_1_Rc
  params = {'OPCD':31, 'XO':476}

class negx(Instruction):
  machine_inst = OPCD_D_A_OE_XO_2_Rc
  params = {'OPCD':31, 'XO':104}

class norx(Instruction):
  machine_inst = OPCD_S_A_B_XO_1_Rc
  params = {'OPCD':31, 'XO':124}

class orx(Instruction):
  machine_inst = OPCD_S_A_B_XO_1_Rc
  params = {'OPCD':31, 'XO':444}

class orcx(Instruction):
  machine_inst = OPCD_S_A_B_XO_1_Rc
  params = {'OPCD':31, 'XO':412}

class ori(Instruction):
  machine_inst = OPCD_S_A_UIMM
  params = {'OPCD':24}

class oris(Instruction):
  machine_inst = OPCD_S_A_UIMM
  params = {'OPCD':25}

class rfi(Instruction):
  machine_inst = OPCD_XO_1
  params = {'OPCD':19, 'XO':50}

class rlwimix(Instruction):
  machine_inst = OPCD_S_A_SH_MB_ME_Rc
  params = {'OPCD':20}

class rlwinmx(Instruction):
  machine_inst = OPCD_S_A_SH_MB_ME_Rc
  params = {'OPCD':21}

class rlwnmx(Instruction):
  machine_inst = OPCD_S_A_B_MB_ME_Rc
  params = {'OPCD':23}

class sc(Instruction):
  machine_inst = OPCD_SC_ONE
  params = {'OPCD':17}

class slwx(Instruction):
  machine_inst = OPCD_S_A_B_XO_1_Rc
  params = {'OPCD':31, 'XO':24}

class srawx(Instruction):
  machine_inst = OPCD_S_A_B_XO_1_Rc
  params = {'OPCD':31, 'XO':792}

class srawix(Instruction):
  machine_inst = OPCD_S_A_SH_XO_1_Rc
  params = {'OPCD':31, 'XO':824}

class srwx(Instruction):
  machine_inst = OPCD_S_A_B_XO_1_Rc
  params = {'OPCD':31, 'XO':536}

class stb(Instruction):
  machine_inst = OPCD_S_A_d
  params = {'OPCD':38}

class stbu(Instruction):
  machine_inst = OPCD_S_A_d
  params = {'OPCD':39}

class stbux(Instruction):
  machine_inst = OPCD_S_A_B_XO_1
  params = {'OPCD':31, 'XO':247}

class stbx(Instruction):
  machine_inst = OPCD_S_A_B_XO_1
  params = {'OPCD':31, 'XO':215}

class stfd(Instruction):
  machine_inst = OPCD_S_A_d
  params = {'OPCD':54}

class stfdu(Instruction):
  machine_inst = OPCD_S_A_d
  params = {'OPCD':55}

class stfdux(Instruction):
  machine_inst = OPCD_S_A_B_XO_1
  params = {'OPCD':31, 'XO':759}

class stfdx(Instruction):
  machine_inst = OPCD_S_A_B_XO_1
  params = {'OPCD':31, 'XO':727}

class stfiwx(Instruction):
  machine_inst = OPCD_S_A_B_XO_1
  params = {'OPCD':31, 'XO':983}

class stfs(Instruction):
  machine_inst = OPCD_S_A_d
  params = {'OPCD':52}

class stfsu(Instruction):
  machine_inst = OPCD_S_A_d
  params = {'OPCD':53}

class stfsux(Instruction):
  machine_inst = OPCD_S_A_B_XO_1
  params = {'OPCD':31, 'XO':695}

class stfsx(Instruction):
  machine_inst = OPCD_S_A_B_XO_1
  params = {'OPCD':31, 'XO':663}

class sth(Instruction):
  machine_inst = OPCD_S_A_d
  params = {'OPCD':44}

class sthbrx(Instruction):
  machine_inst = OPCD_S_A_B_XO_1
  params = {'OPCD':31, 'XO':918}

class sthu(Instruction):
  machine_inst = OPCD_S_A_d
  params = {'OPCD':45}

class sthux(Instruction):
  machine_inst = OPCD_S_A_B_XO_1
  params = {'OPCD':31, 'XO':439}

class sthx(Instruction):
  machine_inst = OPCD_S_A_B_XO_1
  params = {'OPCD':31, 'XO':407}

class stmw(Instruction):
  machine_inst = OPCD_S_A_d
  params = {'OPCD':47}

class stswi(Instruction):
  machine_inst = OPCD_S_A_NB_XO_1
  params = {'OPCD':31, 'XO':725}

class stswx(Instruction):
  machine_inst = OPCD_S_A_B_XO_1
  params = {'OPCD':31, 'XO':661}

class stw(Instruction):
  machine_inst = OPCD_S_A_d
  params = {'OPCD':36}

class stwbrx(Instruction):
  machine_inst = OPCD_S_A_B_XO_1
  params = {'OPCD':31, 'XO':662}

class stwcx_(Instruction):
  machine_inst = OPCD_S_A_B_XO_1_STWCX_ONE
  params = {'OPCD':31, 'XO':150}

class stwu(Instruction):
  machine_inst = OPCD_S_A_d
  params = {'OPCD':37}

class stwux(Instruction):
  machine_inst = OPCD_S_A_B_XO_1
  params = {'OPCD':31, 'XO':183}

class stwx(Instruction):
  machine_inst = OPCD_S_A_B_XO_1
  params = {'OPCD':31, 'XO':151}

class subfx(Instruction):
  machine_inst = OPCD_D_A_B_OE_XO_2_Rc
  params = {'OPCD':31, 'XO':40}

class subfcx(Instruction):
  machine_inst = OPCD_D_A_B_OE_XO_2_Rc
  params = {'OPCD':31, 'XO':8}

class subfex(Instruction):
  machine_inst = OPCD_D_A_B_OE_XO_2_Rc
  params = {'OPCD':31, 'XO':136}

class subfic(Instruction):
  machine_inst = OPCD_D_A_SIMM
  params = {'OPCD':8}

class subfmex(Instruction):
  machine_inst = OPCD_D_A_OE_XO_2_Rc
  params = {'OPCD':31, 'XO':232}

class subfzex(Instruction):
  machine_inst = OPCD_D_A_OE_XO_2_Rc
  params = {'OPCD':31, 'XO':200}

class sync(Instruction):
  machine_inst = OPCD_XO_1
  params = {'OPCD':31, 'XO':598}

class tlbia(Instruction):
  machine_inst = OPCD_XO_1
  params = {'OPCD':31, 'XO':370}

class tlbie(Instruction):
  machine_inst = OPCD_B_XO_1
  params = {'OPCD':31, 'XO':306}

class tlbsync1(Instruction):
  machine_inst = OPCD_XO_1
  params = {'OPCD':31, 'XO':566}

class tw(Instruction):
  machine_inst = OPCD_TO_A_B_XO_1
  params = {'OPCD':31, 'XO':4}

class twi(Instruction):
  machine_inst = OPCD_TO_A_SIMM
  params = {'OPCD':3}

class xorx(Instruction):
  machine_inst = OPCD_S_A_B_XO_1_Rc
  params = {'OPCD':31, 'XO':316}

class xori(Instruction):
  machine_inst = OPCD_S_A_UIMM
  params = {'OPCD':26}

class xoris(Instruction):
  machine_inst = OPCD_S_A_UIMM
  params = {'OPCD':27}


if __name__=='__main__':
  from corepy.spre.syn_util import DecToBin
  a = addx(1,2,3)
  d = a.render()
  print '0x%08X: %s' % (d, DecToBin(d))
