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

from corepy.spre.spe import Instruction, DispatchInstruction, Label
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

def ppc_type(op):
  if isinstance(op, (int, long)):
    return IMM_DUMMY
  elif isinstance(op, Label):
    return LILBL
  else:
    raise Exception("unhandled op %s %s" % (str(op), str(type(op))))
  return

class PPCInstruction(Instruction):
  instruction_type = None
  params = {}
  def __init__(self, *operands, **koperands):
    self.machine_inst = self.instruction_type
    Instruction.__init__(self, *operands, **koperands)
    return

class PPCDispatchInstruction(DispatchInstruction):
  type_id = [ppc_type]
  
class addx(PPCInstruction):
  instruction_type = OPCD_D_A_B_OE_XO_2_Rc
  params = {'OPCD':31, 'XO':266}

class addcx(PPCInstruction):
  instruction_type = OPCD_D_A_B_OE_XO_2_Rc
  params = {'OPCD':31, 'XO':10}

class addex(PPCInstruction):
  instruction_type = OPCD_D_A_B_OE_XO_2_Rc
  params = {'OPCD':31, 'XO':138}

class addi(PPCInstruction):
  instruction_type = OPCD_D_A_SIMM
  params = {'OPCD':14}

class addic(PPCInstruction):
  instruction_type = OPCD_D_A_SIMM
  params = {'OPCD':12}

class addic_(PPCInstruction):
  instruction_type = OPCD_D_A_SIMM
  params = {'OPCD':13}

class addis(PPCInstruction):
  instruction_type = OPCD_D_A_SIMM
  params = {'OPCD':15}

class addmex(PPCInstruction):
  instruction_type = OPCD_D_A_OE_XO_2_Rc
  params = {'OPCD':31, 'XO':234}

class addzex(PPCInstruction):
  instruction_type = OPCD_D_A_OE_XO_2_Rc
  params = {'OPCD':31, 'XO':202}

class andx(PPCInstruction):
  instruction_type = OPCD_S_A_B_XO_1_Rc
  params = {'OPCD':31, 'XO':28}

class andcx(PPCInstruction):
  instruction_type = OPCD_S_A_B_XO_1_Rc
  params = {'OPCD':31, 'XO':60}

class andi(PPCInstruction):
  instruction_type = OPCD_S_A_UIMM
  params = {'OPCD':28}

class andis(PPCInstruction):
  instruction_type = OPCD_S_A_UIMM
  params = {'OPCD':29}

class bx(PPCDispatchInstruction):
  dispatch = (
    (OPCD_LI_AA_LK,     {'OPCD':18}),
    (OPCD_LILBL_AA_LK,  {'OPCD':18}))

class bcx(PPCDispatchInstruction):
  dispatch = (
    (OPCD_BO_BI_BD_AA_LK,    {'OPCD':16}),
    (OPCD_BO_BI_BDLBL_AA_LK, {'OPCD':16}))

class bcctrx(PPCInstruction):
  instruction_type = OPCD_BO_BI_XO_1_LK
  params = {'OPCD':19, 'XO':528}

class bclrx(PPCInstruction):
  instruction_type = OPCD_BO_BI_XO_1_LK
  params = {'OPCD':19, 'XO':16}

class cmp_(PPCInstruction):
  instruction_type = OPCD_crfD_L_A_B
  params = {'OPCD':31}

class cmpi(PPCInstruction):
  instruction_type = OPCD_crfD_L_A_SIMM
  params = {'OPCD':11}

class cmpl(PPCInstruction):
  instruction_type = OPCD_crfD_L_A_B_XO_1
  params = {'OPCD':31, 'XO':32}

class cmpli(PPCInstruction):
  instruction_type = OPCD_crfD_L_A_UIMM
  params = {'OPCD':10}

class cntlzwx(PPCInstruction):
  instruction_type = OPCD_S_A_XO_1_Rc
  params = {'OPCD':31, 'XO':26}

class crand(PPCInstruction):
  instruction_type = OPCD_crbD_crbA_crbB_XO_1
  params = {'OPCD':19, 'XO':257}

class crandc(PPCInstruction):
  instruction_type = OPCD_crbD_crbA_crbB_XO_1
  params = {'OPCD':19, 'XO':129}

class creqv(PPCInstruction):
  instruction_type = OPCD_crbD_crbA_crbB_XO_1
  params = {'OPCD':19, 'XO':289}

class crnand(PPCInstruction):
  instruction_type = OPCD_crbD_crbA_crbB_XO_1
  params = {'OPCD':19, 'XO':225}

class crnor(PPCInstruction):
  instruction_type = OPCD_crbD_crbA_crbB_XO_1
  params = {'OPCD':19, 'XO':33}

class cror(PPCInstruction):
  instruction_type = OPCD_crbD_crbA_crbB_XO_1
  params = {'OPCD':19, 'XO':449}

class crorc(PPCInstruction):
  instruction_type = OPCD_crbD_crbA_crbB_XO_1
  params = {'OPCD':19, 'XO':417}

class crxor(PPCInstruction):
  instruction_type = OPCD_crbD_crbA_crbB_XO_1
  params = {'OPCD':19, 'XO':193}

class dcba(PPCInstruction):
  instruction_type = OPCD_A_B_XO_1
  params = {'OPCD':31, 'XO':758}

class dcbf(PPCInstruction):
  instruction_type = OPCD_A_B_XO_1
  params = {'OPCD':31, 'XO':86}

class dcbi(PPCInstruction):
  instruction_type = OPCD_A_B_XO_1
  params = {'OPCD':31, 'XO':470}

class dcbst(PPCInstruction):
  instruction_type = OPCD_A_B_XO_1
  params = {'OPCD':31, 'XO':54}

class dcbt(PPCInstruction):
  instruction_type = OPCD_A_B_XO_1
  params = {'OPCD':31, 'XO':278}

class dcbtst(PPCInstruction):
  instruction_type = OPCD_A_B_XO_1
  params = {'OPCD':31, 'XO':246}

class dcbz(PPCInstruction):
  instruction_type = OPCD_A_B_XO_1
  params = {'OPCD':31, 'XO':1014}

class divwx(PPCInstruction):
  instruction_type = OPCD_D_A_B_OE_XO_2_Rc
  params = {'OPCD':31, 'XO':491}

class divwux(PPCInstruction):
  instruction_type = OPCD_D_A_B_OE_XO_2_Rc
  params = {'OPCD':31, 'XO':459}

class eciwx(PPCInstruction):
  instruction_type = OPCD_D_A_B_XO_1
  params = {'OPCD':31, 'XO':310}

class ecowx(PPCInstruction):
  instruction_type = OPCD_S_A_B_XO_1
  params = {'OPCD':31, 'XO':438}

class eieio(PPCInstruction):
  instruction_type = OPCD_XO_1
  params = {'OPCD':31, 'XO':854}

class eqvx(PPCInstruction):
  instruction_type = OPCD_S_A_B_XO_1_Rc
  params = {'OPCD':31, 'XO':284}

class extsbx(PPCInstruction):
  instruction_type = OPCD_S_A_XO_1_Rc
  params = {'OPCD':31, 'XO':954}

class extshx(PPCInstruction):
  instruction_type = OPCD_S_A_XO_1_Rc
  params = {'OPCD':31, 'XO':922}

class fabsx(PPCInstruction):
  instruction_type = OPCD_D_B_XO_1_Rc
  params = {'OPCD':63, 'XO':264}

class faddx(PPCInstruction):
  instruction_type = OPCD_D_A_B_XO_3_Rc
  params = {'OPCD':63, 'XO':21}

class faddsx(PPCInstruction):
  instruction_type = OPCD_D_A_B_XO_3_Rc
  params = {'OPCD':59, 'XO':21}

class fcmpo(PPCInstruction):
  instruction_type = OPCD_crfD_A_B_XO_1
  params = {'OPCD':63, 'XO':32}

class fcmpu(PPCInstruction):
  instruction_type = OPCD_crfD_A_B
  params = {'OPCD':63}

class fctiwx(PPCInstruction):
  instruction_type = OPCD_D_B_XO_1_Rc
  params = {'OPCD':63, 'XO':14}

class fctiwzx(PPCInstruction):
  instruction_type = OPCD_D_B_XO_1_Rc
  params = {'OPCD':63, 'XO':15}

class fdivx(PPCInstruction):
  instruction_type = OPCD_D_A_B_XO_3_Rc
  params = {'OPCD':63, 'XO':18}

class fdivsx(PPCInstruction):
  instruction_type = OPCD_D_A_B_XO_3_Rc
  params = {'OPCD':59, 'XO':18}

class fmaddx(PPCInstruction):
  instruction_type = OPCD_D_A_B_C_XO_3_Rc
  params = {'OPCD':63, 'XO':29}

class fmaddsx(PPCInstruction):
  instruction_type = OPCD_D_A_B_C_XO_3_Rc
  params = {'OPCD':59, 'XO':29}

class fmrx(PPCInstruction):
  instruction_type = OPCD_D_B_XO_1_Rc
  params = {'OPCD':63, 'XO':72}

class fmsubx(PPCInstruction):
  instruction_type = OPCD_D_A_B_C_XO_3_Rc
  params = {'OPCD':63, 'XO':28}

class fmsubsx(PPCInstruction):
  instruction_type = OPCD_D_A_B_C_XO_3_Rc
  params = {'OPCD':59, 'XO':28}

class fmulx(PPCInstruction):
  instruction_type = OPCD_D_A_C_XO_3_Rc
  params = {'OPCD':63, 'XO':25}

class fmulsx(PPCInstruction):
  instruction_type = OPCD_D_A_C_XO_3_Rc
  params = {'OPCD':59, 'XO':25}

class fnabsx(PPCInstruction):
  instruction_type = OPCD_D_B_XO_1_Rc
  params = {'OPCD':63, 'XO':136}

class fnegx(PPCInstruction):
  instruction_type = OPCD_D_B_XO_1_Rc
  params = {'OPCD':63, 'XO':40}

class fnmaddx(PPCInstruction):
  instruction_type = OPCD_D_A_B_C_XO_3_Rc
  params = {'OPCD':63, 'XO':31}

class fnmaddsx(PPCInstruction):
  instruction_type = OPCD_D_A_B_C_XO_3_Rc
  params = {'OPCD':59, 'XO':31}

class fnmsubx(PPCInstruction):
  instruction_type = OPCD_D_A_B_C_XO_3_Rc
  params = {'OPCD':63, 'XO':30}

class fnmsubsx(PPCInstruction):
  instruction_type = OPCD_D_A_B_C_XO_3_Rc
  params = {'OPCD':59, 'XO':30}

class fresx(PPCInstruction):
  instruction_type = OPCD_D_B_XO_3_Rc
  params = {'OPCD':59, 'XO':24}

class frspx(PPCInstruction):
  instruction_type = OPCD_D_B_XO_1_Rc
  params = {'OPCD':63, 'XO':12}

class frsqrtex(PPCInstruction):
  instruction_type = OPCD_D_B_XO_3_Rc
  params = {'OPCD':63, 'XO':26}

class fselx(PPCInstruction):
  instruction_type = OPCD_D_A_B_C_XO_3_Rc
  params = {'OPCD':63, 'XO':23}

class fsqrtx(PPCInstruction):
  instruction_type = OPCD_D_B_XO_3_Rc
  params = {'OPCD':63, 'XO':22}

class fsqrtsx(PPCInstruction):
  instruction_type = OPCD_D_B_XO_3_Rc
  params = {'OPCD':59, 'XO':22}

class fsubx(PPCInstruction):
  instruction_type = OPCD_D_A_B_XO_3_Rc
  params = {'OPCD':63, 'XO':20}

class fsubsx(PPCInstruction):
  instruction_type = OPCD_D_A_B_XO_3_Rc
  params = {'OPCD':59, 'XO':20}

class icbi(PPCInstruction):
  instruction_type = OPCD_A_B_XO_1
  params = {'OPCD':31, 'XO':982}

class isync(PPCInstruction):
  instruction_type = OPCD_XO_1
  params = {'OPCD':19, 'XO':150}

class lbz(PPCInstruction):
  instruction_type = OPCD_D_A_d
  params = {'OPCD':34}

class lbzu(PPCInstruction):
  instruction_type = OPCD_D_A_d
  params = {'OPCD':35}

class lbzux(PPCInstruction):
  instruction_type = OPCD_D_A_B_XO_1
  params = {'OPCD':31, 'XO':119}

class lbzx(PPCInstruction):
  instruction_type = OPCD_D_A_B_XO_1
  params = {'OPCD':31, 'XO':87}

class lfd(PPCInstruction):
  instruction_type = OPCD_D_A_d
  params = {'OPCD':50}

class lfdu(PPCInstruction):
  instruction_type = OPCD_D_A_d
  params = {'OPCD':51}

class lfdux(PPCInstruction):
  instruction_type = OPCD_D_A_B_XO_1
  params = {'OPCD':31, 'XO':631}

class lfdx(PPCInstruction):
  instruction_type = OPCD_D_A_B_XO_1
  params = {'OPCD':31, 'XO':599}

class lfs(PPCInstruction):
  instruction_type = OPCD_D_A_d
  params = {'OPCD':48}

class lfsu(PPCInstruction):
  instruction_type = OPCD_D_A_d
  params = {'OPCD':49}

class lfsux(PPCInstruction):
  instruction_type = OPCD_D_A_B_XO_1
  params = {'OPCD':31, 'XO':567}

class lfsx(PPCInstruction):
  instruction_type = OPCD_D_A_B_XO_1
  params = {'OPCD':31, 'XO':535}

class lha(PPCInstruction):
  instruction_type = OPCD_D_A_d
  params = {'OPCD':42}

class lhau(PPCInstruction):
  instruction_type = OPCD_D_A_d
  params = {'OPCD':43}

class lhaux(PPCInstruction):
  instruction_type = OPCD_D_A_B_XO_1
  params = {'OPCD':31, 'XO':375}

class lhax(PPCInstruction):
  instruction_type = OPCD_D_A_B_XO_1
  params = {'OPCD':31, 'XO':343}

class lhbrx(PPCInstruction):
  instruction_type = OPCD_D_A_B_XO_1
  params = {'OPCD':31, 'XO':790}

class lhz(PPCInstruction):
  instruction_type = OPCD_D_A_d
  params = {'OPCD':40}

class lhzu(PPCInstruction):
  instruction_type = OPCD_D_A_d
  params = {'OPCD':41}

class lhzux(PPCInstruction):
  instruction_type = OPCD_D_A_B_XO_1
  params = {'OPCD':31, 'XO':311}

class lhzx(PPCInstruction):
  instruction_type = OPCD_D_A_B_XO_1
  params = {'OPCD':31, 'XO':279}

class lmw(PPCInstruction):
  instruction_type = OPCD_D_A_d
  params = {'OPCD':46}

class lswi(PPCInstruction):
  instruction_type = OPCD_D_A_NB_XO_1
  params = {'OPCD':31, 'XO':597}

class lswx(PPCInstruction):
  instruction_type = OPCD_D_A_B_XO_1
  params = {'OPCD':31, 'XO':533}

class lwarx(PPCInstruction):
  instruction_type = OPCD_D_A_B_XO_1
  params = {'OPCD':31, 'XO':20}

class lwbrx(PPCInstruction):
  instruction_type = OPCD_D_A_B_XO_1
  params = {'OPCD':31, 'XO':534}

class lwz(PPCInstruction):
  instruction_type = OPCD_D_A_d
  params = {'OPCD':32}

class lwzu(PPCInstruction):
  instruction_type = OPCD_D_A_d
  params = {'OPCD':33}

class lwzux(PPCInstruction):
  instruction_type = OPCD_D_A_B_XO_1
  params = {'OPCD':31, 'XO':55}

class lwzx(PPCInstruction):
  instruction_type = OPCD_D_A_B_XO_1
  params = {'OPCD':31, 'XO':23}

class mcrf(PPCInstruction):
  instruction_type = OPCD_crfD_crfS
  params = {'OPCD':19}

class mcrfs(PPCInstruction):
  instruction_type = OPCD_crfD_crfS_XO_1
  params = {'OPCD':63, 'XO':64}

class mcrxr(PPCInstruction):
  instruction_type = OPCD_crfD_XO_1
  params = {'OPCD':31, 'XO':512}

class mfcr(PPCInstruction):
  instruction_type = OPCD_D_XO_1
  params = {'OPCD':31, 'XO':19}

class mffsx(PPCInstruction):
  instruction_type = OPCD_D_XO_1_Rc
  params = {'OPCD':63, 'XO':583}

class mfmsr(PPCInstruction):
  instruction_type = OPCD_D_XO_1
  params = {'OPCD':31, 'XO':83}

class mfspr(PPCInstruction):
  instruction_type = OPCD_D_spr_XO_1
  params = {'OPCD':31, 'XO':339}

class mfsr(PPCInstruction):
  instruction_type = OPCD_D_SR_XO_1
  params = {'OPCD':31, 'XO':595}

class mfsrin(PPCInstruction):
  instruction_type = OPCD_D_B_XO_1
  params = {'OPCD':31, 'XO':659}

class mftb(PPCInstruction):
  instruction_type = OPCD_D_tbr_XO_1
  params = {'OPCD':31, 'XO':371}

class mtcrf(PPCInstruction):
  instruction_type = OPCD_S_CRM_XO_1
  params = {'OPCD':31, 'XO':144}

class mtfsb0x(PPCInstruction):
  instruction_type = OPCD_crbD_XO_1_Rc
  params = {'OPCD':63, 'XO':70}

class mtfsb1x(PPCInstruction):
  instruction_type = OPCD_crbD_XO_1_Rc
  params = {'OPCD':63, 'XO':38}

class mtfsfx(PPCInstruction):
  instruction_type = OPCD_FM_B_XO_1_Rc
  params = {'OPCD':63, 'XO':711}

class mtfsfix(PPCInstruction):
  instruction_type = OPCD_crfD_IMM_XO_1_Rc
  params = {'OPCD':63, 'XO':134}

class mtmsr(PPCInstruction):
  instruction_type = OPCD_S_XO_1
  params = {'OPCD':31, 'XO':146}

class mtspr(PPCInstruction):
  instruction_type = OPCD_S_spr_XO_1
  params = {'OPCD':31, 'XO':467}

class mtsr(PPCInstruction):
  instruction_type = OPCD_S_SR_XO_1
  params = {'OPCD':31, 'XO':210}

class mtsrin(PPCInstruction):
  instruction_type = OPCD_S_B_XO_1
  params = {'OPCD':31, 'XO':242}

class mulhwx(PPCInstruction):
  instruction_type = OPCD_D_A_B_XO_2_Rc
  params = {'OPCD':31, 'XO':75}

class mulhwux(PPCInstruction):
  instruction_type = OPCD_D_A_B_XO_2_Rc
  params = {'OPCD':31, 'XO':11}

class mulli(PPCInstruction):
  instruction_type = OPCD_D_A_SIMM
  params = {'OPCD':7}

class mullwx(PPCInstruction):
  instruction_type = OPCD_D_A_B_OE_XO_2_Rc
  params = {'OPCD':31, 'XO':235}

class nandx(PPCInstruction):
  instruction_type = OPCD_S_A_B_XO_1_Rc
  params = {'OPCD':31, 'XO':476}

class negx(PPCInstruction):
  instruction_type = OPCD_D_A_OE_XO_2_Rc
  params = {'OPCD':31, 'XO':104}

class norx(PPCInstruction):
  instruction_type = OPCD_S_A_B_XO_1_Rc
  params = {'OPCD':31, 'XO':124}

class orx(PPCInstruction):
  instruction_type = OPCD_S_A_B_XO_1_Rc
  params = {'OPCD':31, 'XO':444}

class orcx(PPCInstruction):
  instruction_type = OPCD_S_A_B_XO_1_Rc
  params = {'OPCD':31, 'XO':412}

class ori(PPCInstruction):
  instruction_type = OPCD_S_A_UIMM
  params = {'OPCD':24}

class oris(PPCInstruction):
  instruction_type = OPCD_S_A_UIMM
  params = {'OPCD':25}

class rfi(PPCInstruction):
  instruction_type = OPCD_XO_1
  params = {'OPCD':19, 'XO':50}

class rlwimix(PPCInstruction):
  instruction_type = OPCD_S_A_SH_MB_ME_Rc
  params = {'OPCD':20}

class rlwinmx(PPCInstruction):
  instruction_type = OPCD_S_A_SH_MB_ME_Rc
  params = {'OPCD':21}

class rlwnmx(PPCInstruction):
  instruction_type = OPCD_S_A_B_MB_ME_Rc
  params = {'OPCD':23}

class sc(PPCInstruction):
  instruction_type = OPCD_SC_ONE
  params = {'OPCD':17}

class slwx(PPCInstruction):
  instruction_type = OPCD_S_A_B_XO_1_Rc
  params = {'OPCD':31, 'XO':24}

class srawx(PPCInstruction):
  instruction_type = OPCD_S_A_B_XO_1_Rc
  params = {'OPCD':31, 'XO':792}

class srawix(PPCInstruction):
  instruction_type = OPCD_S_A_SH_XO_1_Rc
  params = {'OPCD':31, 'XO':824}

class srwx(PPCInstruction):
  instruction_type = OPCD_S_A_B_XO_1_Rc
  params = {'OPCD':31, 'XO':536}

class stb(PPCInstruction):
  instruction_type = OPCD_S_A_d
  params = {'OPCD':38}

class stbu(PPCInstruction):
  instruction_type = OPCD_S_A_d
  params = {'OPCD':39}

class stbux(PPCInstruction):
  instruction_type = OPCD_S_A_B_XO_1
  params = {'OPCD':31, 'XO':247}

class stbx(PPCInstruction):
  instruction_type = OPCD_S_A_B_XO_1
  params = {'OPCD':31, 'XO':215}

class stfd(PPCInstruction):
  instruction_type = OPCD_S_A_d
  params = {'OPCD':54}

class stfdu(PPCInstruction):
  instruction_type = OPCD_S_A_d
  params = {'OPCD':55}

class stfdux(PPCInstruction):
  instruction_type = OPCD_S_A_B_XO_1
  params = {'OPCD':31, 'XO':759}

class stfdx(PPCInstruction):
  instruction_type = OPCD_S_A_B_XO_1
  params = {'OPCD':31, 'XO':727}

class stfiwx(PPCInstruction):
  instruction_type = OPCD_S_A_B_XO_1
  params = {'OPCD':31, 'XO':983}

class stfs(PPCInstruction):
  instruction_type = OPCD_S_A_d
  params = {'OPCD':52}

class stfsu(PPCInstruction):
  instruction_type = OPCD_S_A_d
  params = {'OPCD':53}

class stfsux(PPCInstruction):
  instruction_type = OPCD_S_A_B_XO_1
  params = {'OPCD':31, 'XO':695}

class stfsx(PPCInstruction):
  instruction_type = OPCD_S_A_B_XO_1
  params = {'OPCD':31, 'XO':663}

class sth(PPCInstruction):
  instruction_type = OPCD_S_A_d
  params = {'OPCD':44}

class sthbrx(PPCInstruction):
  instruction_type = OPCD_S_A_B_XO_1
  params = {'OPCD':31, 'XO':918}

class sthu(PPCInstruction):
  instruction_type = OPCD_S_A_d
  params = {'OPCD':45}

class sthux(PPCInstruction):
  instruction_type = OPCD_S_A_B_XO_1
  params = {'OPCD':31, 'XO':439}

class sthx(PPCInstruction):
  instruction_type = OPCD_S_A_B_XO_1
  params = {'OPCD':31, 'XO':407}

class stmw(PPCInstruction):
  instruction_type = OPCD_S_A_d
  params = {'OPCD':47}

class stswi(PPCInstruction):
  instruction_type = OPCD_S_A_NB_XO_1
  params = {'OPCD':31, 'XO':725}

class stswx(PPCInstruction):
  instruction_type = OPCD_S_A_B_XO_1
  params = {'OPCD':31, 'XO':661}

class stw(PPCInstruction):
  instruction_type = OPCD_S_A_d
  params = {'OPCD':36}

class stwbrx(PPCInstruction):
  instruction_type = OPCD_S_A_B_XO_1
  params = {'OPCD':31, 'XO':662}

class stwcx_(PPCInstruction):
  instruction_type = OPCD_S_A_B_XO_1_STWCX_ONE
  params = {'OPCD':31, 'XO':150}

class stwu(PPCInstruction):
  instruction_type = OPCD_S_A_d
  params = {'OPCD':37}

class stwux(PPCInstruction):
  instruction_type = OPCD_S_A_B_XO_1
  params = {'OPCD':31, 'XO':183}

class stwx(PPCInstruction):
  instruction_type = OPCD_S_A_B_XO_1
  params = {'OPCD':31, 'XO':151}

class subfx(PPCInstruction):
  instruction_type = OPCD_D_A_B_OE_XO_2_Rc
  params = {'OPCD':31, 'XO':40}

class subfcx(PPCInstruction):
  instruction_type = OPCD_D_A_B_OE_XO_2_Rc
  params = {'OPCD':31, 'XO':8}

class subfex(PPCInstruction):
  instruction_type = OPCD_D_A_B_OE_XO_2_Rc
  params = {'OPCD':31, 'XO':136}

class subfic(PPCInstruction):
  instruction_type = OPCD_D_A_SIMM
  params = {'OPCD':8}

class subfmex(PPCInstruction):
  instruction_type = OPCD_D_A_OE_XO_2_Rc
  params = {'OPCD':31, 'XO':232}

class subfzex(PPCInstruction):
  instruction_type = OPCD_D_A_OE_XO_2_Rc
  params = {'OPCD':31, 'XO':200}

class sync(PPCInstruction):
  instruction_type = OPCD_XO_1
  params = {'OPCD':31, 'XO':598}

class tlbia(PPCInstruction):
  instruction_type = OPCD_XO_1
  params = {'OPCD':31, 'XO':370}

class tlbie(PPCInstruction):
  instruction_type = OPCD_B_XO_1
  params = {'OPCD':31, 'XO':306}

class tlbsync1(PPCInstruction):
  instruction_type = OPCD_XO_1
  params = {'OPCD':31, 'XO':566}

class tw(PPCInstruction):
  instruction_type = OPCD_TO_A_B_XO_1
  params = {'OPCD':31, 'XO':4}

class twi(PPCInstruction):
  instruction_type = OPCD_TO_A_SIMM
  params = {'OPCD':3}

class xorx(PPCInstruction):
  instruction_type = OPCD_S_A_B_XO_1_Rc
  params = {'OPCD':31, 'XO':316}

class xori(PPCInstruction):
  instruction_type = OPCD_S_A_UIMM
  params = {'OPCD':26}

class xoris(PPCInstruction):
  instruction_type = OPCD_S_A_UIMM
  params = {'OPCD':27}


if __name__=='__main__':
  from corepy.spre.syn_util import DecToBin
  a = addx(1,2,3)
  d = a.render()
  print '0x%08X: %s' % (d, DecToBin(d))
