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

# PowerPC Machine Instructions

# Notes:
#  params are: OPCD, XO

from corepy.spre.spe import MachineInstruction, Label
from ppc_fields import *

class OPCD_S_A_B_XO_1_STWCX_ONE(MachineInstruction):
  """ 
  Instructions: (1)  stwcx_
  """ 
  signature = (S, A, B, STWCX_ONE)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | S.render(operands['S']) | A.render(operands['A']) | B.render(operands['B']) | XO_1.render(params['XO']) | STWCX_ONE.render(operands['STWCX_ONE'])
  render = staticmethod(_render)
  
class OPCD_S_A_B_XO_1_Rc(MachineInstruction):
  """ 
  Instructions: (11)  andx, andcx, eqvx, nandx, norx, orx, orcx, slwx, srawx, srwx, xorx
  """ 
  signature = (A, S, B)
  opt_kw = (Rc,)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | S.render(operands['S']) | A.render(operands['A']) | B.render(operands['B']) | XO_1.render(params['XO']) | Rc.render(operands['Rc'])
  render = staticmethod(_render)
  
class OPCD_crfD_IMM_XO_1_Rc(MachineInstruction):
  """ 
  Instructions: (1)  mtfsfix
  """ 
  signature = (crfD, IMM)
  opt_kw = (Rc,)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | crfD.render(operands['crfD']) | IMM.render(operands['IMM']) | XO_1.render(params['XO']) | Rc.render(operands['Rc'])
  render = staticmethod(_render)
  
class OPCD_S_A_SH_XO_1_Rc(MachineInstruction):
  """ 
  Instructions: (1)  srawix
  """ 
  signature = (A, S, SH)
  opt_kw = (Rc,)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | S.render(operands['S']) | A.render(operands['A']) | SH.render(operands['SH']) | XO_1.render(params['XO']) | Rc.render(operands['Rc'])
  render = staticmethod(_render)
  
class OPCD_A_B_XO_1(MachineInstruction):
  """ 
  Instructions: (8)  dcba, dcbf, dcbi, dcbst, dcbt, dcbtst, dcbz, icbi
  """ 
  signature = (A, B)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | A.render(operands['A']) | B.render(operands['B']) | XO_1.render(params['XO'])
  render = staticmethod(_render)
  
class OPCD_D_B_XO_1_Rc(MachineInstruction):
  """ 
  Instructions: (7)  fabsx, fctiwx, fctiwzx, fmrx, fnabsx, fnegx, frspx
  """ 
  signature = (D, B)
  opt_kw = (Rc,)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | D.render(operands['D']) | B.render(operands['B']) | XO_1.render(params['XO']) | Rc.render(operands['Rc'])
  render = staticmethod(_render)
  
class OPCD_D_A_B_XO_1(MachineInstruction):
  """ 
  Instructions: (17)  eciwx, lbzux, lbzx, lfdux, lfdx, lfsux, lfsx, lhaux, lhax, lhbrx, lhzux, lhzx, lswx, lwarx, lwbrx, lwzux, lwzx
  """ 
  signature = (D, A, B)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | D.render(operands['D']) | A.render(operands['A']) | B.render(operands['B']) | XO_1.render(params['XO'])
  render = staticmethod(_render)
  
class OPCD_SC_ONE(MachineInstruction):
  """ 
  Instructions: (1)  sc
  """ 
  signature = (SC_ONE,)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | SC_ONE.render(operands['SC_ONE'])
  render = staticmethod(_render)
  
class OPCD_D_A_C_XO_3_Rc(MachineInstruction):
  """ 
  Instructions: (2)  fmulx, fmulsx
  """ 
  signature = (D, A, C)
  opt_kw = (Rc,)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | D.render(operands['D']) | A.render(operands['A']) | C.render(operands['C']) | XO_3.render(params['XO']) | Rc.render(operands['Rc'])
  render = staticmethod(_render)
  
class OPCD_D_A_B_XO_2_Rc(MachineInstruction):
  """ 
  Instructions: (2)  mulhwx, mulhwux
  """ 
  signature = (D, A, B)
  opt_kw = (Rc,)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | D.render(operands['D']) | A.render(operands['A']) | B.render(operands['B']) | XO_2.render(params['XO']) | Rc.render(operands['Rc'])
  render = staticmethod(_render)
  
class OPCD_crfD_L_A_UIMM(MachineInstruction):
  """ 
  Instructions: (1)  cmpli
  """ 
  signature = (crfD, L, A, UIMM)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | crfD.render(operands['crfD']) | L.render(operands['L']) | A.render(operands['A']) | UIMM.render(operands['UIMM'])
  render = staticmethod(_render)
  
class OPCD_crfD_A_B_XO_1(MachineInstruction):
  """ 
  Instructions: (1)  fcmpo
  """ 
  signature = (crfD, A, B)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | crfD.render(operands['crfD']) | A.render(operands['A']) | B.render(operands['B']) | XO_1.render(params['XO'])
  render = staticmethod(_render)
  
class OPCD_D_A_B_C_XO_3_Rc(MachineInstruction):
  """ 
  Instructions: (9)  fmaddx, fmaddsx, fmsubx, fmsubsx, fnmaddx, fnmaddsx, fnmsubx, fnmsubsx, fselx
  """ 
  signature = (D, A, C, B)
  opt_kw = (Rc,)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | D.render(operands['D']) | A.render(operands['A']) | B.render(operands['B']) | C.render(operands['C']) | XO_3.render(params['XO']) | Rc.render(operands['Rc'])
  render = staticmethod(_render)
  
class OPCD_S_A_B_MB_ME_Rc(MachineInstruction):
  """ 
  Instructions: (1)  rlwnmx
  """ 
  signature = (A, S, B, MB, ME)
  opt_kw = (Rc,)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | S.render(operands['S']) | A.render(operands['A']) | B.render(operands['B']) | MB.render(operands['MB']) | ME.render(operands['ME']) | Rc.render(operands['Rc'])
  render = staticmethod(_render)
  
class OPCD_S_A_XO_1_Rc(MachineInstruction):
  """ 
  Instructions: (3)  cntlzwx, extsbx, extshx
  """ 
  signature = (A, S)
  opt_kw = (Rc,)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | S.render(operands['S']) | A.render(operands['A']) | XO_1.render(params['XO']) | Rc.render(operands['Rc'])
  render = staticmethod(_render)
  
class OPCD_D_A_NB_XO_1(MachineInstruction):
  """ 
  Instructions: (1)  lswi
  """ 
  signature = (D, A, NB)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | D.render(operands['D']) | A.render(operands['A']) | NB.render(operands['NB']) | XO_1.render(params['XO'])
  render = staticmethod(_render)
  
class OPCD_D_B_XO_1(MachineInstruction):
  """ 
  Instructions: (1)  mfsrin
  """ 
  signature = (D, B)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | D.render(operands['D']) | B.render(operands['B']) | XO_1.render(params['XO'])
  render = staticmethod(_render)
  
class OPCD_S_B_XO_1(MachineInstruction):
  """ 
  Instructions: (1)  mtsrin
  """ 
  signature = (S, B)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | S.render(operands['S']) | B.render(operands['B']) | XO_1.render(params['XO'])
  render = staticmethod(_render)
  
class OPCD_D_SR_XO_1(MachineInstruction):
  """ 
  Instructions: (1)  mfsr
  """ 
  signature = (D, SR)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | D.render(operands['D']) | SR.render(operands['SR']) | XO_1.render(params['XO'])
  render = staticmethod(_render)
  
class OPCD_D_A_B_OE_XO_2_Rc(MachineInstruction):
  """ 
  Instructions: (9)  addx, addcx, addex, divwx, divwux, mullwx, subfx, subfcx, subfex
  """ 
  signature = (D, A, B)
  opt_kw = (OE, Rc)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | D.render(operands['D']) | A.render(operands['A']) | B.render(operands['B']) | OE.render(operands['OE']) | XO_2.render(params['XO']) | Rc.render(operands['Rc'])
  render = staticmethod(_render)
  
class OPCD_S_A_d(MachineInstruction):
  """ 
  Instructions: (11)  stb, stbu, stfd, stfdu, stfs, stfsu, sth, sthu, stmw, stw, stwu
  """ 
  signature = (S, A, d)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | S.render(operands['S']) | A.render(operands['A']) | d.render(operands['d'])
  render = staticmethod(_render)
  
class OPCD_crbD_crbA_crbB_XO_1(MachineInstruction):
  """ 
  Instructions: (8)  crand, crandc, creqv, crnand, crnor, cror, crorc, crxor
  """ 
  signature = (crbD, crbA, crbB)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | crbD.render(operands['crbD']) | crbA.render(operands['crbA']) | crbB.render(operands['crbB']) | XO_1.render(params['XO'])
  render = staticmethod(_render)
  
class OPCD_S_A_UIMM(MachineInstruction):
  """ 
  Instructions: (6)  andi, andis, ori, oris, xori, xoris
  """ 
  signature = (A, S, UIMM)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | S.render(operands['S']) | A.render(operands['A']) | UIMM.render(operands['UIMM'])
  render = staticmethod(_render)
  
class OPCD_D_tbr_XO_1(MachineInstruction):
  """ 
  Instructions: (1)  mftb
  """ 
  signature = (D, tbr)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | D.render(operands['D']) | tbr.render(operands['tbr']) | XO_1.render(params['XO'])
  render = staticmethod(_render)
  
class OPCD_D_A_d(MachineInstruction):
  """ 
  Instructions: (13)  lbz, lbzu, lfd, lfdu, lfs, lfsu, lha, lhau, lhz, lhzu, lmw, lwz, lwzu
  """ 
  signature = (D, A, d)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | D.render(operands['D']) | A.render(operands['A']) | d.render(operands['d'])
  render = staticmethod(_render)
  
class OPCD_D_B_XO_3_Rc(MachineInstruction):
  """ 
  Instructions: (4)  fresx, frsqrtex, fsqrtx, fsqrtsx
  """ 
  signature = (D, B)
  opt_kw = (Rc,)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | D.render(operands['D']) | B.render(operands['B']) | XO_3.render(params['XO']) | Rc.render(operands['Rc'])
  render = staticmethod(_render)
  
class OPCD_crfD_L_A_SIMM(MachineInstruction):
  """ 
  Instructions: (1)  cmpi
  """ 
  signature = (crfD, L, A, SIMM)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | crfD.render(operands['crfD']) | L.render(operands['L']) | A.render(operands['A']) | SIMM.render(operands['SIMM'])
  render = staticmethod(_render)
  
class OPCD_crfD_XO_1(MachineInstruction):
  """ 
  Instructions: (1)  mcrxr
  """ 
  signature = (crfD,)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | crfD.render(operands['crfD']) | XO_1.render(params['XO'])
  render = staticmethod(_render)
  
class OPCD_S_XO_1(MachineInstruction):
  """ 
  Instructions: (1)  mtmsr
  """ 
  signature = (S,)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | S.render(operands['S']) | XO_1.render(params['XO'])
  render = staticmethod(_render)
  
class OPCD_TO_A_B_XO_1(MachineInstruction):
  """ 
  Instructions: (1)  tw
  """ 
  signature = (TO, A, B)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | TO.render(operands['TO']) | A.render(operands['A']) | B.render(operands['B']) | XO_1.render(params['XO'])
  render = staticmethod(_render)
  
class OPCD_BO_BI_XO_1_LK(MachineInstruction):
  """ 
  Instructions: (2)  bcctrx, bclrx
  """ 
  signature = (BO, BI)
  opt_kw = (LK,)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | BO.render(operands['BO']) | BI.render(operands['BI']) | XO_1.render(params['XO']) | LK.render(operands['LK'])
  render = staticmethod(_render)
  
class OPCD_D_spr_XO_1(MachineInstruction):
  """ 
  Instructions: (1)  mfspr
  """ 
  signature = (D, spr)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | D.render(operands['D']) | spr.render(operands['spr']) | XO_1.render(params['XO'])
  render = staticmethod(_render)
  
class OPCD_D_XO_1_Rc(MachineInstruction):
  """ 
  Instructions: (1)  mffsx
  """ 
  signature = (D,)
  opt_kw = (Rc,)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | D.render(operands['D']) | XO_1.render(params['XO']) | Rc.render(operands['Rc'])
  render = staticmethod(_render)
  
class OPCD_TO_A_SIMM(MachineInstruction):
  """ 
  Instructions: (1)  twi
  """ 
  signature = (TO, A, SIMM)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | TO.render(operands['TO']) | A.render(operands['A']) | SIMM.render(operands['SIMM'])
  render = staticmethod(_render)
  
class OPCD_crfD_L_A_B_XO_1(MachineInstruction):
  """ 
  Instructions: (1)  cmpl
  """ 
  signature = (crfD, L, A, B)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | crfD.render(operands['crfD']) | L.render(operands['L']) | A.render(operands['A']) | B.render(operands['B']) | XO_1.render(params['XO'])
  render = staticmethod(_render)
  
class OPCD_crfD_L_A_B(MachineInstruction):
  """ 
  Instructions: (1)  cmp_
  """ 
  signature = (crfD, L, A, B)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | crfD.render(operands['crfD']) | L.render(operands['L']) | A.render(operands['A']) | B.render(operands['B'])
  render = staticmethod(_render)
  
class OPCD_S_A_B_XO_1(MachineInstruction):
  """ 
  Instructions: (15)  ecowx, stbux, stbx, stfdux, stfdx, stfiwx, stfsux, stfsx, sthbrx, sthux, sthx, stswx, stwbrx, stwux, stwx
  """ 
  signature = (S, A, B)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | S.render(operands['S']) | A.render(operands['A']) | B.render(operands['B']) | XO_1.render(params['XO'])
  render = staticmethod(_render)
  
class OPCD_S_spr_XO_1(MachineInstruction):
  """ 
  Instructions: (1)  mtspr
  """ 
  signature = (spr, S)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | S.render(operands['S']) | spr.render(operands['spr']) | XO_1.render(params['XO'])
  render = staticmethod(_render)
  
class OPCD_S_CRM_XO_1(MachineInstruction):
  """ 
  Instructions: (1)  mtcrf
  """ 
  signature = (CRM, S)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | S.render(operands['S']) | CRM.render(operands['CRM']) | XO_1.render(params['XO'])
  render = staticmethod(_render)
  
class OPCD_crbD_XO_1_Rc(MachineInstruction):
  """ 
  Instructions: (2)  mtfsb0x, mtfsb1x
  """ 
  signature = (crbD,)
  opt_kw = (Rc,)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | crbD.render(operands['crbD']) | XO_1.render(params['XO']) | Rc.render(operands['Rc'])
  render = staticmethod(_render)
  
class OPCD_B_XO_1(MachineInstruction):
  """ 
  Instructions: (1)  tlbie
  """ 
  signature = (B,)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | B.render(operands['B']) | XO_1.render(params['XO'])
  render = staticmethod(_render)
  
class OPCD_LI_AA_LK(MachineInstruction):
  """ 
  Instructions: (1)  bx
  """ 
  signature = (LI,)
  opt_kw = (AA, LK)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | LI.render(operands['LI']) | AA.render(operands['AA']) | LK.render(operands['LK'])
  render = staticmethod(_render)
  
class OPCD_LILBL_AA_LK(MachineInstruction):
  """ 
  Instructions: (1)  bx
  """ 
  signature = (LILBL,)
  opt_kw = (AA, LK)

  def _render(params, operands):
    # Not supporting AA=1 for labels right now, die if that happens
    if operands['AA'] != 0:
      raise Exception("AA=1 not supported with label operands")

    offset = operands['LILBL'].position - operands['position']
    return OPCD.render(params['OPCD']) | LILBL.render(offset) | AA.render(operands['AA']) | LK.render(operands['LK'])
  render = staticmethod(_render)
  
class OPCD_FM_B_XO_1_Rc(MachineInstruction):
  """ 
  Instructions: (1)  mtfsfx
  """ 
  signature = (FM, B)
  opt_kw = (Rc,)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | FM.render(operands['FM']) | B.render(operands['B']) | XO_1.render(params['XO']) | Rc.render(operands['Rc'])
  render = staticmethod(_render)
  
class OPCD_S_SR_XO_1(MachineInstruction):
  """ 
  Instructions: (1)  mtsr
  """ 
  signature = (SR, S)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | S.render(operands['S']) | SR.render(operands['SR']) | XO_1.render(params['XO'])
  render = staticmethod(_render)
  
class OPCD_crfD_crfS_XO_1(MachineInstruction):
  """ 
  Instructions: (1)  mcrfs
  """ 
  signature = (crfD, crfS)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | crfD.render(operands['crfD']) | crfS.render(operands['crfS']) | XO_1.render(params['XO'])
  render = staticmethod(_render)
  
class OPCD_crfD_crfS(MachineInstruction):
  """ 
  Instructions: (1)  mcrf
  """ 
  signature = (crfD, crfS)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | crfD.render(operands['crfD']) | crfS.render(operands['crfS'])
  render = staticmethod(_render)
  
class OPCD_D_A_B_XO_3_Rc(MachineInstruction):
  """ 
  Instructions: (6)  faddx, faddsx, fdivx, fdivsx, fsubx, fsubsx
  """ 
  signature = (D, A, B)
  opt_kw = (Rc,)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | D.render(operands['D']) | A.render(operands['A']) | B.render(operands['B']) | XO_3.render(params['XO']) | Rc.render(operands['Rc'])
  render = staticmethod(_render)
  
class OPCD_D_A_OE_XO_2_Rc(MachineInstruction):
  """ 
  Instructions: (5)  addmex, addzex, negx, subfmex, subfzex
  """ 
  signature = (D, A)
  opt_kw = (OE, Rc)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | D.render(operands['D']) | A.render(operands['A']) | OE.render(operands['OE']) | XO_2.render(params['XO']) | Rc.render(operands['Rc'])
  render = staticmethod(_render)
  
class OPCD_S_A_SH_MB_ME_Rc(MachineInstruction):
  """ 
  Instructions: (2)  rlwimix, rlwinmx
  """ 
  signature = (A, S, SH, MB, ME)
  opt_kw = (Rc,)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | S.render(operands['S']) | A.render(operands['A']) | SH.render(operands['SH']) | MB.render(operands['MB']) | ME.render(operands['ME']) | Rc.render(operands['Rc'])
  render = staticmethod(_render)
  
class OPCD_S_A_NB_XO_1(MachineInstruction):
  """ 
  Instructions: (1)  stswi
  """ 
  signature = (S, A, NB)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | S.render(operands['S']) | A.render(operands['A']) | NB.render(operands['NB']) | XO_1.render(params['XO'])
  render = staticmethod(_render)
  
class OPCD_BO_BI_BD_AA_LK(MachineInstruction):
  """ 
  Instructions: (1)  bcx
  """ 
  signature = (BO, BI, BD)
  opt_kw = (AA, LK)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | BO.render(operands['BO']) | BI.render(operands['BI']) | BD.render(operands['BD']) | AA.render(operands['AA']) | LK.render(operands['LK'])
  render = staticmethod(_render)
  
class OPCD_BO_BI_BDLBL_AA_LK(MachineInstruction):
  """ 
  Instructions: (1)  bcx
  """ 
  signature = (BO, BI, BDLBL)
  opt_kw = (AA, LK)

  def _render(params, operands):
    if operands['AA'] != 0:
      raise Exception("AA=1 not supported with label operands")

    offset = operands['BDLBL'].position - operands['position']
    return OPCD.render(params['OPCD']) | BO.render(operands['BO']) | BI.render(operands['BI']) | BD.render(offset) | AA.render(operands['AA']) | LK.render(operands['LK'])
  render = staticmethod(_render)
  
class OPCD_D_XO_1(MachineInstruction):
  """ 
  Instructions: (2)  mfcr, mfmsr
  """ 
  signature = (D,)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | D.render(operands['D']) | XO_1.render(params['XO'])
  render = staticmethod(_render)
  
class OPCD_XO_1(MachineInstruction):
  """ 
  Instructions: (6)  eieio, isync, rfi, sync, tlbia, tlbsync1
  """ 
  signature = ()

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | XO_1.render(params['XO'])
  render = staticmethod(_render)
  
class OPCD_D_A_SIMM(MachineInstruction):
  """ 
  Instructions: (6)  addi, addic, addic_, addis, mulli, subfic
  """ 
  signature = (D, A, SIMM)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | D.render(operands['D']) | A.render(operands['A']) | SIMM.render(operands['SIMM'])
  render = staticmethod(_render)
  
class OPCD_crfD_A_B(MachineInstruction):
  """ 
  Instructions: (1)  fcmpu
  """ 
  signature = (crfD, A, B)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | crfD.render(operands['crfD']) | A.render(operands['A']) | B.render(operands['B'])
  render = staticmethod(_render)
  
