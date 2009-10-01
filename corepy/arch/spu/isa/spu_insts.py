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

# Cell SPU Machine Instructions

# Notes:
#  params are: OPCD, XO

from corepy.spre.spe import MachineInstruction
from spu_fields import *

class OPCD(MachineInstruction):
  signature = ()

  def _render(params, operands):
    return OPCD11.render(params['OPCD'])
  render = staticmethod(_render)


class OPCD_T(MachineInstruction):
  signature = (T3,)

  def _render(params, operands):
    return OPCD11.render(params['OPCD']) | T3.render(operands['T3'])
  render = staticmethod(_render)


class OPCD_CF(MachineInstruction):
  signature = (CF,)

  def _render(params, operands):
    return OPCD11.render(params['OPCD']) | CF.render(operands['CF'])
  render = staticmethod(_render)


class OPCD_STOP_SIG(MachineInstruction):
  signature = (STOP_SIG,)

  def _render(params, operands):
    return OPCD11.render(params['OPCD']) | STOP_SIG.render(operands['STOP_SIG'])
  render = staticmethod(_render)


class OPCD_A_T(MachineInstruction):
  signature = (T3, A)

  def _render(params, operands):
    return OPCD11.render(params['OPCD']) | A.render(operands['A']) | T3.render(operands['T3'])
  render = staticmethod(_render)


class OPCD_SA_T(MachineInstruction):
  signature = (T3, SA)

  def _render(params, operands):
    return OPCD11.render(params['OPCD']) | SA.render(operands['SA']) | T3.render(operands['T3'])
  render = staticmethod(_render)


class OPCD_RO_A_P(MachineInstruction):
  signature = (RO, A, P)

  def _render(params, operands):
    return OPCD11.render(params['OPCD']) | RO.render(operands['RO']) | A.render(operands['A']) | P.render(operands['P'])
  render = staticmethod(_render)


class OPCD_LBL9_A_P(MachineInstruction):
  signature = (LBL9, A, P)

  def _render(params, operands):
    offset = (operands['LBL9'].position - operands['position']) >> 2
    return OPCD11.render(params['OPCD']) | RO.render(offset) | A.render(operands['A']) | P.render(operands['P'])
  render = staticmethod(_render)


class OPCD_ROA_I16(MachineInstruction):
  signature = (ROA, I16)

  def _render(params, operands):
    return OPCD7.render(params['OPCD']) | ROA.render(operands['ROA']) | I16.render(operands['I16'])
  render = staticmethod(_render)


class OPCD_LBL9_I16(MachineInstruction):
  signature = (LBL9, I16)

  def _render(params, operands):
    offset = (operands['LBL9'].position - operands['position']) >> 2
    return OPCD7.render(params['OPCD']) | ROA.render(offset) | I16.render(operands['I16'])
  render = staticmethod(_render)


class OPCD_LBL9_LBL16(MachineInstruction):
  signature = (LBL9, LBL16)

  def _render(params, operands):
    off9 = (operands['LBL9'].position - operands['position']) >> 2
    off16 = (operands['LBL16'].position - operands['position']) >> 2

    if abs(off9) > 255:
      print RuntimeWarning("SPU hint offset to branch is too large: " + str(off9))
    if abs(off16) > 32767:
      print RuntimeWarning("SPU hint offset to branch target is too large: " + str(off16))

    return OPCD7.render(params['OPCD']) | ROA.render(off9) | I16.render(off16)
  render = staticmethod(_render)


class OPCD_B_A_T(MachineInstruction):
  signature = (T3, A, B)

  def _render(params, operands):
    return OPCD11.render(params['OPCD']) | B.render(operands['B']) | A.render(operands['A']) | T3.render(operands['T3'])
  render = staticmethod(_render)


class OPCD_A_D_E(MachineInstruction):
  """ TODO - test  instructions that use A/D/E i.e. bi """
  signature = (A, D, E)

  def _render(params, operands):
    return OPCD11.render(params['OPCD']) | A.render(operands['A']) | D.render(operands['D']) | E.render(operands['E'])
  render = staticmethod(_render)


class OPCD_A_T_D_E(MachineInstruction):
  """ TODO - test  instructions that use A/T/D/E i.e. bisled """
  signature = (T3, A, D, E)

  def _render(params, operands):
    return OPCD11.render(params['OPCD']) | A.render(operands['A']) | T3.render(operands['T3']), D.render(operands['D']) | E.render(operands['E'])
  render = staticmethod(_render)


class OPCD_T_B_A_C(MachineInstruction):
  signature = (T4, A, B, C)

  def _render(params, operands):
    return OPCD4.render(params['OPCD']) | T4.render(operands['T4']) | B.render(operands['B']) | A.render(operands['A']) | C.render(operands['C'])
  render = staticmethod(_render)


class OPCD_I7_A_T(MachineInstruction):
  signature = (T3, A, I7)

  def _render(params, operands):
    return OPCD11.render(params['OPCD']) | I7.render(operands['I7']) | A.render(operands['A']) | T3.render(operands['T3'])
  render = staticmethod(_render)


class OPCD_I8_A_T(MachineInstruction):
  signature = (T3, A, I8)

  def _render(params, operands):
    return OPCD10.render(params['OPCD']) | I8.render(operands['I8']) | A.render(operands['A']) | T3.render(operands['T3'])
  render = staticmethod(_render)


class OPCD_I10_A_T(MachineInstruction):
  signature = (T3, A, I10)

  def _render(params, operands):
    return OPCD8.render(params['OPCD']) | I10.render(operands['I10']) | A.render(operands['A']) | T3.render(operands['T3'])
  render = staticmethod(_render)


class OPCD_I16(MachineInstruction):
  signature = (I16,)

  def _render(params, operands):
    return OPCD9.render(params['OPCD']) | I16.render(operands['I16'])
  render = staticmethod(_render)


class OPCD_LBL16(MachineInstruction):
  signature = (LBL16,)

  def _render(params, operands):
    offset = (operands['LBL16'].position - operands['position']) >> 2
    return OPCD9.render(params['OPCD']) | I16.render(offset)
  render = staticmethod(_render)


class OPCD_I16_T(MachineInstruction):
  signature = (T3, I16)

  def _render(params, operands):
    return OPCD9.render(params['OPCD']) | I16.render(operands['I16']) | T3.render(operands['T3'])
  render = staticmethod(_render)


class OPCD_LBL16_T(MachineInstruction):
  signature = (T3, LBL16)

  def _render(params, operands):
    offset = (operands['LBL16'].position - operands['position']) >> 2
    return OPCD9.render(params['OPCD']) | I16.render(offset) | T3.render(operands['T3'])
  render = staticmethod(_render)


class OPCD_I18_T(MachineInstruction):
  signature = (T3, I18)

  def _render(params, operands):
    return OPCD7.render(params['OPCD']) | I18.render(operands['I18']) | T3.render(operands['T3'])
  render = staticmethod(_render)


#class OPCD_I16_A_T(MachineInstruction):
#  signature = (I16, A, T3)

#  def _render(params, operands):
#    return OPCD9.render(params['OPCD9']) | I16.render(operands['I16']) | A.render(operands['A']) | T3.render(operands['T3'])
#  render = staticmethod(_render)

