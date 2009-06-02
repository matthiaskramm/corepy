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

# VMX Machine Instructions

from corepy.spre.spe import MachineInstruction
from vmx_fields import *

class OPCD_vD_vA_vB_vC_XO(MachineInstruction):
  signature = (vD, vA, vB, vC)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | vD.render(operands['vD']) | vA.render(operands['vA']) | vB.render(operands['vB']) | vC.render(operands['vC']) | VA_XO.render(params['XO'])
  render = staticmethod(_render)


class OPCD_vD_vA_vB_SH_XO(MachineInstruction):
  signature = (vD, vA, vB, SH)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | vD.render(operands['vD']) | vA.render(operands['vA']) | vB.render(operands['vB']) | SH.render(operands['SH']) | VA_XO.render(params['XO'])
  render = staticmethod(_render)


class OPCD_vD_vA_vB_XO(MachineInstruction):
  signature = (vD, vA, vB)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | vD.render(operands['vD']) | vA.render(operands['vA']) | vB.render(operands['vB']) | VX_XO.render(params['XO'])
  render = staticmethod(_render)


class OPCD_vD_XO(MachineInstruction):
  signature = (vD)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | vD.render(operands['vD']) | VX_XO.render(params['XO'])
  render = staticmethod(_render)


class OPCD_vB_XO(MachineInstruction):
  signature = (vB)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | vB.render(operands['vB']) | VX_XO.render(params['XO'])
  render = staticmethod(_render)


class OPCD_vD_vB_XO(MachineInstruction):
  signature = (vD, vB)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | vD.render(operands['vD']) | vB.render(operands['vB']) | VX_XO.render(params['XO'])
  render = staticmethod(_render)


class OPCD_vD_UIMM_vB_XO(MachineInstruction):
  signature = (vD, UIMM, vB)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | vD.render(operands['vD']) | UIMM.render(operands['UIMM']) | vB.render(operands['vB']) | VX_XO.render(params['XO'])
  render = staticmethod(_render)


class OPCD_vD_SIMM_XO(MachineInstruction):
  signature = (vD, SIMM)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | vD.render(operands['vD']) | SIMM.render(operands['SIMM']) | VX_XO.render(params['XO'])
  render = staticmethod(_render)


class OPCD_T_STRM_A_B_XO(MachineInstruction):
  signature = (A, B, STRM)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | T.render(params['T']) | STRM.render(operands['STRM']) | A.render(operands['vD']) | B.render(operands['B']) | X_XO.render(params['XO'])
  render = staticmethod(_render)


class OPCD_T_STRM_XO(MachineInstruction):
  signature = (STRM)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | T.render(0) | STRM.render(operands['STRM']) | A.render(operands['vD']) | B.render(operands['B']) | X_XO.render(params['XO'])
  render = staticmethod(_render)


class OPCD_T_XO(MachineInstruction):
  signature = ()

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | T.render(1) | X_XO.render(params['XO'])
  render = staticmethod(_render)


class OPCD_vD_A_B_XO(MachineInstruction):
  signature = (vD, A, B)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | vD.render(operands['vD']) | A.render(operands['A']) | B.render(operands['B']) | X_XO.render(params['XO'])
  render = staticmethod(_render)


class OPCD_vD_vA_vB_RC_XO(MachineInstruction):
  signature = (vD, vA, vB)
  opt_kw = (Rc,)

  def _render(params, operands):
    return OPCD.render(params['OPCD']) | vD.render(operands['vD']) | vA.render(operands['vA']) | vB.render(operands['vB']) | Rc.render(operands['Rc']) | VXR_XO.render(params['XO'])
  render = staticmethod(_render)



