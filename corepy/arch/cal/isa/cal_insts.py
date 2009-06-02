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
from cal_fields import *

class OPCD(MachineInstruction):
  signature = ()

  def _render(params, operands):
    return OPCD0.render(params['OPCD'])
  render = staticmethod(_render)

class OPCD_RESOURCE_TYPE_UNNORM_FMT_0_0(MachineInstruction):
  signature = (RESOURCEID, TYPE, FMT)
  opt_kw = (UNNORM,)

  def _render(params, operands):
    return OPCD0.render(params['OPCD']) + RESOURCEID.render(operands['RESOURCEID']) + '_type(' + TYPE.render(operands['TYPE']) + UNNORM.render(operands['UNNORM']) + ')' + FMT.render(operands['FMT'])
  render = staticmethod(_render)

class OPCD_THREADS_LDS_MEMORY_SR_0_0(MachineInstruction):
  signature = ()
  opt_kw = (THREADS, LDS, MEMORY, SR)

  def _render(params, operands):
    return OPCD0.render(params['OPCD']) + THREADS.render(operands['THREADS']) + LDS.render(operands['LDS']) + MEMORY.render(operands['MEMORY']) + SR.render(operands['SR'])
  render = staticmethod(_render)

class OPCD_STAGE_TYPE_COORDMODE_0_0(MachineInstruction):
  signature = (STAGE, TYPE, COORDMODE)

  def _render(params, operands):
    return OPCD0.render(params['OPCD']) + STAGE.render(operands['STAGE']) + '_type(' + TYPE.render(operands['TYPE'])  + ')' + COORDMODE.render(operands['COORDMODE'])
  render = staticmethod(_render)

class OPCD_0_0_TOPOLOGY(MachineInstruction):
  signature = (TOPOLOGY)

  def _render(params, operands):
    return OPCD0.render(params['OPCD']) + ' ' + TOPOLOGY.render(operands['TOPOLOGY'])
  render = staticmethod(_render)

class OPCD_0_0_OUTPUTTOPOLOGY(MachineInstruction):
  signature = (OUTPUTTOPOLOGY)

  def _render(params, operands):
    return OPCD0.render(params['OPCD']) + ' ' + OUTPUTTOPOLOGY.render(operands['OUTPUTTOPOLOGY'])
  render = staticmethod(_render)

class OPCD_0_0_IL(MachineInstruction):
  signature = (IL0)

  def _render(params, operands):
    return OPCD0.render(params['OPCD']) + ' ' + IL0.render(operands['IL0'])
  render = staticmethod(_render)

class OPCD_0_0_LBL(MachineInstruction):
  signature = (LBL,)

  def _render(params, operands):
    return OPCD0.render(params['OPCD']) + ' ' + LBL.render(operands['LBL'])
  render = staticmethod(_render)

class OPCD_0_0(MachineInstruction):
  signature = ()

  def _render(params, operands):
    return OPCD0.render(params['OPCD'])
  render = staticmethod(_render)

class OPCD_XYZWImport_CENTER_BIAS_INVERT_CENTERED_0_1(MachineInstruction):
  signature = (XYZWImport,TRGT)
  opt_kw = (CENTER,BIAS,INVERT,CENTERED)

  def _render(params, operands):
    return OPCD0.render(params['OPCD']) + XYZWImport.render(operands['XYZWImport']) + CENTER.render(operands['CENTER']) + BIAS.render(operands['BIAS']) + INVERT.render(operands['INVERT']) + CENTERED.render(operands['CENTERED']) + ' ' + TRGT.render(operands['TRGT'])
  render = staticmethod(_render)

class OPCD_USAGE_USAGEINDEX_XYZWImport_0_1(MachineInstruction):
  """
  Instructions: (1) dclvout
  """
  signature = (USAGE, USAGEINDEX, XYZWImport, TRGT)

  def _render(params, operands):
    return OPCD0.render(params['OPCD']) + USAGE.render(operands['USAGE']) + USAGEINDEX.render(operands['USAGEINDEX']) + XYZWImport.render(operands['XYZWImport']) + ' ' + TRGT.render(operands['TRGT'])
  render = staticmethod(_render)

class OPCD_USAGE_XYZWImport_CENTROID_0_1(MachineInstruction):
  """
  Instructions: (1) dclpin
  """
  signature = (USAGE, XYZWImport, TRGT)
  opt_kw = (CENTROID, ORIGIN)

  def _render(params, operands):
    return OPCD0.render(params['OPCD']) + USAGE.render(operands['USAGE']) + XYZWImport.render(operands['XYZWImport']) + CENTROID.render(operands['CENTROID']) + ORIGIN.render(operands['ORIGIN']) + ' ' + TRGT.render(operands['TRGT'])
  render = staticmethod(_render)

class OPCD_USAGE_INTERP_0_1(MachineInstruction):
  signature = (TRGT,)
  opt_kw = (USAGE, INTERP)

  def _render(params, operands):
    return OPCD0.render(params['OPCD']) + USAGE.render(operands['USAGE']) + INTERP.render(operands['INTERP']) + ' ' + TRGT.render(operands['TRGT'])
  render = staticmethod(_render)

class OPCD_VELEM_XYZWImport_0_1(MachineInstruction):
  """
  Instructions: (1) dclv
  """
  signature = (VELEM, XYZWImport, TRGT)

  def _render(params, operands):
    return OPCD0.render(params['OPCD']) + VELEM.render(operands['VELEM']) + XYZWImport.render(operands['XYZWImport']) + ' ' + TRGT.render(operands['TRGT'])
  render = staticmethod(_render)

class OPCD_PARAM_0_1(MachineInstruction):
  """
  Instructions: (1) dclpp
  """
  signature = (PARAM, TRGT)

  def _render(params, operands):
    return OPCD0.render(params['OPCD']) + PARAM.render(operands['PARAM']) + ' ' + TRGT.render(operands['TRGT'])
  render = staticmethod(_render)

class OPCD_USAGE_0_1(MachineInstruction):
  """
  Instructions: (1) dcl_output
  """
  signature = (TRGT,)
  opt_kw = (USAGE,)

  def _render(params, operands):
    return OPCD0.render(params['OPCD']) + USAGE.render(operands['USAGE']) + ' ' + TRGT.render(operands['TRGT'])
  render = staticmethod(_render)

class OPCD_XYZWDefault_0_1(MachineInstruction):
  """
  Instructions: (1) dcldef
  """
  signature = (XYZWDefault, TRGT)

  def _render(params, operands):
    return OPCD0.render(params['OPCD']) + XYZWDefault.render(operands['XYZWDefault']) + ' ' + TRGT.render(operands['TRGT'])
  render = staticmethod(_render)

class OPCD_0_1_L4(MachineInstruction):
  """
  Instructions: (1) def_
  """
  signature = (TRGT, L0, L1, L2, L3)

  def _render(params, operands):
    return OPCD0.render(params['OPCD']) + ' ' + TRGT.render(operands['TRGT']) + ', ' + L0.render(operands['L0']) + ', ' + L1.render(operands['L1']) + ', ' + L2.render(operands['L2']) + ', ' + L3.render(operands['L3'])
  render = staticmethod(_render)

class OPCD_LOGICOP_1_0(MachineInstruction):
  signature = (LOGICOP, TRGT)

  def _render(params, operands):
    return OPCD0.render(params['OPCD']) + LOGICOP.render(operands['LOGICOP']) + ' ' + TRGT.render(operands['TRGT'])
  render = staticmethod(_render)

class OPCD_1_0_REPEAT(MachineInstruction):
  signature = (TRGT,)
  opt_kw = (REPEAT,)

  def _render(params, operands):
    return OPCD0.render(params['OPCD']) + REPEAT.render(operands['REPEAT']) + ' ' + TRGT.render(operands['TRGT'])
  render = staticmethod(_render)

class OPCD_STAGE_SAMPLE_1_0(MachineInstruction):
  signature = (TRGT,)
  opt_kw = (STAGE, SAMPLE)

  def _render(params, operands):
    return OPCD0.render(params['OPCD']) + STAGE.render(operands['STAGE']) + SAMPLE.render(operands['SAMPLE']) + ' ' + TRGT.render(operands['TRGT'])
  render = staticmethod(_render)

class OPCD_1_0_L4(MachineInstruction):
  signature = (SRC0, L0, L1, L2, L3)

  def _render(params, operands):
    return OPCD0.render(params['OPCD']) + ' ' + SRC0.render(operands['SRC0']) + ', ' + L0.render(operands['L0']) + ', ' + L1.render(operands['L1']) + ', ' + L2.render(operands['L2']) + ', ' + L3.render(operands['L3'])
  render = staticmethod(_render)

class OPCD_0_1_BOOL(MachineInstruction):
  signature = (TRGT, BOOL)

  def _render(params, operands):
    return OPCD0.render(params['OPCD']) + ' ' + TRGT.render(operands['TRGT']) + ', ' + BOOL.render(operands['BOOL'])
  render = staticmethod(_render)


class OPCD_1_0(MachineInstruction):
  signature = (TRGT,)

  def _render(params, operands):
    return OPCD0.render(params['OPCD']) + ' ' + TRGT.render(operands['TRGT'])
  render = staticmethod(_render)

class OPCD_1_0_LBL(MachineInstruction):
  signature = (SRC0, LBL)

  def _render(params, operands):
    return OPCD0.render(params['OPCD']) + ' ' + SRC0.render(operands['SRC0']) + ', ' + LBL.render(operands['LBL'])
  render = staticmethod(_render)

class OPCD_ZEROOP_1_1(MachineInstruction):
  """
  Instructions: (6) ln, log, logp, nrm, rcp, rsq
  """
  signature = (TRGT, SRC0)
  opt_kw = (ZEROOP, SHIFT, SAT)

  def _render(params, operands):
    return OPCD0.render(params['OPCD']) + ZEROOP.render(operands['ZEROOP']) + SHIFT.render(operands['SHIFT']) + SAT.render(operands['SAT']) + ' ' + TRGT.render(operands['TRGT']) + ', ' + SRC0.render(operands['SRC0'])
  render = staticmethod(_render)

class OPCD_RESOURCE_SAMPLER_AOFFIMMI_1_1(MachineInstruction):
  signature = (RESOURCE, SAMPLER, TRGT, SRC0)
  opt_kw = (AOFFIMMI,SHIFT,SAT)

  def _render(params, operands):
    return OPCD0.render(params['OPCD']) + RESOURCE.render(operands['RESOURCE']) + SAMPLER.render(operands['SAMPLER']) + AOFFIMMI.render(operands['AOFFIMMI']) + SHIFT.render(operands['SHIFT']) + SAT.render(operands['SAT']) + ' ' + TRGT.render(operands['TRGT']) + ', ' + SRC0.render(operands['SRC0'])
  render = staticmethod(_render)

class OPCD_RESOURCE_AOFFIMMI_1_1(MachineInstruction):
  signature = (RESOURCE, TRGT, SRC0)
  opt_kw = (AOFFIMMI,SHIFT,SAT)

  def _render(params, operands):
    return OPCD0.render(params['OPCD']) + RESOURCE.render(operands['RESOURCE']) + AOFFIMMI.render(operands['AOFFIMMI']) + SHIFT.render(operands['SHIFT']) + SAT.render(operands['SAT']) + ' ' + TRGT.render(operands['TRGT']) + ', ' + SRC0.render(operands['SRC0'])
  render = staticmethod(_render)

class OPCD_RESOURCE_UINT_1_1(MachineInstruction):
  signature = (RESOURCE, TRGT, SRC0)
  opt_kw = (UINT,SHIFT,SAT)

  def _render(params, operands):
    return OPCD0.render(params['OPCD']) + RESOURCE.render(operands['RESOURCE']) + UINT.render(operands['UINT']) + SHIFT.render(operands['SHIFT']) + SAT.render(operands['SAT']) + ' ' + TRGT.render(operands['TRGT']) + ', ' + SRC0.render(operands['SRC0'])
  render = staticmethod(_render)

class OPCD_NEIGHBOREXCH_SHARINGMODE_1_1(MachineInstruction):
  """
  Instructions: (1) lds_read_vec
  """
  signature = (TRGT, SRC0)
  opt_kw = (NEIGHBOREXCH, SHARINGMODE, SHIFT,SAT)
  
  def _render(params, operands):
    return OPCD0.render(params['OPCD']) + NEIGHBOREXCH.render(operands['NEIGHBOREXCH']) + SHARINGMODE.render(operands['SHARINGMODE']) + SHIFT.render(operands['SHIFT']) + SAT.render(operands['SAT']) + ' ' + TRGT.render(operands['TRGT']) + ', ' + SRC0.render(operands['SRC0'])
  render = staticmethod(_render)

class OPCD_LOFFSET_SHARINGMODE_1_1(MachineInstruction):
  """
  Instructions: (1) lds_write_vec
  """
  signature = (TRGT, SRC0)
  opt_kw = (LOFFSET, SHARINGMODE, SHIFT,SAT)

  def _render(params, operands):
    return OPCD0.render(params['OPCD']) + LOFFSET.render(operands['LOFFSET']) + SHARINGMODE.render(operands['SHARINGMODE']) + SHIFT.render(operands['SHIFT']) + SAT.render(operands['SAT']) + ' ' + TRGT.render(operands['TRGT']) + ', ' + SRC0.render(operands['SRC0'])
  render = staticmethod(_render)

class OPCD_STAGE_1_1(MachineInstruction):
  """
  Instructions: (2) lod, texweight
  """ 
  signature = (STAGE, TRGT, SRC0)
  opt_kw = (SHIFT,SAT)

  def _render(params, operands):
    return OPCD0.render(params['OPCD']) + STAGE.render(operands['STAGE']) + SHIFT.render(operands['SHIFT']) + SAT.render(operands['SAT']) + ' ' + TRGT.render(operands['TRGT']) + ', ' + SRC0.render(operands['SRC0'])
  render = staticmethod(_render)

class OPCD_ELEM_1_1(MachineInstruction):
  """
  Instructions: (1) memimport
  """
  signature = (ELEM, TRGT, SRC0)
  opt_kw = (SHIFT,SAT)

  def _render(params, operands):
    return OPCD0.render(params['OPCD']) + ELEM.render(operands['ELEM']) + SHIFT.render(operands['SHIFT']) + SAT.render(operands['SAT']) + ' ' + TRGT.render(operands['TRGT']) + ', ' + SRC0.render(operands['SRC0'])
  render = staticmethod(_render)

class OPCD_NOISETYPE_1_1(MachineInstruction):
  """
  Instructions: (1) noise
  """
  signature = (NOISETYPE, TRGT, SRC0)
  opt_kw = (SHIFT, SAT)

  def _render(params, operands):
    return OPCD0.render(params['OPCD']) + NOISETYPE.render(operands['NOISETYPE']) + SHIFT.render(operands['SHIFT']) + SAT.render(operands['SAT']) + ' ' + TRGT.render(operands['TRGT']) + ', ' + SRC0.render(operands['SRC0'])
  render = staticmethod(_render)

class OPCD_CENTROID_1_1(MachineInstruction):
  """
  Instructions: (2) dsx, dsy
  """
  signature = (TRGT, SRC0)
  opt_kw = (CENTROID, SHIFT, SAT)

  def _render(params, operands):
    return OPCD0.render(params['OPCD']) + CENTROID.render(operands['CENTROID']) + SHIFT.render(operands['SHIFT']) + SAT.render(operands['SAT']) + ' ' + TRGT.render(operands['TRGT']) + ', ' + SRC0.render(operands['SRC0'])
  render = staticmethod(_render)

class OPCD_ROUND_1_1(MachineInstruction):
  """
  Instructions: (1) mova
  """
  signature = (TRGT, SRC0)
  opt_kw = (ROUND, SHIFT, SAT)

  def _render(params, operands):
    return OPCD0.render(params['OPCD']) + ROUND.render(operands['ROUND']) + SHIFT.render(operands['SHIFT']) + SAT.render(operands['SAT']) + ' ' + TRGT.render(operands['TRGT']) + ', ' + SRC0.render(operands['SRC0'])
  render = staticmethod(_render)

class OPCD_1_1(MachineInstruction):
  signature = (TRGT, SRC0)
  opt_kw = (SHIFT,SAT)

  def _render(params, operands):
    return OPCD0.render(params['OPCD']) + SHIFT.render(operands['SHIFT']) + SAT.render(operands['SAT']) + ' ' + TRGT.render(operands['TRGT']) + ', ' + SRC0.render(operands['SRC0'])
  render = staticmethod(_render)

class OPCD_STREAM_OFFSET_2_0(MachineInstruction):
  signature = (STREAM, OFFSET, SRC0, SRC1)

  def _render(params, operands):
    return OPCD0.render(params['OPCD']) + STREAM.render(operands['STREAM']) + OFFSET.render(operands['OFFSET']) + ' ' + SRC0.render(operands['SRC0']) + ', ' + SRC1.render(operands['SRC1'])
  render = staticmethod(_render)

class OPCD_2_0(MachineInstruction):
  """
  Instructions: (1) dclarray
  """
  signature = (SRC0, SRC1)

  def _render(params, operands):
    return OPCD0.render(params['OPCD']) + ' ' + SRC0.render(operands['SRC0']) + ', ' + SRC1.render(operands['SRC1'])
  render = staticmethod(_render)

class OPCD_RELOP_2_0(MachineInstruction):
  signature = (RELOP, SRC0, SRC1)

  def _render(params, operands):
    return OPCD0.render(params['OPCD']) + RELOP.render(operands['RELOP']) +  ' ' + SRC0.render(operands['SRC0']) + ', ' + SRC1.render(operands['SRC1'])
  render = staticmethod(_render)

class OPCD_IEEE_2_1(MachineInstruction):
  signature = (TRGT, SRC0, SRC1)
  opt_kw = (IEEE,SHIFT,SAT)

  def _render(params, operands):
    return OPCD0.render(params['OPCD']) + IEEE.render(operands['IEEE']) + SHIFT.render(operands['SHIFT']) + SAT.render(operands['SAT']) + ' ' + TRGT.render(operands['TRGT']) + ', ' + SRC0.render(operands['SRC0']) + ', ' + SRC1.render(operands['SRC1'])
  render = staticmethod(_render)

class OPCD_NORMALIZE_2_1(MachineInstruction):
  signature = (TRGT, SRC0, SRC1)
  opt_kw = (NORMALIZE,SHIFT,SAT)

  def _render(params, operands):
    return OPCD0.render(params['OPCD']) + NORMALIZE.render(operands['NORMALIZE']) + SHIFT.render(operands['SHIFT']) + SAT.render(operands['SAT']) + ' ' + TRGT.render(operands['TRGT']) + ', ' + SRC0.render(operands['SRC0']) + ', ' + SRC1.render(operands['SRC1'])
  render = staticmethod(_render)

class OPCD_RELOP_2_1(MachineInstruction):
  signature = (RELOP, TRGT, SRC0, SRC1)
  opt_kw = (SHIFT,SAT)

  def _render(params, operands):
    return OPCD0.render(params['OPCD'])  + RELOP.render(operands['RELOP']) + SHIFT.render(operands['SHIFT']) + SAT.render(operands['SAT']) + ' ' + TRGT.render(operands['TRGT']) + ', ' + SRC0.render(operands['SRC0']) + ', ' + SRC1.render(operands['SRC1'])
  render = staticmethod(_render)

class OPCD_ZEROOP_2_1(MachineInstruction):
  signature = (TRGT, SRC0, SRC1)
  opt_kw = (ZEROOP, SHIFT,SAT)

  def _render(params, operands):
    return OPCD0.render(params['OPCD'])  + ZEROOP.render(operands['ZEROOP']) + SHIFT.render(operands['SHIFT']) + SAT.render(operands['SAT']) + ' ' + TRGT.render(operands['TRGT']) + ', ' + SRC0.render(operands['SRC0']) + ', ' + SRC1.render(operands['SRC1'])
  render = staticmethod(_render)

class OPCD_RESOURCE_SAMPLER_AOFFIMMI_2_1(MachineInstruction):
  signature = (RESOURCE, SAMPLER, TRGT, SRC0, SRC1)
  opt_kw = (AOFFIMMI,SHIFT,SAT)

  def _render(params, operands):
    return OPCD0.render(params['OPCD']) + RESOURCE.render(operands['RESOURCE']) + SAMPLER.render(operands['SAMPLER']) + AOFFIMMI.render(operands['AOFFIMMI']) + SHIFT.render(operands['SHIFT']) + SAT.render(operands['SAT']) + ' ' + TRGT.render(operands['TRGT']) + ', ' + SRC0.render(operands['SRC0']) + ', ' + SRC1.render(operands['SRC1'])
  render = staticmethod(_render)

class OPCD_RESOURCE_SAMPLER_2_1(MachineInstruction):
  signature = (RESOURCE, SAMPLER, TRGT, SRC0, SRC1)
  opt_kw = (SHIFT,SAT)

  def _render(params, operands):
    return OPCD0.render(params['OPCD']) + RESOURCE.render(operands['RESOURCE']) + SAMPLER.render(operands['SAMPLER']) + SHIFT.render(operands['SHIFT']) + SAT.render(operands['SAT']) + ' ' + TRGT.render(operands['TRGT']) + ', ' + SRC0.render(operands['SRC0']) + ', ' + SRC1.render(operands['SRC1'])
  render = staticmethod(_render)

class OPCD_MATRIX_2_1(MachineInstruction):
  signature = (MATRIX, TRGT, SRC0, SRC1)
  opt_kw = (SHIFT,SAT)

  def _render(params, operands):
    return OPCD0.render(params['OPCD'])  + MATRIX.render(operands['MATRIX']) + SHIFT.render(operands['SHIFT']) + SAT.render(operands['SAT']) + ' ' + TRGT.render(operands['TRGT']) + ', ' + SRC0.render(operands['SRC0']) + ', ' + SRC1.render(operands['SRC1'])
  render = staticmethod(_render)

class OPCD_2_1(MachineInstruction):
  signature = (TRGT, SRC0, SRC1)
  opt_kw = (SHIFT,SAT)

  def _render(params, operands):
    return OPCD0.render(params['OPCD']) + SHIFT.render(operands['SHIFT']) + SAT.render(operands['SAT']) + ' ' + TRGT.render(operands['TRGT']) + ', ' + SRC0.render(operands['SRC0']) + ', ' + SRC1.render(operands['SRC1'])
  render = staticmethod(_render)

class OPCD_RELOP_CMPVAL_3_1(MachineInstruction):
  signature = (RELOP, CMPVAL, TRGT, SRC0, SRC1, SRC2)
  opt_kw = (SHIFT, SAT)

  def _render(params, operands):
    return OPCD0.render(params['OPCD'])  + RELOP.render(operands['RELOP']) + CMPVAL.render(operands['CMPVAL'])  + SHIFT.render(operands['SHIFT']) + SAT.render(operands['SAT']) + ' ' + TRGT.render(operands['TRGT']) + ', ' + SRC0.render(operands['SRC0']) + ', ' + SRC1.render(operands['SRC1']) + ', ' + SRC2.render(operands['SRC2'])
  render = staticmethod(_render)

class OPCD_IEEE_3_1(MachineInstruction):
  signature = (TRGT, SRC0, SRC1, SRC2)
  opt_kw = (IEEE,SHIFT,SAT)

  def _render(params, operands):
    return OPCD0.render(params['OPCD']) + IEEE.render(operands['IEEE']) + SHIFT.render(operands['SHIFT']) + SAT.render(operands['SAT']) + ' ' + TRGT.render(operands['TRGT']) + ', ' + SRC0.render(operands['SRC0']) + ', ' + SRC1.render(operands['SRC1']) + ', ' + SRC2.render(operands['SRC2'])
  render = staticmethod(_render)

class OPCD_RESOURCE_SAMPLER_AOFFIMMI_3_1(MachineInstruction):
  signature = (RESOURCE, SAMPLER, TRGT, SRC0, SRC1, SRC2)
  opt_kw = (AOFFIMMI,SHIFT,SAT)

  def _render(params, operands):
    return OPCD0.render(params['OPCD']) + RESOURCE.render(operands['RESOURCE']) + SAMPLER.render(operands['SAMPLER']) + AOFFIMMI.render(operands['AOFFIMMI']) + SHIFT.render(operands['SHIFT']) + SAT.render(operands['SAT']) + ' ' + TRGT.render(operands['TRGT']) + ', ' + SRC0.render(operands['SRC0']) + ', ' + SRC1.render(operands['SRC1']) + ', ' + SRC2.render(operands['SRC2'])
  render = staticmethod(_render)

class OPCD_3_1(MachineInstruction):
  signature = (TRGT, SRC0, SRC1, SRC2)
  opt_kw = (SHIFT,SAT)

  def _render(params, operands):
    return OPCD0.render(params['OPCD']) + SHIFT.render(operands['SHIFT']) + SAT.render(operands['SAT']) + ' ' + TRGT.render(operands['TRGT']) + ', ' + SRC0.render(operands['SRC0']) + ', ' + SRC1.render(operands['SRC1']) + ', ' + SRC2.render(operands['SRC2'])
  render = staticmethod(_render)

class OPCD_RESOURCE_SAMPLER_AOFFIMMI_4_1(MachineInstruction):
  signature = (RESOURCE, SAMPLER, TRGT, SRC0, SRC1, SRC2, SRC3)
  opt_kw = (AOFFIMMI,SHIFT,SAT)

  def _render(params, operands):
    return OPCD0.render(params['OPCD']) + RESOURCE.render(operands['RESOURCE']) + SAMPLER.render(operands['SAMPLER']) + AOFFIMMI.render(operands['AOFFIMMI']) + SHIFT.render(operands['SHIFT']) + SAT.render(operands['SAT']) + ' ' + TRGT.render(operands['TRGT']) + ', ' + SRC0.render(operands['SRC0']) + ', ' + SRC1.render(operands['SRC1']) + ', ' + SRC2.render(operands['SRC2']) + ', ' + SRC3.render(operands['SRC3'])
  render = staticmethod(_render)
