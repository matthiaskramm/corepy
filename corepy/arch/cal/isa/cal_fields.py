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

from corepy.spre.spe import Register, InstructionOperand#, Variable, Label


class CALField(InstructionOperand):
  def __init__(self, name, default = None):
    InstructionOperand.__init__(self, name, default)
    return

  def render(self, value):
    return value

#  def __eq__(self, other):
#    return type(self) == type(other)

class CALFlagField(CALField):
  def __init__(self, name, ilstr, default = None):
    self.ilstr = ilstr
    InstructionOperand.__init__(self, name, default)
    return

  def check(self, value):
    return value in (True, False)

  def render(self, value):
    if value:
      return self.ilstr
    return ''

#  def __eq__(self, other):
#    return type(self) == type(other) and self.name == other.name

class RegisterField(CALField):
  def check(self, value):
    return True

  def render(self, value):
    if type(value) == type(Register):
      return value.render()
    else:
      return str(value)

class RELOPField(CALField):
  def check(self, value):
    return value in ('eq', 'ge', 'gt', 'le', 'lt', 'ne')

  def render(self, value):
    return '_relop(' + value + ')'

class ZEROOPField(CALField):
  def check(self, value):
    return value in ('zero', 'fltmax', 'inf_else_max', 'infinity')

  def render(self, value):
    return '_zeroop(' + value + ')'

class LOGICOPField(CALField):
  def check(self, value):
    return value in ('eq', 'ne')

  def render(self, value):
    return '_logicop(' + value + ')'

class USAGEField(CALField):
  def check(self, value):
    #if value == None:
    #  return True
    return value in ('backcolor', 'color', 'fog', 'generic', 'pointsize', 'pos', 'wincoord', None)

  def render(self, value):
    if value == None:
      return ''
    else:
      return '_usage(' + value + ')'

class INTERPField(CALField):
  def check(self, value):
    return value in ('constant', 'linear', 'centroid', 'noperspective', 'noper_centroid', 'noper_sample', 'sample', 'notused', 'linear_centroid', 'linear_noperspective', 'linear_noper_centroid', 'linear_noper_sample', 'linear_sample')

  def render(self, value):
    if value != 'notused':
      return '_interp(' + value + ')'
    else:
      return ''

class SHARINGMODEField(CALField):
  def check(self, value):
    return value in ('rel', 'abs', None)

  def render(self, value):
    if value == None:
      return ''
    else:
      return '_sharingMode(' + value + ')'

class TYPEField(CALField):
  def check(self, value):
    return value in (1, 2, 3, '1d', '2d', '2dms_array', '2dmsaa', '3d', 'cubemap', 'cubemaparray', 'unkown', 'buffer')

  def render(self, value):
    if type(value)==str:
      return value 
    else:
      return str(value) + 'd'
    # this one is different beccause of UNNORM option - '_type(' and ')' is handled by inst
    # Furthermore, some instructions handle this differently, such as dclpt which does not
    # have the unnorm flag

class RESOURCEField(CALField):
  def check(self, value):
    return isinstance(value, (int, long)) and value >= 0 and value < 256
  
  def render(self, value):
    return '_resource(' + str(value) + ')'

class RESOURCEIDField(CALField):
  def check(self, value):
    return isinstance(value, (int, long)) and value >= 0 and value < 256
  
  def render(self, value):
    return '_id(' + str(value) + ')'

class SAMPLERField(CALField):
  def check(self, value):
    return isinstance(value, (int, long)) and value >= 0 and value < 16
  
  def render(self, value):
    return '_sampler(' + str(value) + ')'

#class IEEEField(CALFlagField):
#  def render(self, value):
#    if value == True:
#      return '_ieee'
#    else:
#      return ''

#class UINTField(CALFlagField):
#  def render(self, value):
#    if value == True:
#      return '_uint'
#    else:
#      return ''

#class REPEATField(CALFlagField):
#  def render(self, value):
#    if value == True:
#      return ' repeat'
#    else:
#      return ''

#class NEIGHBOREXCHField(CALFlagField):
#  def render(self, value):
#    if value == True:
#      return '_neighborExch'
#    else:
#      return ''

#class UNNORMField(CALFlagField):
#  def render(self, value):
#    if value == True:
#      return ',unnorm'
#    else:
#      return ''

#class THREADSField(CALFlagField):
#  def render(self, value):
#    if value == True:
#      return '_threads'
#    else:
#      return ''

#class LDSField(CALFlagField):
#  def render(self, value):
#    if value == True:
#      return '_lds'
#    else:
#      return ''

#class MEMORYField(CALFlagField):
#  def render(self, value):
#    if value == True:
#      return '_memory'
#    else:
#      return ''

#class SRField(CALFlagField):
#  def render(self, value):
#    if value == True:
#      return '_sr'
#    else:
#      return ''

#class CENTERField(CALFlagField):
#  def render(self, value):
#    if value == True:
#      return '_center'
#    else:
#      return ''

#class BIASField(CALFlagField):
#  def render(self, value):
#    if value == True:
#      return '_bias'
#    else:
#      return ''

#class INVERTField(CALFlagField):
#  def render(self, value):
#    if value == True:
#      return '_invert'
#    else:
#      return ''

#class CENTEREDField(CALFlagField):
#  def render(self, value):
#    if value == True:
#      return '_centered'
#    else:
#      return ''

class STAGEField(CALField):
  def check(self, value):
    if value == None:
      return True
    if isinstance(value, (int, long)) and value >= 0 and value < 256:
      return True
    return False
  
  def render(self, value):
    if value == None:
      return ''
    else:
      return '_stage(' + value + ')'

class LOFFSETField(CALField):
  def check(self, value):
    if value == None:
      return True
    if value in (0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60):
      return True
    return False
  
  def render(self, value):
    if value == None:
      return ''
    else:
      return '_lOffset(' + value + ')'

#class SAMPLEField(CALFlagField):
#  def render(self, value):
#    if value == True:
#      return '_sample'
#    else:
#      return ''

class STREAMField(CALField):
  def check(self, value):
    if isinstance(value, (int, long)) and value >= 0 and value < 16:
      return True
    return False
  
  def render(self, value):
    return '_exportStream(' + str(value) + ')'

class OFFSETField(CALField):
  def check(self, value):
    if isinstance(value, (int, long)) and value >= 0 and value < 512:
      return True
    return False
  
  def render(self, value):
    return '_elemOffset(' + str(value) + ')'

class ELEMField(CALField):
  def check(self, value):
    if isinstance(value, (int, long)) and value >= 0 and value < 16:
      return True
    return False
  
  def render(self, value):
    return '_elem(' + str(value) + ')'

class VELEMField(CALField):
  def check(self, value):
    if isinstance(value, (int, long)) and value >= 0 and value < 64:
      return True
    return False
  
  def render(self, value):
    return '_elem(' + str(value) + ')'

class AOFFIMMIField(CALField):
  def check(self, value):
    if value == None or value == ():
      return True

    if len(value) != 3:
      return False

    p = 2 ** 16
    return False not in [isinstance(v, (int, long)) and v >= 0 and v < p for v in value]

    #try:
    #  valid = True
    #  if len(value) == 3:
    #    if not isinstance(value[0], (int, long)) or value[0] < 0 or value[0] > pow(2, 16):
    #      valid = False
    #    if not isinstance(value[1], (int, long)) or value[1] < 0 or value[1] > pow(2, 16):
    #      valid = False
    #    if not isinstance(value[2], (int, long)) or value[2] < 0 or value[2] > pow(2, 16):
    #      valid = False
    #    return valid
    #  return False
    #except:
    #  return False

  def render(self, value):
    if value == None or value == ():
      return ''
    else:
      return '_aoffimmi(' + str(value[0]) + ',' + str(value[1]) + ',' + str(value[2]) + ')'

class FMTField(CALField):
  def check(self, value):
    vals = ('float', 'mixed', 'sint', 'snorm', 'srgb', 'uint', 'unkown', 'unorm')

    if value in vals:
      return True
    if len(value) != 4:
      return False
    return False not in [v not in vals for v in value]
    #else:
    #  valid = True
    #  if len(value) == 4:
    #    if not value[0] in vals:
    #      valid = False
    #    if not value[1] in vals:
    #      valid = False
    #    if not value[2] in vals:
    #      valid = False
    #    if not value[3] in vals:
    #      valid = False
    #    return valid
    #  return False

  def render(self, value):
    if value == None:
      return ''
    else:
      if type(value) == str:
        return '_fmtx(' + value + ')_fmty(' + value + ')_fmtz(' + value + ')_fmtw(' + value + ')'
      else:
        return '_fmtx(' + value[0] + ')_fmty(' + value[1] + ')_fmtz(' + value[2] + ')_fmtw(' + value[3] + ')'

class XYZWDefaultField(CALField):
  def check(self, value):

    if len(value) != 4:
      return False

    return False not in [v not in (0.0, 1.0, None) for v in value]
    #valid = True
    #if not(value[0] == 0.0 or value[0] == 1.0 or value[0] == None):
    #  valid = False
    #if not(value[1] == 0.0 or value[1] == 1.0 or value[1] == None):
    #  valid = False
    #if not(value[2] == 0.0 or value[2] == 1.0 or value[2] == None):
    #  valid = False
    #if not(value[3] == 0.0 or value[3] == 1.0 or value[3] == None):
    #  valid = False
    #return valid

  def render(self, value):
    strvalue = ['' for i in range(4)]
    for i, val in enumerate(value):
      if val == None:
        strvalue[i] = '*'
      else:
        strvalue[i] = str(val)
    return '_x(' + strvalue[0] + ')_y(' + strvalue[1] + ')_z(' + strvalue[2] + ')_w(' + strvalue[3] + ')'

class XYZWImportField(CALField):
  def check(self, value):
    valid = True

    if len(value) != 4:
      return False

    #return False not in [v not in ('0', '1', '*', '-', None) for v in value]
    if not(value[0] == '0' or value[0] == '1' or value[0] == '*' or value[0] == '-' or value[0] == None):
      valid = False
    if not(value[1] == '0' or value[1] == '1' or value[1] == '*' or value[1] == '-' or value[1] == None):
      valid = False
    if not(value[2] == '0' or value[2] == '1' or value[2] == '*' or value[2] == '-' or value[2] == None):
      valid = False
    if not(value[3] == '0' or value[3] == '1' or value[3] == '*' or value[3] == '-' or value[3] == None):
      valid = False
    return valid


  def render(self, value):
    strvalue = ['' for i in range(4)]
    for i, val in enumerate(value):
      if val == None:
        strvalue[i] = '*'
      else:
        strvalue[i] = str(val)
    return '_x(' + strvalue[0] + ')_y(' + strvalue[1] + ')_z(' + strvalue[2] + ')_w(' + strvalue[3] + ')'

class TOPOLOGYField(CALField):
  def check(self, value):
    return value in ('line', 'line_adj', 'point', 'triangle', 'triangle_adj')

  def render(self, value):
    return str(value)

class OUTPUTTOPOLOGYField(CALField):
  def check(self, value):
    return value in ('linestrip', 'pointlist', 'trianglestrip')

  def render(self, value):
    return str(value)

class LiteralField(CALField):
  def check(self, value):
    return isinstance(value, (int, long, float))
  
  def render(self, value):
    return str(value)

class IntegerLiteralField(CALField):
  def check(self, value):
    return isinstance(value, (int, long))
  
  def render(self, value):
    return str(value)

class IntegerLabelField(IntegerLiteralField):
  def check(self, value):
    return isinstance(value, (int, long)) and value >= 0

  def render(self, value):
    return str(value)

class SHIFTField(CALField):
  def check(self, value):
    return value in ('', 'x2', 'x4', 'x8', 'd2', 'd4', 'd8')

  def render(self, value):
    if value != '':
      return '_' + value
    else:
      return ''

#class SATField(CALFlagField):
#  def render(self, value):
#    if value == True:
#      return '_sat'
#    else:
#      return ''

class MATRIXField(CALField):
  def check(self, value):
    return value in ('3x2', '3x3', '3x4', '4x3', '4x4')

  def render(self, value):
    return '_matrix(' + value + ')'

class USAGEINDEXField(CALField):
  def check(self, value):
    return isinstance(value, (int, long)) and value >= 0 and value < 256
  
  def render(self, value):
    return '_usageIndex(' + str(value) + ')'

class PARAMField(CALField):
  def check(self, value):
    return isinstance(value, (int, long)) and value >= 0 and value < 256
  
  def render(self, value):
    return '_param(' + str(value) + ')'

class COORDMODEField(CALField):
  def check(self, value):
    return value in ('normalized', 'unkown', 'unnormalized')

  def render(self, value):
    return '_coordmode(' + value + ')'

class BOOLField(CALField):
  def check(self, value):
    return isinstance(value, (int, long)) and value >= 0 and value < 2
  
  def render(self, value):
    return str(value)

class NOISETYPEField(CALField):
  def check(self, value):
    return value in ('perlin1D', 'perlin2D', 'perlin3D', 'perlin4D')

  def render(self, value):
    return '_type(' + value + ')'

class CMPVALField(CALField):
  def check(self, value):
    return value in (0.0, 0.5, 1.0, -0.5, -1.0)

  def render(self, value):
    return '_cmpval(' + value + ')'
  
OPCD0 = CALField("OPCD0")
TRGT = RegisterField("TRGT")
SRC0 = RegisterField("SRC0")
SRC1 = RegisterField("SRC1")
SRC2 = RegisterField("SRC2")
SRC3 = RegisterField("SRC3")
RELOP = RELOPField("RELOP")
ZEROOP = ZEROOPField("ZEROOP", 'inf_else_max')
LOGICOP = LOGICOPField("LOGICOP")
INTERP = INTERPField("INTERP", 'notused')
RESOURCE = RESOURCEField("RESOURCE")
RESOURCEID = RESOURCEIDField("RESOURCEID") # this would be binary identical to RESOURCE, but the text name is different for some instructions
SAMPLER = SAMPLERField("SAMPLER")
LBL = IntegerLabelField("LBL", 0)
IEEE = CALFlagField("IEEE", "_ieee", False)
UINT = CALFlagField("UINT", "_uint", False)
REPEAT = CALFlagField("REPEAT", " repeat", False)
STAGE = STAGEField("STAGE", 0)
SAMPLE = CALFlagField("SAMPLE", "_sample", False)
AOFFIMMI = AOFFIMMIField("AOFFIMMI", ())
USAGE = USAGEField("USAGE", 'interp')
TYPE = TYPEField("TYPE")
UNNORM = CALFlagField("UNNORM", ",unnorm", False)
FMT = FMTField("FMT")
STREAM = STREAMField("STREAM")
OFFSET = OFFSETField("OFFSET")
ELEM = ELEMField("ELEM")
VELEM = VELEMField("VELEM")
NEIGHBOREXCH = CALFlagField("NEIGHBOREXCH", "_neighborExch", False)
SHARINGMODE = SHARINGMODEField("SHARINGMODE", False)
LOFFSET = LOFFSETField("LOFFSET", 0)
THREADS = CALFlagField("THREADS", "_threads", False)
LDS = CALFlagField("LDS", "_lds", False)
MEMORY = CALFlagField("MEMORY", "_memory", False)
SR = CALFlagField("SR", "_sr", False)
XYZWDefault = XYZWDefaultField("XYZWDefault")
XYZWImport = XYZWImportField("XYZWImport")
TOPOLOGY = TOPOLOGYField("TOPOLOGY")
OUTPUTTOPOLOGY = OUTPUTTOPOLOGYField("OUTPUTTOPOLOGY")
L0 = LiteralField("L0")
L1 = LiteralField("L1")
L2 = LiteralField("L2")
L3 = LiteralField("L3")
IL0 = IntegerLiteralField("IL0")
SHIFT = SHIFTField("SHIFT", '')
SAT = CALFlagField("SAT", "_sat", False)
MATRIX = MATRIXField("MATRIX")
CENTER = CALFlagField("CENTER", "_center", True)
BIAS = CALFlagField("BIAS", "_bias", False)
INVERT = CALFlagField("INVERT", "_invert", False)
CENTERED = CALFlagField("CENTERED", "_centered", False)
USAGEINDEX = USAGEINDEXField("USAGEINDEX")
CENTROID = CALFlagField("CENTROID", "_centroid", False)
ORIGIN = CALFlagField("ORIGIN", "_origin", False) # used in dclpin - maybe...
PARAM = PARAMField("PARAM")
BOOL = BOOLField("BOOL")
ROUND = CALFlagField("ROUND", "_round", False)
NOISETYPE = NOISETYPEField("NOISETYPE")
COORDMODE = NOISETYPEField("COORDMODE")
NORMALIZE = CALFlagField("NORMALIZE", "_normalize", False)
CMPVAL = CMPVALField("CMPVAL")
