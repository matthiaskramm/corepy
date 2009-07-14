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

# CAL Il instructions

from corepy.spre.spe import Instruction, DispatchInstruction
from cal_insts import *

__doc__="""
ISA for AMD's Compute Abstraction Layer Intermediate Language.
"""

class relop:
  eq = 'eq'
  ge = 'ge'
  gt = 'gt'
  le = 'le'
  lt = 'lt'
  ne = 'ne'

class zeroop:
  zero = 'zero'
  fltmax = 'fltmax'
  inf_else_max = 'inf_else_max'
  infinity = 'infinity'

class logicop:
  eq = 'eq'
  ne = 'ne'

class usage:
  backcolor = 'backcolor'
  color = 'color'
  fog = 'fog'
  generic = 'generic'
  pointsize = 'pointsize'
  pos = 'pos'
  wincoord = 'wincoord'

class pixtex_type:
  oned = '1d'
  twod = '2d'
  twodms_array = '2dms_array'
  twodmsaa = '2dmsaa'
  threed = '3d'
  cubemap = 'cubemap'
  cubemaparray = 'cubemaparray'
  unkown = 'unkown'
  buffer = 'buffer'

class fmt:
  float = 'float'
  mixed = 'mixed'
  sint = 'sint'
  snorm = 'snorm'
  srgb = 'srgb'
  uint = 'uint'
  unkown = 'unkown'
  unorm = 'unorm'

class sharingMode:
  abs = 'abs'
  rel = 'rel'

class interp:
  constant = 'constant'
  linear = 'linear'
  centroid = 'centroid'
  linear_noperspective = 'linear_noperspective'
  noper_centroid = 'noper_centroid'
  noper_sample = 'noper_sample'
  sample = 'sample'
  notused = 'notused'

class topology:
  line = 'line'
  line_adj = 'line_adj'
  point = 'point'
  triangle = 'triangle'
  triangle_adj = 'triangle_adj'

class output_topology:
  linestrip = 'linestrip'
  pointlist = 'pointlist'
  trianglestrip = 'trianglestrip'

class coordmode:
  normalized = 'normalized'
  unkown = 'unkown'
  unnormalized = 'unnormalized'

class noise_type:
  perlin1D = 'perlin1D'
  perlin2D = 'perlin2D'
  perlin3D = 'perlin3D'
  perlin4D = 'perlin4D'

IL_OP_BREAK = 'break'
IL_OP_BREAKC = 'breakc'
IL_OP_BREAK_LOGICALZ = 'break_logicalz'
IL_OP_BREAK_LOGICALNZ = 'break_logicalnz'
IL_OP_CALL = 'call'
IL_OP_CALLNZ = 'callnz'
IL_OP_CALL_LOGICALZ = 'call_logicalz'
IL_OP_CALL_LOGICALNZ = 'call_logicalnz'
IL_OP_CASE = 'case'
IL_OP_CONTINUE = 'continue'
IL_OP_CONTINUEC = 'continuec'
IL_OP_CONTINUE_LOGICALZ = 'continue_logicalz'
IL_OP_CONTINUE_LOGICALNZ = 'continue_logicalnz'
IL_OP_DEFAULT = 'default'
IL_OP_ELSE = 'else'
IL_OP_ENDSWITCH = 'endswitch'
IL_OP_ENDMAIN = 'endmain'
IL_OP_END = 'end'
IL_OP_ENDFUNC = 'endfunc'
IL_OP_ENDIF = 'endif'
IL_OP_ENDLOOP = 'endloop'
IL_OP_FUNC = 'func'
IL_OP_IFC = 'ifc'
IL_OP_IFNZ = 'ifnz'
IL_OP_IF_LOGICALNZ = 'if_logicalnz'
IL_OP_IF_LOGICALZ = 'if_logicalz'
IL_OP_LOOP = 'loop'
IL_OP_WHILE = 'whileloop'
IL_OP_SWITCH = 'switch'
IL_OP_RET = 'ret'
IL_OP_RET_DYN = 'ret_dyn'
IL_OP_RET_LOGICALNZ = 'ret_logicalnz'
IL_OP_RET_LOGICALZ = 'ret_logicalz'
IL_OP_DCLARRAY = 'dclarray'
IL_DCL_CONST_BUFFER = 'dcl_cb'
IL_OP_DCLDEF = 'dcldef'
IL_OP_DEF = 'def'
IL_OP_DEFB = 'defb'
IL_DCL_INDEXED_TEMP_ARRAY = 'dcl_indexed_temp_array'
IL_DCL_INPUT = 'dcl_input'
IL_DCL_INPUTPRIMITIVE = 'dcl_input_primitive'
IL_DCL_LITERAL = 'dcl_literal'
IL_DCL_MAX_OUTPUT_VERTEX_COUNT = 'dcl_max_output_vertex_count'
IL_DCL_ODEPTH = 'dcl_odepth'
IL_DCL_OUTPUT_TOPOLOGY = 'dcl_output_topology'
IL_DCL_OUTPUT = 'dcl_output'
IL_OP_DCL_VPRIM = 'dcl_vprim'
IL_OP_DCL_SHARED_TEMP = 'dcl_shared_temp'
IL_OP_DCL_LDS_SIZE_PER_THREAD = 'dcl_lds_size_per_thread'
IL_OP_DCL_LDS_SHARING_MODE = 'dcl_lds_sharing_mode'
IL_OP_DCL_NUM_THREAD_PER_GROUP = 'dcl_num_thread_per_group'
IL_OP_DCLPI = 'dclpi'
IL_OP_DCLPIN = 'dclpin'
IL_OP_DCLPP = 'dclpp'
IL_OP_DCLPT = 'dclpt'
IL_OP_DCL_RESOURCE = 'dcl_resource'
IL_OP_DCLV = 'dclv'
IL_OP_DCLVOUT = 'dclvout'
IL_OP_CUT = 'cut'
IL_OP_KILL = 'kill'
IL_OP_DISCARD_LOGICALNZ = 'discard_logicalnz'
IL_OP_DISCARD_LOGICALZ = 'discard_logicalz'
IL_OP_EMIT = 'emit'
IL_OP_EMIT_THEN_CUT = 'emitcut'
IL_OP_LOAD = 'load'
IL_OP_LOD = 'lod'
IL_OP_MEMEXPORT = 'memexport'
IL_OP_MEMIMPORT = 'memimport'
IL_OP_RESINFO = 'resinfo'
IL_OP_SAMPLEINFO = 'sampleinfo'
IL_OP_SAMPLEPOS = 'samplepos'
IL_OP_SAMPLE = 'sample'
IL_OP_SAMPLE_B = 'sample_b'
IL_OP_SAMPLE_G = 'sample_g'
IL_OP_SAMPLE_L = 'sample_l'
IL_OP_SAMPLE_C_LZ= 'sample_c_lz'
IL_OP_SAMPLE_C = 'sample_c'
IL_OP_SAMPLE_C_G = 'sample_c_g'
IL_OP_SAMPLE_C_L = 'sample_c_l'
IL_OP_TEXLD = 'texld'
IL_OP_TEXLDB = 'texldb'
IL_OP_TEXLDD = 'texldd'
IL_OP_TEXLDMS = 'texldms'
IL_OP_TEXWEIGHT = 'texlweight'
IL_OP_LDS_READ_VEC = 'lds_read_vec'
IL_OP_LDS_WRITE_VEC = 'lds_write_vec'
IL_OP_FENCE = 'fence'
IL_OP_IAND = 'iand'
IL_OP_I_NOT = 'inot'
IL_OP_I_OR = 'ior'
IL_OP_I_XOR = 'ixor'
IL_OP_I_ADD = 'iadd'
IL_OP_I_MAD = 'imad'
IL_OP_I_MAX = 'imax'
IL_OP_I_MIN = 'imin'
IL_OP_I_MUL = 'imul'
IL_OP_I_MUL_HIGH = 'imul_high'
IL_OP_I_EQ = 'ieq'
IL_OP_I_GE = 'ige'
IL_OP_I_LT = 'ilt'
IL_OP_I_NE = 'ine'
IL_OP_I_NEGATE = 'inegate'
IL_OP_I_SHL = 'ishl'
IL_OP_I_SHR = 'ishr'
IL_OP_U_SHR = 'ushr'
IL_OP_U_DIV = 'udiv'
IL_OP_U_MOD = 'umod'
IL_OP_U_MAD = 'umad'
IL_OP_U_MAX = 'umax'
IL_OP_U_MIN = 'umin'
IL_OP_U_GE = 'uge'
IL_OP_U_LT = 'ult'
IL_OP_U_MUL = 'umul'
IL_OP_U_MUL_HIGH = 'umul_high'
IL_OP_FTOI = 'ftoi'
IL_OP_FTOU = 'ftou'
IL_OP_ITOF = 'itof'
IL_OP_UTOF = 'utof'
IL_OP_D_2_F = 'd2f'
IL_OP_F_2_D = 'f2d'
IL_OP_ABS = 'abs'
IL_OP_ADD = 'add'
IL_OP_ACOS = 'acos'
IL_OP_AND = 'and'
IL_OP_ASIN = 'asin'
IL_OP_ATAN = 'atan'
IL_OP_CLAMP = 'clamp'
IL_OP_CLG = 'clg'
IL_OP_CMOV = 'cmov'
IL_OP_CMOV_LOGICAL = 'cmov_logical'
IL_OP_CMP = 'cmp'
IL_OP_COLORCLAMP = 'colorclamp'
IL_OP_COS = 'cos'
IL_OP_CRS = 'crs'
IL_OP_DIST = 'dist'
IL_OP_DIV = 'div'
IL_OP_DP2ADD = 'dp2add'
IL_OP_DP2 = 'dp2'
IL_OP_DP3 = 'dp3'
IL_OP_DP4 = 'dp4'
IL_OP_DST = 'dst'
IL_OP_DSX = 'dsx'
IL_OP_DSY = 'dsy'
IL_OP_DXSINCOS = 'dxsincos'
IL_OP_EQ = 'eq'
IL_OP_EXN = 'exn'
IL_OP_EXP = 'exp'
IL_OP_EXP_VEC = 'exp_vec'
IL_OP_EXPP = 'expp'
IL_OP_FACEFORWARD = 'faceforward'
IL_OP_FLR = 'flr'
IL_OP_FRC = 'frc'
IL_OP_FWIDTH = 'fwidth'
IL_OP_GE = 'ge'
IL_OP_LEN = 'len'
IL_OP_LIT = 'lit'
IL_OP_LN = 'ln'
IL_OP_LOG = 'log'
IL_OP_LOG_VEC = 'log_vec'
IL_OP_LOGP = 'logp'
IL_OP_LRP = 'lrp'
IL_OP_LT = 'lt'
IL_OP_MAD = 'mad'
IL_OP_MAX = 'max'
IL_OP_MIN = 'min'
IL_OP_MMUL = 'mmul'
IL_OP_MOD = 'mod'
IL_OP_INVARIANT_MOV = 'invariant_move'
IL_OP_MOV = 'mov'
IL_OP_MOVA = 'mova'
IL_OP_MUL = 'mul'
IL_OP_NE = 'ne'
IL_OP_NOISE = 'noise'
IL_OP_NRM = 'nrm'
IL_OP_PIREDUCE = 'pireduce'
IL_OP_POW = 'pow'
IL_OP_RCP = 'rcp'
IL_OP_REFLECT = 'reflect'
IL_OP_RND = 'rnd'
IL_OP_ROUND_NEAR = 'round_nearest'
IL_OP_ROUND_NEG_INF = 'round_neginf'
IL_OP_ROUND_POS_INF = 'round_posinf'
IL_OP_ROUND_ZERO = 'round_z'
IL_OP_RSQ_VEC = 'rsq_vec'
IL_OP_RSQ = 'rsq'
IL_OP_SET = 'set'
IL_OP_SGN = 'sgn'
IL_OP_SIN = 'sin'
IL_OP_SINCOS = 'sincos'
IL_OP_SIN_VEC = 'sin_vec'
IL_OP_COS_VEC = 'cos_vec'
IL_OP_SQRT = 'sqrt'
IL_OP_SQRT_VEC = 'sqrt_vec'
IL_OP_SUB = 'sub'
IL_OP_TAN = 'tan'
IL_OP_TRANSPOSE = 'transpose'
IL_OP_TRC = 'trc'
IL_OP_D_NE = 'dne'
IL_OP_D_EQ = 'deq'
IL_OP_D_GE = 'dge'
IL_OP_D_LT = 'dlt'
IL_OP_D_FREXP = 'dfrexp'
IL_OP_D_ADD = 'dadd'
IL_OP_D_MUL = 'dmul'
IL_OP_D_DIV = 'ddiv'
IL_OP_D_LDEXP = 'dldexp'
IL_OP_D_FRAC = 'dfrac'
IL_OP_D_MULADD = 'dmad'


class break_(Instruction):
  name = 'break'
  machine_inst = OPCD_0_0
  params = {'OPCD':IL_OP_BREAK}

class breakc(Instruction):
  name = 'breakc'
  machine_inst = OPCD_RELOP_2_0
  params = {'OPCD':IL_OP_BREAKC}

class break_logicalz(Instruction):
  name = 'break_logicalz'
  machine_inst = OPCD_1_0
  params = {'OPCD':IL_OP_BREAK_LOGICALZ}

class break_logicalnz(Instruction):
  name = 'break_logicalnz'
  machine_inst = OPCD_1_0
  params = {'OPCD':IL_OP_BREAK_LOGICALNZ}

class call(Instruction):
  name = 'call'
  machine_inst = OPCD_0_0_LBL
  params = {'OPCD':IL_OP_CALL}

class callnz(Instruction):
  name = 'callnz'
  machine_inst = OPCD_1_0_LBL
  params = {'OPCD':IL_OP_CALLNZ}

class call_logicalz(Instruction):
  name = 'call_logicalz'
  machine_inst = OPCD_1_0_LBL
  params = {'OPCD':IL_OP_CALL_LOGICALZ}

class call_logicalnz(Instruction):
  name = 'call_logicalnz'
  machine_inst = OPCD_1_0_LBL
  params = {'OPCD':IL_OP_CALL_LOGICALNZ}

class case(Instruction):
  name = 'case'
  machine_inst = OPCD_1_0
  params = {'OPCD':IL_OP_CASE}

class continue_(Instruction):
  name = 'continue'
  machine_inst = OPCD_0_0
  params = {'OPCD':IL_OP_CONTINUE}

class continuec(Instruction):
  name = 'continuec'
  machine_inst = OPCD_RELOP_2_0
  params = {'OPCD':IL_OP_CONTINUEC}

class continue_logicalz(Instruction):
  name = 'continue_logicalz'
  machine_inst = OPCD_LOGICOP_1_0
  params = {'OPCD':IL_OP_CONTINUE_LOGICALZ}

class continue_logicalnz(Instruction):
  name = 'continue_logicalnz'
  machine_inst = OPCD_LOGICOP_1_0
  params = {'OPCD':IL_OP_CONTINUE_LOGICALNZ}

class default(Instruction):
  name = 'default'
  machine_inst = OPCD_0_0
  params = {'OPCD':IL_OP_DEFAULT}

class else_(Instruction):
  name = 'else'
  machine_inst = OPCD_0_0
  params = {'OPCD':IL_OP_ELSE}

class endswitch(Instruction):
  name = 'endswitch'
  machine_inst = OPCD_0_0
  params = {'OPCD':IL_OP_ENDSWITCH}

class endmain(Instruction):
  name = 'endmain'
  machine_inst = OPCD_0_0
  params = {'OPCD':IL_OP_ENDMAIN}

class end(Instruction):
  name = 'end'
  machine_inst = OPCD_0_0
  params = {'OPCD':IL_OP_END}

class endfunc(Instruction):
  name = 'endfunc'
  machine_inst = OPCD_0_0
  params = {'OPCD':IL_OP_ENDFUNC}

class endif(Instruction):
  name = 'endif'
  machine_inst = OPCD_0_0
  params = {'OPCD':IL_OP_ENDIF}

class endloop(Instruction):
  name = 'endloop'
  machine_inst = OPCD_0_0
  params = {'OPCD':IL_OP_ENDLOOP}

class func(Instruction):
  name = 'func'
  machine_inst = OPCD_0_0_LBL
  params = {'OPCD':IL_OP_FUNC}

class ifc(Instruction):
  name = 'ifc'
  machine_inst = OPCD_RELOP_2_0
  params = {'OPCD':IL_OP_IFC}

class ifnz(Instruction):
  name = 'ifnz'
  machine_inst = OPCD_1_0
  params = {'OPCD':IL_OP_IFNZ}

class if_logicalnz(Instruction):
  name = 'iflogicalnz'
  machine_inst = OPCD_1_0
  params = {'OPCD':IL_OP_IF_LOGICALNZ}

class if_logicalz(Instruction):
  name = 'iflogicalz'
  machine_inst = OPCD_1_0
  params = {'OPCD':IL_OP_IF_LOGICALZ}

class loop(Instruction):
  name = 'loop'
  machine_inst = OPCD_1_0_REPEAT
  params = {'OPCD':IL_OP_LOOP}

class whileloop(Instruction):
  name = 'whileloop'
  machine_inst = OPCD_0_0
  params = {'OPCD':IL_OP_WHILE}

class switch(Instruction):
  name = 'switch'
  machine_inst = OPCD_1_0
  params = {'OPCD':IL_OP_SWITCH}

class ret(Instruction):
  name = 'ret'
  machine_inst = OPCD_0_0
  params = {'OPCD':IL_OP_RET}

class ret_dyn(Instruction):
  name = 'ret_dyn'
  machine_inst = OPCD_0_0
  params = {'OPCD':IL_OP_RET_DYN}

class ret_logicalnz(Instruction):
  name = 'ret_logicalnz'
  machine_inst = OPCD_1_0
  params = {'OPCD':IL_OP_RET_LOGICALNZ}

class ret_logicalz(Instruction):
  name = 'ret_logicalz'
  machine_inst = OPCD_1_0
  params = {'OPCD':IL_OP_RET_LOGICALZ}

# 6.4

class dclarray(Instruction):
  name = 'dclarray'
  machine_inst = OPCD_2_0
  params = {'OPCD':IL_OP_DCLARRAY}

class dcl_cb(Instruction):
  name = 'dcl_cb'
  machine_inst = OPCD_1_0
  params = {'OPCD':IL_DCL_CONST_BUFFER}

class dcldef(Instruction):
  name = 'dcldef'
  machine_inst = OPCD_XYZWDefault_0_1
  params = {'OPCD':IL_OP_DCLDEF}

class def_(Instruction):
  name = 'def'
  machine_inst = OPCD_0_1_L4
  params = {'OPCD':IL_OP_DEF}

class defb(Instruction):
  name = 'defb'
  machine_inst = OPCD_0_1_BOOL
  params = {'OPCD':IL_OP_DEFB}

class dcl_indexed_temp_array(Instruction):
  name = 'dcl_indexed_temp_array'
  machine_inst = OPCD_1_0
  params = {'OPCD':IL_DCL_INDEXED_TEMP_ARRAY}

class dcl_input(Instruction):
  name = 'dcl_input'
  machine_inst = OPCD_USAGE_INTERP_0_1
  params = {'OPCD':IL_DCL_INPUT}

class dcl_input_primitive(Instruction):
  name = 'dcl_input_primitive'
  machine_inst = OPCD_0_0_TOPOLOGY
  params = {'OPCD':IL_DCL_INPUTPRIMITIVE}

class dcl_literal(Instruction):
  name = 'dcl_literal'
  machine_inst = OPCD_1_0_L4
  params = {'OPCD':IL_DCL_LITERAL}

class dcl_max_output_vertex_count(Instruction):
  name = 'dcl_max_output_vertex_count'
  machine_inst = OPCD_0_0_IL
  params = {'OPCD':IL_DCL_LITERAL}

class dcl_odepth(Instruction):
  name = 'dcl_odepth'
  machine_inst = OPCD_0_0
  params = {'OPCD':IL_DCL_ODEPTH}

class dcl_output_topology(Instruction):
  name = 'dcl_output_topology'
  machine_inst = OPCD_0_0_OUTPUTTOPOLOGY
  params = {'OPCD':IL_DCL_OUTPUT_TOPOLOGY}

class dcl_output(Instruction):
  name = 'dcl_output'
  machine_inst = OPCD_USAGE_0_1
  params = {'OPCD':IL_DCL_OUTPUT}

class dcl_vprim(Instruction):
  name = 'dcl_vprim'
  machine_inst = OPCD_0_0
  params = {'OPCD':IL_OP_DCL_VPRIM}

class dcl_shared_temp(Instruction):
  name = 'dcl_shared_temp'
  machine_inst = OPCD_1_0
  params = {'OPCD':IL_OP_DCL_SHARED_TEMP}

# TODO: dcl_lds_size_per_thread
# TODO: dcl_lds_sharing_mode
# TODO: dcl_num_thread_per_group

class dclpi(Instruction):
  name = 'dclpi'
  machine_inst = OPCD_XYZWImport_CENTER_BIAS_INVERT_CENTERED_0_1
  params = {'OPCD':IL_OP_DCLPI}

class dclpin(Instruction):
  name = 'dclpin'
  machine_inst = OPCD_USAGE_XYZWImport_CENTROID_0_1
  params = {'OPCD':IL_OP_DCLPIN}

class dclpp(Instruction):
  name = 'dclpp'
  machine_inst = OPCD_PARAM_0_1
  params = {'OPCD':IL_OP_DCLPP}

class dclpt(Instruction):
  name = 'dclpt'
  machine_inst = OPCD_STAGE_TYPE_COORDMODE_0_0
  params = {'OPCD':IL_OP_DCLPT}

class dcl_resource(Instruction):
  name = 'dcl_resource'
  machine_inst = OPCD_RESOURCE_TYPE_UNNORM_FMT_0_0
  params = {'OPCD':IL_OP_DCL_RESOURCE}

class dclv(Instruction):
  name = 'dclv'
  machine_inst = OPCD_VELEM_XYZWImport_0_1
  params = {'OPCD':IL_OP_DCLV}

class dclvout(Instruction):
  name = 'dclvout'
  machine_inst = OPCD_USAGE_USAGEINDEX_XYZWImport_0_1
  params = {'OPCD':IL_OP_DCLVOUT}

# 6.5
class cut(Instruction):
  name = 'cut'
  machine_inst = OPCD_0_0
  params = {'OPCD':IL_OP_CUT}

class kill(Instruction):
  name = 'kill'
  machine_inst = OPCD_STAGE_SAMPLE_1_0
  params = {'OPCD':IL_OP_CUT}

class discard_logicalnz(Instruction):
  name = 'discard_logicalnz'
  machine_inst = OPCD_1_0
  params = {'OPCD':IL_OP_DISCARD_LOGICALNZ}

class discard_logicalz(Instruction):
  name = 'discard_logicalz'
  machine_inst = OPCD_1_0
  params = {'OPCD':IL_OP_DISCARD_LOGICALZ}

class emit(Instruction):
  name = 'emit'
  machine_inst = OPCD_0_0
  params = {'OPCD':IL_OP_EMIT}

class emitcut(Instruction):
  name = 'emitcut'
  machine_inst = OPCD_0_0
  params = {'OPCD':IL_OP_EMIT_THEN_CUT}

class load(Instruction):
  name = 'load'
  machine_inst = OPCD_RESOURCE_AOFFIMMI_1_1
  params = {'OPCD':IL_OP_LOAD}

class lod(Instruction):
  name = 'lod'
  machine_inst = OPCD_STAGE_1_1
  params = {'OPCD':IL_OP_LOD}

class memexport(Instruction):
  name = 'memexport'
  machine_inst = OPCD_STREAM_OFFSET_2_0
  params = {'OPCD':IL_OP_MEMEXPORT}

class memimport(Instruction):
  name = 'memimport'
  machine_inst = OPCD_ELEM_1_1
  params = {'OPCD':IL_OP_MEMIMPORT}

class resinfo(Instruction):
  name = 'resinfo'
  machine_inst = OPCD_RESOURCE_UINT_1_1
  params = {'OPCD':IL_OP_RESINFO}

class sampleinfo(Instruction):
  name = 'sampleinfo'
  machine_inst = OPCD_RESOURCE_UINT_1_1
  params = {'OPCD':IL_OP_SAMPLEINFO}

class samplepos(Instruction):
  name = 'samplepos'
  machine_inst = OPCD_RESOURCE_UINT_1_1
  params = {'OPCD':IL_OP_SAMPLEPOS}

# TODO: Figure out how to handle 3 operand case for sample
class sample(Instruction):
  name = 'sample'
  machine_inst = OPCD_RESOURCE_SAMPLER_AOFFIMMI_1_1
  params = {'OPCD':IL_OP_SAMPLE}

# TODO: Figure out how to handle 4 operand case for sample_b
class sample_b(Instruction):
  name = 'sample_b'
  machine_inst = OPCD_RESOURCE_SAMPLER_AOFFIMMI_2_1
  params = {'OPCD':IL_OP_SAMPLE_B}

# is the aoffimmi legitimate? BDM
class sample_g(Instruction):
  name = 'sample_g'
  machine_inst = OPCD_RESOURCE_SAMPLER_AOFFIMMI_3_1
  params = {'OPCD':IL_OP_SAMPLE_G}

class sample_l(Instruction):
  name = 'sample_l'
  machine_inst = OPCD_RESOURCE_SAMPLER_2_1
  params = {'OPCD':IL_OP_SAMPLE_L}

# TODO: Figure out how to handle 4 operand case for sample_c_lz
# is the aoffimmi legitimate? BDM
class sample_c_lz(Instruction):
  name = 'sample_c_lz'
  machine_inst = OPCD_RESOURCE_SAMPLER_AOFFIMMI_2_1
  params = {'OPCD':IL_OP_SAMPLE_C_LZ}

# TODO: Figure out how to handle 4 operand case for sample_c
# is the aoffimmi legitimate? BDM
class sample_c(Instruction):
  name = 'sample_c'
  machine_inst = OPCD_RESOURCE_SAMPLER_AOFFIMMI_2_1
  params = {'OPCD':IL_OP_SAMPLE_C}

# is the aoffimmi legitimate? BDM
class sample_c_g(Instruction):
  name = 'sample_c_g'
  machine_inst = OPCD_RESOURCE_SAMPLER_AOFFIMMI_4_1
  params = {'OPCD':IL_OP_SAMPLE_C_G}

# is the aoffimmi legitimate? BDM
class sample_c_l(Instruction):
  name = 'sample_c_l'
  machine_inst = OPCD_RESOURCE_SAMPLER_AOFFIMMI_3_1
  params = {'OPCD':IL_OP_SAMPLE_C_L}

# TODO: TEXLD
# TODO: TEXLDB
# TODO: TEXLDD
# TODO: TEXLDMS

class texweight(Instruction):
  name = 'texweight'
  machine_inst = OPCD_STAGE_1_1
  params = {'OPCD':IL_OP_TEXWEIGHT}

class lds_read_vec(Instruction):
  name = 'lds_read_vec'
  machine_inst = OPCD_NEIGHBOREXCH_SHARINGMODE_1_1
  params = {'OPCD':IL_OP_LDS_READ_VEC}

class lds_write_vec(Instruction):
  name = 'lds_write_vec'
  machine_inst = OPCD_LOFFSET_SHARINGMODE_1_1
  params = {'OPCD':IL_OP_LDS_WRITE_VEC}

class fence(Instruction):
  name = 'fence'
  machine_inst = OPCD_THREADS_LDS_MEMORY_SR_0_0
  params = {'OPCD':IL_OP_FENCE}

# 6.6
class iand(Instruction):
  name = 'iand'
  machine_inst = OPCD_2_1
  params = {'OPCD':IL_OP_IAND}

class inot(Instruction):
  name = 'inot'
  machine_inst = OPCD_1_1
  params = {'OPCD':IL_OP_I_NOT}

class ior(Instruction):
  name = 'ior'
  machine_inst = OPCD_2_1
  params = {'OPCD':IL_OP_I_OR}

class ixor(Instruction):
  name = 'ixor'
  machine_inst = OPCD_2_1
  params = {'OPCD':IL_OP_I_XOR}

class iadd(Instruction):
  name = 'iadd'
  machine_inst = OPCD_2_1
  params = {'OPCD':IL_OP_I_ADD}

class imad(Instruction):
  name = 'imad'
  machine_inst = OPCD_3_1
  params = {'OPCD':IL_OP_I_MAD}

class imax(Instruction):
  name = 'imax'
  machine_inst = OPCD_2_1
  params = {'OPCD':IL_OP_I_MAX}

class imin(Instruction):
  name = 'imin'
  machine_inst = OPCD_2_1
  params = {'OPCD':IL_OP_I_MIN}

class imul(Instruction):
  name = 'imul'
  machine_inst = OPCD_3_1
  params = {'OPCD':IL_OP_I_MUL}

class imul_high(Instruction):
  name = 'imul_high'
  machine_inst = OPCD_2_1
  params = {'OPCD':IL_OP_I_MUL_HIGH}

class ieq(Instruction):
  name = 'ieq'
  machine_inst = OPCD_2_1
  params = {'OPCD':IL_OP_I_EQ}

class ige(Instruction):
  name = 'ige'
  machine_inst = OPCD_2_1
  params = {'OPCD':IL_OP_I_GE}

class ilt(Instruction):
  name = 'ilt'
  machine_inst = OPCD_2_1
  params = {'OPCD':IL_OP_I_LT}

class ine(Instruction):
  name = 'ine'
  machine_inst = OPCD_2_1
  params = {'OPCD':IL_OP_I_NE}

class inegate(Instruction):
  name = 'inegate'
  machine_inst = OPCD_1_1
  params = {'OPCD':IL_OP_I_NEGATE}

class ishl(Instruction):
  name = 'ishl'
  machine_inst = OPCD_2_1
  params = {'OPCD':IL_OP_I_SHL}

class ishr(Instruction):
  name = 'ishr'
  machine_inst = OPCD_2_1
  params = {'OPCD':IL_OP_I_SHR}

# 6.7
class ushr(Instruction):
  name = 'ushr'
  machine_inst = OPCD_2_1
  params = {'OPCD':IL_OP_U_SHR}

class udiv(Instruction):
  name = 'udiv'
  machine_inst = OPCD_2_1
  params = {'OPCD':IL_OP_U_DIV}


class umod(Instruction):
  name = 'umod'
  machine_inst = OPCD_2_1
  params = {'OPCD':IL_OP_U_MOD}

class umad(Instruction):
  name = 'umad'
  machine_inst = OPCD_3_1
  params = {'OPCD':IL_OP_U_MAD}

class umax(Instruction):
  name = 'iuax'
  machine_inst = OPCD_2_1
  params = {'OPCD':IL_OP_U_MAX}

class umin(Instruction):
  name = 'umin'
  machine_inst = OPCD_2_1
  params = {'OPCD':IL_OP_U_MIN}

class uge(Instruction):
  name = 'uge'
  machine_inst = OPCD_2_1
  params = {'OPCD':IL_OP_U_GE}

class ult(Instruction):
  name = 'ult'
  machine_inst = OPCD_2_1
  params = {'OPCD':IL_OP_U_LT}

class umul(Instruction):
  name = 'umul'
  machine_inst = OPCD_2_1
  params = {'OPCD':IL_OP_U_MUL}

class umul_high(Instruction):
  name = 'umul_high'
  machine_inst = OPCD_2_1
  params = {'OPCD':IL_OP_U_MUL_HIGH}

# 6.8

class ftoi(Instruction):
  name = 'ftoi'
  machine_inst = OPCD_1_1
  params = {'OPCD':IL_OP_FTOI}

class ftou(Instruction):
  name = 'ftou'
  machine_inst = OPCD_1_1
  params = {'OPCD':IL_OP_FTOU}

class itof(Instruction):
  name = 'itof'
  machine_inst = OPCD_1_1
  params = {'OPCD':IL_OP_ITOF}

class utof(Instruction):
  name = 'utof'
  machine_inst = OPCD_1_1
  params = {'OPCD':IL_OP_UTOF}

class d2f(Instruction):
  name = 'd2f'
  machine_inst = OPCD_1_1
  params = {'OPCD':IL_OP_D_2_F}

class f2d(Instruction):
  name = 'f2d'
  machine_inst = OPCD_1_1
  params = {'OPCD':IL_OP_F_2_D}

# 6.9

class abs(Instruction):
  name = 'abs'
  machine_inst = OPCD_1_1
  params = {'OPCD':IL_OP_ABS}

class add(Instruction):
  name = 'add'
  machine_inst = OPCD_2_1
  params = {'OPCD':IL_OP_ADD}

class acos(Instruction):
  name = 'acos'
  machine_inst = OPCD_1_1
  params = {'OPCD':IL_OP_ACOS}

class and_(Instruction):
  name = 'and'
  machine_inst = OPCD_2_1
  params = {'OPCD':IL_OP_AND}

class asin(Instruction):
  name = 'asin'
  machine_inst = OPCD_1_1
  params = {'OPCD':IL_OP_ASIN}

class atan(Instruction):
  name = 'atan'
  machine_inst = OPCD_1_1
  params = {'OPCD':IL_OP_ATAN}

class clamp(Instruction):
  name = 'clamp'
  machine_inst = OPCD_3_1
  params = {'OPCD':IL_OP_CLAMP}

class clg(Instruction):
  name = 'clg'
  machine_inst = OPCD_1_1
  params = {'OPCD':IL_OP_CLG}

class cmov(Instruction):
  name = 'cmov'
  machine_inst = OPCD_2_1
  params = {'OPCD':IL_OP_CMOV}

class cmov_logical(Instruction):
  name = 'cmov_logical'
  machine_inst = OPCD_3_1
  params = {'OPCD':IL_OP_CMOV_LOGICAL}

class cmp(Instruction):
  name = 'cmp'
  machine_inst = OPCD_RELOP_CMPVAL_3_1
  params = {'OPCD':IL_OP_CMP}

class colorclamp(Instruction):
  name = 'colorclamp'
  machine_inst = OPCD_1_1
  params = {'OPCD':IL_OP_COLORCLAMP}

class cos(Instruction):
  name = 'cos'
  machine_inst = OPCD_1_1
  params = {'OPCD':IL_OP_COS}

class crs(Instruction):
  name = 'crs'
  machine_inst = OPCD_2_1
  params = {'OPCD':IL_OP_CRS}

class dist(Instruction):
  name = 'dist'
  machine_inst = OPCD_2_1
  params = {'OPCD':IL_OP_DIST}
 
class div(Instruction):
  name = 'div'
  machine_inst = OPCD_ZEROOP_2_1
  params = {'OPCD':IL_OP_DIV}

class dp2add(Instruction):
  name = 'dp2add'
  machine_inst = OPCD_IEEE_3_1
  params = {'OPCD':IL_OP_DP2ADD}

class dp2(Instruction):
  name = 'dp2'
  machine_inst = OPCD_IEEE_2_1
  params = {'OPCD':IL_OP_DP2}

class dp3(Instruction):
  name = 'dp3'
  machine_inst = OPCD_IEEE_2_1
  params = {'OPCD':IL_OP_DP3}

class dp4(Instruction):
  name = 'dp4'
  machine_inst = OPCD_IEEE_2_1
  params = {'OPCD':IL_OP_DP4}

class dst(Instruction):
  name = 'dst'
  machine_inst = OPCD_2_1
  params = {'OPCD':IL_OP_DST}

class dsx(Instruction):
  name = 'dsx'
  machine_inst = OPCD_CENTROID_1_1
  params = {'OPCD':IL_OP_DSX}

class dsy(Instruction):
  name = 'dsy'
  machine_inst = OPCD_CENTROID_1_1
  params = {'OPCD':IL_OP_DSY}

class dxsincos(Instruction):
  name = 'dxsincos'
  machine_inst = OPCD_3_1
  params = {'OPCD':IL_OP_DXSINCOS}

class eq(Instruction):
  name = 'eq'
  machine_inst = OPCD_2_1
  params = {'OPCD':IL_OP_EQ}

class exn(Instruction):
  name = 'exn'
  machine_inst = OPCD_1_1
  params = {'OPCD':IL_OP_EXN}

class exp(Instruction):
  name = 'exp'
  machine_inst = OPCD_1_1
  params = {'OPCD':IL_OP_EXP}

class exp_vec(Instruction):
  name = 'exp_vec'
  machine_inst = OPCD_1_1
  params = {'OPCD':IL_OP_EXP_VEC}

class expp(Instruction):
  name = 'expp'
  machine_inst = OPCD_1_1
  params = {'OPCD':IL_OP_EXPP}

class faceforward(Instruction):
  name = 'faceforward'
  machine_inst = OPCD_3_1
  params = {'OPCD':IL_OP_FACEFORWARD}

class flr(Instruction):
  name = 'flr'
  machine_inst = OPCD_1_1
  params = {'OPCD':IL_OP_FLR}

class frc(Instruction):
  name = 'frc'
  machine_inst = OPCD_1_1
  params = {'OPCD':IL_OP_FRC}

class fwidth(Instruction):
  name = 'fwidth'
  machine_inst = OPCD_1_1
  params = {'OPCD':IL_OP_FWIDTH}

class ge(Instruction):
  name = 'ge'
  machine_inst = OPCD_2_1
  params = {'OPCD':IL_OP_GE}

class len(Instruction):
  name = 'len'
  machine_inst = OPCD_1_1
  params = {'OPCD':IL_OP_LEN}

class lit(Instruction):
  name = 'lit'
  machine_inst = OPCD_1_1
  params = {'OPCD':IL_OP_LIT}

class ln(Instruction):
  name = 'ln'
  machine_inst = OPCD_ZEROOP_1_1
  params = {'OPCD':IL_OP_LN}

class log(Instruction):
  name = 'log'
  machine_inst = OPCD_ZEROOP_1_1
  params = {'OPCD':IL_OP_LOG}

class log_vec(Instruction):
  name = 'log_vec'
  machine_inst = OPCD_1_1
  params = {'OPCD':IL_OP_LOG_VEC}

class logp(Instruction):
  name = 'logp'
  machine_inst = OPCD_ZEROOP_1_1
  params = {'OPCD':IL_OP_LOGP}

class lrp(Instruction):
  name = 'lrp'
  machine_inst = OPCD_3_1
  params = {'OPCD':IL_OP_LRP}

class lt(Instruction):
  name = 'lt'
  machine_inst = OPCD_2_1
  params = {'OPCD':IL_OP_LT}

class mad(Instruction):
  name = 'mad'
  machine_inst = OPCD_IEEE_3_1
  params = {'OPCD':IL_OP_MAD}

class max(Instruction):
  name = 'max'
  machine_inst = OPCD_IEEE_2_1
  params = {'OPCD':IL_OP_MAX}

class min(Instruction):
  name = 'min'
  machine_inst = OPCD_IEEE_2_1
  params = {'OPCD':IL_OP_MIN}

class mmul(Instruction):
  name = 'mmul'
  machine_inst = OPCD_MATRIX_2_1
  params = {'OPCD':IL_OP_MMUL}

class mod(Instruction):
  name = 'mod'
  machine_inst = OPCD_2_1
  params = {'OPCD':IL_OP_MOD}

class invariant_move(Instruction):
  name = 'invariant_move'
  machine_inst = OPCD_1_1
  params = {'OPCD':IL_OP_INVARIANT_MOV}

class mov(Instruction):
  name = 'mov'
  machine_inst = OPCD_1_1
  params = {'OPCD':IL_OP_MOV}

class mova(Instruction):
  name = 'mova'
  machine_inst = OPCD_ROUND_1_1
  params = {'OPCD':IL_OP_MOVA}

class mul(Instruction):
  name = 'mul'
  machine_inst = OPCD_IEEE_2_1
  params = {'OPCD':IL_OP_MUL}

class ne(Instruction):
  name = 'ne'
  machine_inst = OPCD_2_1
  params = {'OPCD':IL_OP_NE}

class noise(Instruction):
  name = 'noise'
  machine_inst = OPCD_NOISETYPE_1_1
  params = {'OPCD':IL_OP_NOISE}


class nrm(Instruction):
  name = 'nrm'
  machine_inst = OPCD_ZEROOP_1_1
  params = {'OPCD':IL_OP_NRM}


class pireduce(Instruction):
  name = 'pireduce'
  machine_inst = OPCD_1_1
  params = {'OPCD':IL_OP_PIREDUCE}

class pow(Instruction):
  name = 'pow'
  machine_inst = OPCD_2_1
  params = {'OPCD':IL_OP_POW}

class rcp(Instruction):
  name = 'rcp'
  machine_inst = OPCD_ZEROOP_1_1
  params = {'OPCD':IL_OP_RCP}

class reflect(Instruction):
  name = 'reflect'
  machine_inst = OPCD_NORMALIZE_2_1
  params = {'OPCD':IL_OP_REFLECT}

class rnd(Instruction):
  name = 'rnd'
  machine_inst = OPCD_1_1
  params = {'OPCD':IL_OP_RND}

class round_nearest(Instruction):
  name = 'round_nearest'
  machine_inst = OPCD_1_1
  params = {'OPCD':IL_OP_ROUND_NEAR}

class round_neginf(Instruction):
  name = 'round_neginf'
  machine_inst = OPCD_1_1
  params = {'OPCD':IL_OP_ROUND_NEG_INF}

class round_posinf(Instruction):
  name = 'round_posinf'
  machine_inst = OPCD_1_1
  params = {'OPCD':IL_OP_ROUND_POS_INF}

class round_z(Instruction):
  name = 'round_z'
  machine_inst = OPCD_1_1
  params = {'OPCD':IL_OP_ROUND_ZERO}

class rsq_vec(Instruction):
  name = 'rsq_vec'
  machine_inst = OPCD_1_1
  params = {'OPCD':IL_OP_RSQ_VEC}

class rsq(Instruction):
  name = 'rsq'
  machine_inst = OPCD_ZEROOP_1_1
  params = {'OPCD':IL_OP_RSQ}

class set(Instruction):
  name = 'set'
  machine_inst = OPCD_RELOP_2_1
  params = {'OPCD':IL_OP_SET}

class sgn(Instruction):
  name = 'sgn'
  machine_inst = OPCD_1_1
  params = {'OPCD':IL_OP_SGN}

class sin(Instruction):
  name = 'sin'
  machine_inst = OPCD_1_1
  params = {'OPCD':IL_OP_SIN}

class sincos(Instruction):
  name = 'sincos'
  machine_inst = OPCD_1_1
  params = {'OPCD':IL_OP_SINCOS}

class sin_vec(Instruction):
  name = 'sin_vec'
  machine_inst = OPCD_1_1
  params = {'OPCD':IL_OP_SIN_VEC}

class cos_vec(Instruction):
  name = 'cos_vec'
  machine_inst = OPCD_1_1
  params = {'OPCD':IL_OP_COS_VEC}

class sqrt(Instruction):
  name = 'sqrt'
  machine_inst = OPCD_1_1
  params = {'OPCD':IL_OP_SQRT}

class sqrt_vec(Instruction):
  name = 'sqrt_vec'
  machine_inst = OPCD_1_1
  params = {'OPCD':IL_OP_SQRT_VEC}

class sub(Instruction):
  name = 'sub'
  machine_inst = OPCD_2_1
  params = {'OPCD':IL_OP_SUB}

class tan(Instruction):
  name = 'tan'
  machine_inst = OPCD_1_1
  params = {'OPCD':IL_OP_TAN}

class transpose(Instruction):
  name = 'transpose'
  machine_inst = OPCD_1_1
  params = {'OPCD':IL_OP_TRANSPOSE}

class trc(Instruction):
  name = 'trc'
  machine_inst = OPCD_1_1
  params = {'OPCD':IL_OP_TRC}

# 6.10

class dne(Instruction):
  name = 'dne'
  machine_inst = OPCD_2_1
  params = {'OPCD':IL_OP_D_NE}

class deq(Instruction):
  name = 'deq'
  machine_inst = OPCD_2_1
  params = {'OPCD':IL_OP_D_EQ}

class dge(Instruction):
  name = 'dge'
  machine_inst = OPCD_2_1
  params = {'OPCD':IL_OP_D_GE}

class dlt(Instruction):
  name = 'dlt'
  machine_inst = OPCD_2_1
  params = {'OPCD':IL_OP_D_LT}

class dfrexp(Instruction):
  name = 'dfrexp'
  machine_inst = OPCD_1_1
  params = {'OPCD':IL_OP_D_FREXP}

class dadd(Instruction):
  name = 'dadd'
  machine_inst = OPCD_2_1
  params = {'OPCD':IL_OP_D_ADD}

class dmul(Instruction):
  name = 'dmul'
  machine_inst = OPCD_2_1
  params = {'OPCD':IL_OP_D_MUL}

class ddiv(Instruction):
  name = 'ddiv'
  machine_inst = OPCD_2_1
  params = {'OPCD':IL_OP_D_DIV}

class dldexp(Instruction):
  name = 'dldexp'
  machine_inst = OPCD_2_1
  params = {'OPCD':IL_OP_D_LDEXP}

class dfrac(Instruction):
  name = 'dfrac'
  machine_inst = OPCD_1_1
  params = {'OPCD':IL_OP_D_FRAC}

class dmad(Instruction):
  name = 'dmad'
  machine_inst = OPCD_3_1
  params = {'OPCD':IL_OP_D_MULADD}


















