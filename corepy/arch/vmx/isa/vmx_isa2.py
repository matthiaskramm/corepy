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
from vmx_insts import *

__doc__="""
VMX/AltiVec Instruction Set Architectre (ISA) 

To use, import this module and call the Instructions as Python
functions to generate a properly coded version.  For example, to
create an addx instruction:

import vmx_isa as av

# add the vectors of unsigned bytes in v2 and v3 and put the result in v3
inst = av.vaddubs(1, 2, 3) 

Operands are in the same order as presented in the Programming
Environments Manual.

For a complete reference and details for all instructions, please
referer to: 

'AltiVec Technology Programming Environments Manual' from Freescale
Semiconductor. 

URL (valid as of June 1, 2006):
http://www.freescale.com/files/32bit/doc/ref_manual/ALTIVECPEM.pdf
"""

class VMXInstruction(Instruction): pass

class vmhaddshs(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_vC_XO
  params = {'OPCD':4, 'XO':32}

class vmhraddshs(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_vC_XO
  params = {'OPCD':4, 'XO':33}

class vmladduhm(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_vC_XO
  params = {'OPCD':4, 'XO':34}

class vmsumubm(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_vC_XO
  params = {'OPCD':4, 'XO':36}

class vmsummbm(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_vC_XO
  params = {'OPCD':4, 'XO':37}

class vmsumuhm(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_vC_XO
  params = {'OPCD':4, 'XO':38}

class vmsumuhs(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_vC_XO
  params = {'OPCD':4, 'XO':39}

class vmsumshm(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_vC_XO
  params = {'OPCD':4, 'XO':40}

class vmsumshs(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_vC_XO
  params = {'OPCD':4, 'XO':41}

class vsel(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_vC_XO
  params = {'OPCD':4, 'XO':42}

class vperm(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_vC_XO
  params = {'OPCD':4, 'XO':43}

class vsldoi(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_SH_XO
  params = {'OPCD':4, 'XO':44}

class vmaddfp(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_vC_XO
  params = {'OPCD':4, 'XO':46}

class vnmsubfp(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_vC_XO
  params = {'OPCD':4, 'XO':47}

class vaddubm(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':0}

class vadduhm(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':64}

class vadduwm(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':128}

class vaddcuw(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':384}

class vaddubs(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':512}

class vadduhs(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':576}

class vadduws(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':640}

class vaddsbs(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':768}

class vaddshs(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':832}

class vaddsws(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':896}

class vaddubm(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':1024}

class vadduhm(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':1088}

class vadduwm(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':1152}

class vsubcuw(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':1408}

class vsububs(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':1536}

class vsubuhs(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':1600}

class vsubuws(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':1664}

class vsubsbs(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':1792}

class vsubshs(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':1856}

class vsubsws(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':1920}

class vmaxub(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':2}

class vmaxuh(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':66}

class vmaxuw(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':130}

class vmaxsb(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':258}

class vmaxsh(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':322}

class vmaxsw(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':386}

class vminub(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':514}

class vminuh(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':578}

class vminuw(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':642}

class vminsb(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':770}

class vminsh(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':834}

class vminsw(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':898}

class vavgub(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':1026}

class vavguh(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':1090}

class vavguw(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':1154}

class vavgsb(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':1282}

class vavgsh(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':1346}

class vavgsw(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':1410}

class vrlb(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':4}

class vrlh(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':68}

class vrlw(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':132}

class vslb(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':260}

class vslh(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':324}

class vslw(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':388}

class vsl(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':452}

class vsrb(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':516}

class vsrh(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':580}

class vsrw(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':644}

class vsr(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':708}

class vsrab(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':772}

class vsrah(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':836}

class vsraw(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':900}

class vand(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':1028}

class vandc(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':1092}

class vor(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':1156}

class vnor(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':1284}

class mfvscr(VMXInstruction):
  machine_inst = OPCD_vD_XO
  params = {'OPCD':4, 'XO':1540}

class mtvscr(VMXInstruction):
  machine_inst = OPCD_vB_XO
  params = {'OPCD':4, 'XO':1604}

class vmuloub(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':8}

class vmulouh(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':72}

class vmulosb(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':264}

class vmulosh(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':328}

class vmuleub(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':520}

class vmuleuh(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':584}

class vmulesb(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':776}

class vmulesh(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':840}

class vsum4ubs(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':1544}

class vsum4sbs(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':1800}

class vsum4shs(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':1608}

class vsum2sws(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':1672}

class vsumsws(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':1928}

class vaddfp(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':10}

class vsubfp(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':74}

class vrefp(VMXInstruction):
  machine_inst = OPCD_vD_vB_XO
  params = {'OPCD':4, 'XO':266}

class vsqrtefp(VMXInstruction):
  machine_inst = OPCD_vD_vB_XO
  params = {'OPCD':4, 'XO':330}

class vexptefp(VMXInstruction):
  machine_inst = OPCD_vD_vB_XO
  params = {'OPCD':4, 'XO':394}

class vlogefp(VMXInstruction):
  machine_inst = OPCD_vD_vB_XO
  params = {'OPCD':4, 'XO':458}

class vrfin(VMXInstruction):
  machine_inst = OPCD_vD_vB_XO
  params = {'OPCD':4, 'XO':522}

class vrfiz(VMXInstruction):
  machine_inst = OPCD_vD_vB_XO
  params = {'OPCD':4, 'XO':586}

class vrfip(VMXInstruction):
  machine_inst = OPCD_vD_vB_XO
  params = {'OPCD':4, 'XO':650}

class vrfim(VMXInstruction):
  machine_inst = OPCD_vD_vB_XO
  params = {'OPCD':4, 'XO':714}

class vcfux(VMXInstruction):
  machine_inst = OPCD_vD_UIMM_vB_XO
  params = {'OPCD':4, 'XO':778}

class vcfsx(VMXInstruction):
  machine_inst = OPCD_vD_UIMM_vB_XO
  params = {'OPCD':4, 'XO':842}

class vctuxs(VMXInstruction):
  machine_inst = OPCD_vD_UIMM_vB_XO
  params = {'OPCD':4, 'XO':906}

class vctsx(VMXInstruction):
  machine_inst = OPCD_vD_UIMM_vB_XO
  params = {'OPCD':4, 'XO':970}

class vmaxfp(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':1034}

class vminfp(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':1098}

class vmrghb(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':12}

class vmrghh(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':76}

class vmrghw(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':140}

class vmrglb(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':268}

class vmrglh(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':332}

class vmrglw(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':396}

class vspltb(VMXInstruction):
  machine_inst = OPCD_vD_UIMM_vB_XO
  params = {'OPCD':4, 'XO':524}

class vsplth(VMXInstruction):
  machine_inst = OPCD_vD_UIMM_vB_XO
  params = {'OPCD':4, 'XO':588}

class vspltw(VMXInstruction):
  machine_inst = OPCD_vD_UIMM_vB_XO
  params = {'OPCD':4, 'XO':652}

class vspltisb(VMXInstruction):
  machine_inst = OPCD_vD_SIMM_XO
  params = {'OPCD':4, 'XO':780}

class vspltish(VMXInstruction):
  machine_inst = OPCD_vD_SIMM_XO
  params = {'OPCD':4, 'XO':844}

class vspltisw(VMXInstruction):
  machine_inst = OPCD_vD_SIMM_XO
  params = {'OPCD':4, 'XO':908}

class vslo(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':1036}

class vsro(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':1100}

class vpkuhum(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':14}

class vpkuwum(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':78}

class vpkuhus(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':142}

class vpkuwus(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':206}

class vpkshus(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':270}

class vpkswus(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':334}

class vpkshss(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':398}

class vpkswss(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':462}

class vupkhsb(VMXInstruction):
  machine_inst = OPCD_vD_vB_XO
  params = {'OPCD':4, 'XO':526}

class vupkhsh(VMXInstruction):
  machine_inst = OPCD_vD_vB_XO
  params = {'OPCD':4, 'XO':590}

class vupkisb(VMXInstruction):
  machine_inst = OPCD_vD_vB_XO
  params = {'OPCD':4, 'XO':654}

class vupkish(VMXInstruction):
  machine_inst = OPCD_vD_vB_XO
  params = {'OPCD':4, 'XO':718}

class vpkpx(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':782}

class vupkhpx(VMXInstruction):
  machine_inst = OPCD_vD_vB_XO
  params = {'OPCD':4, 'XO':846}

class vupklpx(VMXInstruction):
  machine_inst = OPCD_vD_vB_XO
  params = {'OPCD':4, 'XO':974}

class vxor(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_XO
  params = {'OPCD':4, 'XO':1220}

class dst(VMXInstruction):
  machine_inst = OPCD_T_STRM_A_B_XO
  params = {'OPCD':31, 'T':0, 'XO':342}

class dstt(VMXInstruction):
  machine_inst = OPCD_T_STRM_A_B_XO
  params = {'OPCD':31, 'T':1, 'XO':342}

class dstst(VMXInstruction):
  machine_inst = OPCD_T_STRM_A_B_XO
  params = {'OPCD':31, 'T':0, 'XO':374}

class dststt(VMXInstruction):
  machine_inst = OPCD_T_STRM_A_B_XO
  params = {'OPCD':31, 'T':1, 'XO':374}

class dss(VMXInstruction):
  machine_inst = OPCD_T_STRM_XO
  params = {'OPCD':31, 'T':0, 'XO':822}

class dssall(VMXInstruction):
  machine_inst = OPCD_T_XO
  params = {'OPCD':31, 'T':1, 'XO':822}

class lvebx(VMXInstruction):
  machine_inst = OPCD_vD_A_B_XO
  params = {'OPCD':31, 'XO':7}

class lvehx(VMXInstruction):
  machine_inst = OPCD_vD_A_B_XO
  params = {'OPCD':31, 'XO':39}

class lvewx(VMXInstruction):
  machine_inst = OPCD_vD_A_B_XO
  params = {'OPCD':31, 'XO':71}

class lvsl(VMXInstruction):
  machine_inst = OPCD_vD_A_B_XO
  params = {'OPCD':31, 'XO':6}

class lvsr(VMXInstruction):
  machine_inst = OPCD_vD_A_B_XO
  params = {'OPCD':31, 'XO':38}

class lvx(VMXInstruction):
  machine_inst = OPCD_vD_A_B_XO
  params = {'OPCD':31, 'XO':103}

class lvxl(VMXInstruction):
  machine_inst = OPCD_vD_A_B_XO
  params = {'OPCD':31, 'XO':359}

class stvebx(VMXInstruction):
  machine_inst = OPCD_vD_A_B_XO
  params = {'OPCD':31, 'XO':135}

class stvehx(VMXInstruction):
  machine_inst = OPCD_vD_A_B_XO
  params = {'OPCD':31, 'XO':167}

class stvewx(VMXInstruction):
  machine_inst = OPCD_vD_A_B_XO
  params = {'OPCD':31, 'XO':199}

class stvx(VMXInstruction):
  machine_inst = OPCD_vD_A_B_XO
  params = {'OPCD':31, 'XO':231}

class stvxl(VMXInstruction):
  machine_inst = OPCD_vD_A_B_XO
  params = {'OPCD':31, 'XO':487}

class vcmpbfpx(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_RC_XO
  params = {'OPCD':4, 'XO':966}

class vcmpeqfpx(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_RC_XO
  params = {'OPCD':4, 'XO':198}

class vcmpequbx(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_RC_XO
  params = {'OPCD':4, 'XO':6}

class vcmpequhx(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_RC_XO
  params = {'OPCD':4, 'XO':70}

class vcmpequwx(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_RC_XO
  params = {'OPCD':4, 'XO':134}

class vcmpgefpx(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_RC_XO
  params = {'OPCD':4, 'XO':454}

class vcmpgtfpx(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_RC_XO
  params = {'OPCD':4, 'XO':710}

class vcmpgtswx(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_RC_XO
  params = {'OPCD':4, 'XO':902}

class vcmpgtubx(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_RC_XO
  params = {'OPCD':4, 'XO':518}

class vcmpgtuhx(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_RC_XO
  params = {'OPCD':4, 'XO':582}

class vcmpgtuwx(VMXInstruction):
  machine_inst = OPCD_vD_vA_vB_RC_XO
  params = {'OPCD':4, 'XO':646}






