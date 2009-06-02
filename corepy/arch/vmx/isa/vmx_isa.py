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

from corepy.spre.isa_syn import *

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

# ------------------------------
# Custom Fields
# ------------------------------

class OneBit(Field):
  callTemplate = "def CallFunc(value):\n  return (1 << %(shift)d)\nself.__call__ = CallFunc\n"

# ------------------------------
# VMX Fields
# ------------------------------

Fields = (
  ("OPCD", (Field, (0,5))),
  ("rA",   (Field, (11,15))),
  ("A",    (Field, (11,15))),
  ("rB",   (Field, (16,20))),
  ("B",    (Field, (16,20))),
  ("Rc",   (Field, (31), 0)),
  ("vA",   (Field, (11,15))),
  ("vB",   (Field, (16,20))),
  ("vC",   (Field, (21,25))),
  ("vD",   (Field, (6,10))),
  ("vS",   (Field, (6,10))),
  ("SHB",  (Field, (22,25))),
  ("SIMM", (Field, (11,15))),
  ("UIMM", (Field, (11,15))),
  ("XO_1", (Opcode, (21,31))),
  ("XO_2", (Opcode, (22,31))),
  ("XO_3", (Opcode, (26,31))),
  ("ONE_BIT", (OneBit, (6))), # Custom field to handle the ds* instructions
  ("STRM",  (Field, (9,10))), # Custom field to handle the ds* instructions
  )

# Create the Field objects
SynthesizeFields(Fields, globals())

# ------------------------------
# VMX Instructions
# ------------------------------

# Common machine->assembly mappings
# Note: Use {..., 'asm': None, ... }  for instructions that use machine order for asm order

ASM_ABStrm = (A, B, STRM)
ASM_vDBUimm = (vD, vB, UIMM)
ASM_vDACB = (vD, vA, vC, vB)

# In the first few instructions, ( ..., XOPCODE, 0) has been replaced with (XOPCODE << 1)
# '0' is a reserved bit and this only happens for the memory instructions.

VMX_ISA = (
  ("dss",        {'binary': (31, 0, "0_0", STRM, "0_0000", "0000_0", (822 << 1)), 'asm': None }),
  ("dssall",     {'binary': (31, ONE_BIT, "0_0", STRM, "0_0000", "0000_0", (822 << 1)), 'asm': None }),
  ("dst",        {'binary': (31, 0, "0_0", STRM, A, B, (342 << 1)), 'asm': ASM_ABStrm }),
  ("dstst",      {'binary': (31, 0, "0_0", STRM, A, B, (374 << 1)), 'asm': ASM_ABStrm }),
  ("dststt",     {'binary': (31, ONE_BIT, "0_0", STRM, A, B, (374 << 1)), 'asm': ASM_ABStrm }),
  ("dstt",       {'binary': (31, ONE_BIT, "0_0", STRM, A, B, (342 << 1)), 'asm': ASM_ABStrm }),
  ("lvebx",      {'binary': (31, vD, A, B, (7 << 1)), 'asm': None }),
  ("lvehx",      {'binary': (31, vD, A, B, (39 << 1)), 'asm': None }),
  ("lvewx",      {'binary': (31, vD, A, B, (71 << 1)), 'asm': None }),
  ("lvsl",       {'binary': (31, vD, A, B, (6 << 1)), 'asm': None }),
  ("lvsr",       {'binary': (31, vD, A, B, (38 << 1)), 'asm': None }),
  ("lvx",        {'binary': (31, vD, A, B, (103 << 1)), 'asm': None }),
  ("lvxl",       {'binary': (31, vD, A, B, (359 << 1)), 'asm': None }),
  ("mfvscr",     {'binary': (4,  vD, "0_0000", "0000_0", 1540), 'asm': None }),
  ("mtvscr",     {'binary': (4,  "00_000", "0_0000", vB, 1604), 'asm': None }),
  ("stvebx",     {'binary': (31, vS, A, B, (135 << 1)), 'asm': None }),
  ("stvehx",     {'binary': (31, vS, A, B, (167 << 1)), 'asm': None }),
  ("stvewx",     {'binary': (31, vS, A, B, (199 << 1)), 'asm': None }),
  ("stvx",       {'binary': (31, vS, A, B, (231 << 1)), 'asm': None }),
  ("stvxl",      {'binary': (31, vS, A, B, (487 << 1)), 'asm': None }),
  ("vaddcuw",    {'binary': (4,  vD, vA, vB, 384), 'asm': None }),
  ("vaddfp",     {'binary': (4,  vD, vA, vB, 10), 'asm': None }),
  ("vaddsbs",    {'binary': (4,  vD, vA, vB, 768), 'asm': None }),
  ("vaddshs",    {'binary': (4,  vD, vA, vB, 832), 'asm': None }),
  ("vaddsws",    {'binary': (4,  vD, vA, vB, 896), 'asm': None }),
  ("vaddubm",    {'binary': (4,  vD, vA, vB, 0), 'asm': None }),
  ("vaddubs",    {'binary': (4,  vD, vA, vB, 512), 'asm': None }),
  ("vadduhm",    {'binary': (4,  vD, vA, vB, 64), 'asm': None }),
  ("vadduhs",    {'binary': (4,  vD, vA, vB, 576), 'asm': None }),
  ("vadduwm",    {'binary': (4,  vD, vA, vB, 128), 'asm': None }),
  ("vadduws",    {'binary': (4,  vD, vA, vB, 640), 'asm': None }),
  ("vand",       {'binary': (4,  vD, vA, vB, 1028), 'asm': None }),
  ("vandc",      {'binary': (4,  vD, vA, vB, 1092), 'asm': None }),
  ("vavgsb",     {'binary': (4,  vD, vA, vB, 1282), 'asm': None }),
  ("vavgsh",     {'binary': (4,  vD, vA, vB, 1346), 'asm': None }),
  ("vavgsw",     {'binary': (4,  vD, vA, vB, 1410), 'asm': None }),
  ("vavgub",     {'binary': (4,  vD, vA, vB, 1026), 'asm': None }),
  ("vavguh",     {'binary': (4,  vD, vA, vB, 1090), 'asm': None }),
  ("vavguw",     {'binary': (4,  vD, vA, vB, 1154), 'asm': None }),
  ("vcfsx",      {'binary': (4,  vD, UIMM, vB, 842), 'asm': ASM_vDBUimm }),
  ("vcfux",      {'binary': (4,  vD, UIMM, vB, 778), 'asm': ASM_vDBUimm }),
  ("vcmpbfpx",   {'binary': (4,  vD, vA, vB, Rc, 966), 'asm': None }),
  ("vcmpeqfpx",  {'binary': (4,  vD, vA, vB, Rc, 198), 'asm': None }),
  ("vcmpequbx",  {'binary': (4,  vD, vA, vB, Rc, 6), 'asm': None }),
  ("vcmpequhx",  {'binary': (4,  vD, vA, vB, Rc, 70), 'asm': None }),
  ("vcmpequwx",  {'binary': (4,  vD, vA, vB, Rc, 134), 'asm': None }),
  ("vcmpgefpx",  {'binary': (4,  vD, vA, vB, Rc, 454), 'asm': None }),
  ("vcmpgtfpx",  {'binary': (4,  vD, vA, vB, Rc, 710), 'asm': None }),
  ("vcmpgtsbx",  {'binary': (4,  vD, vA, vB, Rc, 774), 'asm': None }),
  ("vcmpgtshx",  {'binary': (4,  vD, vA, vB, Rc, 838), 'asm': None }),
  ("vcmpgtswx",  {'binary': (4,  vD, vA, vB, Rc, 902), 'asm': None }),
  ("vcmpgtubx",  {'binary': (4,  vD, vA, vB, Rc, 518), 'asm': None }),
  ("vcmpgtuhx",  {'binary': (4,  vD, vA, vB, Rc, 582), 'asm': None }),
  ("vcmpgtuwx",  {'binary': (4,  vD, vA, vB, Rc, 646), 'asm': None }),
  ("vctsxs",     {'binary': (4,  vD, UIMM, vB, 970), 'asm': ASM_vDBUimm }),
  ("vctuxs",     {'binary': (4,  vD, UIMM, vB, 906), 'asm': ASM_vDBUimm }),
  ("vexptefp",   {'binary': (4,  vD, "0_0000", vB, 394), 'asm': None }),
  ("vlogefp",    {'binary': (4,  vD, "0_0000", vB, 458), 'asm': None }),
  ("vmaddfp",    {'binary': (4,  vD, vA, vB, vC, 46), 'asm': ASM_vDACB }),
  ("vmaxfp",     {'binary': (4,  vD, vA, vB, 1034), 'asm': None }),
  ("vmaxsb",     {'binary': (4,  vD, vA, vB, 258), 'asm': None }),
  ("vmaxsh",     {'binary': (4,  vD, vA, vB, 322), 'asm': None }),
  ("vmaxsw",     {'binary': (4,  vD, vA, vB, 386), 'asm': None }),
  ("vmaxub",     {'binary': (4,  vD, vA, vB, 2), 'asm': None }),
  ("vmaxuh",     {'binary': (4,  vD, vA, vB, 66), 'asm': None }),
  ("vmaxuw",     {'binary': (4,  vD, vA, vB, 130), 'asm': None }),
  ("vmhaddshs",  {'binary': (4,  vD, vA, vB, vC, 32), 'asm': None }),
  ("vmhraddshs", {'binary': (4,  vD, vA, vB, vC, 33), 'asm': None }),
  ("vminfp",     {'binary': (4,  vD, vA, vB, 1098), 'asm': None }),
  ("vminsb",     {'binary': (4,  vD, vA, vB, 770), 'asm': None }),
  ("vminsh",     {'binary': (4,  vD, vA, vB, 834), 'asm': None }),
  ("vminsw",     {'binary': (4,  vD, vA, vB, 898), 'asm': None }),
  ("vminub",     {'binary': (4,  vD, vA, vB, 514), 'asm': None }),
  ("vminuh",     {'binary': (4,  vD, vA, vB, 578), 'asm': None }),
  ("vminuw",     {'binary': (4,  vD, vA, vB, 642), 'asm': None }),
  ("vmladduhm",  {'binary': (4,  vD, vA, vB, vC, 34), 'asm': None }),
  ("vmrghb",     {'binary': (4,  vD, vA, vB, 12), 'asm': None }),
  ("vmrghh",     {'binary': (4,  vD, vA, vB, 76), 'asm': None }),
  ("vmrghw",     {'binary': (4,  vD, vA, vB, 140), 'asm': None }),
  ("vmrglb",     {'binary': (4,  vD, vA, vB, 268), 'asm': None }),
  ("vmrglh",     {'binary': (4,  vD, vA, vB, 332), 'asm': None }),
  ("vmrglw",     {'binary': (4,  vD, vA, vB, 396), 'asm': None }),
  ("vmsummbm",   {'binary': (4,  vD, vA, vB, vC, 37), 'asm': None }),
  ("vmsumshm",   {'binary': (4,  vD, vA, vB, vC, 40), 'asm': None }),
  ("vmsumshs",   {'binary': (4,  vD, vA, vB, vC, 41), 'asm': None }),
  ("vmsumubm",   {'binary': (4,  vD, vA, vB, vC, 36), 'asm': None }),
  ("vmsumuhm",   {'binary': (4,  vD, vA, vB, vC, 38), 'asm': None }),
  ("vmsumuhs",   {'binary': (4,  vD, vA, vB, vC, 39), 'asm': None }),
  ("vmulesb",    {'binary': (4,  vD, vA, vB, 776), 'asm': None }),
  ("vmulesh",    {'binary': (4,  vD, vA, vB, 840), 'asm': None }),
  ("vmuleub",    {'binary': (4,  vD, vA, vB, 520), 'asm': None }),
  ("vmuleuh",    {'binary': (4,  vD, vA, vB, 584), 'asm': None }),
  ("vmulosb",    {'binary': (4,  vD, vA, vB, 264), 'asm': None }),
  ("vmulosh",    {'binary': (4,  vD, vA, vB, 328), 'asm': None }),
  ("vmuloub",    {'binary': (4,  vD, vA, vB, 8), 'asm': None }),
  ("vmulouh",    {'binary': (4,  vD, vA, vB, 72), 'asm': None }),
  ("vnmsubfp",   {'binary': (4,  vD, vA, vB, vC, 47), 'asm': ASM_vDACB }),
  ("vnor",       {'binary': (4,  vD, vA, vB, 1284), 'asm': None }),
  ("vor",        {'binary': (4,  vD, vA, vB, 1156), 'asm': None }),
  ("vperm",      {'binary': (4,  vD, vA, vB, vC, 43), 'asm': None }),
  ("vpkpx",      {'binary': (4,  vD, vA, vB, 782), 'asm': None }),
  ("vpkshss",    {'binary': (4,  vD, vA, vB, 398), 'asm': None }),
  ("vpkshus",    {'binary': (4,  vD, vA, vB, 270), 'asm': None }),
  ("vpkswss",    {'binary': (4,  vD, vA, vB, 462), 'asm': None }),
  ("vpkswus",    {'binary': (4,  vD, vA, vB, 334), 'asm': None }),
  ("vpkuhum",    {'binary': (4,  vD, vA, vB, 14), 'asm': None }),
  ("vpkuhus",    {'binary': (4,  vD, vA, vB, 142), 'asm': None }),
  ("vpkuwum",    {'binary': (4,  vD, vA, vB, 78), 'asm': None }),
  ("vpkuwus",    {'binary': (4,  vD, vA, vB, 206), 'asm': None }),
  ("vrefp",      {'binary': (4,  vD, "0_0000", vB, 266), 'asm': None }),
  ("vrfim",      {'binary': (4,  vD, "0_0000", vB, 714), 'asm': None }),
  ("vrfin",      {'binary': (4,  vD, "0_0000", vB, 522), 'asm': None }),
  ("vrfip",      {'binary': (4,  vD, "0_0000", vB, 650), 'asm': None }),
  ("vrfiz",      {'binary': (4,  vD, "0_0000", vB, 586), 'asm': None }),
  ("vrlb",       {'binary': (4,  vD, vA, vB, 4), 'asm': None }),
  ("vrlh",       {'binary': (4,  vD, vA, vB, 68), 'asm': None }),
  ("vrlw",       {'binary': (4,  vD, vA, vB, 132), 'asm': None }),
  ("vrsqrtefp",  {'binary': (4,  vD, "0_0000", vB, 330), 'asm': None }),
  ("vsel",       {'binary': (4,  vD, vA, vB, vC, 42), 'asm': None }),
  ("vsl",        {'binary': (4,  vD, vA, vB, 452), 'asm': None }),
  ("vslb",       {'binary': (4,  vD, vA, vB, 260), 'asm': None }),
  ("vsldoi",     {'binary': (4,  vD, vA, vB, 0, SHB, 44), 'asm': None }),
  ("vslh",       {'binary': (4,  vD, vA, vB, 324), 'asm': None }),
  ("vslo",       {'binary': (4,  vD, vA, vB, 1036), 'asm': None }),
  ("vslw",       {'binary': (4,  vD, vA, vB, 388), 'asm': None }),
  ("vspltb",     {'binary': (4,  vD, UIMM, vB, 524), 'asm': ASM_vDBUimm }),
  ("vsplth",     {'binary': (4,  vD, UIMM, vB, 588), 'asm': ASM_vDBUimm }),
  ("vspltisb",   {'binary': (4,  vD, SIMM, "0000_0", 780), 'asm': None }),
  ("vspltish",   {'binary': (4,  vD, SIMM, "0000_0", 844), 'asm': None }),
  ("vspltisw",   {'binary': (4,  vD, SIMM, "0000_0", 908), 'asm': None }),
  ("vspltw",     {'binary': (4,  vD, UIMM, vB, 652), 'asm': ASM_vDBUimm }),
  ("vsr",        {'binary': (4,  vD, vA, vB, 708), 'asm': None }),
  ("vsrab",      {'binary': (4,  vD, vA, vB, 772), 'asm': None }),
  ("vsrah",      {'binary': (4,  vD, vA, vB, 836), 'asm': None }),
  ("vsraw",      {'binary': (4,  vD, vA, vB, 900), 'asm': None }),
  ("vsrb",       {'binary': (4,  vD, vA, vB, 516), 'asm': None }),
  ("vsrh",       {'binary': (4,  vD, vA, vB, 580), 'asm': None }),
  ("vsro",       {'binary': (4,  vD, vA, vB, 1100), 'asm': None }),
  ("vsrw",       {'binary': (4,  vD, vA, vB, 644), 'asm': None }),
  ("vsubcuw",    {'binary': (4,  vD, vA, vB, 1408), 'asm': None }),
  ("vsubfp",     {'binary': (4,  vD, vA, vB, 74), 'asm': None }),
  ("vsubsbs",    {'binary': (4,  vD, vA, vB, 1792), 'asm': None }),
  ("vsubshs",    {'binary': (4,  vD, vA, vB, 1856), 'asm': None }),
  ("vsubsws",    {'binary': (4,  vD, vA, vB, 1920), 'asm': None }),
  ("vsububm",    {'binary': (4,  vD, vA, vB, 1024), 'asm': None }),
  ("vsububs",    {'binary': (4,  vD, vA, vB, 1536), 'asm': None }),
  ("vsubuhm",    {'binary': (4,  vD, vA, vB, 1088), 'asm': None }),
  ("vsubuhs",    {'binary': (4,  vD, vA, vB, 1600), 'asm': None }),
  ("vsubuwm",    {'binary': (4,  vD, vA, vB, 1152), 'asm': None }),
  ("vsubuws",    {'binary': (4,  vD, vA, vB, 1664), 'asm': None }),
  ("vsumsws",    {'binary': (4,  vD, vA, vB, 1928), 'asm': None }),
  ("vsum2sws",   {'binary': (4,  vD, vA, vB, 1672), 'asm': None }),
  ("vsum4sbs",   {'binary': (4,  vD, vA, vB, 1800), 'asm': None }),
  ("vsum4shs",   {'binary': (4,  vD, vA, vB, 1608), 'asm': None }),
  ("vsum4ubs",   {'binary': (4,  vD, vA, vB, 1544), 'asm': None }),
  ("vupkhpx",    {'binary': (4,  vD, "0_0000", vB, 846), 'asm': None }),
  ("vupkhsb",    {'binary': (4,  vD, "0_0000", vB, 526), 'asm': None }),
  ("vupkhsh",    {'binary': (4,  vD, "0_0000", vB, 590), 'asm': None }),
  ("vupklpx",    {'binary': (4,  vD, "0_0000", vB, 974), 'asm': None }),
  ("vupklsb",    {'binary': (4,  vD, "0_0000", vB, 654), 'asm': None }),
  ("vupklsh",    {'binary': (4,  vD, "0_0000", vB, 718), 'asm': None }),
  ("vxor",       {'binary': (4,  vD, vA, vB, 1220), 'asm': None }),
  )


# Create the Instructions
opcodes = {0:OPCD, 21:XO_1, 22:XO_2, 26:XO_3} # bit position: extended opcode
SynthesizeISA(VMX_ISA, globals(), opcodes)
