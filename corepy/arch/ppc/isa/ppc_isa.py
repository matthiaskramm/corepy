# Copyright 2006-2007 The Trustees of Indiana University.

# This software is available for evaluation purposes only.  It may not be
# redistirubted or used for any other purposes without express written
# permission from the authors.

# Author:
#   Christopher Mueller

from corepy.spre.isa_syn import *

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

# ------------------------------
# Custom Fields
# ------------------------------

class L(Field):
  callTemplate = "def CallFunc(value):\n  return 0\nself.__call__ = CallFunc"
  codeTemplate = "0"
  
class SC_ONE(Field):
  callTemplate = "def CallFunc(value):\n  return 2\nself.__call__ = CallFunc"
  codeTemplate = "2"
  
class STWCX_ONE(Field):
  callTemplate = "def CallFunc(value):\n  return 1\nself.__call__ = CallFunc"
  codeTemplate = "1"


# ------------------------------
# PPC Fields
# ------------------------------

Fields = (
  # Shorthand for GP/FP registers
  ("A",  (Field, (11,15))),
  ("B",  (Field, (16,20))),
  ("C",  (Field, (21,25))),
  ("D",  (Field, (6,10))),
  ("S",  (Field, (6,10))),

  # Other fields
  ("AA",   (Field, (30), 0)),
  ("BD",   (MaskedField_14, (16,29))),
  ("BI",   (Field, (11,15))),
  ("BO",   (Field, (6,10))),
  ("crbA", (Field, (11,15))),
  ("crbB", (Field, (16,20))),
  ("crbD", (Field, (6,10))),
  ("crfD", (Field, (6,8))),
  ("crfS", (Field, (11,13))),
  ("CRM",  (Field, (12,19))),
  ("d",    (MaskedField_16, (16,31))),
  ("FM",   (Field, (7,14))),
  ("frA",  (Field, (11,15))),
  ("frB",  (Field, (16,20))),
  ("frC",  (Field, (21,25))),
  ("frD",  (Field, (6,10))),
  ("frS",  (Field, (6,10))),
  ("IMM",  (Field, (16,19))),
  ("L",    (L,     (10))),
  ("LI",   (MaskedField_LI, (6,29))),
  ("LK",   (Field, (31), 0)),
  ("MB",   (Field, (21,25))),
  ("ME",   (Field, (26,30))),
  ("NB",   (Field, (16,20))),
  ("OE",   (Field, (21), 0)),
  ("OPCD", (Opcode, (0,5))),
  ("rA",   (Field, (11,15))),
  ("rB",   (Field, (16,20))),
  ("Rc",   (Field, (31), 0)),
  ("rD",   (Field, (6,10))),
  ("rS",   (Field, (6,10))),
  ("SH",   (Field, (16,20))),
  ("SIMM", (MaskedField_16, (16,31))),
  ("spr",  (SplitField, (11,20))),  # split field 
  ("SR",   (Field, (12,15))),
  ("tbr",  (SplitField, (11,20))),       # Note: this may need specialization
  ("TO",   (Field, (6,10))),  
  ("UIMM", (MaskedField_16, (16,31))),
  ("XO_1", (Opcode, (21,30))),
  ("XO_2", (Opcode, (22,30))),
  ("XO_3", (Opcode, (26,30))),
  ("SC_ONE", (SC_ONE, (30))),  # Custom field for the '1' bit that's set in sc
  ("STWCX_ONE", (STWCX_ONE, (31))),  # Custom field for the '1' bit that's set in stwxc.
  )


# Create the Field objects
SynthesizeFields(Fields, globals())


# ------------------------------
# PPC Instructions
# ------------------------------

# Common machine->assembly mappings
# Note: Use {..., 'asm': None, ... }  for instructions that use machine order for asm order

ASM_AS = (A, S)
ASM_ASB = (A, S, B)
ASM_ASUm = (A, S, UIMM)
ASM_ASSh = (A, S, SH)
ASM_CrmS = (CRM, S)
ASM_SprS = (spr, S)
ASM_SrS = (SR, S)
ASM_ASShMbMe = (A, S, SH, MB, ME)
ASM_ASBMbMe = (A, S, B, MB, ME)
ASM_DACB = (D, A, C, B)

PPC_ISA = (
  ("addx",    {'binary': (31, D, A, B, OE, 266, Rc), 'asm': None }),
  ("addcx",   {'binary': (31, D, A, B, OE, 10, Rc), 'asm': None }),
  ("addex",   {'binary': (31, D, A, B, OE, 138, Rc), 'asm': None }),
  ("addi",    {'binary': (14, D, A, SIMM), 'asm': None }), #"0" * 21)), # D, A, SIMM)),
  ("addic",   {'binary': (12, D, A, SIMM), 'asm': None  }),
  ("addic_",  {'binary': (13, D, A, SIMM), 'asm': None }),
  ("addis",   {'binary': (15, D, A, SIMM), 'asm': None }),
  ("addmex",  {'binary': (31, D, A, 0, 0, 0, 0, 0, OE, 234, Rc), 'asm': None }),
  ("addzex",  {'binary': (31, D, A, 0, 0, 0, 0, 0, OE, 202, Rc), 'asm': None }),
  ("andx",    {'binary': (31, S, A, B, 28, Rc), 'asm': ASM_ASB }),
  ("andcx",   {'binary': (31, S, A, B, 60, Rc), 'asm': ASM_ASB }),
  ("andi",    {'binary': (28, S, A, UIMM), 'asm': ASM_ASUm }),
  ("andis",   {'binary': (29, S, A, UIMM), 'asm': ASM_ASUm }),
  ("bx",      {'binary': (18, LI, AA, LK), 'asm': None }),
  ("bcx",     {'binary': (16, BO, BI, BD, AA, LK), 'asm': None }),
  ("bcctrx",  {'binary': (19, BO, BI, 0, 0, 0, 0, 0, 528, LK), 'asm': None }),
  ("bclrx",   {'binary': (19, BO, BI, 0, 0, 0, 0, 0, 16, LK), 'asm': None }),
  ("cmp_",    {'binary': (31, crfD, 0, L, A, B, 0, 0), 'asm': None }), # Note: 'cmp' is a builtin in Python
  ("cmpi",    {'binary': (11, crfD, 0, L, A, SIMM), 'asm': None }),
  ("cmpl",    {'binary': (31, crfD, 0, L, A, B, 32, 0), 'asm': None }),
  ("cmpli",   {'binary': (10, crfD, 0, L, A, UIMM), 'asm': None }),
  ("cntlzwx", {'binary': (31, S, A, 0, 0, 0, 0, 0, 26, Rc), 'asm': ASM_AS }),
  ("crand",   {'binary': (19, crbD, crbA, crbB, 257, 0), 'asm': None }),
  ("crandc",  {'binary': (19, crbD, crbA, crbB, 129, 0), 'asm': None }),
  ("creqv",   {'binary': (19, crbD, crbA, crbB, 289, 0), 'asm': None }),
  ("crnand",  {'binary': (19, crbD, crbA, crbB, 225, 0), 'asm': None }),
  ("crnor",   {'binary': (19, crbD, crbA, crbB, 33, 0), 'asm': None }),
  ("cror",    {'binary': (19, crbD, crbA, crbB, 449, 0), 'asm': None }),
  ("crorc",   {'binary': (19, crbD, crbA, crbB, 417, 0), 'asm': None }),
  ("crxor",   {'binary': (19, crbD, crbA, crbB, 193, 0), 'asm': None }),
  ("dcba",    {'binary': (31, 0, 0, 0, 0, 0, A, B, 758, 0), 'asm': None }),
  ("dcbf",    {'binary': (31, 0, 0, 0, 0, 0, A, B, 86, 0), 'asm': None }),
  ("dcbi",    {'binary': (31, 0, 0, 0, 0, 0, A, B, 470, 0), 'asm': None }),
  ("dcbst",   {'binary': (31, 0, 0, 0, 0, 0, A, B, 54, 0), 'asm': None }),
  ("dcbt",    {'binary': (31, 0, 0, 0, 0, 0, A, B, 278, 0), 'asm': None }),
  ("dcbtst",  {'binary': (31, 0, 0, 0, 0, 0, A, B, 246, 0), 'asm': None }),
  ("dcbz",    {'binary': (31, 0, 0, 0, 0, 0, A, B, 1014, 0), 'asm': None }),
  ("divwx",   {'binary': (31, D, A, B, OE, 491, Rc), 'asm': None }),
  ("divwux",  {'binary': (31, D, A, B, OE, 459, Rc), 'asm': None }),
  ("eciwx",   {'binary': (31, D, A, B, 310, 0), 'asm': None }),
  ("ecowx",   {'binary': (31, S, A, B, 438, 0), 'asm': None }),
  ("eieio",   {'binary': (31, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 854, 0), 'asm': None }),
  ("eqvx",    {'binary': (31, S, A, B, 284, Rc), 'asm': ASM_ASB }),
  ("extsbx",  {'binary': (31, S, A, 0, 0, 0, 0, 0, 954, Rc), 'asm': ASM_AS }),
  ("extshx",  {'binary': (31, S, A, 0, 0, 0, 0, 0, 922, Rc), 'asm': ASM_AS }),
  ("fabsx",   {'binary': (63, D, 0, 0, 0, 0, 0, B, 264, Rc), 'asm': None }),
  ("faddx",   {'binary': (63, D, A, B, 0, 0, 0, 0, 0, 21, Rc), 'asm': None }),
  ("faddsx",  {'binary': (59, D, A, B, 0, 0, 0, 0, 0, 21, Rc), 'asm': None }),
  ("fcmpo",   {'binary': (63, crfD, 0, 0, A, B, 32, 0), 'asm': None }),
  ("fcmpu",   {'binary': (63, crfD, 0, 0, A, B, 0, 0), 'asm': None }),
  ("fctiwx",  {'binary': (63, D, 0, 0, 0, 0, 0, B, 14, Rc), 'asm': None }),
  ("fctiwzx", {'binary': (63, D, 0, 0, 0, 0, 0, B, 15, Rc), 'asm': None }),
  ("fdivx",   {'binary': (63, D, A, B, 0, 0, 0, 0, 0, 18, Rc), 'asm': None }),
  ("fdivsx",  {'binary': (59, D, A, B, 0, 0, 0, 0, 0, 18, Rc), 'asm': None }),
  ("fmaddx",  {'binary': (63, D, A, B, C, 29, Rc), 'asm': ASM_DACB }),
  ("fmaddsx", {'binary': (59, D, A, B, C, 29, Rc), 'asm': ASM_DACB }),
  ("fmrx",    {'binary': (63, D, 0, 0, 0, 0, 0, B, 72, Rc), 'asm': None }),
  ("fmsubx",  {'binary': (63, D, A, B, C, 28, Rc), 'asm': ASM_DACB }),
  ("fmsubsx", {'binary': (59, D, A, B, C, 28, Rc), 'asm': ASM_DACB }),
  ("fmulx",   {'binary': (63, D, A, 0, 0, 0, 0, 0, C, 25, Rc), 'asm': None }),
  ("fmulsx",  {'binary': (59, D, A, 0, 0, 0, 0, 0, C, 25, Rc), 'asm': None }),
  ("fnabsx",  {'binary': (63, D, 0, 0, 0, 0, 0, B, 136, Rc), 'asm': None }),
  ("fnegx",   {'binary': (63, D, 0, 0, 0, 0, 0, B, 40, Rc), 'asm': None }),
  ("fnmaddx", {'binary': (63, D, A, B, C, 31, Rc), 'asm': ASM_DACB }),
  ("fnmaddsx",{'binary': (59, D, A, B, C, 31, Rc), 'asm': ASM_DACB }),
  ("fnmsubx", {'binary': (63, D, A, B, C, 30, Rc), 'asm': ASM_DACB }),
  ("fnmsubsx",{'binary': (59, D, A, B, C, 30, Rc), 'asm': ASM_DACB }),
  ("fresx",   {'binary': (59, D, 0, 0, 0, 0, 0, B, 0, 0, 0, 0, 0, 24, Rc), 'asm': None }),
  ("frspx",   {'binary': (63, D, 0, 0, 0, 0, 0, B, 12, Rc), 'asm': None }),
  ("frsqrtex",{'binary': (63, D, 0, 0, 0, 0, 0, B, 0, 0, 0, 0, 0, 26, Rc), 'asm': None }),
  ("fselx",   {'binary': (63, D, A, B, C, 23, Rc), 'asm': ASM_DACB }),
  ("fsqrtx",  {'binary': (63, D, 0, 0, 0, 0, 0, B, 0, 0, 0, 0, 0, 22, Rc), 'asm': None }),
  ("fsqrtsx", {'binary': (59, D, 0, 0, 0, 0, 0, B, 0, 0, 0, 0, 0, 22, Rc), 'asm': None }),
  ("fsubx",   {'binary': (63, D, A, B, 0, 0, 0, 0, 0, 20, Rc), 'asm': None }),
  ("fsubsx",  {'binary': (59, D, A, B, 0, 0, 0, 0, 0, 20, Rc), 'asm': None }),
  ("icbi",    {'binary': (31, 0, 0, 0, 0, 0, A, B, 982, 0), 'asm': None }),
  ("isync",   {'binary': (19, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 150, 0), 'asm': None }),
  ("lbz",     {'binary': (34, D, A, d), 'asm': None }),
  ("lbzu",    {'binary': (35, D, A, d), 'asm': None }),
  ("lbzux",   {'binary': (31, D, A, B, 119, 0), 'asm': None }),
  ("lbzx",    {'binary': (31, D, A, B, 87, 0), 'asm': None }),
  ("lfd",     {'binary': (50, D, A, d), 'asm': None }),
  ("lfdu",    {'binary': (51, D, A, d), 'asm': None }),
  ("lfdux",   {'binary': (31, D, A, B, 631, 0), 'asm': None }),
  ("lfdx",    {'binary': (31, D, A, B, 599, 0), 'asm': None }),
  ("lfs",     {'binary': (48, D, A, d), 'asm': None }),
  ("lfsu",    {'binary': (49, D, A, d), 'asm': None }),
  ("lfsux",   {'binary': (31, D, A, B, 567, 0), 'asm': None }),
  ("lfsx",    {'binary': (31, D, A, B, 535, 0), 'asm': None }),
  ("lha",     {'binary': (42, D, A, d), 'asm': None }),
  ("lhau",    {'binary': (43, D, A, d), 'asm': None }),
  ("lhaux",   {'binary': (31, D, A, B, 375, 0), 'asm': None }),
  ("lhax",    {'binary': (31, D, A, B, 343, 0), 'asm': None }),
  ("lhbrx",   {'binary': (31, D, A, B, 790, 0), 'asm': None }),
  ("lhz",     {'binary': (40, D, A, d), 'asm': None }),
  ("lhzu",    {'binary': (41, D, A, d), 'asm': None }),
  ("lhzux",   {'binary': (31, D, A, B, 311, 0), 'asm': None }),
  ("lhzx",    {'binary': (31, D, A, B, 279, 0), 'asm': None }),
  ("lmw",     {'binary': (46, D, A, d), 'asm': None }),
  ("lswi",    {'binary': (31, D, A, NB, 597, 0), 'asm': None }),
  ("lswx",    {'binary': (31, D, A, B, 533, 0), 'asm': None }),
  ("lwarx",   {'binary': (31, D, A, B, 20, 0), 'asm': None }),
  ("lwbrx",   {'binary': (31, D, A, B, 534, 0), 'asm': None }),
  ("lwz",     {'binary': (32, D, A, d), 'asm': None }),
  ("lwzu",    {'binary': (33, D, A, d), 'asm': None }),
  ("lwzux",   {'binary': (31, D, A, B, 55, 0), 'asm': None }),
  ("lwzx",    {'binary': (31, D, A, B, 23, 0), 'asm': None }),
  ("mcrf",    {'binary': (19, crfD, 0, 0, crfS, 0, 0, 0, 0, 0, 0, 0, 0, 0), 'asm': None }),
  ("mcrfs",   {'binary': (63, crfD, 0, 0, crfS, 0, 0, 0, 0, 0, 0, 0, 64, 0), 'asm': None }),
  ("mcrxr",   {'binary': (31, crfD, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 512, 0), 'asm': None }),
  ("mfcr",    {'binary': (31, D, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19, 0), 'asm': None }),
  ("mffsx",   {'binary': (63, D, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 583, Rc), 'asm': None }),
  ("mfmsr",   {'binary': (31, D, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 83, 0), 'asm': None }),
  ("mfspr",   {'binary': (31, D, spr, 339, 0), 'asm': None }),
  ("mfsr",    {'binary': (31, D, 0, SR, 0, 0, 0, 0, 0, 595, 0), 'asm': None }),
  ("mfsrin",  {'binary': (31, D, 0, 0, 0, 0, 0, B, 659, 0), 'asm': None }),
  ("mftb",    {'binary': (31, D, tbr, 371, 0), 'asm': None }),
  ("mtcrf",   {'binary': (31, S, 0, CRM, 0, 144, 0), 'asm': ASM_CrmS }),
  ("mtfsb0x", {'binary': (63, crbD, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 70, Rc), 'asm': None }),
  ("mtfsb1x", {'binary': (63, crbD, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 38, Rc), 'asm': None }),
  ("mtfsfx",  {'binary': (63, 0, FM, 0, B, 711, Rc), 'asm': None }),
  ("mtfsfix", {'binary': (63, crfD, 0, 0, 0, 0, 0, 0, 0, IMM, 0, 134, Rc), 'asm': None }),
  ("mtmsr",   {'binary': (31, S, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 146, 0), 'asm': None }),
  ("mtspr",   {'binary': (31, S, spr, 467, 0), 'asm': ASM_SprS }),
  ("mtsr",    {'binary': (31, S, 0, SR, 0, 0, 0, 0, 0, 210, 0), 'asm': ASM_SrS }),
  ("mtsrin",  {'binary': (31, S, 0, 0, 0, 0, 0, B, 242, 0), 'asm': None }),
  ("mulhwx",  {'binary': (31, D, A, B, 0, 75, Rc), 'asm': None }),
  ("mulhwux", {'binary': (31, D, A, B, 0, 11, Rc), 'asm': None }),
  ("mulli",   {'binary': (7,  D, A, SIMM), 'asm': None }),
  ("mullwx",  {'binary': (31, D, A, B, OE, 235, Rc), 'asm': None }),
  ("nandx",   {'binary': (31, S, A, B, 476, Rc), 'asm': ASM_ASB }),
  ("negx",    {'binary': (31, D, A, 0, 0, 0, 0, 0, OE, 104, Rc), 'asm': None }),
  ("norx",    {'binary': (31, S, A, B, 124, Rc), 'asm': ASM_ASB }),
  ("orx",     {'binary': (31, S, A, B, 444, Rc), 'asm': ASM_ASB }),
  ("orcx",    {'binary': (31, S, A, B, 412, Rc), 'asm': ASM_ASB }),
  ("ori",     {'binary': (24, S, A, UIMM), 'asm': ASM_ASUm }),
  ("oris",    {'binary': (25, S, A, UIMM), 'asm': ASM_ASUm }),
  ("rfi",     {'binary': (19, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 50, 0), 'asm': None }),
  ("rlwimix", {'binary': (20, S, A, SH, MB, ME, Rc), 'asm': ASM_ASShMbMe }),
  ("rlwinmx", {'binary': (21, S, A, SH, MB, ME, Rc), 'asm': ASM_ASShMbMe }),
  ("rlwnmx",  {'binary': (23, S, A, B, MB, ME, Rc), 'asm': ASM_ASBMbMe }),
  ("sc",      {'binary': (17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, SC_ONE, 0), 'asm': None }),
  ("slwx",    {'binary': (31, S, A, B, 24, Rc), 'asm': ASM_ASB }),
  ("srawx",   {'binary': (31, S, A, B, 792, Rc), 'asm': ASM_ASB }),
  ("srawix",  {'binary': (31, S, A, SH, 824, Rc), 'asm': ASM_ASSh }),
  ("srwx",    {'binary': (31, S, A, B, 536, Rc), 'asm': ASM_ASB }),
  ("stb",     {'binary': (38, S, A, d), 'asm': None }),
  ("stbu",    {'binary': (39, S, A, d), 'asm': None }),
  ("stbux",   {'binary': (31, S, A, B, 247, 0), 'asm': None }),
  ("stbx",    {'binary': (31, S, A, B, 215, 0), 'asm': None }),
  ("stfd",    {'binary': (54, S, A, d), 'asm': None }),
  ("stfdu",   {'binary': (55, S, A, d), 'asm': None }),
  ("stfdux",  {'binary': (31, S, A, B, 759, 0), 'asm': None }),
  ("stfdx",   {'binary': (31, S, A, B, 727, 0), 'asm': None }),
  ("stfiwx",  {'binary': (31, S, A, B, 983, 0), 'asm': None }),
  ("stfs",    {'binary': (52, S, A, d), 'asm': None }),
  ("stfsu",   {'binary': (53, S, A, d), 'asm': None }),
  ("stfsux",  {'binary': (31, S, A, B, 695, 0), 'asm': None }),
  ("stfsx",   {'binary': (31, S, A, B, 663, 0), 'asm': None }),
  ("sth",     {'binary': (44, S, A, d), 'asm': None }),
  ("sthbrx",  {'binary': (31, S, A, B, 918, 0), 'asm': None }),
  ("sthu",    {'binary': (45, S, A, d), 'asm': None }),
  ("sthux",   {'binary': (31, S, A, B, 439, 0), 'asm': None }),
  ("sthx",    {'binary': (31, S, A, B, 407, 0), 'asm': None }),
  ("stmw",    {'binary': (47, S, A, d), 'asm': None }),
  ("stswi",   {'binary': (31, S, A, NB, 725, 0), 'asm': None }),
  ("stswx",   {'binary': (31, S, A, B, 661, 0), 'asm': None }),
  ("stw",     {'binary': (36, S, A, d), 'asm': None }),
  ("stwbrx",  {'binary': (31, S, A, B, 662, 0), 'asm': None }),
  ("stwcx_",  {'binary': (31, S, A, B, 150, STWCX_ONE), 'asm': None }),
  ("stwu",    {'binary': (37, S, A, d), 'asm': None }),
  ("stwux",   {'binary': (31, S, A, B, 183, 0), 'asm': None }),
  ("stwx",    {'binary': (31, S, A, B, 151, 0), 'asm': None }),
  ("subfx",   {'binary': (31, D, A, B, OE, 40, Rc), 'asm': None }),
  ("subfcx",  {'binary': (31, D, A, B, OE, 8, Rc), 'asm': None }),
  ("subfex",  {'binary': (31, D, A, B, OE, 136, Rc), 'asm': None }),
  ("subfic",  {'binary': (8 , D, A, SIMM), 'asm': None }),
  ("subfmex", {'binary': (31, D, A, 0, 0, 0, 0, 0, OE, 232, Rc), 'asm': None }),
  ("subfzex", {'binary': (31, D, A, 0, 0, 0, 0, 0, OE, 200, Rc), 'asm': None }),
  ("sync",    {'binary': (31, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 598, 0), 'asm': None }),
  ("tlbia",   {'binary': (31, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 370, 0), 'asm': None }),
  ("tlbie",   {'binary': (31, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, B, 306, 0), 'asm': None }),
  ("tlbsync1",{'binary': (31, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 566, 0), 'asm': None }),
  ("tw",      {'binary': (31, TO, A, B, 4, 0), 'asm': None }),
  ("twi",     {'binary': (3,  TO, A, SIMM), 'asm': None }),
  ("xorx",    {'binary': (31, S, A, B, 316, Rc), 'asm': ASM_ASB }),
  ("xori",    {'binary': (26, S, A, UIMM), 'asm': ASM_ASUm }),
  ("xoris",   {'binary': (27, S, A, UIMM), 'asm': ASM_ASUm })
  )


# Create the Instructions
opcodes = {0:OPCD, 21:XO_1, 22:XO_2, 26:XO_3} # bit position: extended opcode
SynthesizeISA(PPC_ISA, globals(), opcodes)

