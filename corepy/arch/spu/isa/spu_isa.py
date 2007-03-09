# Copyright 2006-2007 The Trustees of Indiana University.

# This software is available for evaluation purposes only.  It may not be
# redistirubted or used for any other purposes without express written
# permission from the authors.

# Authors:
#   Christopher Mueller (chemuell@cs.indiana.edu)
#   Andrew Lumsdaine    (lums@cs.indiana.edu)


from corepy.spre.isa_syn import *

__doc__="""
ISA for the Cell Broadband Engine's SPU.
"""

# machine = {} # name: MachineInstruction

# ------------------------------
# PPC Fields
# ------------------------------

MASK_2  = 0x3;   # 0011
MASK_7  = 0x7F;  # 0111 1111
MASK_8  = 0xFFF; # 1111 1111
MASK_10 = 0x3FF; # 0011 1111 1111
MASK_16 = 0xFFFF; # 1111 1111 1111 1111
MASK_18 = 0x3FFFF; # 0011 1111 1111 1111 1111

Fields = (
  # Shorthand for GP/FP registers
  ("A",  (Field, (18,24))),
  ("B",  (Field, (11,17))),
  ("C",  (Field, (4,10))),
  ("T",  (Field, (25,31))),

  # Other fields
  ("OPRR", (Opcode, (0,10))),
  ("OPRRR", (Opcode, (0,3))),
  ("OPI7", (Opcode, (0,10))),
  ("OPI8", (Opcode, (0,9))),  
  ("OPI10", (Opcode, (0,7))),
  ("OPI16", (Opcode, (0,8))),
  ("OPI18", (Opcode, (0,6))),
  ("RA",  (Field, (18,24))),
  ("RB",  (Field, (11,17))),
  ("RC",  (Field, (25,31))),
  ("RT",  (Field, (25,31))),

  # RT for RRR instructions
  ("RRR_RT",  (Field, (4,10))),  

  ("CA",  (Field, (18,24))),
  ("SA",  (Field, (18,24))),    

  ("D",  (Field, (12), 0)),
  ("E",  (Field, (13), 0)),

  ("P",  (Field, (11), 0)),

  ("ROH",  (MaskedField, (16,17), MASK_2 )),
  ("ROHA",  (MaskedField, (7,8), MASK_2 )),        
  ("ROL",  (Field, (25,31))),

  ("STOP_SIG",  (Field, (18,31))),

  ("_C", (Field, (11), 0)),
  
  ("I7",  (MaskedField, (11,17), MASK_7)),
  ("I8",  (MaskedField, (10,17), MASK_8)),  
  ("I10",  (MaskedField, (8,17), MASK_10)),
  ("I16",  (MaskedField, (9,24), MASK_16)),  
  ("I18",  (MaskedField, (7,24), MASK_18)),  
  )

# Reserved fields
R_RA = '0000000'
R_RB = '0000000'
R_RT = '0000000'

# Create the Field objects
SynthesizeFields(Fields, globals())

def RR(op):
  return (OPRR(BinToDec(op)),RB, RA, RT)

def RRR(op):
  return (OPRRR(BinToDec(op)), RRR_RT, RB, RA, RC)

def RI7(op):
  return (OPI7(BinToDec(op)), I7, RA, RT)

def RI10(op):
  return (OPI10(BinToDec(op)), I10, RA, RT)

def RI16(op):
  return (OPI16(BinToDec(op)), I16, RT)

def RI18(op):
  return (OPI18(BinToDec(op)), I18, RT)


# Common machine->assembly mappings
ASM_RR = (RT, RA, RB)
ASM_XR = (RT, RA) # Null RB
ASM_XX = (RT, )   # Null RB, RA
ASM_RR_Branch = (RT, RA)
ASM_RR_Channel = (RT, CA)
ASM_RRR = (RRR_RT, RA, RB, RC)

ASM_I7 = (RT, RA, I7)
ASM_I8 = (RT, RA, I8)
ASM_I10 = (RT, RA, I10)
ASM_I16 = (RT, I16)
ASM_XI16 = (I16,) # Null RT
ASM_I18 = (RT, I18)

# SPU Instruction Metadata
SPU_ISA = (
  # OPRR
  ('lqx', {'binary': (OPRR(BinToDec('00111000100')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (1, 6, 0) }),
  ('stqx', {'binary': (OPRR(BinToDec('00101000100')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (1, 6, 0) }),
  ('cbx', {'binary': (OPRR(BinToDec('00111010100')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (1, 4, 0) }),
  ('chx', {'binary': (OPRR(BinToDec('00111010101')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (1, 4, 0) }),
  ('cwx', {'binary': (OPRR(BinToDec('00111010110')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (1, 4, 0) }),
  ('cdx', {'binary': (OPRR(BinToDec('00111010111')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (1, 4, 0) }),
  ('ah', {'binary': (OPRR(BinToDec('00011001000')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 2, 0) }),
  ('a', {'binary': (OPRR(BinToDec('00011000000')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 2, 0) }),
  ('sfh', {'binary': (OPRR(BinToDec('00001001000')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 2, 0) }),
  ('sf', {'binary': (OPRR(BinToDec('00001000000')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 2, 0) }),
  ('addx', {'binary': (OPRR(BinToDec('01101000000')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 2, 0) }),
  ('cg', {'binary': (OPRR(BinToDec('00011000010')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 2, 0) }),
  ('cgx', {'binary': (OPRR(BinToDec('01101000010')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 2, 0) }),
  ('sfx', {'binary': (OPRR(BinToDec('01101000001')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 2, 0) }),
  ('bg', {'binary': (OPRR(BinToDec('00001000010')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 2, 0) }),
  ('bgx', {'binary': (OPRR(BinToDec('01101000011')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 2, 0) }),
  ('mpy', {'binary': (OPRR(BinToDec('01111000100')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 7, 0) }),
  ('mpyu', {'binary': (OPRR(BinToDec('01111001100')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 7, 0) }),
  ('mpyh', {'binary': (OPRR(BinToDec('01111000101')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 7, 0) }),
  ('mpys', {'binary': (OPRR(BinToDec('01111000111')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 7, 0) }),
  ('mpyhh', {'binary': (OPRR(BinToDec('01111000110')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 7, 0) }),
  ('mpyhha', {'binary': (OPRR(BinToDec('01101000110')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 7, 0) }),
  ('mpyhhu', {'binary': (OPRR(BinToDec('01111001110')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 7, 0) }),
  ('mpyhhau', {'binary': (OPRR(BinToDec('01101001110')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 7, 0) }),
  ('clz', {'binary': (OPRR(BinToDec('01010100101')), R_RB, RA, RT), 'asm': ASM_XR, 'cycles': (0, 2, 0) }),
  ('cntb', {'binary': (OPRR(BinToDec('01010110100')), R_RB, RA, RT), 'asm': ASM_XR, 'cycles': (0, 4, 0) }),
  ('fsmb', {'binary': (OPRR(BinToDec('00110110110')), R_RB, RA, RT), 'asm': ASM_XR, 'cycles': (1, 4, 0) }),
  ('fsmh', {'binary': (OPRR(BinToDec('00110110101')), R_RB, RA, RT), 'asm': ASM_XR, 'cycles': (1, 4, 0) }),
  ('fsm', {'binary': (OPRR(BinToDec('00110110100')), R_RB, RA, RT), 'asm': ASM_XR, 'cycles': (1, 4, 0) }),
  ('gbb', {'binary': (OPRR(BinToDec('00110110010')), R_RB, RA, RT), 'asm': ASM_XR, 'cycles': (1, 4, 0) }),
  ('gbh', {'binary': (OPRR(BinToDec('00110110001')), R_RB, RA, RT), 'asm': ASM_XR, 'cycles': (1, 4, 0) }),
  ('gb', {'binary': (OPRR(BinToDec('00110110000')), R_RB, RA, RT), 'asm': ASM_XR, 'cycles': (1, 4, 0) }),
  ('avgb', {'binary': (OPRR(BinToDec('00011010011')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 4, 0) }),
  ('absdb', {'binary': (OPRR(BinToDec('00001010011')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 4, 0) }),
  ('sumb', {'binary': (OPRR(BinToDec('01001010011')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 4, 0) }),
  ('xsbh', {'binary': (OPRR(BinToDec('01010110110')), R_RB, RA, RT), 'asm': ASM_XR, 'cycles': (0, 2, 0) }),
  ('xshw', {'binary': (OPRR(BinToDec('01010101110')), R_RB, RA, RT), 'asm': ASM_XR, 'cycles': (0, 2, 0) }),
  ('xswd', {'binary': (OPRR(BinToDec('01010100110')), R_RB, RA, RT), 'asm': ASM_XR, 'cycles': (0, 2, 0) }),
  ('and_', {'binary': (OPRR(BinToDec('00011000001')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 2, 0) }),
  ('andc', {'binary': (OPRR(BinToDec('01011000001')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 2, 0) }),
  ('or_', {'binary': (OPRR(BinToDec('00001000001')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 2, 0) }),
  ('orc', {'binary': (OPRR(BinToDec('01011001001')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 2, 0) }),
  ('orx', {'binary': (OPRR(BinToDec('00111110000')), R_RB, RA, RT), 'asm': ASM_XR, 'cycles': (1, 4, 0) }),
  ('xor', {'binary': (OPRR(BinToDec('01001000001')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 2, 0) }),
  ('nand', {'binary': (OPRR(BinToDec('00011001001')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 2, 0) }),
  ('nor', {'binary': (OPRR(BinToDec('00001001001')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 2, 0) }),
  ('eqv', {'binary': (OPRR(BinToDec('01001001001')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 2, 0) }),
  ('shlh', {'binary': (OPRR(BinToDec('00001011111')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 4, 0) }),
  ('shl', {'binary': (OPRR(BinToDec('00001011011')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 4, 0) }),
  ('shlqbi', {'binary': (OPRR(BinToDec('00111011011')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (1, 4, 0) }),
  ('shlqby', {'binary': (OPRR(BinToDec('00111011111')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (1, 4, 0) }),
  ('shlqbybi', {'binary': (OPRR(BinToDec('00111001111')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (1, 4, 0) }),
  ('roth', {'binary': (OPRR(BinToDec('00001011100')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 4, 0) }),
  ('rot', {'binary': (OPRR(BinToDec('00001011000')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 4, 0) }),
  ('rotqby', {'binary': (OPRR(BinToDec('00111011100')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (1, 4, 0) }),
  ('rotqbybi', {'binary': (OPRR(BinToDec('00111001100')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (1, 4, 0) }),
  ('rotqbi', {'binary': (OPRR(BinToDec('00111011000')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (1, 4, 0) }),
  ('rothm', {'binary': (OPRR(BinToDec('00001011101')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 4, 0) }),
  ('rotm', {'binary': (OPRR(BinToDec('00001011001')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 4, 0) }),
  ('rotqmby', {'binary': (OPRR(BinToDec('00111011101')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (1, 4, 0) }),
  ('rotqmbybi', {'binary': (OPRR(BinToDec('00111001101')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (1, 4, 0) }),
  ('rotqmbi', {'binary': (OPRR(BinToDec('00111011001')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (1, 4, 0) }),
  ('rotmah', {'binary': (OPRR(BinToDec('00001011110')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 4, 0) }),
  ('rotma', {'binary': (OPRR(BinToDec('00001011010')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 4, 0) }),
  ('heq', {'binary': (OPRR(BinToDec('01111011000')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 2, 0) }),
  ('hgt', {'binary': (OPRR(BinToDec('01001011000')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 2, 0) }),
  ('hlgt', {'binary': (OPRR(BinToDec('01011011000')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 2, 0) }),
  ('ceqb', {'binary': (OPRR(BinToDec('01111010000')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 2, 0) }),
  ('ceqh', {'binary': (OPRR(BinToDec('01111001000')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 2, 0) }),
  ('ceq', {'binary': (OPRR(BinToDec('01111000000')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 2, 0) }),
  ('cgtb', {'binary': (OPRR(BinToDec('01001010000')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 2, 0) }),
  ('cgth', {'binary': (OPRR(BinToDec('01001001000')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 2, 0) }),
  ('cgt', {'binary': (OPRR(BinToDec('01001000000')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 2, 0) }),
  ('clgtb', {'binary': (OPRR(BinToDec('01011010000')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 2, 0) }),
  ('clgth', {'binary': (OPRR(BinToDec('01011001000')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 2, 0) }),
  ('clgt', {'binary': (OPRR(BinToDec('01011000000')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 2, 0) }),
  ('bi', {'binary': (OPRR(BinToDec('00110101000')), '0', D, E, '0000', RA, R_RT), 'asm': ASM_RR_Branch, 'cycles': (1, 4, 0) }),
  ('iret', {'binary': (OPRR(BinToDec('00110101010')), '0', D, E, '0000', RA, R_RT), 'asm': ASM_RR, 'cycles': (1, 4, 0) }),
  ('bisled', {'binary': (OPRR(BinToDec('00110101011')), '0', D, E, '0000', RA, RT), 'asm': ASM_RR_Branch, 'cycles': (1, 4, 0) }),
  ('bisl', {'binary': (OPRR(BinToDec('00110101001')), '0', D, E, '0000', RA, RT), 'asm': ASM_RR_Branch, 'cycles': (1, 4, 0) }),
  ('biz', {'binary': (OPRR(BinToDec('00100101000')), '0', D, E, '0000', RA, RT), 'asm': ASM_RR_Branch, 'cycles': (1, 4, 0) }),
  ('binz', {'binary': (OPRR(BinToDec('00100101001')), '0', D, E, '0000', RA, RT), 'asm': ASM_RR_Branch, 'cycles': (1, 4, 0) }),
  ('bihz', {'binary': (OPRR(BinToDec('00100101010')), '0', D, E, '0000', RA, RT), 'asm': ASM_RR_Branch, 'cycles': (1, 4, 0) }),
  ('bihnz', {'binary': (OPRR(BinToDec('00100101011')), '0', D, E, '0000', RA, RT), 'asm': ASM_RR_Branch, 'cycles': (1, 4, 0) }),
  ('hbr', {'binary': (OPRR(BinToDec('00110101100')), P, '0000', ROH, RA, ROL), 'asm': None, 'cycles': (1, 15, 0) }),
  ('fa', {'binary': (OPRR(BinToDec('01011000100')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 6, 0) }),
  ('dfa', {'binary': (OPRR(BinToDec('01011001100')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 13, 6) }),
  ('fs', {'binary': (OPRR(BinToDec('01011000101')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 6, 0) }),
  ('dfs', {'binary': (OPRR(BinToDec('01011001101')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 13, 6) }),
  ('fm', {'binary': (OPRR(BinToDec('01011000110')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 6, 0) }),
  ('dfm', {'binary': (OPRR(BinToDec('01011001110')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 13, 6) }),
  ('dfma', {'binary': (OPRR(BinToDec('01101011100')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 13, 6) }),
  ('dfnms', {'binary': (OPRR(BinToDec('01101011110')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 13, 6) }),
  ('dfms', {'binary': (OPRR(BinToDec('01101011101')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 13, 6) }),
  ('dfnma', {'binary': (OPRR(BinToDec('01101011111')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 13, 6) }),
  ('frest', {'binary': (OPRR(BinToDec('00110111000')), R_RB, RA, RT), 'asm': ASM_XR, 'cycles': (1, 4, 0) }),
  ('frsqest', {'binary': (OPRR(BinToDec('00110111001')), R_RB, RA, RT), 'asm': ASM_XR, 'cycles': (1, 4, 0) }),
  ('fi', {'binary': (OPRR(BinToDec('01111010100')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 7, 0) }),
  ('frds', {'binary': (OPRR(BinToDec('01110111001')), R_RB, RA, RT), 'asm': ASM_XR, 'cycles': (0, 13, 6) }),
  ('fesd', {'binary': (OPRR(BinToDec('01110111000')), R_RB, RA, RT), 'asm': ASM_XR, 'cycles': (0, 13, 6) }),
  ('fceq', {'binary': (OPRR(BinToDec('01111000010')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 2, 0) }),
  ('fcmeq', {'binary': (OPRR(BinToDec('01111001010')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 2, 0) }),
  ('fcgt', {'binary': (OPRR(BinToDec('01011000010')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 2, 0) }),
  ('fcmgt', {'binary': (OPRR(BinToDec('01011001010')), RB, RA, RT), 'asm': ASM_RR, 'cycles': (0, 2, 0) }),
  ('fscrwr', {'binary': (OPRR(BinToDec('01110111010')), R_RB, RA, RT), 'asm': ASM_XR, 'cycles': (0, 7, 0) }),
  ('fscrrd', {'binary': (OPRR(BinToDec('01110011000')), R_RB, R_RA, RT), 'asm': ASM_XX, 'cycles': (0, 13, 6) }),
  ('stop', {'binary': (OPRR(BinToDec('00000000000')), R_RB, STOP_SIG), 'asm': (STOP_SIG,), 'cycles': (1, 4, 0) }),
  ('stopd', {'binary': (OPRR(BinToDec('01010000000')), RB, RA, RC), 'asm': (RC, RA, RB), 'cycles': (1, 4, 0) }),
  ('lnop', {'binary': (OPRR(BinToDec('0000000001')), R_RB, R_RA, RT), 'asm': ASM_XX, 'cycles': (1, 0, 0) }),
  ('nop', {'binary': (OPRR(BinToDec('1000000001')),  R_RB, R_RA, RT), 'asm': ASM_XX, 'cycles': (0, 0, 0) }),
  ('sync', {'binary': (OPRR(BinToDec('0000000010')), _C, '00000000000000000000'), 'asm': None, 'cycles': (1, 4, 0) }),
  ('dsync', {'binary': (OPRR(BinToDec('0000000011')), R_RB, R_RA, R_RT), 'asm': None, 'cycles': (1, 4, 0) }),
  ('mfspr', {'binary': (OPRR(BinToDec('00000001100')), R_RB, SA, RT), 'asm': (RT, SA), 'cycles': (1, 6, 0) }),
  ('mtspr', {'binary': (OPRR(BinToDec('00100001100')), R_RB, SA, RT), 'asm': (SA, RT), 'cycles': (1, 6, 0) }),
  ('rdch', {'binary': (OPRR(BinToDec('00000001101')), R_RB, CA, RT), 'asm': ASM_RR_Channel, 'cycles': (1, 6, 0) }),
  ('rchcnt', {'binary': (OPRR(BinToDec('00000001111')), R_RB, CA, RT), 'asm': ASM_RR_Channel, 'cycles': (1, 6, 0) }),
  ('wrch', {'binary': (OPRR(BinToDec('00100001101')), R_RB, CA, RT), 'asm': ASM_RR_Channel, 'cycles': (1, 6, 0) }),
  
  # OPRRR
  ('mpya', {'binary': (OPRRR(BinToDec('1100')), RRR_RT, RB, RA, RC), 'asm': ASM_RRR, 'cycles': (0, 7, 0) }),
  ('selb', {'binary': (OPRRR(BinToDec('1000')), RRR_RT, RB, RA, RC), 'asm': ASM_RRR, 'cycles': (0, 2, 0) }),
  ('shufb', {'binary': (OPRRR(BinToDec('1011')), RRR_RT, RB, RA, RC), 'asm': ASM_RRR, 'cycles': (1, 4, 0) }),
  ('fma', {'binary': (OPRRR(BinToDec('1110')), RRR_RT, RB, RA, RC), 'asm': ASM_RRR, 'cycles': (0, 6, 0) }),
  ('fnms', {'binary': (OPRRR(BinToDec('1101')), RRR_RT, RB, RA, RC), 'asm': ASM_RRR, 'cycles': (0, 6, 0) }),
  ('fms', {'binary': (OPRRR(BinToDec('1111')), RRR_RT, RB, RA, RC), 'asm': ASM_RRR, 'cycles': (0, 6, 0) }),

  # OPI7
  ('cbd', {'binary': (OPI7(BinToDec('00111110100')), I7, RA, RT), 'asm': ASM_I7, 'cycles': (1, 4, 0) }),
  ('chd', {'binary': (OPI7(BinToDec('00111110101')), I7, RA, RT), 'asm': ASM_I7, 'cycles': (1, 4, 0) }),
  ('cwd', {'binary': (OPI7(BinToDec('00111110110')), I7, RA, RT), 'asm': ASM_I7, 'cycles': (1, 4, 0) }),
  ('cdd', {'binary': (OPI7(BinToDec('00111110111')), I7, RA, RT), 'asm': ASM_I7, 'cycles': (1, 4, 0) }),
  ('shlhi', {'binary': (OPI7(BinToDec('00001111111')), I7, RA, RT), 'asm': ASM_I7, 'cycles': (0, 4, 0) }),
  ('shli', {'binary': (OPI7(BinToDec('00001111011')), I7, RA, RT), 'asm': ASM_I7, 'cycles': (0, 4, 0) }),
  ('shlqbii', {'binary': (OPI7(BinToDec('00111111011')), I7, RA, RT), 'asm': ASM_I7, 'cycles': (1, 4, 0) }),
  ('shlqbyi', {'binary': (OPI7(BinToDec('00111111111')), I7, RA, RT), 'asm': ASM_I7, 'cycles': (1, 4, 0) }),
  ('rothi', {'binary': (OPI7(BinToDec('00001111100')), I7, RA, RT), 'asm': ASM_I7, 'cycles': (0, 4, 0) }),
  ('roti', {'binary': (OPI7(BinToDec('00001111000')), I7, RA, RT), 'asm': ASM_I7, 'cycles': (0, 4, 0) }),
  ('rotqbyi', {'binary': (OPI7(BinToDec('00111111100')), I7, RA, RT), 'asm': ASM_I7, 'cycles': (1, 4, 0) }),
  ('rotqbii', {'binary': (OPI7(BinToDec('00111111000')), I7, RA, RT), 'asm': ASM_I7, 'cycles': (1, 4, 0) }),
  ('rothmi', {'binary': (OPI7(BinToDec('00001111101')), I7, RA, RT), 'asm': ASM_I7, 'cycles': (0, 4, 0) }),
  ('rotmi', {'binary': (OPI7(BinToDec('00001111001')), I7, RA, RT), 'asm': ASM_I7, 'cycles': (0, 4, 0) }),
  ('rotqmbyi', {'binary': (OPI7(BinToDec('00111111101')), I7, RA, RT), 'asm': ASM_I7, 'cycles': (1, 4, 0) }),
  ('rotqmbii', {'binary': (OPI7(BinToDec('00111111001')), I7, RA, RT), 'asm': ASM_I7, 'cycles': (1, 4, 0) }),
  ('rotmahi', {'binary': (OPI7(BinToDec('00001111110')), I7, RA, RT), 'asm': ASM_I7, 'cycles': (0, 4, 0) }),
  ('rotmai', {'binary': (OPI7(BinToDec('00001111010')), I7, RA, RT), 'asm': ASM_I7, 'cycles': (0, 4, 0) }),

  # OPI8
  ('csflt', {'binary': (OPI8(BinToDec('0111011010')), I8, RA, RT), 'asm': ASM_I8, 'cycles': (0, 7, 0) }),
  ('cflts', {'binary': (OPI8(BinToDec('0111011000')), I8, RA, RT), 'asm': ASM_I8, 'cycles': (0, 7, 0) }),
  ('cuflt', {'binary': (OPI8(BinToDec('0111011011')), I8, RA, RT), 'asm': ASM_I8, 'cycles': (0, 7, 0) }),
  ('cfltu', {'binary': (OPI8(BinToDec('0111011001')), I8, RA, RT), 'asm': ASM_I8, 'cycles': (0, 7, 0) }),

  # OPI10
  ('lqd', {'binary': (OPI10(BinToDec('00110100')), I10, RA, RT), 'asm': ASM_I10, 'cycles': (1, 6, 0) }),
  ('stqd', {'binary': (OPI10(BinToDec('00100100')), I10, RA, RT), 'asm': ASM_I10, 'cycles': (1, 6, 0) }),
  ('ahi', {'binary': (OPI10(BinToDec('00011101')), I10, RA, RT), 'asm': ASM_I10, 'cycles': (0, 2, 0) }),
  ('ai', {'binary': (OPI10(BinToDec('00011100')), I10, RA, RT), 'asm': ASM_I10, 'cycles': (0, 2, 0) }),
  ('sfhi', {'binary': (OPI10(BinToDec('00001101')), I10, RA, RT), 'asm': ASM_I10, 'cycles': (0, 2, 0) }),
  ('sfi', {'binary': (OPI10(BinToDec('00001100')), I10, RA, RT), 'asm': ASM_I10, 'cycles': (0, 2, 0) }),
  ('mpyi', {'binary': (OPI10(BinToDec('01110100')), I10, RA, RT), 'asm': ASM_I10, 'cycles': (0, 7, 0) }),
  ('mpyui', {'binary': (OPI10(BinToDec('01110101')), I10, RA, RT), 'asm': ASM_I10, 'cycles': (0, 7, 0) }),
  ('andbi', {'binary': (OPI10(BinToDec('00010110')), I10, RA, RT), 'asm': ASM_I10, 'cycles': (0, 2, 0) }),
  ('andhi', {'binary': (OPI10(BinToDec('00010101')), I10, RA, RT), 'asm': ASM_I10, 'cycles': (0, 2, 0) }),
  ('andi', {'binary': (OPI10(BinToDec('00010100')), I10, RA, RT), 'asm': ASM_I10, 'cycles': (0, 2, 0) }),
  ('orbi', {'binary': (OPI10(BinToDec('00000110')), I10, RA, RT), 'asm': ASM_I10, 'cycles': (0, 2, 0) }),
  ('orhi', {'binary': (OPI10(BinToDec('00000101')), I10, RA, RT), 'asm': ASM_I10, 'cycles': (0, 2, 0) }),
  ('ori', {'binary': (OPI10(BinToDec('00000100')), I10, RA, RT), 'asm': ASM_I10, 'cycles': (0, 2, 0) }),
  ('xorbi', {'binary': (OPI10(BinToDec('01000110')), I10, RA, RT), 'asm': ASM_I10, 'cycles': (0, 2, 0) }),
  ('xorhi', {'binary': (OPI10(BinToDec('01000101')), I10, RA, RT), 'asm': ASM_I10, 'cycles': (0, 2, 0) }),
  ('xori', {'binary': (OPI10(BinToDec('01000100')), I10, RA, RT), 'asm': ASM_I10, 'cycles': (0, 2, 0) }),
  ('heqi', {'binary': (OPI10(BinToDec('01111111')), I10, RA, RT), 'asm': ASM_I10, 'cycles': (0, 2, 0) }),
  ('hgti', {'binary': (OPI10(BinToDec('01001111')), I10, RA, RT), 'asm': ASM_I10, 'cycles': (0, 2, 0) }),
  ('hlgti', {'binary': (OPI10(BinToDec('01011111')), I10, RA, RT), 'asm': ASM_I10, 'cycles': (0, 2, 0) }),
  ('ceqbi', {'binary': (OPI10(BinToDec('01111110')), I10, RA, RT), 'asm': ASM_I10, 'cycles': (0, 2, 0) }),
  ('ceqhi', {'binary': (OPI10(BinToDec('01111101')), I10, RA, RT), 'asm': ASM_I10, 'cycles': (0, 2, 0) }),
  ('ceqi', {'binary': (OPI10(BinToDec('01111100')), I10, RA, RT), 'asm': ASM_I10, 'cycles': (0, 2, 0) }),
  ('cgtbi', {'binary': (OPI10(BinToDec('01001110')), I10, RA, RT), 'asm': ASM_I10, 'cycles': (0, 2, 0) }),
  ('cgthi', {'binary': (OPI10(BinToDec('01001101')), I10, RA, RT), 'asm': ASM_I10, 'cycles': (0, 2, 0) }),
  ('cgti', {'binary': (OPI10(BinToDec('01001100')), I10, RA, RT), 'asm': ASM_I10, 'cycles': (0, 2, 0) }),
  ('clgtbi', {'binary': (OPI10(BinToDec('01011110')), I10, RA, RT), 'asm': ASM_I10, 'cycles': (0, 2, 0) }),
  ('clgthi', {'binary': (OPI10(BinToDec('01011101')), I10, RA, RT), 'asm': ASM_I10, 'cycles': (0, 2, 0) }),
  ('clgti', {'binary': (OPI10(BinToDec('01011100')), I10, RA, RT), 'asm': ASM_I10, 'cycles': (0, 2, 0) }),

  
  ('lqa', {'binary': (OPI16(BinToDec('001100001')), I16, RT), 'asm': ASM_I16, 'cycles': (1, 6, 0) }),
  ('lqr', {'binary': (OPI16(BinToDec('001100111')), I16, RT), 'asm': ASM_I16, 'cycles': (1, 6, 0) }),
  ('stqa', {'binary': (OPI16(BinToDec('001000001')), I16, RT), 'asm': ASM_I16, 'cycles': (1, 6, 0) }),
  ('stqr', {'binary': (OPI16(BinToDec('001000111')), I16, RT), 'asm': ASM_I16, 'cycles': (1, 6, 0) }),
  ('ilh', {'binary': (OPI16(BinToDec('010000011')), I16, RT), 'asm': ASM_I16, 'cycles': (0, 2, 0) }),
  ('ilhu', {'binary': (OPI16(BinToDec('010000010')), I16, RT), 'asm': ASM_I16, 'cycles': (0, 2, 0) }),
  ('il', {'binary': (OPI16(BinToDec('010000001')), I16, RT), 'asm': ASM_I16, 'cycles': (0, 2, 0) }),
  ('iohl', {'binary': (OPI16(BinToDec('011000001')), I16, RT), 'asm': ASM_I16, 'cycles': (0, 2, 0) }),
  ('fsmbi', {'binary': (OPI16(BinToDec('001100101')), I16, RT), 'asm': ASM_I16, 'cycles': (1, 4, 0) }),
  ('br', {'binary': (OPI16(BinToDec('001100100')), I16, R_RT), 'asm': ASM_XI16, 'cycles': (1, 4, 0) }),
  ('bra', {'binary': (OPI16(BinToDec('001100000')), I16, R_RT), 'asm': ASM_XI16, 'cycles': (1, 4, 0) }),
  ('brsl', {'binary': (OPI16(BinToDec('001100110')), I16, RT), 'asm': ASM_I16, 'cycles': (1, 4, 0) }),
  ('brasl', {'binary': (OPI16(BinToDec('001100010')), I16, RT), 'asm': ASM_I16, 'cycles': (1, 4, 0) }),
  ('brnz', {'binary': (OPI16(BinToDec('001000010')), I16, RT), 'asm': ASM_I16, 'cycles': (1, 4, 0) }),
  ('brz', {'binary': (OPI16(BinToDec('001000000')), I16, RT), 'asm': ASM_I16, 'cycles': (1, 4, 0) }),
  ('brhnz', {'binary': (OPI16(BinToDec('001000110')), I16, RT), 'asm': ASM_I16, 'cycles': (1, 4, 0) }),
  ('brhz', {'binary': (OPI16(BinToDec('001000100')), I16, RT), 'asm': ASM_I16, 'cycles': (1, 4, 0) }),
  ('hbra',  {'binary': (OPI18(BinToDec('0001000')), ROHA, I16, ROL), 'asm': None, 'cycles': (1, 15, 0) }),
  ('hbrr', {'binary': (OPI18(BinToDec('0001001')), ROHA, I16, ROL), 'asm': None, 'cycles': (1, 15, 0) }),
   
  ('ila', {'binary': (OPI18(BinToDec('0100001')), I18, RT), 'asm': ASM_I18, 'cycles': (0, 2, 0) })
)

SynthesizeISA(SPU_ISA, globals(), None)

