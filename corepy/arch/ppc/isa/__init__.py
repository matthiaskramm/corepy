
# import platform_conf
import ppc_isa as machine
import corepy.spre.spe as spe

# Nothing to see here, move along... ;)
__active_code = None

def set_active_code(code):
  global __active_code

  if __active_code is not None:
    __active_code.set_active_callback(None)

  __active_code = code

  if code is not None:
    code.set_active_callback(set_active_code)
  return

# Property version
def __get_active_code(self):
  global __active_code
  return __active_code

# Free function version
def get_active_code():
  global __active_code
  return __active_code

# _ppc_active_code_prop = property(get_active_code)

# Build the instructions
for inst in ppc_isa.PPC_ISA:
  name = inst[0]
  machine_inst = getattr(machine, name)
  
  # asm_order = inst[1]['asm']
  members = {}
  for key in inst[1].keys():
    members[key] = inst[1][key]

  members['asm_order'] =  members['asm']
  members['machine_inst'] =  machine_inst
  members['active_code']  = property(__get_active_code) # _ppc_active_code_prop
  globals()[inst[0]] = type(name, (spe.Instruction,), members)
                                                       
# ------------------------------
# Mnemonics
# ------------------------------

# TODO: Find a better place for these...
def add(D, A, SIMM): return addx(D, A, SIMM, 0, 0)
def b(LI):   return bx(LI, 0, 0)
def ba(LI):   return bx(LI, 1, 0)
def bdnz(BD): return bcx(0x10, 0, BD, 0, 0)
def bgt(BD):  return bcx(0x0D, 1, BD, 0, 0)   # bo = 011zy -> 01101 branch if true (> 0), likely to be taken
def blt(BD):  return bcx(0x0D, 0, BD, 0, 0)   # bo = 011zy -> 01101 branch if true (> 0), likely to be taken
def bne(BD):  return bcx(4, 2, BD, 0, 0)
def beq(BD):  return bcx(12, 2, BD, 0, 0)
def cmpw(crfD, A, B): return cmp_(crfD, 0, A, B)
def divw(D, A, B): return divwx(D, A, B, 0, 0)
def li(D, SIMM): return addi(D, 0, SIMM)
def mftbl(D): return mftb(D, 268)
def mftbu(D): return mftb(D, 269)
def mullw(D, A, B): return mullwx(D, A, B, 0, 0)
def mtctr(S): return mtspr(9, S)
def mtvrsave(S): return mtspr(256, S)
def mfvrsave(S): return mfspr(S, 256)
def noop(): return ori(0,0,0) # preferred PPC noop (CWG p14)
def subf(D, A, B): return subfx(D, A, B, 0, 0)

def Illegal(): return 0;
# def blr(): return (19 << 26) | (20 << 21) | (0 << 16) | (0 << 11) | (16 << 1)
def blr(): return bclrx(20, 0, 0)

# Copyright 2006 The Trustees of Indiana University.

# This software is available for evaluation purposes only.  It may not be
# redistirubted or used for any other purposes without express written
# permission from the authors.

# Authors:
#   Christopher Mueller (chemuell@cs.indiana.edu)
#   Andrew Lumsdaine    (lums@cs.indiana.edu)

