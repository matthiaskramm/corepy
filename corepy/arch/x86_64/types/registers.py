
# Copyright 2006-2007 The Trustees of Indiana University.

# This software is available for evaluation purposes only.  It may not be
# redistirubted or used for any other purposes without express written
# permission from the authors.

# Authors:
#   Andrew Friedley     (afriedle@cs.indiana.edu)
#   Christopher Mueller (chemuell@cs.indiana.edu)
#   Andrew Lumsdaine    (lums@cs.indiana.edu)

import corepy.spre.spe as spe


# ------------------------------
# Registers
# ------------------------------

class x86_64Register(spe.Register):
  def __init__(self, reg, rex, name = None):
    self.rex = rex
    spe.Register.__init__(self, reg, name = name)
    return

  def __eq__(self, other):
    return type(self) == type(other) and self.reg == other.reg and self.rex == other.rex and self.name == other.name

class GPRegister8(x86_64Register): pass
class GPRegister16(x86_64Register): pass
class GPRegister32(x86_64Register): pass
class GPRegister64(x86_64Register): pass
class FPRegister(x86_64Register): pass
class MMXRegister(x86_64Register): pass
class XMMRegister(x86_64Register): pass
class IPRegister(x86_64Register): pass


# Set up an instance for each register
# TODO - ah-dh registers are accessible only w/o a REX,
# and sil/dil/bpl/spl are accessible only WITH a REX.  How do I enforce this?
# encoding wise, a REX value of 0x40 is needed for si/dil/bpl.

# Use a bool for indicating whether an REX prefix is needed.  This doesn't mean
# there are bits in the REX to set, just that one is needed.  Render methods
# can then conditionally add the REX prefix based on this boolean.  Registers
# can use their 4-bit values, but still in the comparison checks for gp8 the
# REX boolean will need to be considered.
# Is there a way to do this without the REX prefix?
# Slightly cleaner might be to use 0/1/None, with the same idea above.
# 0 or 1 indicates the value of the bit to go in the REX prefix.  This
# simplifies the bitwise logic to create the REX prefix, no need to extract the
# bit from the register value.  Would then need to always compare the REX bit
# when doing register comparisons.

# All the GP and XMM regs need to have a rex field, FP and MMX do not.

gp8 =  ((0, 0, "al"),   (3, 0, "bl"),   (1, 0, "cl"),   (2, 0, "dl"),
        (4, 0, "ah"),   (7, 0, "bh"),   (5, 0, "ch"),   (6, 0, "dh"),
        (6, 0, "sil"),  (7, 0, "dil"),  (5, 0, "bpl"),  (4, 0, "spl"),
        (0, 1, "r8b"),  (1, 1, "r9b"),  (2, 1, "r10b"), (3, 1, "r11b"),
        (4, 1, "r12b"), (5, 1, "r13b"), (6, 1, "r14b"), (7, 1, "r15b"))
gp16 = ((0, 0, "ax"),   (3, 0, "bx"),   (1, 0, "cx"),   (2, 0, "dx"),
        (4, 0, "sp"),   (5, 0, "bp"),   (6, 0, "si"),   (7, 0, "di"),
        (0, 1, "r8w"),  (1, 1, "r9w"),  (2, 1, "r10w"), (3, 1, "r11w"),
        (4, 1, "r12w"), (5, 1, "r13w"), (6, 1, "r14w"), (7, 1, "r15w"))
gp32 = ((0, 0, "eax"),  (3, 0, "ebx"),  (1, 0, "ecx"),  (2, 0, "edx"),
        (4, 0, "esp"),  (5, 0, "ebp"),  (6, 0, "esi"),  (7, 0, "edi"),
        (0, 1, "r8d"),  (1, 1, "r9d"),  (2, 1, "r10d"), (3, 1, "r11d"),
        (4, 1, "r12d"), (5, 1, "r13d"), (6, 1, "r14d"), (7, 1, "r15d"))
gp64 = ((0, 0, "rax"),  (3, 0, "rbx"),  (1, 0, "rcx"),  (2, 0, "rdx"),
        (4, 0, "rsp"),  (5, 0, "rbp"),  (6, 0, "rsi"),  (7, 0, "rdi"),
        (0, 1, "r8"),   (1, 1, "r9"),   (2, 1, "r10"),  (3, 1, "r11"),
        (4, 1, "r12"),  (5, 1, "r13"),  (6, 1, "r14"),  (7, 1, "r15"))

gp8_array = []
gp16_array = []
gp32_array = []
gp64_array = []
st_array = []
mm_array = []
xmm_array = []

# Set up RIP register.  This register is only useable with a displacement in a memory
# reference, and nowhere else.
globals()["rip"] = IPRegister(8, 1, "rip")
globals()["eip"] = IPRegister(8, 0, "eip")

# Set up GP registers
for (regs, cls, arr) in ((gp8, GPRegister8, gp8_array), (gp16, GPRegister16, gp16_array), (gp32, GPRegister32, gp32_array), (gp64, GPRegister64, gp64_array)):
  for (reg, rex, name) in regs:
    globals()[name] = cls(reg, rex, name = name)
    arr.append(globals()[name])

# Set up x87, MMX, and SSE registers
for i in range(0, 8):
  stri = str(i)

  name = "st" + stri
  globals()[name] = FPRegister(i, None, name = name)
  st_array.append(globals()[name])

  name = "mm" + stri
  globals()[name] = MMXRegister(i, None, name = name)
  mm_array.append(globals()[name])

  name = "xmm" + stri
  globals()[name] = XMMRegister(i, 0, name = name)
  xmm_array.append(globals()[name])

  # Set up 8 more SSE registers, with REX = 1
  name = "xmm" + str(i + 8)
  globals()[name] = XMMRegister(i, 1, name = name)
  xmm_array.append(globals()[name])
  
