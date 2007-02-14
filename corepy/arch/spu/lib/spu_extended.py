
import corepy.arch.spu.isa as spu
import corepy.spre.spe as spe

__doc__="""
SPU Extended Instructions
"""

class SPUExt(spe.ExtendedInstruction):
  isa_module = spu

class shr(SPUExt):
  """
  Shift-right word.

  Shift the value in register a to the right by the number of bits
  specified by the value in register b.  Store the result in register
  d. 
  """
  def block(self, d, a, b):
    # Based on example on p133 of SPU ISA manual
    code = self.get_active_code()
    temp = code.acquire_register()
    spu.sfi(temp, b, 0)
    spu.rotm(d, a, temp)
    code.release_register(temp)
    return

class cneq(SPUExt):
  """
  Compare the word values in registers a and b.  If the operands are
  not equal, register d contains all ones.
  """
  def block(self, d, a, b):
    spu.ceq(d, a, a)
    spu.nor(d, d, d)
    return


class cge(SPUExt):
  """
  Word compare greater than equal.
  """

  def block(self, d, a, b):
    code = self.get_active_code()
    temp = code.acquire_register()
    spu.cgt(temp, a, b)
    spu.ceq(d, a, b)
    spu.or_(d, d, temp)
    code.release_register(temp)
    return

class cgei(SPUExt):
  """
  Word compare greater than equal immediate.
  """

  def block(self, d, a, b):
    code = self.get_active_code()
    temp = code.acquire_register()
    spu.cgti(temp, a, b)
    spu.ceqi(d, a, b)
    spu.or_(d, d, temp)
    code.release_register(temp)
    return


class lt(SPUExt):
  """
  Word compare less than
  """

  def block(self, d, a, b):
    spu.cgt(d, b, a)
    return

class lti(SPUExt):
  """
  Word compare less than
  """

  def block(self, d, a, b):
    temp = code.acquire_register()
    spu.cgti(temp, a, b)
    spu.ceqi(d, a, b)
    spu.nor(d, d, temp)
    code.release_register(temp)
    
    return

# ------------------------------------------------------------
# Unit Tests
# ------------------------------------------------------------

def TestAll():
  import corepy.arch.spu.platform as env

  code = env.InstructionStream()
  spu.set_active_code(code)

  a = code.acquire_register()
  b = code.acquire_register()
  c = code.acquire_register()
  
  srw(c, a, b)
  cneq(c, a, b)

  code.print_code()
  proc = env.Processor()
  proc.execute(code)
  
  return

if __name__=='__main__':
  TestAll()
