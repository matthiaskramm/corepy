# Copyright (c) 2006-2008 The Trustees of Indiana University.                   
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

# SPU Log2
import corepy.arch.spu.platform as synspu
import corepy.arch.spu.isa as spu
import corepy.arch.spu.types.spu_types as var
import corepy.arch.spu.lib.dma as dma
from corepy.arch.spu.lib.iterators import syn_iter

constants = {
  'LOG2EA':  0x3EE2A8ED,  # 4.4269505143E-01
  'SQRTH':   0x3F3504F3,  # 7.0710676908E-01
  'MINLOGF': 0xC2B17218,  # -8.8722839355E+01
  'LOGE2F':  0x3F317218,  # 6.9314718246E-01
  'SQRTHF':  0x3F3504F3,  # 7.0710676908E-01
  'C1':  0x3D9021BB,  # 7.0376835763E-02
  'C2':  0x3DEBD1B8,  # 1.1514610052E-01
  'C3':  0x3DEF251A,  # 1.1676998436E-01
  'C4':  0x3DFE5D4F,  # 1.2420140952E-01
  'C5':  0x3E11E9BF,  # 1.4249323308E-01
  'C6':  0x3E2AAE50,  # 1.6668057442E-01
  'C7':  0x3E4CCEAC,  # 2.0000714064E-01
  'C8':  0x3E7FFFFC,  # 2.4999994040E-01
  'C9':  0x3EAAAAAA,  # 3.3333331347E-01
  'C10': 0xBF000000,  # -5.0000000000E-01
  'M1':  0x807FFFFF,  # Mantissa extract masks
  'M2':  0x3F000000,  #  ""   ""
  'ONE': 0x3F800000,  # 1.0
  '_23': 0x17         # 23
}


class SPULog:
  """
  Compute log2f(x).

  Based on netlib log2f.
  """

  def __init__(self):
    self.consts = None
    self.x = None
    self.result = None
    return

  def set_x(self, x): self.x = x
  def set_result(self, r): self.result = r

  def setup(self, code):
    old_code = spu.get_active_code()
    spu.set_active_code(code)
    
    self.consts = {}
    for const in constants.keys():
      self.consts[const] = var.Word(constants[const])

    spu.set_active_code(old_code)
    return

  def synthesize(self, code):
    old_code = spu.get_active_code()
    spu.set_active_code(code)
    
    if self.x is None: raise Exception("Please set x")
    if self.result is None: raise Exception("Please set result")

    # exponent
    e = var.Word()
    
    # Working values    
    x = var.Word()
    y = var.Word()
    z = var.Word()

    cmp = var.Bits()
    tmp = var.Word()

    spu.xor(cmp, cmp, cmp)
    spu.xor(tmp, tmp, tmp)    

    # Set the working x
    x.v = self.x

    # Extract the exponent
    # int e = (((*(unsigned int *) &x) >> 23) & 0xff) - 0x7e;
    e.v = x >> self.consts['_23']
    e.v = spu.andi.ex(e, 0xff)
    e.v = spu.ai.ex(e, 0x382) # 0x382 == (- 0x7E) using 10 bits
    # 0b 111 1110

    # Extract the mantissa
    x.v = x & self.consts['M1'] # *(unsigned int*)&x &= 0x807fffff;
    x.v = x | self.consts['M2'] # *(unsigned int*)&x |= 0x3f000000;

    # Normalize
    x1, x2, e1 = y, z, tmp
    
    # if (x < SQRTHF) 
    cmp.v = spu.fcgt.ex(self.consts['SQRTHF'], x)

    # (True) { ... }
    e1.v = spu.ai.ex(e, -1)                  #   e -= 1;
    x1.v = spu.fa.ex(x, x)                   #   x = x + x - 1.0;
    x1.v = spu.fs.ex(x1, self.consts['ONE']) #     ""  ""

    # (False) { ... }
    x2.v = spu.fs.ex(x, self.consts['ONE'])  #   x = x - 1.0;

    # Select the True/False values based on cmp
    e.v = spu.selb.ex(e,  e1, cmp)
    x.v = spu.selb.ex(x2, x1, cmp)

    # Compute polynomial
    z.v = spu.fm.ex(x, x)                      #  z = x * x;
    
    y.v = spu.fms.ex(self.consts['C1'], x,     #  y = (((((((( 7.0376836292E-2 * x  
                     self.consts['C2'])        #	       - 1.1514610310E-1) * x      
    y.v = spu.fma.ex(y, x, self.consts['C3'])  #	     + 1.1676998740E-1) * x        
    y.v = spu.fms.ex(y, x, self.consts['C4'])  #	    - 1.2420140846E-1) * x         
    y.v = spu.fma.ex(y, x, self.consts['C5'])  #	   + 1.4249322787E-1) * x          
    y.v = spu.fms.ex(y, x, self.consts['C6'])  #	  - 1.6668057665E-1) * x           
    y.v = spu.fma.ex(y, x, self.consts['C7'])  #	 + 2.0000714765E-1) * x            
    y.v = spu.fms.ex(y, x, self.consts['C8'])  #	- 2.4999993993E-1) * x             
    y.v = spu.fma.ex(y, x, self.consts['C9'])  #       + 3.3333331174E-1) 
    y.v = spu.fm.ex(y, x)                      #   * x 
    y.v = spu.fm.ex(y, z)                      #   * z;   
    
    y.v = spu.fma.ex(self.consts['C10'], z, y) #  y += -0.5 * z;

    # Convert to log base 2
    z.v = spu.fm.ex( y, self.consts['LOG2EA'])     # z = y * LOG2EA;
    z.v = spu.fma.ex(x, self.consts['LOG2EA'], z)  # z += x * LOG2EA;
    z.v = spu.fa.ex(z, y)                          # z += y;
    z.v = spu.fa.ex(z, x)                          # z += x;
    e.v = spu.csflt.ex(e, 155)                     # z += (float) e;
    z.v = spu.fa.ex(z, e)                          #  ""  ""
    
    spu.ai(self.result, z, 0)       # return z

    spu.set_active_code(old_code)
    return

  def cleanup(self, code):
    for const in constants.values():
      code.release_register(const.reg)
    return



def TestLog():
  code = synspu.InstructionStream()
  proc = synspu.Processor()

  spu.set_active_code(code)
  # Create a simple SPU program that computes log for 10 values and
  # sends the result back using the mailbox

  log = SPULog()
  
  values = []
  result = code.acquire_register()

  N = 10
  
  x = 1
  for i in range(N):
    val = var.Word(x)
    spu.cuflt(val, val, 155)
    values.append(val)
    x = x * 10
    
  log.setup(code)
  log.set_result(result)

  for i in range(N):

    log.set_x(values[i])
    log.synthesize(code)

    spu.wrch(result, dma.SPU_WrOutMbox)
    
  spe_id = proc.execute(code, mode = 'async')

  x = 1
  for i in range(N):
    while synspu.spu_exec.stat_out_mbox(spe_id) == 0: pass
    print 'log said: 0x%08X  (%d)' %(synspu.spu_exec.read_out_mbox(spe_id), x)
    x = x * 10

  proc.join(spe_id)

  return


def _sp_to_float(x):
  """
  Convert a binary single precision floating point number to a Python float
  """
  # Extract the exponent
  # int e = (((*(unsigned int *) &x) >> 23) & 0xff) - 0x7e;
  
  # Extract the mantissa
  # x.v = x & self.consts['M1'] # *(unsigned int*)&x &= 0x807fffff;
  # x.v = x | self.consts['M2'] # *(unsigned int*)&x |= 0x3f000000;
  
  sign = (x >> 31) & 1
  # exp  = (x >> 23) & 0xFF - 126
  exp  = ((x >> 23) & 0xFF) - 127
  # print hex(x), ((x >> 23) & 0xFF) - 127
  m    = x & 0x7ffffff

  # print sign, exp, m, 1.0 + (1.0 / m)

  mv = 1.0
  p = 2.0
  for i in range(23):
    if (m >> (22 - i)) & 1:
      mv += (1.0 / p)
    p *= 2.0
    
  value = (2 ** (exp)) * mv # (1.0 + (1.0 / m))
  
  if sign == 1: 
    value = value * -1

  # print '%.12f' % value
  return value

# _sp_to_float(0x41200000)
# _sp_to_float(0xBEF0A3D7)

def TestFloats():
  import math
  
  code = synspu.InstructionStream()
  proc = synspu.Processor()

  spu.set_active_code(code)

  code.set_debug(True)
  
  # Create a simple SPU program that computes log for all values bettween
  # .01 and 10.0 with .01 increments

  start = .65
  stop  = .75
  inc   = .01

  sp_step = 0x3C23D70A
  # r_current = var.Word(0x3C23D70A) # .01 in single precision
  r_current = var.Word(0x3F266666)
  r_step  = var.Word(sp_step)    # .01 in single precision
  result  = var.Word(0)
  log = SPULog()

  log.setup(code)
  log.set_result(result)
  log.set_x(r_current)
  
  log_iter = syn_iter(code, int((stop - start) / inc))

  for i in log_iter:
    
    log.synthesize(code)
    spu.fa(r_current, r_current, r_step)
    spu.wrch(result, dma.SPU_WrOutMbox)

  # code.print_code()
  spe_id = proc.execute(code, mode = 'async')

  x = start
  for i in range(int((stop - start) / inc)):
    while synspu.spu_exec.stat_out_mbox(spe_id) == 0: pass
    slog = synspu.spu_exec.read_out_mbox(spe_id)
    print '%.3f 0x%08X  %.08f %.08f ' % (x, slog, _sp_to_float(slog), math.log(x, 2))
    x += inc

  proc.join(spe_id)

  return


if __name__=='__main__':
  # TestLog()
  TestFloats()
  pass
