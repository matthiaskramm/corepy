# SPU Log2
import corepy.arch.spu.platform as synspu
import corepy.arch.spu.isa as spu
import corepy.arch.spu.types.spu_types as var
import corepy.arch.spu.lib.dma as dma

# Constants
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
    e.v = spu.cuflt.ex(e, 155)                     # z += (float) e;
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


if __name__=='__main__':
  TestLog()
  
  
