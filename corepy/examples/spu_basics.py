import corepy.arch.spu.isa as spu
from corepy.arch.spu.platform import InstructionStream, Processor
  
def TestInt():
  code = InstructionStream()
  proc = Processor()

  spu.set_active_code(code)
  
  r13 = code.acquire_register(reg = 13)
  r20 = code.acquire_register(reg = 20)
  spu.ai(r20, r20, 13)
  spu.ai(r13, r13, 13)
  spu.ai(r13, r13, 13)
  spu.ai(r13, r13, 13)
  spu.ai(r13, r13, 13)
  spu.ai(r13, r13, 13)
  
  spu.stop(0x200D)
  
  r = proc.execute(code) # , debug = True)
  print 'int result:', r
  assert(r == 13)

  # while True:
  #   pass
  return

if __name__=='__main__':
  TestInt()
