
def foo():
  return

import sys
sys.path.append('../..')


from ispu import ISPU
import corepy.arch.spu.isa as spu

cli = ISPU()

cli.start()

ia = 127
ib = 126

fa = 125
fb = 124

y0 = 120
y1 = 121
y2 = 122

t1 = 119

result  = 118

ione = 110
fone = 111

insts = [
  # Create fone = 1.0, fa = 2.0 and fb = 4.0
  spu.ai(ione, 0, 1),  
  spu.ai(ia, 0, 2),
  spu.ai(ib, 0, 4),
  spu.cuflt(fone, ione, 155),  
  spu.cuflt(fa, ia, 155),
  spu.cuflt(fb, ib, 155),

  # Compute 1/fb
  spu.frest(y0, fb),
  spu.fi(y1, fb, y0),
  spu.fnms(t1, fb, y1, fone),
  spu.fma(y2, t1, y1, y1),

  spu.fm(result, fa, y2)
  ]

for inst in insts:
  cli.execute(inst)

regs = cli.get_regs()

for reg in (ione, fone, ia, ib, fa, fb, y0, y1, y2, t1, result):
  print reg, '0x%08X 0x%08X 0x%08X 0x%08X' % regs[reg]

cli.stop()
