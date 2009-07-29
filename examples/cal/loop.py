import corepy.lib.extarray as extarray
import corepy.arch.cal.isa as cal
import corepy.arch.cal.types.registers as reg
import corepy.arch.cal.platform as env

import math
import ctypes

def generate(prgm):
  code = prgm.get_stream()
  cal.set_active_code(code)

  r_count = prgm.acquire_register()
  #r_cx = prgm.acquire_register()
  #r_cy = prgm.acquire_register()
  r_sum = prgm.acquire_register()
  r_limit = prgm.acquire_register((64.0,) * 4)
  r_cmp = prgm.acquire_register()

  cal.dcl_output(reg.o0, USAGE=cal.usage.generic)

  cal.mov(r_count, r_count('0000'))
  #cal.mov(r_cx, r_cx('0000'))
  #cal.mov(r_cy, r_cy('0000'))
  cal.mov(r_sum, r_sum('0000'))

  cal.whileloop()
  #cal.ge(r_cmp, r_count.x, r_limit)
  #cal.break_logicalnz(r_cmp)
  cal.breakc(cal.relop.ge, r_count.x, r_limit)

  cal.mov(r_count, r_count('x0zw'))

  cal.whileloop()
  #cal.ge(r_cmp, r_count.y, r_limit)
  #cal.break_logicalnz(r_cmp)
  cal.breakc(cal.relop.ge, r_count.y, r_limit)

  cal.add(r_sum, r_sum, r_sum('1111'))

  cal.add(r_count, r_count, r_count('0100'))
  cal.endloop()

  cal.add(r_count, r_count, r_count('1000'))
  cal.endloop()
  
  cal.mov(reg.o0, r_sum)

  return code

if __name__ == '__main__':
  SIZE = 64

  prgm = env.Program()
  prgm.add(generate(prgm))

  proc = env.Processor(0)

  out = proc.alloc_remote('f', 4, SIZE, SIZE)
  prgm.set_binding(reg.o0, out)

  proc.execute(prgm, (0, 0, SIZE, SIZE))

  print out
  prgm.print_code()

