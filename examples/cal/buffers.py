import corepy.lib.extarray as extarray
import corepy.arch.cal.isa as cal
import corepy.arch.cal.types.registers as reg
import corepy.arch.cal.platform as env

def test_4comp():
  proc = env.Processor(0)
  code = env.InstructionStream()

  inp = proc.alloc_remote('i', 1, 4, 1)
  out = proc.alloc_remote('i', 4, 1, 1)

  for i in xrange(0, 4):
    inp[i] = i + 1
    out[i] = 0

  print "inp", inp
  print "out", out
  
  cal.set_active_code(code)

  cal.dcl_output(reg.o0, USAGE=cal.usage.generic)
  cal.dcl_resource(0, cal.pixtex_type.oned, cal.fmt.float, UNNORM=True) # positions

  r_cnt = code.acquire_register()
  r = code.acquire_registers(4)

  cal.mov(r_cnt, r_cnt('0000'))

  for i in xrange(0, 4):
    cal.sample(0, 0, r[i].x000, r_cnt.x)
    cal.add(r_cnt, r_cnt, r_cnt('1111'))

  cal.iadd(r[0], r[0], r[1]('0x00'))
  cal.iadd(r[0], r[0], r[2]('00x0'))
  cal.iadd(r[0], r[0], r[3]('000x'))
  cal.iadd(r[0], r[0], r[0])
  cal.mov(reg.o0, r[0])

  code.set_remote_binding(reg.i0, inp)
  code.set_remote_binding(reg.o0, out)

  code.cache_code()
  print code.render_string
  proc.execute(code, (0, 0, 1, 1))

  print "inp", inp
  print "out", out
  for i in xrange(0, 4):
    assert(out[i] == (i + 1) * 2)
  return


def test_1comp():
  proc = env.Processor(0)
  code = env.InstructionStream()

  inp = proc.alloc_remote('i', 4, 1, 1)
  out = proc.alloc_remote('i', 1, 4, 1)

  for i in xrange(0, 4):
    inp[i] = i + 1
    out[i] = 0

  print "inp", inp
  print "out", out
  
  cal.set_active_code(code)

  cal.dcl_output(reg.o0, USAGE=cal.usage.generic)
  cal.dcl_resource(0, cal.pixtex_type.oned, cal.fmt.float, UNNORM=True) # positions

  r = code.acquire_register()

  cal.sample(0, 0, r.x000, r('0000'))

  #cal.iadd(r[0], r[0], r[1]('0x00'))
  #cal.iadd(r[0], r[0], r[2]('00x0'))
  #cal.iadd(r[0], r[0], r[3]('000x'))
  cal.iadd(r, r, r)

  cal.mov(reg.o0.x, r)

  code.set_remote_binding(reg.i0, inp)
  code.set_remote_binding(reg.o0, out)

  code.cache_code()
  print code.render_string
  proc.execute(code, (0, 0, 4, 1))

  print "inp", inp
  print "out", out
  for i in xrange(0, 4):
    assert(out[i] == 2)
  
  return


def test_foo():
  proc = env.Processor(0)
  code = env.InstructionStream()

  cal.set_active_code(code)

  cb = proc.alloc_remote('i', 1, 4, 1)
  out = proc.alloc_remote('i', 4, 1, 1)
  gb = proc.alloc_remote('i', 1, 4, 1, True)

  for i in xrange(0, 4):
    cb[i] = i + 1
    out[i] = 42
    gb[i] = 67

  cal.dcl_output(reg.o0, USAGE=cal.usage.generic)
  cal.dcl_cb('cb0[4]')

  cal.mov('r0', 'cb0[0]')
  cal.mov('r1', 'cb0[1]')
  #cal.mov('r2', 'cb0[2]')
  #cal.mov('r3', 'cb0[3]')

  cal.mov('o0', 'r0')
  cal.mov('g[0]', 'r0')

  code.set_remote_binding('cb0', cb)
  code.set_remote_binding('o0', out)
  code.set_remote_binding('g[]', gb)

  code.cache_code()
  print code.render_string
  proc.execute(code, (0, 0, 1, 1))

  print "cb ", cb
  print "out", out
  print "gb ", gb
  return


if __name__ == '__main__':
  #test_4comp()
  #test_1comp()
  test_foo()

