import corepy.lib.extarray as extarray
import corepy.arch.cal.isa as cal
import corepy.arch.cal.types.registers as reg
import corepy.arch.cal.platform as env


proc = env.Processor(0)
code = env.InstructionStream()

inp = proc.alloc_remote('f', 4, 64)
out = proc.alloc_remote('f', 4, 64)

out.clear()
for i in xrange(0, 64):
  inp[i] = float(i + 1)

cal.set_active_code(code)

cal.dcl_input(reg.v0.x, USAGE=cal.usage.pos)
cal.dcl_resource(0, cal.pixtex_type.oned, cal.fmt.float, UNNORM = True)
cal.dcl_output(reg.o0, USAGE=cal.usage.generic)

cal.sample(0, 0, reg.o0, reg.v0.x)

code.set_binding(reg.i0, inp)
code.set_binding(reg.o0, out)
proc.execute(code)

print "inp", inp
print "out", out

