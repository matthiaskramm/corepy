import corepy.lib.extarray as extarray
import corepy.arch.spu.isa as spu
import corepy.arch.spu.lib.util as util
import corepy.arch.spu.platform as env

prgm = env.Program()
code = prgm.get_stream()
proc = env.Processor()

# Generate substream
# Multiply gp_return by 2, add 1
subcode = prgm.get_stream()
subcode.add(spu.shli(subcode.gp_return, subcode.gp_return, 1))
subcode.add(spu.ai(subcode.gp_return, subcode.gp_return, 1))

# Initialize gp_return, insert code
code.add(spu.il(code.gp_return, 5))
code.add(subcode)

# Add 3, insert again
code.add(spu.ai(code.gp_return, code.gp_return, 3))
code.add(subcode)

#code.print_code()

prgm.add(code)
prgm.print_code() # TODO  - support print prgm instead?

ret = proc.execute(prgm, mode = 'int')
print "ret", ret

prgm = env.Program()
code = prgm.get_stream()

r_add = prgm.acquire_register()

# Generate substream
# Multiply gp_return by 2, add 1
subcode = prgm.get_stream()
subcode.add(spu.shli(subcode.gp_return, subcode.gp_return, 1))
subcode.add(spu.a(subcode.gp_return, subcode.gp_return, r_add))

# Initialize gp_return, insert code
code.add(spu.il(r_add, 1))
code.add(spu.il(code.gp_return, 5))
code.add(subcode)

# Add 3, insert again
code.add(spu.il(r_add, 2))
code.add(spu.ai(code.gp_return, code.gp_return, 3))
code.add(subcode)


prgm.add(code)
prgm.print_code()

ret = proc.execute(prgm, mode = 'int')
print "ret", ret

