import corepy.arch.x86_64.isa as x86
from corepy.arch.x86_64.types.registers import *
import corepy.arch.x86_64.platform as env
from corepy.arch.x86_64.lib.memory import MemRef
import corepy.lib.extarray as extarray
import corepy.arch.x86_64.lib.util as util
import time

ITERS = 1000000
THREADS = 4

data = extarray.extarray('l', 1)
dbi = data.buffer_info()


# This first case is intentionally wrong to show what happens w/o locking.
data[0] = 0

prgm = env.Program()
code = prgm.get_stream()
x86.set_active_code(code)

x86.mov(rax, 1)
x86.mov(rcx, ITERS)
x86.mov(rdi, dbi[0])

lbl_loop = prgm.get_unique_label("loop")
code.add(lbl_loop)

x86.add(MemRef(rdi), rax)
x86.dec(rcx)
x86.jnz(lbl_loop)

prgm += code
prgm.print_code(hex = True)

proc = env.Processor()
t1 = time.time()
ids = [proc.execute(prgm, async = True) for i in xrange(0, THREADS)]
[proc.join(i) for i in ids]
t2 = time.time()

print "time", t2 - t1
print "val", data[0], ITERS * THREADS
print "passed?", data[0] == ITERS * THREADS


# This case locks like it should, so should be correct.
data[0] = 0

prgm = env.Program()
code = prgm.get_stream()
x86.set_active_code(code)

x86.mov(rax, 1)
x86.mov(rcx, ITERS)
x86.mov(rdi, dbi[0])

lbl_loop = prgm.get_unique_label("loop")
code.add(lbl_loop)

x86.add(MemRef(rdi), rax, lock = True)
x86.dec(rcx)
x86.jnz(lbl_loop)

prgm += code

proc = env.Processor()
t1 = time.time()
ids = [proc.execute(prgm, async = True) for i in xrange(0, THREADS)]
[proc.join(i) for i in ids]
t2 = time.time()

print
print "time", t2 - t1
print "val", data[0], ITERS * THREADS
print "passed?", data[0] == ITERS * THREADS


# Same thing again, just using cmpxchg to do the work.
data[0] = 0

prgm = env.Program()
code = prgm.get_stream()
x86.set_active_code(code)

x86.mov(rcx, ITERS)
x86.mov(rdi, dbi[0])

lbl_loop = prgm.get_unique_label("loop")
code.add(lbl_loop)

# Read the data into rax
# cmploop:
#   add rax+1, storing in say rbx
#   cmpxchg rbx, data
#   jz cmploop

x86.mov(rax, MemRef(rdi))

lbl_cmpxchg = prgm.get_unique_label("cmpxchg")
code.add(lbl_cmpxchg)

x86.mov(rbx, rax)
x86.add(rbx, 1)
x86.cmpxchg(MemRef(rdi), rbx, lock = True)
x86.jnz(lbl_cmpxchg)

x86.dec(rcx)
x86.jnz(lbl_loop)

prgm += code

proc = env.Processor()
t1 = time.time()
ids = [proc.execute(prgm, async = True) for i in xrange(0, THREADS)]
[proc.join(i) for i in ids]
t2 = time.time()

print
print "time", t2 - t1
print "val", data[0], ITERS * THREADS
print "passed?", data[0] == ITERS * THREADS


# Use a float value in an SSE register.
data = extarray.extarray('f', 1)
dbi = data.buffer_info()

data[0] = 0.0
data[1] = 1.0

prgm = env.Program()
code = prgm.get_stream()
x86.set_active_code(code)

x86.mov(rcx, ITERS)
x86.mov(rdi, dbi[0])
x86.movss(xmm0, MemRef(rdi, 4, data_size = 32))

lbl_loop = prgm.get_unique_label("loop")
code.add(lbl_loop)

# Read the data into rax
# cmploop:
#   add rax+1, storing in say rbx
#   cmpxchg rbx, data
#   jz cmploop

x86.mov(eax, MemRef(rdi, data_size = 32))

lbl_cmpxchg = prgm.get_unique_label("cmpxchg")
code.add(lbl_cmpxchg)

x86.movd(xmm1, eax)
x86.addss(xmm1, xmm0)
x86.movd(ebx, xmm1)
x86.cmpxchg(MemRef(rdi, data_size = 32), ebx, lock = True)
x86.jnz(lbl_cmpxchg)

x86.dec(rcx)
x86.jnz(lbl_loop)

prgm += code

proc = env.Processor()
t1 = time.time()
ids = [proc.execute(prgm, mode = 'fp', async = True) for i in xrange(0, THREADS)]
ret = [proc.join(i) for i in ids]
t2 = time.time()

print "ret", ret

print
print "time", t2 - t1
print "val", data[0], ITERS * THREADS
print "passed?", data[0] == ITERS * THREADS

