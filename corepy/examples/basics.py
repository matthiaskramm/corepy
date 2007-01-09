import array

import corepy.arch.ppc.isa as ppc
import corepy.arch.vmx.isa as vmx
import corepy.arch.ppc.platform as env
import corepy.arch.ppc.types.ppc_types as vars
from   corepy.arch.ppc.lib.util import load_word

# code is the current Synthetic Programm
code = env.InstructionStream()

# proc is a platform-specific execution environemnt 
proc = env.Processor()

# Setting the active code allows you call instructions directly
# and automatically add them to the instruction stream.
#
# Add instruction without active code:
#   code.add(ppc.addi(...))
#
# Add instruction wit active code:
#   ppc.addi(...)
ppc.set_active_code(code)

ppc.addi(code.gp_return, 0, 12)
r = proc.execute(code, debug=True)

assert(r == 12)
print 'int result:', r

code.reset()

a = array.array('d', [3.14])

load_word(code, code.gp_return, a.buffer_info()[0])
ppc.lfd(code.fp_return, code.gp_return, 0)

r = proc.execute(code, mode='fp', debug=True)
assert(r == 3.14)
print 'float result:', r


# ------------------------------------------------------------
# Variables/Exprsesions
# ------------------------------------------------------------

# Without active code
ppc.set_active_code(None)
code.reset()

a = vars.SignedWord(11, code)
b = vars.SignedWord(31, reg = code.acquire_register())
c = vars.SignedWord(reg = code.gp_return)

byte_mask = vars.Bits(0xFF, code)

# c.v = a + SignedWord.cast(b & byte_mask) + 12
c.v = a + (byte_mask & b) + 12

r = proc.execute(code)
print 'result:', r
assert(r == (42 + 12))
  
# With active code
code.reset()

ppc.set_active_code(code)
  
a = vars.SignedWord(11)
b = vars.SignedWord(31)
c = vars.SignedWord(reg = code.gp_return)

byte_mask = vars.Bits(0xFF)

c.v = a + (b & byte_mask)

ppc.set_active_code(None)
r = proc.execute(code)
assert(r == 42)

code.print_code()

# ------------------------------------------------------------
# VMX
# ------------------------------------------------------------

code.reset()
ppc.set_active_code(code)
vmx.set_active_code(code)

v_x = code.acquire_register('vector')

result = array.array('I', [0,0,0,0])
r_addr = code.acquire_register()
load_word(code, r_addr, result.buffer_info()[0])

vmx.vspltisw(v_x, 4)
vmx.stvx(v_x, 0, r_addr)

ppc.set_active_code(None)
vmx.set_active_code(None)
r = proc.execute(code)
print result
  
