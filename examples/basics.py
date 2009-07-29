# Copyright (c) 2006-2009 The Trustees of Indiana University.                   
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

import sys
import array

import corepy.arch.ppc.isa as ppc
import corepy.arch.vmx.isa as vmx
import corepy.arch.ppc.platform as env
import corepy.arch.ppc.types.ppc_types as vars
from   corepy.arch.ppc.lib.util import load_word

# prgm is the current Synthetic Program
# code is the current InstructionStream, where new code is added
prgm = env.Program()
code = prgm.get_stream()

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

ppc.addi(prgm.gp_return, 0, 12)
ppc.b(prgm.lbl_epilogue)

prgm.add(code)
prgm.print_code(pro=True, epi=True, binary=True)

r = proc.execute(prgm, debug=True)

print 'int result:', r
assert(r == 12)

code.reset()

a = array.array('d', [3.14])

load_word(code, prgm.gp_return, a.buffer_info()[0])
ppc.lfd(prgm.fp_return, prgm.gp_return, 0)

r = proc.execute(prgm, mode='fp', debug=True)
assert(r == 3.14)
print 'float result:', r


code.reset()

load_word(code, prgm.gp_return, 0xFFFFFFFF)

r = proc.execute(prgm, mode='int', debug=True)
print "int result:",r
assert(r == -1)


code.reset()

ppc.addi(prgm.gp_return, 0, 16)
ppc.mtctr(prgm.gp_return)
ppc.addi(prgm.gp_return, 0, 0)

lbl_loop = prgm.get_label("LOOP")
code.add(lbl_loop)
ppc.addi(prgm.gp_return, prgm.gp_return, 2)
ppc.bdnz(lbl_loop)

#prgm.print_code(hex = True)
r = proc.execute(prgm, mode='int', debug=True)
print "int result:",r
assert(r == 32)

#sys.exit(0)

# ------------------------------------------------------------
# Variables/Exprsesions
# ------------------------------------------------------------

# Without active code
ppc.set_active_code(None)
code.reset()

a = vars.SignedWord(11, code)
b = vars.SignedWord(31, code, reg = prgm.acquire_register())
c = vars.SignedWord(code = code, reg = prgm.gp_return)

byte_mask = vars.Bits(0xFF, code)

# c.v = a + SignedWord.cast(b & byte_mask) + 12
c.v = a + (byte_mask & b) + 12


r = proc.execute(prgm, debug = True)
print 'result:', r
assert(r == (42 + 12))

# With active code
code.reset()

ppc.set_active_code(code)

a = vars.SignedWord(11)
b = vars.SignedWord(31)
c = vars.SignedWord(reg = prgm.gp_return)

byte_mask = vars.Bits(0xFF)

c.v = a + (b & byte_mask)

ppc.set_active_code(None)
r = proc.execute(prgm)
assert(r == 42)

code.print_code()

# ------------------------------------------------------------
# VMX
# ------------------------------------------------------------

code.reset()
ppc.set_active_code(code)
vmx.set_active_code(code)

v_x = prgm.acquire_register('vector')

result = array.array('I', [0,0,0,0,0,0])
r_addr = prgm.acquire_register()

# Minor hack to align the address to a 16-byte boundary.
# Note that enough space was allocated in the array to
# ensure the save location is valid.
# (we are working on a cleaner memory management interface
#  for CorePy that will fix these annoying alignment issues)
addr = result.buffer_info()[0]
if addr % 16 != 0:
  addr += 16 - (addr % 16)
  print 'aligning addr'

load_word(code, r_addr, addr)

vmx.vspltisw(v_x, 4)
vmx.stvx(v_x, 0, r_addr)

ppc.set_active_code(None)
vmx.set_active_code(None)
r = proc.execute(prgm) # , debug = True)
# code.print_code(pro = True, epi = True)

print result
