# Copyright (c) 2006-2008 The Trustees of Indiana University.                   
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

import corepy.lib.extarray as extarray

import corepy.arch.x86_64.isa as x86
from corepy.arch.x86_64.types.registers import *

import corepy.arch.x86_64.platform as env
from corepy.arch.x86_64.lib.memory import MemRef
import corepy.arch.x86_64.lib.iterators as iter


prgm = env.Program()
code = prgm.get_stream()
proc = env.Processor()

# Generate sub-stream
# Multiple rax by 2, add 1
subcode = prgm.get_stream()
subcode.add(x86.shl(rax, 1))
subcode.add(x86.add(rax, 1))

# Initialize a register, insert code
code.add(x86.mov(rax, 5))
code.add(subcode)

# Add 3, insert again
code.add(x86.add(rax, 3))
code.add(subcode)

prgm.add(code)
prgm.print_code()
ret = proc.execute(prgm, mode = 'int')
print "ret", ret


prgm = env.Program()
code = prgm.get_stream()

# Use a register from the parent code in the subcode directly
r_add = prgm.acquire_register()
r_sub = prgm.acquire_register(reg_type = GPRegister32)

# Generate sub-stream
# Multiple rax by 2, add 1
subcode = prgm.get_stream()
subcode.add(x86.shl(rax, 1))
subcode.add(x86.add(rax, r_add))
subcode.add(x86.mov(r_sub, 72))

# Initialize a register, insert code
code.add(x86.mov(r_add, 1))
code.add(x86.mov(rax, 5))
code.add(subcode)

# Add 3, insert again
code.add(x86.mov(r_add, 2))
code.add(x86.add(rax, 3))
code.add(subcode)

prgm.release_register(r_add)
prgm.release_register(r_sub)


prgm.add(code)
prgm.print_code()
ret = proc.execute(prgm, mode = 'int')
print "ret", ret


# Generate code to copy data from one array to another
def copy_array(prgm, r_dst, r_src, r_len):
  r_cnt = prgm.acquire_register()
  r_data = prgm.acquire_register()
  code = prgm.get_stream()

  # Define the loop body
  def loop_body(r_i):
    body = prgm.get_stream()
    body.add(x86.mov(r_data, MemRef(r_src, 0, r_i, scale = 8)))
    body.add(x86.mov(MemRef(r_dst, 0, r_i, scale = 8), r_data))
    return body

  # Use a synthetic iterator to generate the loop
  for r_i in iter.syn_iter(code, r_len, count_reg = r_cnt):
    body = loop_body(r_i)

    # Perform the loop body twice per loop iteration
    code.add(body)
    code.add(x86.inc(r_i))
    code.add(body)

  prgm.release_register(r_cnt)
  prgm.release_register(r_data)

  code.print_code()
  return code


prgm = env.Program()
code = prgm.get_stream()

# Load
# No longer needed - params are passed by register
#code.add(x86.mov(rdi, MemRef(rbp, 16)))
#code.add(x86.mov(rsi, MemRef(rbp, 24)))
#code.add(x86.mov(rdx, MemRef(rbp, 32)))

code.add(copy_array(prgm, rdi, rsi, rdx))

prgm.add(code)

a = extarray.extarray('i', range(0, 32))
b = extarray.extarray('i', (0,) * 32)

params = env.ExecParams()
params.p1 = b.buffer_info()[0]
params.p2 = a.buffer_info()[0]
params.p3 = 32

print a
print b

proc.execute(prgm, params = params)

print a
print b

