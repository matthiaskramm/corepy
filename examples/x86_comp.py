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

import array

import corepy.arch.x86.isa as x86
from corepy.arch.x86.types.registers import *

import corepy.arch.x86.platform as env
from corepy.arch.x86.lib.memory import MemRef


prgm = env.Program()
code = prgm.get_stream()
proc = env.Processor()

# Generate sub-stream
# Multiple eax by 2, add 1
subcode = prgm.get_stream()
subcode.add(x86.shl(eax, 1))
subcode.add(x86.add(eax, 1))

# Initialize a register, insert code
code.add(x86.mov(eax, 5))
code.add(subcode)

# Add 3, insert again
code.add(x86.add(eax, 3))
code.add(subcode)

prgm.add(code)
prgm.print_code()
ret = proc.execute(prgm, mode = 'int')
print "ret", ret


prgm = env.Program()
code = prgm.get_stream()

# Use a register from the parent code in the subcode directly
r_add = prgm.acquire_register()
print "got reg", r_add

# Generate sub-stream
# Multiple eax by 2, add 1
subcode = prgm.get_stream()
subcode.add(x86.shl(eax, 1))
subcode.add(x86.add(eax, r_add))

# Initialize a register, insert code
code.add(x86.mov(r_add, 1))
code.add(x86.mov(eax, 5))
code.add(subcode)

# Add 3, insert again
code.add(x86.mov(r_add, 2))
code.add(x86.add(eax, 3))
code.add(subcode)

prgm.add(code)
prgm.print_code()
ret = proc.execute(prgm, mode = 'int')
print "ret", ret


