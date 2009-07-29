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

import corepy.arch.ppc.isa as ppc
import corepy.arch.vmx.isa as vmx
import corepy.arch.ppc.platform as env

prgm = env.Program()
code = prgm.get_stream()
proc = env.Processor()

# Generate sub-stream
# Multiple eax by 2, add 1
subcode = prgm.get_stream()
subcode.add(ppc.mulli(prgm.gp_return, prgm.gp_return, 2))
subcode.add(ppc.addi(prgm.gp_return, prgm.gp_return, 1))

# Initialize a register, insert code
code.add(ppc.addi(prgm.gp_return, 0, 5))
code.add(subcode)

# Add 3, insert again
code.add(ppc.addi(prgm.gp_return, prgm.gp_return, 3))
code.add(subcode)

prgm.add(code)
prgm.print_code()
ret = proc.execute(prgm, mode = 'int')
print "ret", ret


prgm = env.Program()
code = prgm.get_stream()

# Use a register from the parent code in the subcode directly
r_add = prgm.acquire_register()

# Generate sub-stream
# Multiple eax by 2, add 1
subcode = prgm.get_stream()
subcode.add(ppc.mulli(prgm.gp_return, prgm.gp_return, 2))
subcode.add(ppc.add(prgm.gp_return, prgm.gp_return, r_add))

# Initialize a register, insert code
code.add(ppc.addi(r_add, 0, 1))
code.add(ppc.addi(prgm.gp_return, 0, 5))
code.add(subcode)

# Add 3, insert again
code.add(ppc.addi(r_add, 0, 2))
code.add(ppc.addi(prgm.gp_return, prgm.gp_return, 3))
code.add(subcode)

prgm.add(code)
prgm.print_code()
ret = proc.execute(prgm, mode = 'int')
print "ret", ret



