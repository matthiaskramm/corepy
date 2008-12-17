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

import os

sys_info = os.uname()

# Very basic architecture detection...
if sys_info[0] == 'Darwin' and sys_info[-1] == 'Power Macintosh':
  OS = 'osx'
  ARCH = 'ppc'
  #BITS = 32
elif sys_info[0] == 'Darwin' and sys_info[-1] == 'i386':
  OS = 'osx'
  import sys
  if sys.maxint == 9223372036854775807: # 64bit python?
    ARCH = 'x86_64'
    #BITS = 64
  else: # assumed 32bit
    ARCH = 'x86'
    #BITS = 32
elif sys_info[0] == 'Linux' and sys_info[-1] == 'ppc64':
  OS = 'linux'
  ARCH = 'ppc'
  #BITS = 64

  #cpus = [line.split(':')[1] for line in open('/proc/cpuinfo').readlines() if line[:3] == 'cpu']
  #if len(cpus) > 0 and cpus[0][:5] == ' Cell':
  #  ARCH = 'cell'
  #  OS = 'linux'
elif sys_info[0] == 'Linux' and sys_info[-1] == 'i686':
  OS = 'linux'
  ARCH = 'x86'
  #BITS = 32
elif sys_info[0] == 'Linux' and sys_info[-1] == 'x86_64':
  OS = 'linux'
  ARCH = 'x86_64'
  #BITS = 64
else:
  print "Unsupported architecture: Using 'dummy' settings"
  OS = 'dummy'
  ARCH = 'dummy'
  #BITS = 0


