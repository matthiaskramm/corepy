#!/usr/bin/env python
# Copyright (c) 2006-2008 The Trustees of Indiana University.                   
# All rights reserved.                                                          
#                                                                               
# Redistribution and use in source and binary forms, with or without            
# modification, are permitted provided that the following conditions are met:   
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

# Build the appropriate CorePy modules with the following command:
# python setup.py build_ext -i

from distutils.core import setup, Extension
from distutils.util import get_platform
import sys
from os import path


ext_modules = [#Extension('corepy.lib.extarray._alloc',
               #          sources=['corepy/lib/extarray/alloc.i'],
               #          depends=['corepy/lib/extarray/alloc.h']),
               #Extension('corepy.lib.nextarray.nextarray',
               #          sources=['corepy/lib/nextarray/nextarray.c'],
               #          depends=['corepy/lib/nextarray/alloc.h'])]
               Extension('corepy.lib.extarray.extarray',
                         sources=['corepy/lib/extarray/extarray.c'],
                         depends=['corepy/lib/extarray/alloc.h'])]


py_platform = get_platform()
print "Python platform:", py_platform

if py_platform == "linux-ppc64":
  OS = 'linux'
  ARCH = 'ppc'
  #BITS = '64'

  # Enable more stuff if libspe2 is available
  if path.exists("/usr/lib/libspe2.so"):
    print "LibSPE2 is available; enabling native SPU code execution support"
    libraries = ['spe2']
    define_macros = [('HAS_LIBSPE2', '1')]
  else:
    print "LibSPE2 NOT available; disabling native SPU code execution support"
    libraries = []
    define_macros = []

  ext_modules.append(
      Extension('corepy.arch.spu.platform.linux_spufs._spu_exec',
                sources=['corepy/arch/spu/platform/linux_spufs/spu_exec.i'],
                depends=['corepy/arch/spu/platform/linux_spufs/spu_exec.h',
                         'corepy/arch/spu/platform/linux_spufs/spufs.h'],
                libraries = libraries, define_macros = define_macros))

elif py_platform == "linux-ppc":
  OS = 'linux'
  ARCH = 'ppc'
  #BITS = '32'
elif py_platform == "linux-x86_64":
  OS = 'linux'
  ARCH = 'x86_64'
  #BITS = '64'
elif py_platform == "linux-i686":
  OS = 'linux'
  ARCH = 'x86'
  #BITS = '32'
elif py_platform[0:6] == 'macosx' or py_platform[0:6] == 'darwin':
  OS = 'osx'
  if py_platform[-3:] == 'ppc':
    ARCH = 'ppc'
    #BITS = 32
  elif py_platform[-3:] == '386':
    if sys.maxint == 2**63 - 1: # 64bit python?
      ARCH = 'x86_64'
      #BITS = 64
    else: # assumed 32bit
      ARCH = 'x86'
      #BITS = 32
  elif py_platform[-3:] == 'fat':
    # distutils says to build universal -- guess machine type from byte
    # order; assume 32-bit?
    #BITS = 32
    if sys.byteorder == 'little':
        ARCH = 'x86'
    else:
        ARCH = 'ppc'
else:
  print "Unsupported Python platform!  Aborting."
  exit(-1)

# TODO - maybe rename the _exec files to not have the arch in them?
template = 'corepy/arch/%s/platform/%s/%s_exec.' % (ARCH, OS, ARCH)
ext_modules.append(
    Extension('corepy.arch.%s.platform.%s._%s_exec' % (ARCH, OS, ARCH),
        sources=['%s%s' % (template, 'i')],
        depends=['%s%s' % (template, 'h')]))

#print "CorePy platform:", ARCH, OS, BITS
print "CorePy platform:", ARCH, OS
print

# New enough python to support swig_opts?
options={}
if sys.version_info[0] >= 2 and sys.version_info[1] >= 4:
  options={'build_ext':{'swig_opts':'-O -Wall'}}

# See http://www.nabble.com/C%2B%2B,-swig,-and-distutils-td1555651.html
# for info on the options line below
setup (name = 'CorePy',
       version = '1.0',
       author      = "Chris Mueller, Andrew Friedley, Andrew Lumsdaine",
       description = """http://www.corepy.org""",
       ext_modules = ext_modules,
#       py_modules = ["corepy"],
       options=options
       )

