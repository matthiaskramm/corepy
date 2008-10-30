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

# Synthetic Programming Environment Makefile for OS X 10.4.6

# To build all extension modules, simple run make:
#   % make
# To build the ppc execute module only:
#   % make ppc
# To build the Numeric support module only:
#   % make numeric

ifndef PYTHON
PYTHON=/usr/include/python2.5
endif

INC=-I$(PYTHON) -I$(PYTHON)/Include

# CPP_LINK_POST=-libmc++ -lstdc++ -lm 

all:  _spu_execmodule.so
cell: _spu_execmodule.so

# ------------------------------------------------------------
# cell_exec module - This is the only required module.
# ------------------------------------------------------------

# new system for getting spu programs to the spus
# spu_syscalls_wrap.cxx: spu_syscalls.i spu_syscalls.h
# 	swig -c++ -python $(INC) spu_syscalls.i

# spu_syscalls_wrap.o: spu_syscalls_wrap.cxx
# 	g++ -fpic $(INC) -c spu_syscalls_wrap.cxx

# _spu_syscallsmodule.so: spu_syscalls_wrap.o
# 	g++ -shared spu_syscalls_wrap.o -o _spu_syscallsmodule.so

# old system
spu_exec_wrap.cxx: spu_exec.i spu_exec.h
	swig -c++ -python $(INC) spu_exec.i

# SWIG_PYTHON_SILENT_MEMLEAK is defined to suppress the error message
# on speid_t * deletion.  This is the only error and it is a result of 
# changes made to swig and (most likely) not a real memory leak.
spu_exec_wrap.o: spu_exec_wrap.cxx 
	 ppu-c++ -m32 -DSWIG_PYTHON_SILENT_MEMLEAK -fpic $(INC) -c spu_exec_wrap.cxx

_spu_execmodule.so: spu_exec_wrap.o
	ppu-c++ -m32 -shared spu_exec_wrap.o -o _spu_execmodule.so $(CPP_LINK_POST) -lspe2


# Framebuffer support
cell_fb_wrap.cxx: cell_fb.i cell_fb.h
	swig -c++ -python $(INC) cell_fb.i

cell_fb_wrap.o: cell_fb_wrap.cxx 
	ppu-c++ -m32 -fpic $(INC) -c cell_fb_wrap.cxx

_cell_fbmodule.so: cell_fb_wrap.o 
	ppu-c++ -m32 -shared cell_fb_wrap.o -o _cell_fbmodule.so $(CPP_LINK_POST)

cell_fb: _cell_fbmodule.so


clean:
	rm -f *.pyc *_wrap.cxx *_wrap.o _*module.so spu_exec.py cell_fb.py


