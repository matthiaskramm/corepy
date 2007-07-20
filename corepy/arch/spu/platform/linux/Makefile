# Copyright 2006-2007 The Trustees of Indiana University.

# This software is available for evaluation purposes only.  It may not be
# redistirubted or used for any other purposes without express written
# permission from the authors.

# Authors:
#   Christopher Mueller (chemuell@cs.indiana.edu)
#   Andrew Lumsdaine    (lums@cs.indiana.edu)

# Synthetic Programming Environment Makefile for OS X 10.4.6

# To build all extension modules, simple run make:
#   % make
# To build the ppc execute module only:
#   % make ppc
# To build the Numeric support module only:
#   % make numeric

ifndef PYTHON
PYTHON=/usr/include/python2.4
endif

INC=-I$(PYTHON) -I$(PYTHON)/Include

# CPP_LINK_POST=-libmc++ -lstdc++ -lm 

all:  _spu_execmodule.so spu_bootstrap.o synbuffermodule.so spu_native_test.o
cell: _spu_execmodule.so synbuffermodule.so
numeric: synnumeric.so

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
	 g++ -DSWIG_PYTHON_SILENT_MEMLEAK -fpic $(INC) -c spu_exec_wrap.cxx

_spu_execmodule.so: spu_exec_wrap.o
	g++  -shared spu_exec_wrap.o -o _spu_execmodule.so $(CPP_LINK_POST) -lspe

spu_bootstrap.o: spu_bootstrap.c spu_constants.h
	spu-gcc spu_bootstrap.c -o spu_bootstrap.o

spu_native_test.o: spu_native.c spu_constants.h
	spu-gcc spu_native.c -o spu_native_test.o

# other stuff
synbuffer.o: synbuffer.c
	 g++ -fpic $(INC) -c synbuffer.c

synbuffermodule.so: synbuffer.o
	g++ -shared synbuffer.o -o synbuffermodule.so 

# Framebuffer support
cell_fb_wrap.cxx: cell_fb.i cell_fb.h
	swig -c++ -python $(INC) cell_fb.i

cell_fb_wrap.o: cell_fb_wrap.cxx 
	g++ -fpic -std=c99 $(INC) -c cell_fb_wrap.cxx

_cell_fbmodule.so: cell_fb_wrap.o 
	g++  -shared cell_fb_wrap.o -o _cell_fbmodule.so $(CPP_LINK_POST)

cell_fb: _cell_fbmodule.so


clean:
	rm -f *.pyc *_wrap.cxx *_wrap.o _*module.so


