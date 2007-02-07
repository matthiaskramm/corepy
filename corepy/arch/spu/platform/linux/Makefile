# Copyright 2006 The Trustees of Indiana University.

# This software is available for evaluation purposes only.  It may not be
# redistirubted or used for any other purposes without express written 
# permission from the authors.

# Author:
#   Christopher Mueller

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

all:  _spu_execmodule.so spu_bootstrap.o synbuffermodule.so
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

spu_exec_wrap.o: spu_exec_wrap.cxx 
	 g++ -fpic $(INC) -c spu_exec_wrap.cxx

_spu_execmodule.so: spu_exec_wrap.o
	g++  -shared spu_exec_wrap.o -o _spu_execmodule.so $(CPP_LINK_POST) -lspe

spu_bootstrap.o: spu_bootstrap.c spu_constants.h
	spu-gcc spu_bootstrap.c -o spu_bootstrap.o

# other stuff
synbuffer.o: synbuffer.c
	 g++ -fpic $(INC) -c synbuffer.c

synbuffermodule.so: synbuffer.o
	g++ -shared synbuffer.o -o synbuffermodule.so 

clean:
	rm -f *.pyc *_wrap.cxx *_wrap.o _*module.so
