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

all: _cell_execmodule.so 
cell: _cell_execmodule.so 
numeric: synnumeric.so

# ------------------------------------------------------------
# cell_exec module - This is the only required module.
# ------------------------------------------------------------

cell_exec_wrap.cxx: cell_exec.i cell_exec.h # Makefile
	swig -c++ -python $(INC) cell_exec.i

cell_exec_wrap.o: cell_exec_wrap.cxx
	 g++ -fpic $(INC) -c cell_exec_wrap.cxx

_cell_execmodule.so: cell_exec_wrap.o
	g++ -shared cell_exec_wrap.o -o _cell_execmodule.so $(CPP_LINK_POST) 

synbuffer.o: synbuffer.c
	 g++ -fpic $(INC) -c synbuffer.c

synbuffermodule.so: synbuffer.o
	g++ -shared synbuffer.o -o synbuffermodule.so 

# # ------------------------------------------------------------
# # synnumeric module - required for using Numeric arrays
# # ------------------------------------------------------------

# synnumeric.o: synnumeric.c
# 	g++ -O3 -c $(FRAMEWORKS) $(INC) synnumeric.c

# synnumeric.so: synnumeric.o
# 	g++ -bundle -bundle_loader /usr/bin/python -framework Python -framework Carbon synnumeric.o -o synnumeric.so


clean:
	rm *.pyc *_wrap.cxx *_wrap.o _*module.so
