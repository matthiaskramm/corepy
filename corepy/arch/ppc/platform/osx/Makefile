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



FRAMEWORKS= -framework Carbon
INC=-I/System/Library/Frameworks/Python.framework/Headers

all: _ppc_execmodule.so synnumeric.so # _cjaccardmodule.so synnumeric.so synbuffer.so
ppc: _ppc_execmodule.so
numeric: synnumeric.so
buffer: synbuffer.so

# ------------------------------------------------------------
# ppc_exec module - This is the only required module.
# ------------------------------------------------------------

ppc_exec_wrap.cxx: ppc_exec.i ppc_exec.h # Makefile
	swig -c++ -python $(INC) ppc_exec.i

ppc_exec_wrap.o: ppc_exec_wrap.cxx
	g++ -c $(FRAMEWORKS) $(INC) ppc_exec_wrap.cxx

_ppc_execmodule.so: ppc_exec_wrap.o
	g++ -bundle -bundle_loader /usr/bin/python -framework Python -framework Carbon ppc_exec_wrap.o -o _ppc_execmodule.so

# ------------------------------------------------------------
# cjaccard module - C++ fingerprint comparison routine
# ------------------------------------------------------------

cjaccard_wrap.cxx: cjaccard.i jaccard.h # Makefile
	swig -c++ -python $(INC) cjaccard.i

cjaccard_wrap.o: cjaccard_wrap.cxx
	g++ -O3  -c $(FRAMEWORKS) $(INC) cjaccard_wrap.cxx

_cjaccardmodule.so: cjaccard_wrap.o
	g++ -bundle -bundle_loader /usr/bin/python -framework Python -framework Carbon cjaccard_wrap.o -o _cjaccardmodule.so


# ------------------------------------------------------------
# synnumeric module - required for using Numeric arrays
# ------------------------------------------------------------

synnumeric.o: synnumeric.c
	g++ -O3 -c $(FRAMEWORKS) $(INC) synnumeric.c

synnumeric.so: synnumeric.o
	g++ -bundle -bundle_loader /usr/bin/python -framework Python -framework Carbon synnumeric.o -o synnumeric.so

# ------------------------------------------------------------
# synbuffer module - access the address of a buffer
# ------------------------------------------------------------

synbuffer.o: synbuffer.c
	g++ -O3 -c $(FRAMEWORKS) $(INC) synbuffer.c

synbuffer.so: synbuffer.o
	g++ -bundle -bundle_loader /usr/bin/python -framework Python -framework Carbon synbuffer.o -o synbuffer.so


clean:
	rm *.pyc *_wrap.cxx *_wrap.o _*module.so

