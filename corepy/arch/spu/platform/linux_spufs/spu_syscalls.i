// -*-c++-*-
// Copyright 2006-2007 The Trustees of Indiana University.

// This software is available for evaluation purposes only.  It may not be
// redistirubted or used for any other purposes without express written
// permission from the authors.

// Author:
//   Ben Martin
//   Christopher Mueller

// Module Name
%module spu_syscalls

// ************************************************************
// Module Includes 
// ************************************************************

// These are copied directly to the .cxx file and are not parsed
// by SWIG.  Include include files or definitions that are required
// for the module to build correctly.

%{ 
#include "spu_syscalls.h"
%}

%include "spu_syscalls.h"
