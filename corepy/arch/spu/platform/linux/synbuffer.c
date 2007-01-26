// Copyright 2006 The Trustees of Indiana University.

// This software is available for evaluation purposes only.  It may not be
// redistirubted or used for any other purposes without express written
// permission from the authors.

// Author:
//   Christopher Mueller

// Helpers for working with Buffers

#include <Python.h>

static PyObject *
synbuffer_buffer_info(PyObject *self, PyObject *args) {
  char *addr;
  int size;
  if (!PyArg_ParseTuple(args, "w#", &addr, &size))
    return NULL;

  return Py_BuildValue("(ki)", (unsigned long)addr, size);
}

static PyMethodDef synbufferMethods[] = {
    {"buffer_info",  synbuffer_buffer_info, METH_VARARGS, "Return the addres and size of a buffer."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyMODINIT_FUNC
initsynbuffer(void)
{
    (void) Py_InitModule("synbuffer", synbufferMethods);
}

