// Copyright 2006 The Trustees of Indiana University.

// This software is available for evaluation purposes only.  It may not be
// redistirubted or used for any other purposes without express written
// permission from the authors.

// Author:
//   Christopher Mueller

// Helpers for working with Numeric data types

#include <Python.h>
#include "Numeric/arrayobject.h"

static PyObject *
synnumeric_array_address(PyObject *self, PyObject *args) {
  PyArrayObject *array;
  if (!PyArg_ParseTuple(args, "O!",
                        &PyArray_Type, &array))
    return NULL;

  return PyInt_FromLong((long)array->data);
}

static PyObject *
synnumeric_array_strides(PyObject *self, PyObject *args) {
  PyArrayObject *array;
  if (!PyArg_ParseTuple(args, "O!",
                        &PyArray_Type, &array))
    return NULL;
  
  int i = 0;
  PyObject* list = PyList_New(array->nd);
  
  for(i = 0; i < array->nd; ++i) {
    PyList_SetItem(list, i, PyInt_FromLong((long)(array->strides[i])));
  }
  
  return list;
}

static PyMethodDef synnumericMethods[] = {
    {"array_address",  synnumeric_array_address, METH_VARARGS, "Return the address of array->data."},
    {"array_strides",  synnumeric_array_strides, METH_VARARGS, "Return the array->strides as a list."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyMODINIT_FUNC
initsynnumeric(void)
{
    (void) Py_InitModule("synnumeric", synnumericMethods);
    import_array();
}

