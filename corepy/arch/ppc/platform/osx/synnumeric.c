/* Copyright (c) 2006-2009 The Trustees of Indiana University.
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * - Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 * 
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * 
 * - Neither the Indiana University nor the names of its contributors may be
 *   used to endorse or promote products derived from this software without
 *   specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

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

