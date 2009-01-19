#include <Python.h>
#include "structmember.h"
//#include <Numeric/arrayobject.h>
#include <stdio.h>
#include "alloc.h"

#ifndef _DEBUG
#define _DEBUG 1
#endif

typedef struct NextArray {
  PyObject_HEAD

  char typecode;
  char huge;
  char _lock;

  int itemsize;
  int _data_len;
  int _alloc_len;
  int _page_size;

  void* _memory;

  void* (*_alloc)(int size);
  void* (*_realloc)(void* mem, int oldsize, int newsize);
  void (*_free)(void* mem);

  int (*_setitem)(struct NextArray* self, int ind, PyObject* val);
  PyObject* (*_getitem)(struct NextArray* self, int ind);
} NextArray;


static PyTypeObject NextArrayType;

static PyObject*
NextArray_new(PyTypeObject* type, PyObject* args, PyObject* kwds)
{
  NextArray* self;

  self = (NextArray*)type->tp_alloc(type, 0);
  if(self != NULL) {
    self->typecode = '\0';
    self->huge = 0;
    self->_lock = 0;

    self->itemsize = 0;
    self->_data_len = 0;
    self->_alloc_len = 0;
    self->_page_size = 0;

    self->_memory = NULL;
    self->_alloc = NULL;
    self->_realloc = NULL;
    self->_free = NULL;

    self->_setitem = NULL;
    self->_getitem = NULL;
  }

  return (PyObject*)self;
}


static int _setitem_schar(NextArray* self, int ind, PyObject* val)
{
  ((char*)self->_memory)[ind] = PyLong_AsLong(val);
  return 0;
}

static int _setitem_uchar(NextArray* self, int ind, PyObject* val)
{
  ((unsigned char*)self->_memory)[ind] = PyLong_AsLong(val);
  return 0;
}

static int _setitem_sshort(NextArray* self, int ind, PyObject* val)
{
  ((short*)self->_memory)[ind] = PyLong_AsLong(val);
  return 0;
}

static int _setitem_ushort(NextArray* self, int ind, PyObject* val)
{
  ((unsigned short*)self->_memory)[ind] = PyLong_AsLong(val);
  return 0;
}

static int _setitem_sint(NextArray* self, int ind, PyObject* val)
{
  ((int*)self->_memory)[ind] = PyLong_AsLong(val);
  return 0;
}

static int _setitem_uint(NextArray* self, int ind, PyObject* val)
{
  ((unsigned int*)self->_memory)[ind] = PyLong_AsLong(val);
  return 0;
}

static int _setitem_slong(NextArray* self, int ind, PyObject* val)
{
  ((long*)self->_memory)[ind] = PyLong_AsLong(val);
  return 0;
}

static int _setitem_ulong(NextArray* self, int ind, PyObject* val)
{
  ((unsigned long*)self->_memory)[ind] = PyLong_AsLong(val);
  return 0;
}

static int _setitem_float(NextArray* self, int ind, PyObject* val)
{
  ((float*)self->_memory)[ind] = PyLong_AsDouble(val);
  return 0;
}

static int _setitem_double(NextArray* self, int ind, PyObject* val)
{
  ((double*)self->_memory)[ind] = PyLong_AsDouble(val);
  return 0;
}

static PyObject* _getitem_schar(NextArray* self, int ind)
{
  return PyLong_FromLong(((char*)self->_memory)[ind]);
}

static PyObject* _getitem_uchar(NextArray* self, int ind)
{
  return PyLong_FromLong(((unsigned char*)self->_memory)[ind]);
}

static PyObject* _getitem_sshort(NextArray* self, int ind)
{
  return PyLong_FromLong(((short*)self->_memory)[ind]);
}

static PyObject* _getitem_ushort(NextArray* self, int ind)
{
  return PyLong_FromLong(((unsigned short*)self->_memory)[ind]);
}

static PyObject* _getitem_sint(NextArray* self, int ind)
{
  return PyLong_FromLong(((int*)self->_memory)[ind]);
}

static PyObject* _getitem_uint(NextArray* self, int ind)
{
  return PyLong_FromLong(((unsigned int*)self->_memory)[ind]);
}

static PyObject* _getitem_slong(NextArray* self, int ind)
{
  return PyLong_FromLong(((long*)self->_memory)[ind]);
}

static PyObject* _getitem_ulong(NextArray* self, int ind)
{
  return PyLong_FromLong(((unsigned long*)self->_memory)[ind]);
}


static int
_set_type_fns(NextArray* self, char typecode)
{
  // TODO - deal with byteswap routines

  switch(typecode) {
  case 'c':
  case 'b':
    self->_setitem = _setitem_schar;
    self->_getitem = _getitem_schar;
    self->itemsize = sizeof(char);
    break;
  case 'B':
    self->_setitem = _setitem_uchar;
    self->_getitem = _getitem_uchar;
    self->itemsize = sizeof(unsigned char);
    break;
  case 'h':
    self->_setitem = _setitem_sshort;
    self->_getitem = _getitem_sshort;
    self->itemsize = sizeof(short);
    break;
  case 'H':
    self->_setitem = _setitem_ushort;
    self->_getitem = _getitem_ushort;
    self->itemsize = sizeof(unsigned short);
    break;
  case 'i':
    self->_setitem = _setitem_sint;
    self->_getitem = _getitem_sint;
    self->itemsize = sizeof(int);
    break;
  case 'I':
    self->_setitem = _setitem_uint;
    self->_getitem = _getitem_uint;
    self->itemsize = sizeof(unsigned int);
    break;
  case 'l':
    self->_setitem = _setitem_slong;
    self->_getitem = _getitem_slong;
    self->itemsize = sizeof(long);
    break;
  case 'L':
    self->_setitem = _setitem_ulong;
    self->_getitem = _getitem_ulong;
    self->itemsize = sizeof(unsigned long);
    break;
  case 'u':
    PyErr_SetString(PyExc_NotImplementedError,
        "Unicode not supported by extarray");
    return -1;
  default:
    // TODO - include specified type in string
    PyErr_SetString(PyExc_TypeError, "Unknown array type specified");
    return -1;
  }

  self->typecode = typecode;
  return 0;
}


static int alloc(NextArray* self, int length)
{
  int size;
  int m;

  if(self->_lock == 1) {
    PyErr_SetString(PyExc_MemoryError,
        "Attempt to allocate with memory lock enabled");
    return -1;
  }

  // Round size to a page
  size = length * self->itemsize;
  m = size % self->_page_size;
  if(m != 0) {
    size += self->_page_size - m;
  }

  if(self->_alloc_len < size) {
    self->_memory = self->_realloc(self->_memory, self->_alloc_len, size);
    self->_alloc_len = size;
    if(length < self->_data_len) {
      self->_data_len = length;
    }

#ifdef _DEBUG
    printf("Allocated %d bytes at %p\n", size, self->_memory);
#endif
  }

  return 0;
}


static int
NextArray_init(NextArray* self, PyObject* args, PyObject* kwds)
{
  //def __init__(self, typecode, init = None, huge = False):
  static char* kwlist[] = {"typecode", "init", "huge", NULL};
  char typecode;
  char huge = 0;
  PyObject* init = Py_None;

  //TODO - use PyArg_ParseTupleAndKeywords()
  if(!PyArg_ParseTupleAndKeywords(args, kwds, "c|Oi",
      kwlist, &typecode, &init, &huge)) {
    return -1;
  }

  self->typecode = typecode;
  self->huge = huge;
  self->_lock = 0;
  self->_alloc_len = 0;
  self->_memory = NULL;

  //TODO - replace has_huge_pages with a define
  if(huge == 1 && has_huge_pages() == 0) {
    PyErr_SetString(PyExc_MemoryError,
        "No huge pages available, try regular pages");
    return -1;
  }

  if(_set_type_fns(self, typecode) != 0) {
    return -1;
  }

  if(huge == 1) {
    self->_alloc = alloc_hugemem;
    self->_realloc = realloc_hugemem;
    self->_free = free_hugemem;
    self->_page_size = get_hugepage_size();
  } else {
    self->_alloc = alloc_mem;
    self->_realloc = realloc_mem;
    self->_free = free_mem;
    self->_page_size = get_page_size();
  }


  // Check the type of init:
  // None means no data to initialize
  // int/long means allocate space for that many elements
  // sequence means copy the sequence elements into the array
  if(init == Py_None) {
    self->_data_len = 0;
    self->_alloc_len = 0;
  } else if(PyInt_Check(init)) {
    self->_data_len = PyInt_AsLong(init);
    alloc(self, self->_data_len);
  } else if(PySequence_Check(init)) {
    self->_data_len = PySequence_Size(init);
    //TODO - call alloc
    alloc(self, self->_data_len);

    //for(i = 0; i < self->_data_len; i++) {
      //Set element i to value of PySequence_ITEM(init, i)
    //}
  } else {
    // Throw an exception?
    return -1;
  }

  return 0;
}


static void
NextArray_dealloc(NextArray* self)
{
  self->ob_type->tp_free((PyObject*)self);
}


static PyMemberDef NextArray_members[] = {
  {"typecode", T_CHAR, offsetof(NextArray, typecode), 0, "typecode"},
  {"itemsize", T_INT, offsetof(NextArray, typecode), 0, "typecode"},
  {NULL}
};


static PyMethodDef NextArray_methods[] = {
  //{"alloc", (PyCFunction)NextArray_alloc, METH_VARARGS, "alloc"},
  {NULL}
};


static int NextArray_length(PyObject* self)
{
  return ((NextArray*)self)->_data_len;
}

static int NextArray_setitem(PyObject* self, int ind, PyObject* val)
{
#if 0
  NextArray* na = (NextArray*)self;
  switch(((NextArray*)self)->typecode) {
  case 'c':
  case 'b':
    return _setitem_schar((NextArray*)self, ind, val);
  case 'B':
    return _setitem_uchar((NextArray*)self, ind, val);
  case 'h':
    return _setitem_sshort((NextArray*)self, ind, val);
  case 'H':
    return _setitem_ushort((NextArray*)self, ind, val);
  case 'i':
    return _setitem_sint((NextArray*)self, ind, val);
  case 'I':
    return _setitem_uint((NextArray*)self, ind, val);
  case 'l':
    return _setitem_slong((NextArray*)self, ind, val);
  case 'L':
    ((unsigned long*)na->_memory)[ind] = PyLong_AsLong(val);
    return 0;
    //return _setitem_ulong((NextArray*)self, ind, val);
  case 'u':
    PyErr_SetString(PyExc_NotImplementedError,
        "Unicode not supported by extarray");
    return -1;
  default:
    // TODO - include specified type in string
    PyErr_SetString(PyExc_TypeError, "Unknown array type specified");
    return -1;
  }
#endif
  return ((NextArray*)self)->_setitem((NextArray*)self, ind, val);
}

static PyObject* NextArray_getitem(PyObject* self, int ind)
{
  return ((NextArray*)self)->_getitem((NextArray*)self, ind);
}


static PySequenceMethods NextArray_seqmethods = {
  NextArray_length,               /*sq_length*/
  0,                              /*sq_concat*/
  0,                              /*sq_repeat*/
  NextArray_getitem,              /*sq_item*/
  0,                              /*sq_slice */
  NextArray_setitem,              /*sq_ass_item*/
  0,                              /*sq_ass_slice*/
  0,                              /*sq_contains*/
  0,                              /*sq_inplace_concat*/
  0                               /*sq_inplace_repeat*/
};

static PyTypeObject NextArrayType = {
  PyObject_HEAD_INIT(NULL)
  0,                              /*ob_size*/
  "nextarray.nextarray",          /*tp_name*/
  sizeof(NextArray),              /*tp_basicsize*/
  0,                              /*tp_itemsize*/
  (destructor)NextArray_dealloc,  /*tp_dealloc*/
  0,                              /*tp_print*/
  0,                              /*tp_getattr*/
  0,                              /*tp_setattr*/
  0,                              /*tp_compare*/
  0,                              /*tp_repr*/
  0,                              /*tp_as_number*/
  &NextArray_seqmethods,          /*tp_as_sequence*/
  0,                              /*tp_as_mapping*/
  0,                              /*tp_hash */
  0,                              /*tp_call*/
  0,                              /*tp_str*/
  0,                              /*tp_getattro*/
  0,                              /*tp_setattro*/
  0,                              /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE, /*tp_flags*/
  "NextArray",                    /*tp_doc */
  0,                              /* tp_traverse */
  0,                              /* tp_clear */
  0,                              /* tp_richcompare */
  0,                              /* tp_weaklistoffset */
  0,                              /* tp_iter */
  0,                              /* tp_iternext */
  NextArray_methods,              /* tp_methods */
  NextArray_members,              /* tp_members */
  0,                              /* tp_getset */
  0,                              /* tp_base */
  0,                              /* tp_dict */
  0,                              /* tp_descr_get */
  0,                              /* tp_descr_set */
  0,                              /* tp_dictoffset */
  (initproc)NextArray_init,       /* tp_init */
  0,                              /* tp_alloc */
  NextArray_new,                  /* tp_new */
};

static PyMethodDef module_methods[] = {
    {NULL}  /* Sentinel */
};



#if 0
static PyMethodDef NextArrayMethods[] = {
  {"spam_hello", spam_hello, METH_VARARGS, "hello"},
  {"get_arr_addr", get_arr_addr, METH_VARARGS, "get_arr_addr"},
  {NULL, NULL, 0, NULL} };
#endif

PyMODINIT_FUNC initnextarray(void)
{
  PyObject* m;

  NextArrayType.tp_new = PyType_GenericNew;
  if(PyType_Ready(&NextArrayType) < 0) {
    return;
  }

  m = Py_InitModule3("nextarray", module_methods, "NextArray");
  Py_INCREF(&NextArrayType);
  PyModule_AddObject(m, "nextarray", (PyObject*)&NextArrayType);


  //Py_InitModule("nextarray", NextArrayMethods);
  //import_array()
  printf("initnextarray\n");
}

