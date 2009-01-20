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
  char lock;

  int itemsize;
  int data_len;
  int alloc_len;
  int page_size;

  void* memory;

  void* (*alloc)(int size);
  void* (*realloc)(void* mem, int oldsize, int newsize);
  void (*free)(void* mem);
} NextArray;


static int NextArray_setitem(PyObject* self, int ind, PyObject* val);
static PyObject* NextArray_getitem(PyObject* self, int ind);


static PyObject*
NextArray_new(PyTypeObject* type, PyObject* args, PyObject* kwds)
{
  NextArray* self;

  self = (NextArray*)type->tp_alloc(type, 0);
  if(self != NULL) {
    self->typecode = '\0';
    self->huge = 0;
    self->lock = 0;

    self->itemsize = 0;
    self->_data_len = 0;
    self->alloc_len = 0;
    self->page_size = 0;

    self->_memory = NULL;
    self->alloc = NULL;
    self->realloc = NULL;
    self->free = NULL;
  }

  return (PyObject*)self;
}


static int _set_type_fns(NextArray* self, char typecode)
{
  switch(typecode) {
  case 'c':
  case 'b':
  case 'B':
    self->itemsize = sizeof(char);
    break;
  case 'h':
  case 'H':
    self->itemsize = sizeof(short);
    break;
  case 'i':
  case 'I':
    self->itemsize = sizeof(int);
    break;
  case 'l':
  case 'L':
    self->itemsize = sizeof(long);
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
  m = size % self->page_size;
  if(m != 0) {
    size += self->page_size - m;
  }

  if(self->alloc_len < size) {
    self->memory = self->realloc(self->memory, self->alloc_len, size);
    self->alloc_len = size;
    if(length < self->data_len) {
      self->data_len = length;
    }

#ifdef _DEBUG
    printf("Allocated %d bytes at %p\n", size, self->memory);
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

  if(!PyArg_ParseTupleAndKeywords(args, kwds, "c|Oi",
      kwlist, &typecode, &init, &huge)) {
    return -1;
  }

  self->huge = huge;
  self->_lock = 0;
  self->alloc_len = 0;
  self->memory = NULL;

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
    self->alloc = alloc_hugemem;
    self->realloc = realloc_hugemem;
    self->free = free_hugemem;
    self->page_size = get_hugepage_size();
  } else {
    self->alloc = alloc_mem;
    self->realloc = realloc_mem;
    self->free = free_mem;
    self->page_size = get_page_size();
  }


  // Check the type of init:
  // None means no data to initialize
  // int/long means allocate space for that many elements
  // sequence means copy the sequence elements into the array
  if(init == Py_None) {
    self->_data_len = 0;
  } else if(PyInt_Check(init)) {
    self->data_len = PyInt_AsLong(init);
    alloc(self, self->data_len);
  } else if(PySequence_Check(init)) {
    int i;

    self->data_len = PySequence_Size(init);
    alloc(self, self->data_len);

    for(i = 0; i < self->_data_len; i++) {
      NextArray_setitem((PyObject*)self, i, PySequence_ITEM(init, i));
    }
  } else {
    // Throw an exception?
    return -1;
  }

  return 0;
}


static void
NextArray_dealloc(NextArray* self)
{
  if(self->_memory != NULL && self->_lock == 0) {
#ifdef _DEBUG
    printf("Freeing memory at %p\n", self->_memory);
#endif
    self->_free(self->_memory);
  }

  self->ob_type->tp_free((PyObject*)self);
}


static PyMemberDef NextArray_members[] = {
  {"typecode", T_CHAR, offsetof(NextArray, typecode), 0, "typecode"},
  {"itemsize", T_INT, offsetof(NextArray, typecode), 0, "typecode"},
  {NULL}
};


static PyObject* NextArray_byteswap(NextArray* self, PyObject* args)
{
  int i;

  //TODO - x86 has a bswap instruction, could use it :)
  switch(self->itemsize) {
  case 2:
  {
    unsigned short* addr = (unsigned short*)self->memory;
    int i;

    for(i = 0; i < self->data_len; ++i) {
        addr[i] = (addr[i] << 8) | (addr[i] >> 8);
    }

    break;
  }
  case 4:
  {
    unsigned int* addr = (unsigned int*)self->memory;

    for(i = 0; i < self->data_len; ++i) {
        addr[i] = (addr[i] << 24) | ((addr[i] >> 8) & 0xFF00) |
                ((addr[i] & 0xFF00) << 8) | (addr[i] >> 24);
    }

    break;
  }
  case 8:
  {
    unsigned long long int* addr = (unsigned long long int*)self->memory;

    for(i = 0; i < self->data_len; ++i) {
        addr[i] = (addr[i] << 56) | ((addr[i] >> 48) & 0xFF00) |
                ((addr[i] >> 24) & 0xFF0000) | ((addr[i] >> 8) & 0xFF000000) |
                ((addr[i] & 0xFF000000) << 8) | ((addr[i] & 0xFF0000) << 24) |
                ((addr[i] & 0xFF00) << 48) | (addr[i] >> 56);
    }
    break;
  }
  }

  return Py_None;
}

static PyMethodDef NextArray_methods[] = {
  {"byteswap", (PyCFunction)NextArray_byteswap, METH_VARARGS, "byteswap"},
  {NULL}
};


static PyObject* NextArray_str(PyObject* self)
{
  NextArray* na = (NextArray*)self;
  PyObject* str;

  if(na->data_len == 0) {
    str = PyString_FromFormat("extarray('%c', [])", na->typecode);
  } else {
    PyObject* sep = PyString_FromString(", ");
    PyObject* tmp = NextArray_getitem(self, 0);
    int i;

    str = PyString_FromFormat("extarray('%c', [", na->typecode);
    PyString_ConcatAndDel(&str, tmp->ob_type->tp_str(tmp));
    Py_XDECREF(tmp);

    for(i = 1; i < na->data_len; i++) {
      PyString_Concat(&str, sep);

      tmp = NextArray_getitem(self, i);
      PyString_ConcatAndDel(&str, tmp->ob_type->tp_str(tmp));
      Py_XDECREF(tmp);
    }

    Py_XDECREF(sep);
    PyString_ConcatAndDel(&str, PyString_FromString("])"));
  }

  return str;
}


static int NextArray_length(PyObject* self)
{
  return ((NextArray*)self)->_data_len;
}

static int NextArray_setitem(PyObject* self, int ind, PyObject* val)
{
  NextArray* na = (NextArray*)self;

  switch(na->typecode) {
  case 'c':
  case 'b':
    ((char*)na->_memory)[ind] = PyLong_AsLong(val);
    return 0;
  case 'B':
    ((unsigned char*)na->_memory)[ind] = PyLong_AsUnsignedLong(val);
    return 0;
  case 'h':
    ((short*)na->_memory)[ind] = PyLong_AsLong(val);
    return 0;
  case 'H':
    ((unsigned short*)na->_memory)[ind] = PyLong_AsUnsignedLong(val);
    return 0;
  case 'i':
    ((int*)na->_memory)[ind] = PyLong_AsLong(val);
    return 0;
  case 'I':
    ((unsigned int*)na->_memory)[ind] = PyLong_AsUnsignedLong(val);
    return 0;
  case 'l':
    ((long*)na->_memory)[ind] = PyLong_AsLong(val);
    return 0;
  case 'L':
    ((unsigned long*)na->_memory)[ind] = PyLong_AsUnsignedLongMask(val);
    return 0;
  case 'u':
    PyErr_SetString(PyExc_NotImplementedError,
        "Unicode not supported by extarray");
    return -1;
  default:
    // TODO - include specified type in string
    PyErr_SetString(PyExc_TypeError, "Unknown array type specified");
    return -1;
  }
}

static PyObject* NextArray_getitem(PyObject* self, int ind)
{
  NextArray* na = (NextArray*)self;

  switch(na->typecode) {
  case 'c':
  case 'b':
    return PyLong_FromLong(((char*)na->_memory)[ind]);
  case 'B':
    return PyLong_FromUnsignedLong(((unsigned char*)na->_memory)[ind]);
  case 'h':
    return PyLong_FromLong(((short*)na->_memory)[ind]);
  case 'H':
    return PyLong_FromUnsignedLong(((unsigned short*)na->_memory)[ind]);
  case 'i':
    return PyLong_FromLong(((int*)na->_memory)[ind]);
  case 'I':
    return PyLong_FromUnsignedLong(((unsigned int*)na->_memory)[ind]);
  case 'l':
    return PyLong_FromLong(((long*)na->_memory)[ind]);
  case 'L':
    return PyLong_FromUnsignedLong(((unsigned long*)na->_memory)[ind]);
  case 'u':
    PyErr_SetString(PyExc_NotImplementedError,
        "Unicode not supported by extarray");
    return NULL;
  default:
    // TODO - include specified type in string
    PyErr_SetString(PyExc_TypeError, "Unknown array type specified");
    return NULL;
  }
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
  NextArray_str,                  /*tp_str*/
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
}

