#include <Python.h>
#include "structmember.h"
#include <stdio.h>
#include "alloc.h"

//#ifndef _DEBUG
//#define _DEBUG 0
//#endif

typedef struct NextArray {
  PyObject_HEAD

  char typecode;
  char huge;
  char lock;

  int itemsize;
  int data_len;
  int alloc_len;
  int page_size;
  int iter;

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
    self->data_len = 0;
    self->alloc_len = 0;
    self->page_size = 0;

    self->memory = NULL;
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
  case 'f':
    self->itemsize = sizeof(float);
    break;
  case 'd':
    self->itemsize = sizeof(double);
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

  if(self->lock == 1) {
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
    //if(self->memory == NULL) {
    //  self->memory = self->alloc(size);
    //} else {
      self->memory = self->realloc(self->memory, self->alloc_len, size);
    //}

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
  self->lock = 0;
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
    self->data_len = 0;
  } else if(PyInt_Check(init)) {
    self->data_len = PyInt_AsLong(init);
    alloc(self, self->data_len);
  } else if(PySequence_Check(init)) {
    PyObject* item;
    int i;

    self->data_len = PySequence_Size(init);
    alloc(self, self->data_len);

    for(i = 0; i < self->data_len; i++) {
      item = PySequence_ITEM(init, i);
      NextArray_setitem((PyObject*)self, i, item);
      Py_DECREF(item);
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
  if(self->memory != NULL && self->lock == 0) {
#ifdef _DEBUG
    printf("Freeing memory at %p\n", self->memory);
#endif
    self->free(self->memory);
  }

  self->ob_type->tp_free((PyObject*)self);
}


static PyObject* NextArray_alloc(NextArray* self, PyObject* arg)
{
  int length;

  if(!PyArg_ParseTuple(arg, "i", &length)) {
    return NULL;
  }

  if(alloc(self, length) == -1) {
    return NULL;
  }

  Py_INCREF(Py_None);
  return Py_None;
}


static PyObject* NextArray_append(NextArray* self, PyObject* val)
{
  self->data_len++;
  if(alloc(self, self->data_len) == -1) {
    return NULL;
  }

  if(NextArray_setitem((PyObject*)self, self->data_len - 1, val) == -1) {
    return NULL;
  }

  Py_INCREF(Py_None);
  return Py_None;
}


static PyObject* NextArray_buffer_info(NextArray* self, PyObject* val)
{
  return PyTuple_Pack(2,
      PyLong_FromUnsignedLong((unsigned long)self->memory),
      PyInt_FromLong(self->data_len));
}


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

  Py_INCREF(Py_None);
  return Py_None;
}


static PyObject* NextArray_changetype(NextArray* self, PyObject* arg)
{
  char typecode;
  char oldcode = self->typecode;

  //TODO - what about when changing from int to short?  should data_len change?
  if(!PyArg_ParseTuple(arg, "c", &typecode)) {
    return NULL;
  }

  _set_type_fns(self, typecode);

  if((self->data_len * self->itemsize) % oldcode != 0) {
    _set_type_fns(self, oldcode);
    PyErr_SetString(PyExc_TypeError,
        "Array length is not a multiple of new type");
  }

  Py_INCREF(Py_None);
  return Py_None;
}


static PyObject* NextArray_clear(NextArray* self, PyObject* args)
{
  memset(self->memory, 0, self->alloc_len);
  Py_INCREF(Py_None);
  return Py_None;
}


static PyObject* NextArray_copy_direct(NextArray* self, PyObject* arg)
{
  char* buf;
  int len;

  if(PyString_AsStringAndSize(arg, &buf, &len) == -1) {
    return NULL;
  }

  //TODO - what if this doesn't divide evenly?
  self->data_len = len / self->itemsize;
  alloc(self, self->data_len);

  memcpy(self->memory, buf, len);
  Py_INCREF(Py_None);
  return Py_None;
}


static PyObject* NextArray_extend(NextArray* self, PyObject* arg)
{
  int data_len = self->data_len;
  int alloc_len = self->alloc_len;
  PyObject* iter = PyObject_GetIter(arg);
  PyObject* item;

  if(iter == NULL) {
    return NULL;
  }

  while((item = PyIter_Next(iter)) != NULL) {
    alloc(self, self->data_len + 1);

    if(NextArray_setitem((PyObject*)self, self->data_len, item) == -1) {
      if(alloc_len < self->alloc_len) {
        self->memory = self->realloc(self->memory, self->alloc_len, alloc_len);
        self->alloc_len = alloc_len;
        self->data_len = data_len;
      }

      Py_DECREF(item);
      Py_DECREF(iter);
      return NULL;
    }

    self->data_len++;
    Py_DECREF(item);
  }

  Py_DECREF(iter);
  Py_INCREF(Py_None);
  return Py_None;
}


static PyObject* NextArray_fromlist(NextArray* self, PyObject* list)
{
  int data_len = self->data_len;
  int alloc_len = self->alloc_len;
  int len = PyList_Size(list);
  int i;

  alloc(self, self->data_len + len);

  for(i = 0; i < len; i++) {
    if(NextArray_setitem((PyObject*)self,
        self->data_len, PyList_GET_ITEM(list, i)) == -1) {
      if(alloc_len < self->alloc_len) {
        self->memory = self->realloc(self->memory, self->alloc_len, alloc_len);
        self->alloc_len = alloc_len;
        self->data_len = data_len;
      }

      return NULL;
    }

    self->data_len++;
  }

  Py_INCREF(Py_None);
  return Py_None;
}


static PyObject* NextArray_fromstring(NextArray* self, PyObject* list)
{
  return NextArray_fromlist(self, list);
}


static PyObject* NextArray_memory_lock(NextArray* self, PyObject* arg)
{
  if(arg == Py_False) {
    self->lock = 0;
  } else if(arg == Py_True) {
    self->lock = 1;
  }

  return PyBool_FromLong(self->lock);
}


static PyObject* NextArray_synchronize(NextArray* self, PyObject* arg)
{
// TODO - other architectures
#ifdef __powerpc__
  asm("lwsync");
#else
#ifndef  SWIG
//#error "No sync primitives for this platform"
#endif
#endif

  Py_INCREF(Py_None);
  return Py_None;
}


static PyObject* NextArray_iter(PyObject* self)
{
  ((NextArray*)self)->iter = 0;
  Py_INCREF(self);
  return self;
}


static PyObject* NextArray_iternext(PyObject* self)
{
   NextArray* na = (NextArray*)self;
  if(na->iter == na->data_len) {
    return NULL;
  }

  na->iter++;
  return NextArray_getitem(self, na->iter - 1);
}


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
    PyString_ConcatAndDel(&str, PyObject_Str(tmp));
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
  return ((NextArray*)self)->data_len;
}


static int NextArray_setitem(PyObject* self, int ind, PyObject* val)
{
  NextArray* na = (NextArray*)self;

  switch(na->typecode) {
  case 'c':
  case 'b':
    ((char*)na->memory)[ind] = PyLong_AsLong(val);
    return 0;
  case 'B':
    ((unsigned char*)na->memory)[ind] = PyLong_AsUnsignedLongMask(val);
    return 0;
  case 'h':
    ((short*)na->memory)[ind] = PyLong_AsLong(val);
    return 0;
  case 'H':
    ((unsigned short*)na->memory)[ind] = PyLong_AsUnsignedLongMask(val);
    return 0;
  case 'i':
    ((int*)na->memory)[ind] = PyLong_AsLong(val);
    return 0;
  case 'I':
    ((unsigned int*)na->memory)[ind] = PyLong_AsUnsignedLongMask(val);
    return 0;
  case 'l':
    ((long*)na->memory)[ind] = PyLong_AsLong(val);
    return 0;
  case 'L':
    ((unsigned long*)na->memory)[ind] = PyLong_AsUnsignedLongMask(val);
    return 0;
  case 'f':
    ((float*)na->memory)[ind] = PyFloat_AsDouble(val);
    return 0;
  case 'd':
    ((double*)na->memory)[ind] = PyFloat_AsDouble(val);
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
    return PyLong_FromLong(((char*)na->memory)[ind]);
  case 'B':
    return PyLong_FromUnsignedLong(((unsigned char*)na->memory)[ind]);
  case 'h':
    return PyLong_FromLong(((short*)na->memory)[ind]);
  case 'H':
    return PyLong_FromUnsignedLong(((unsigned short*)na->memory)[ind]);
  case 'i':
    return PyLong_FromLong(((int*)na->memory)[ind]);
  case 'I':
    return PyLong_FromUnsignedLong(((unsigned int*)na->memory)[ind]);
  case 'l':
    return PyLong_FromLong(((long*)na->memory)[ind]);
  case 'L':
    return PyLong_FromUnsignedLong(((unsigned long*)na->memory)[ind]);
  case 'f':
    return PyFloat_FromDouble(((float*)na->memory)[ind]);
  case 'd':
    return PyFloat_FromDouble(((double*)na->memory)[ind]);
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


static PyMemberDef NextArray_members[] = {
  {"typecode", T_CHAR, offsetof(NextArray, typecode), 0, "typecode"},
  {"itemsize", T_INT, offsetof(NextArray, itemsize), 0, "itemsize"},
  {NULL}
};


static PyMethodDef NextArray_methods[] = {
  {"alloc", (PyCFunction)NextArray_alloc, METH_O, "alloc"},
  {"append", (PyCFunction)NextArray_append, METH_O, "append"},
  {"buffer_info", (PyCFunction)NextArray_buffer_info, METH_NOARGS, "buffer_info"},
  {"byteswap", (PyCFunction)NextArray_byteswap, METH_NOARGS, "byteswap"},
  {"change_type", (PyCFunction)NextArray_changetype, METH_VARARGS, "change_type"},
  {"clear", (PyCFunction)NextArray_clear, METH_VARARGS, "clear"},
  {"extend", (PyCFunction)NextArray_extend, METH_O, "extend"},
  {"copy_direct", (PyCFunction)NextArray_copy_direct, METH_O, "copy_direct"},
  {"fromlist", (PyCFunction)NextArray_fromlist, METH_O, "fromlist"},
  {"fromstring", (PyCFunction)NextArray_fromstring, METH_O, "fromstring"},
  {"memory_lock", (PyCFunction)NextArray_memory_lock, METH_O, "memory_lock"},
  {"synchronize", (PyCFunction)NextArray_synchronize, METH_NOARGS, "synchronize"},
  {NULL}
};


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
  NextArray_iter,                 /* tp_iter */
  NextArray_iternext,             /* tp_iternext */
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

