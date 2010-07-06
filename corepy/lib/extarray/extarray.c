#include <Python.h>
#include "structmember.h"

#if (PY_VERSION_HEX < 0x02050000)
typedef int Py_ssize_t;
#endif

#include <stdio.h>
#include "alloc.h"

#if defined(__APPLE__) & defined( __MACH__)
#include <sys/mman.h>
#endif

//#ifndef _DEBUG
//#define _DEBUG 0
//#endif

//Make sure Py_ssize_t is defined

typedef struct ExtArray {
  PyObject_HEAD

  PyObject* attr_dict;
  char typecode;          //Type of array elements
  unsigned char huge;     //Boolean, 1 if huge pages are used, 0 otherwise
  char lock;              //Boolean, 1 if memory is 'locked' eg no realloc

  int page_size;          //Memory page size in bytes
  int itemsize;           //Size of a single element in bytes
  Py_ssize_t data_len;    //Data length counted in items
  Py_ssize_t alloc_len;   //Allocated memory length counted in bytes
  Py_ssize_t iter;        //Counter for supporting iterating over extarrays

  void* memory;           //Pointer to the memory backing the array

  //Functions for allocating/freeing memory
  void* (*realloc)(void* mem, Py_ssize_t oldsize, Py_ssize_t newsize);
  void (*free)(void* mem);

  //If this extarray is a slice of another extarray, this is a reference to
  // that other extarray
  struct ExtArray* arr_ref;
} ExtArray;


static PyTypeObject ExtArrayType;
static int ExtArray_setitem(PyObject* self, Py_ssize_t ind, PyObject* val);
static PyObject* ExtArray_getitem(PyObject* self, Py_ssize_t ind);


#if 0
static PyObject*
ExtArray_new(PyTypeObject* type, PyObject* args, PyObject* kwds)
{
  ExtArray* self;

  self = (ExtArray*)type->tp_alloc(type, 0);
  if(self != NULL) {
    self->typecode = '\0';
    self->huge = 0;
    self->lock = 0;

    self->itemsize = 0;
    self->data_len = 0;
    self->alloc_len = 0;
    self->page_size = 0;

    self->memory = NULL;
    self->realloc = NULL;
    self->free = NULL;

    self->arr_ref = NULL;
  }

  return (PyObject*)self;
}
#endif


//Set the type and item size
static int set_type(ExtArray* self, char typecode)
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


//Allocate memory for length elements, rounding the allocation up to a
// multiple of a page.
static int alloc(ExtArray* self, Py_ssize_t length)
{
  Py_ssize_t size;
  Py_ssize_t m;

  if(self->lock == 1) {
    PyErr_SetString(PyExc_MemoryError,
        "Attempt to allocate with memory lock enabled");
    return -1;
  }

  //Round size to a page
  size = length * self->itemsize;
  m = size % self->page_size;
  if(m != 0) {
    size += self->page_size - m;
  }

  //Only realloc memory if the new size is larger
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
ExtArray_init(ExtArray* self, PyObject* args, PyObject* kwds)
{
  //Python-level extarray constructor:
  //def __init__(self, typecode, init = None, huge = False):
  static char* kwlist[] = {"typecode", "init", "huge", NULL};
  char typecode;
  unsigned char huge = 0;
  PyObject* init = Py_None;

  if(!PyArg_ParseTupleAndKeywords(args, kwds, "c|Ob",
      kwlist, &typecode, &init, &huge)) {
    return -1;
  }

  self->huge = huge;
  self->lock = 0;
  self->alloc_len = 0;
  self->memory = NULL;

  self->arr_ref = NULL;

  //TODO - replace has_huge_pages with a define
  if(huge == 1 && has_huge_pages() == 0) {
    PyErr_SetString(PyExc_MemoryError,
        "No huge pages available, try regular pages");
    return -1;
  }

  if(set_type(self, typecode) != 0) {
    return -1;
  }

  if(huge == 1) {
    self->realloc = realloc_hugemem;
    self->free = free_hugemem;
    self->page_size = get_hugepage_size();
  } else {
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
  } else if(PyLong_Check(init)) {
    self->data_len = PyLong_AsLong(init);
    alloc(self, self->data_len);
  } else if(PySequence_Check(init)) {
    PyObject* item;
    int i;

    self->data_len = PySequence_Size(init);
    alloc(self, self->data_len);

    for(i = 0; i < self->data_len; i++) {
      item = PySequence_ITEM(init, i);
      ExtArray_setitem((PyObject*)self, i, item);
      Py_DECREF(item);
    }
  } else {
    // Throw an exception?
    return -1;
  }

  return 0;
}


static void
ExtArray_dealloc(ExtArray* self)
{
  //Python-level extarray destructor
  //Free the memory if not locked, decrement references, and go away
  if(self->memory != NULL && self->lock == 0) {
#ifdef _DEBUG
    printf("Freeing memory at %p\n", self->memory);
#endif
    self->free(self->memory);
  }

  if(self->arr_ref != NULL) {
    Py_DECREF(self->arr_ref);
  }

  self->ob_type->tp_free((PyObject*)self);
}


//User-interface function to allocate memory for length elements
static PyObject* ExtArray_alloc(ExtArray* self, PyObject* arg)
{
  Py_ssize_t length;

  if(!PyArg_ParseTuple(arg, "l", &length)) {
    return NULL;
  }

  if(alloc(self, length) == -1) {
    return NULL;
  }

  Py_INCREF(Py_None);
  return Py_None;
}


//Append a value to the array
static PyObject* ExtArray_append(ExtArray* self, PyObject* val)
{
  self->data_len++;
  if(alloc(self, self->data_len) == -1) {
    return NULL;
  }

  if(ExtArray_setitem((PyObject*)self, self->data_len - 1, val) == -1) {
    return NULL;
  }

  Py_INCREF(Py_None);
  return Py_None;
}


//Return a tuple containing the memory address and data length
static PyObject* ExtArray_buffer_info(ExtArray* self, PyObject* val)
{
  PyObject* tuple = PyTuple_New(2);

  PyTuple_SET_ITEM(tuple, 0,
      PyLong_FromUnsignedLong((unsigned long)self->memory));
  PyTuple_SET_ITEM(tuple, 1,
      PyInt_FromLong(self->data_len));

  return tuple;
}


//Change the endianness of all elements in the extarray
static PyObject* ExtArray_byteswap(ExtArray* self, PyObject* args)
{
  Py_ssize_t i;

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


//Change the type of the elements in the array; the array length in bytes
//must be a multiple of the size of the new type
static PyObject* ExtArray_change_type(ExtArray* self, PyObject* arg)
{
  char typecode;
  char itemsize = self->itemsize;
  char oldcode = self->typecode;

  //TODO - what about when changing from int to short?  should data_len change?
  if(!PyArg_ParseTuple(arg, "c", &typecode)) {
    return NULL;
  }

  set_type(self, typecode);

  if((self->data_len * self->itemsize) % itemsize != 0) {
    set_type(self, oldcode);
    PyErr_SetString(PyExc_TypeError,
        "Array length is not a multiple of new type");
    return NULL;
  }

  Py_INCREF(Py_None);
  return Py_None;
}


//Quickly clear the array data to zero
static PyObject* ExtArray_clear(ExtArray* self, PyObject* args)
{
  memset(self->memory, 0, self->alloc_len);
  Py_INCREF(Py_None);
  return Py_None;
}


//Directly copy the data in a character buffer to the array
static PyObject* ExtArray_copy_direct(ExtArray* self, PyObject* arg)
{
  char* buf;
  Py_ssize_t len;

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


//Extend the array with the contents of another iterable object
static PyObject* ExtArray_extend(ExtArray* self, PyObject* arg)
{
  Py_ssize_t data_len = self->data_len;
  Py_ssize_t alloc_len = self->alloc_len;
  PyObject* iter = PyObject_GetIter(arg);
  PyObject* item;

  if(iter == NULL) {
    return NULL;
  }

  while((item = PyIter_Next(iter)) != NULL) {
    alloc(self, self->data_len + 1);

    if(ExtArray_setitem((PyObject*)self, self->data_len, item) == -1) {
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


//Extend the array with data from another list
static PyObject* ExtArray_fromlist(ExtArray* self, PyObject* list)
{
  Py_ssize_t data_len = self->data_len;
  Py_ssize_t alloc_len = self->alloc_len;
  Py_ssize_t len = PyList_Size(list);
  Py_ssize_t i;

  alloc(self, self->data_len + len);

  for(i = 0; i < len; i++) {
    if(ExtArray_setitem((PyObject*)self,
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


//Extend the array with characters of a string
static PyObject* ExtArray_fromstring(ExtArray* self, PyObject* str)
{
  return ExtArray_copy_direct(self, str);
}


//Adjust the memory access to allow execution
static PyObject* ExtArray_make_executable(ExtArray* self, PyObject* arg)
{
// TODO - other architectures
#if defined(__APPLE__) & defined( __MACH__)
  // TODO - AWF - should query for the page size instead of just masking
  //sys_icache_invalidate((char *)addr, size * 4);
  if(mprotect(self->memory, self->alloc_len, 
        PROT_READ | PROT_WRITE | PROT_EXEC) == -1) {
  //if(mprotect((void *)(addr & 0xFFFFF000), size + (addr & 0xFFF), 
  //      PROT_READ | PROT_WRITE | PROT_EXEC) == -1) {
    PyErr_SetString(PyExc_OSError, "mprotect() failed");
  }

#else
//#error "make_executable() not implemented for this platform"
#endif

  Py_RETURN_NONE;
}

//Lock the array memory, preventing it from being re-allocated (moved)
static PyObject* ExtArray_memory_lock(ExtArray* self, PyObject* arg)
{
  if(arg == Py_False) {
    self->lock = 0;
  } else if(arg == Py_True) {
    self->lock = 1;
  }

  return PyBool_FromLong(self->lock);
}


//Set the array length in number of elements.  Memory is allocated if length
//is longer than the current memory allocation.  Any newly allocated memory at
//the end of the array will have undefined values.
static PyObject* ExtArray_set_length(ExtArray* self, PyObject* arg)
{
  unsigned long length;

  length = PyLong_AsLong(arg);

  //Allocate more memory for the length, if needed.
  alloc(self, length);

  self->data_len = length;
  Py_RETURN_NONE;
}


//Set the array's internal memory pointer to user-specified memory
// Memory is locked as part of this operation; this array will not re-allocate
// or free it.
static PyObject* ExtArray_set_memory(ExtArray* self, PyObject* arg)
{
  unsigned long addr;
  Py_ssize_t len;

  if(!PyArg_ParseTuple(arg, "kl", &addr, &len)) {
    return NULL;
  }

  //Check that the length is a multiple of the itemsize.
  if(len % self->itemsize != 0) {
    PyErr_SetString(PyExc_TypeError,
        "Memory length is not a multiple of array type");
    return NULL;
  }

  //Free memory if needed
  if(self->memory != NULL && self->lock != 1) {
    self->free(self->memory);
  }

  self->memory = (void*)addr;
  //self->memory = (void*)PyLong_AsUnsignedLong(arg);
  self->alloc_len = len;
  self->data_len = len / self->itemsize;
  self->lock = 1;

  Py_RETURN_NONE;
}


//Execute an architecture-specific memory fence/synchronization
static PyObject* ExtArray_synchronize(ExtArray* self, PyObject* arg)
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


//Write the raw data to a specified file
static PyObject* ExtArray_tofile(ExtArray* self, PyObject* arg)
{
  PyObject* fobj;
  FILE* fd;
  size_t len;

  if(!PyArg_ParseTuple(arg, "O!", &PyFile_Type, &fobj)) {
    return NULL;
  }

  fd = PyFile_AsFile(fobj);

  len = self->data_len * self->itemsize;
  if(fwrite(self->memory, len, 1, fd) != 1) {
    PyErr_SetString(PyExc_OSError,
        "Fewer bytes written than expected");
    return NULL;
  }

  Py_RETURN_NONE;
}


//Array iteration support (__iter__)
static PyObject* ExtArray_iter(PyObject* self)
{
  ((ExtArray*)self)->iter = 0;
  Py_INCREF(self);
  return self;
}


//Array iteration support
static PyObject* ExtArray_iternext(PyObject* self)
{
   ExtArray* na = (ExtArray*)self;
  if(na->iter == na->data_len) {
    return NULL;
  }

  na->iter++;
  return ExtArray_getitem(self, na->iter - 1);
}


//__str__
static PyObject* ExtArray_str(PyObject* self)
{
  ExtArray* na = (ExtArray*)self;
  PyObject* str;

  if(na->data_len == 0) {
    str = PyString_FromFormat("extarray('%c', [])", na->typecode);
  } else {
    PyObject* sep = PyString_FromString(", ");
    PyObject* tmp = ExtArray_getitem(self, 0);
    Py_ssize_t i;

    str = PyString_FromFormat("extarray('%c', [", na->typecode);
    PyString_ConcatAndDel(&str, PyObject_Str(tmp));
    Py_XDECREF(tmp);

    for(i = 1; i < na->data_len; i++) {
      PyString_Concat(&str, sep);

      tmp = ExtArray_getitem(self, i);
      PyString_ConcatAndDel(&str, tmp->ob_type->tp_str(tmp));
      Py_XDECREF(tmp);
    }

    Py_XDECREF(sep);
    PyString_ConcatAndDel(&str, PyString_FromString("])"));
  }

  return str;
}


//__len__
static Py_ssize_t ExtArray_length(PyObject* self)
{
  return ((ExtArray*)self)->data_len;
}


//__setitem__
static int ExtArray_setitem(PyObject* self, Py_ssize_t ind, PyObject* val)
{
  ExtArray* na = (ExtArray*)self;

  switch(na->typecode) {
  case 'c':
  case 'b':
    ((char*)na->memory)[ind] = PyLong_AsLong(val);
    break;
  case 'B':
    ((unsigned char*)na->memory)[ind] = PyLong_AsUnsignedLongMask(val);
    break;
  case 'h':
    ((short*)na->memory)[ind] = PyLong_AsLong(val);
    break;
  case 'H':
    ((unsigned short*)na->memory)[ind] = PyLong_AsUnsignedLongMask(val);
    break;
  case 'i':
    ((int*)na->memory)[ind] = PyLong_AsLong(val);
    break;
  case 'I':
    ((unsigned int*)na->memory)[ind] = PyLong_AsUnsignedLongMask(val);
    break;
  case 'l':
    ((long*)na->memory)[ind] = PyLong_AsLong(val);
    break;
  case 'L':
    ((unsigned long*)na->memory)[ind] = PyLong_AsUnsignedLongMask(val);
    break;
  case 'f':
    ((float*)na->memory)[ind] = PyFloat_AsDouble(val);
    break;
  case 'd':
    ((double*)na->memory)[ind] = PyFloat_AsDouble(val);
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

  if(PyErr_Occurred() != NULL) {
    PyErr_Format(PyExc_TypeError, "Invalid argument type for array type '%c'", na->typecode);
    return -1;
  }

  return 0;
}


//__getitem__
static PyObject* ExtArray_getitem(PyObject* self, Py_ssize_t ind)
{
  ExtArray* na = (ExtArray*)self;

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


//Array slicing support similar to NumPy array views
static PyObject* ExtArray_getslice(PyObject *self, Py_ssize_t i1, Py_ssize_t i2)
{
  ExtArray* arr = (ExtArray*)self;
  ExtArray* new_arr;

  if(i2 == PY_SSIZE_T_MAX) {
    i2 = arr->data_len;
  }

  //Create a new object using this array's memory backing it, and memory locked
  new_arr = PyObject_New(ExtArray, &ExtArrayType);

  new_arr->typecode = arr->typecode;
  new_arr->itemsize = arr->itemsize;

  //Memory should always be locked
  //TODO - add an additional lock not controllable by the user?
  new_arr->huge = arr->huge;
  new_arr->realloc = arr->realloc;
  new_arr->free = arr->free;
  new_arr->page_size = arr->page_size;

  new_arr->lock = 1;
  new_arr->alloc_len = 0;

  if(i2 <= i1) {
    //Return an empty array
    new_arr->memory = NULL;
    new_arr->data_len = 0;
  } else {
    new_arr->memory = (char*)arr->memory + i1 * arr->itemsize;
    new_arr->data_len = i2 - i1;
  }

  new_arr->arr_ref = arr;
  Py_INCREF(arr);
  return (PyObject*)new_arr;
}


//Set the values of a slice of the array using a sequence
static int ExtArray_setslice(PyObject *self, Py_ssize_t i1, Py_ssize_t i2, PyObject* v)
{
  ExtArray* arr = (ExtArray*)self;
  PyObject* seq;
  PyObject** items;
  int len;
  int i;

  if(i2 == INT_MAX) {
    i2 = arr->data_len;
  }

  seq = PySequence_Fast(v, NULL);
  if(seq == NULL) {
    return -1;
  }

  len = PySequence_Fast_GET_SIZE(seq);

  if(i2 - i1 != len) {
    PyErr_BadArgument();
    Py_DECREF(seq);
    return -1;
  }

  items = PySequence_Fast_ITEMS(seq);

  for(i = 0; i < len; i++) {
    if(ExtArray_setitem(self, i1 + i, items[i]) == -1) {
      Py_DECREF(seq);
      return -1;
    }
  }

  Py_DECREF(seq);
  return 0;
}


static PyMemberDef ExtArray_members[] = {
  {"typecode", T_CHAR, offsetof(ExtArray, typecode), 0, "typecode"},
  {"itemsize", T_INT, offsetof(ExtArray, itemsize), 0, "itemsize"},
  {"data_len", T_INT, offsetof(ExtArray, data_len), 0, "data_len"},
  {"memory", T_ULONG, offsetof(ExtArray, memory), 0, "memory"},
  {NULL}
};


static PyMethodDef ExtArray_methods[] = {
  {"alloc", (PyCFunction)ExtArray_alloc, METH_O, "alloc"},
  {"append", (PyCFunction)ExtArray_append, METH_O, "append"},
  {"buffer_info", (PyCFunction)ExtArray_buffer_info, METH_NOARGS, "buffer_info"},
  {"byteswap", (PyCFunction)ExtArray_byteswap, METH_NOARGS, "byteswap"},
  {"change_type", (PyCFunction)ExtArray_change_type, METH_VARARGS, "change_type"},
  {"clear", (PyCFunction)ExtArray_clear, METH_VARARGS, "clear"},
  {"extend", (PyCFunction)ExtArray_extend, METH_O, "extend"},
  {"copy_direct", (PyCFunction)ExtArray_copy_direct, METH_O, "copy_direct"},
  {"fromlist", (PyCFunction)ExtArray_fromlist, METH_O, "fromlist"},
  {"fromstring", (PyCFunction)ExtArray_fromstring, METH_O, "fromstring"},
  {"make_executable", (PyCFunction)ExtArray_make_executable, METH_NOARGS, "make_executable"},
  {"memory_lock", (PyCFunction)ExtArray_memory_lock, METH_O, "memory_lock"},
  {"set_length", (PyCFunction)ExtArray_set_length, METH_O, "set_length"},
  {"set_memory", (PyCFunction)ExtArray_set_memory, METH_VARARGS, "set_memory"},
  {"synchronize", (PyCFunction)ExtArray_synchronize, METH_NOARGS, "synchronize"},
  {"tofile", (PyCFunction)ExtArray_tofile, METH_VARARGS, "tofile"},
  {NULL}
};


static PySequenceMethods ExtArray_seqmethods = {
  ExtArray_length,                /*sq_length*/
  0,                              /*sq_concat*/
  0,                              /*sq_repeat*/
  ExtArray_getitem,               /*sq_item*/
  ExtArray_getslice,              /*sq_slice */
  ExtArray_setitem,               /*sq_ass_item*/
  ExtArray_setslice,              /*sq_ass_slice*/
  0,                              /*sq_contains*/
  0,                              /*sq_inplace_concat*/
  0                               /*sq_inplace_repeat*/
};


static PyTypeObject ExtArrayType = {
  PyObject_HEAD_INIT(NULL)
  0,                              /*ob_size*/
  "extarray.extarray",            /*tp_name*/
  sizeof(ExtArray),               /*tp_basicsize*/
  0,                              /*tp_itemsize*/
  (destructor)ExtArray_dealloc,   /*tp_dealloc*/
  0,                              /*tp_print*/
  0,                              /*tp_getattr*/
  0,                              /*tp_setattr*/
  0,                              /*tp_compare*/
  0,                              /*tp_repr*/
  0,                              /*tp_as_number*/
  &ExtArray_seqmethods,           /*tp_as_sequence*/
  0,                              /*tp_as_mapping*/
  0,                              /*tp_hash */
  0,                              /*tp_call*/
  ExtArray_str,                   /*tp_str*/
  PyObject_GenericGetAttr,        /*tp_getattro*/
  PyObject_GenericSetAttr,        /*tp_setattro*/
  0,                              /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE, /*tp_flags*/
  "ExtArray",                     /*tp_doc */
  0,                              /* tp_traverse */
  0,                              /* tp_clear */
  0,                              /* tp_richcompare */
  0,                              /* tp_weaklistoffset */
  ExtArray_iter,                  /* tp_iter */
  ExtArray_iternext,              /* tp_iternext */
  ExtArray_methods,               /* tp_methods */
  ExtArray_members,               /* tp_members */
  //ExtArray_getset,                /* tp_getset */
  0,
  0,                              /* tp_base */
  0,                              /* tp_dict */
  0,                              /* tp_descr_get */
  0,                              /* tp_descr_set */
  offsetof(ExtArray, attr_dict),  /* tp_dictoffset */
  (initproc)ExtArray_init,        /* tp_init */
  0,                              /* tp_alloc */
  0, /*(ExtArray_new,*/           /* tp_new */
};

static PyMethodDef module_methods[] = {
    {NULL}  /* Sentinel */
};



typedef struct ExtBuffer {
  PyObject_HEAD;

  void* memory;
  Py_ssize_t data_len;
  Py_ssize_t alloc_len;
  int page_size;
  unsigned char huge;
  unsigned char do_free;
} ExtBuffer;


static int
ExtBuffer_init(ExtBuffer* self, PyObject* args, PyObject* kwds)
{
  //Python-level extbuffer constructor:
  //def __init__(self, data_len, memory = NULL, huge = False)
  static char* kwlist[] = {"data_len", "memory", "huge", NULL};

  self->memory = NULL;
  self->page_size = 0;
  self->huge = 0;
  self->do_free = 0;

  if(!PyArg_ParseTupleAndKeywords(args, kwds, "i|kb",
      kwlist, &self->data_len, &self->memory, &self->huge)) {
    return -1;
  }

  //If memory is NULL and length is not zero, allocate some memory
  if(self->memory != NULL || self->data_len == 0) {
    return 0;
  }

  //Allocate our own memory
  self->do_free = 1;
  if(self->huge == 1) {
    //Huge-page memory
    Py_ssize_t m;

    if(has_huge_pages() == 0) {
      PyErr_SetString(PyExc_MemoryError,
          "No huge pages available, try regular pages");
      return -1;
    }


    self->alloc_len = self->data_len;
    self->page_size = get_hugepage_size();

    //Round the allocation up to a page
    m = self->alloc_len % self->page_size;
    if(m != 0) {
      self->alloc_len += self->page_size - m;
    }

    self->memory = alloc_hugemem(self->alloc_len);
  } else {
    //Regular memory
    Py_ssize_t m;

    self->alloc_len = self->data_len;
    self->page_size = get_page_size();

    //Round the allocation up to a page
    m = self->alloc_len % self->page_size;
    if(m != 0) {
      self->alloc_len += self->page_size - m;
    }

    self->memory = alloc_mem(self->alloc_len);
  }

  return 0; 
}


static void
ExtBuffer_dealloc(ExtBuffer* self)
{
  //Python-level destructor, free the memory and go away
  if(self->do_free == 1) {
    if(self->huge == 1 && self->memory != NULL) {
      free_hugemem(self->memory);
    } else {
      free_mem(self->memory);
    }
  }

  self->ob_type->tp_free((PyObject*)self);
}


//Return a read-only pointer to the memory buffer
Py_ssize_t ExtBuffer_readbuffer(PyObject* self, Py_ssize_t seg, void** ptr)
{
  ExtBuffer* buf = (ExtBuffer*)self;
  *ptr = buf->memory;
  return buf->data_len;
}

//Return a read/write pointer to the memory buffer
Py_ssize_t ExtBuffer_writebuffer(PyObject* self, Py_ssize_t seg, void** ptr)
{
  ExtBuffer* buf = (ExtBuffer*)self;
  *ptr = buf->memory;
  return buf->data_len;
}

//Return the number of memory segments (always 1) and total memory length
Py_ssize_t ExtBuffer_segcount(PyObject* self, Py_ssize_t* len)
{
  ExtBuffer* buf = (ExtBuffer*)self;

  if(len != NULL) {
    *len = buf->data_len;
  }

  return 1;
}


static PyBufferProcs ExtBuffer_bufferprocs = {
  ExtBuffer_readbuffer,
  ExtBuffer_writebuffer,
  ExtBuffer_segcount,
  NULL
};


static PyMemberDef ExtBuffer_members[] = {
  {"huge", T_UBYTE, offsetof(ExtBuffer, huge), 0, "Huge pages used?"},
  {"data_len", T_INT, offsetof(ExtBuffer, data_len), 0, "Data length"},
  {"alloc_len", T_INT, offsetof(ExtBuffer, alloc_len), 0, "Allocated length"},
  {"memory", T_LONG, offsetof(ExtBuffer, memory), 0, "memory"},
  {NULL}
};



static PyTypeObject ExtBufferType = {
  PyObject_HEAD_INIT(NULL)
  0,                              /*ob_size*/
  "extarray.extbuffer",           /*tp_name*/
  sizeof(ExtBuffer),              /*tp_basicsize*/
  0,                              /*tp_itemsize*/
  (destructor)ExtBuffer_dealloc,  /*tp_dealloc*/
  0,                              /*tp_print*/
  0,                              /*tp_getattr*/
  0,                              /*tp_setattr*/
  0,                              /*tp_compare*/
  0,                              /*tp_repr*/
  0,                              /*tp_as_number*/
  0,                              /*tp_as_sequence*/
  0,                              /*tp_as_mapping*/
  0,                              /*tp_hash */
  0,                              /*tp_call*/
  0,                              /*tp_str*/
  0,                              /*tp_getattro*/
  0,                              /*tp_setattro*/
  &ExtBuffer_bufferprocs,         /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT,             /*tp_flags*/
  "ExtBuffer",                    /*tp_doc */
  0,                              /* tp_traverse */
  0,                              /* tp_clear */
  0,                              /* tp_richcompare */
  0,                              /* tp_weaklistoffset */
  0,                              /* tp_iter */
  0,                              /* tp_iternext */
  0,                              /* tp_methods */
  ExtBuffer_members,              /* tp_members */
  0,
  0,                              /* tp_base */
  0,                              /* tp_dict */
  0,                              /* tp_descr_get */
  0,                              /* tp_descr_set */
  0,  /* tp_dictoffset */
  (initproc)ExtBuffer_init,       /* tp_init */
  0,                              /* tp_alloc */
  0,                              /* tp_new */
};


PyMODINIT_FUNC initextarray(void)
{
  PyObject* m;

  ExtArrayType.tp_new = PyType_GenericNew;
  if(PyType_Ready(&ExtArrayType) < 0) {
    return;
  }

  ExtBufferType.tp_new = PyType_GenericNew;
  if(PyType_Ready(&ExtBufferType) < 0) {
    return;
  }

  m = Py_InitModule3("extarray", module_methods, "ExtArray");

  Py_INCREF(&ExtArrayType);
  PyModule_AddObject(m, "extarray", (PyObject*)&ExtArrayType);
  Py_INCREF(&ExtBufferType);
  PyModule_AddObject(m, "extbuffer", (PyObject*)&ExtBufferType);
}

