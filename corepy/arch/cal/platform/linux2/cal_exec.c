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

#include <Python.h>
#include "structmember.h"

#if (PY_VERSION_HEX < 0x02050000)
typedef int Py_ssize_t;
#endif

#include <stdio.h>
#include <sched.h>
#include "cal.h"
#include "calcl.h"

#ifdef HAS_NUMPY
#include <numpy/arrayobject.h>
#endif

//#ifndef _DEBUG
//#define _DEBUG 0
//#endif

#define CAL_ERROR(fn, ret)                                                \
do { PyErr_Format(PyExc_RuntimeError, "%s: %s", fn, calGetErrorString()); \
     return ret; } while(0)

#define CALCL_ERROR(fn, ret)                                                \
do { PyErr_Format(PyExc_RuntimeError, "%s: %s", fn, calclGetErrorString()); \
     return ret; } while(0)



//Buffer object containing CAL-allocated remote memory for integration with
// NumPy or anything else using the buffer interface.
typedef struct CALMemBuffer {
  PyObject_HEAD;

  CALresource res;
  CALuint fmt;
  CALuint width;
  CALuint pitch;
  CALuint height;
  CALuint length;
  CALuint components;

  CALvoid* ptr;
} CALMemBuffer;

static PyTypeObject CALMemBufferType;


//Internal structure for maintaining handles during asynchronous kernel
//execution.  A pointer to this is returned back to python when the kernel
//is started.  Python passes the pointer back in to the join call to finish
//the kernel execution.

struct ThreadInfo {
  //Dictionary of memory bindings (regname -> [CALresource, void*])
  PyObject* bindings;
  CALmem* mem;

  CALcontext ctx;
  CALmodule mod;
  CALevent event;
};


CALuint cal_device_count = 0;
CALdevice* cal_devices = NULL;
CALdeviceinfo* cal_device_info = NULL;


//
// CAL kernel compilation
//

//Take a string, compile it, and return a kernel image ready to execute
static PyObject* cal_compile(PyObject* self, PyObject* arg)
{
  char* kernel = NULL;
  CALobject obj = NULL;
  CALimage img = NULL;

#ifdef _DEBUG
  CALuint ver[3];
  calclGetVersion(&ver[0], &ver[1], &ver[2]);
  printf("CAL Compiler %d.%d.%d\n", ver[0], ver[1], ver[2]);
#endif

  //Does argument type checking
  kernel = PyString_AsString(arg);
  if(kernel == NULL) {
    return NULL;
  }

#ifdef _DEBUG
  printf("got kernel string:\n%s\n", kernel);
#endif

  //Find the GPU revision for the compiler target
  //TODO - allow the user to specify which device?
  if(calclCompile(&obj, CAL_LANGUAGE_IL, kernel, cal_device_info[0].target)
      != CAL_RESULT_OK)
    CALCL_ERROR("calclCompile", NULL);

  if(calclLink(&img, &obj, 1) != CAL_RESULT_OK)
    CALCL_ERROR("calclLink", NULL);

  return PyLong_FromVoidPtr(img);
}


//Free a compiled kernel image
static PyObject* cal_free_image(PyObject* self, PyObject* arg)
{
  CALimage img = NULL;

  img = (CALimage)PyLong_AsUnsignedLong(arg);
  calclFreeImage(img);

  Py_RETURN_NONE;
}


//Return the number of available GPUs
static PyObject* cal_get_num_gpus(PyObject* self, PyObject* arg)
{
  return PyInt_FromLong((unsigned int)cal_device_count);
}


//
// CAL kernel execution
//

//Allocate a CAL context on a particular device.
static PyObject* cal_alloc_ctx(PyObject* self, PyObject* arg)
{
  CALuint dev_num = PyLong_AsUnsignedLong(arg);
  CALcontext ctx;

  if(calCtxCreate(&ctx, cal_devices[dev_num]) != CAL_RESULT_OK)
    CAL_ERROR("calCtxCreate", NULL);

  return PyLong_FromUnsignedLong((unsigned long)ctx);
}


//Release a CAL context.
static PyObject* cal_free_ctx(PyObject* self, PyObject* arg)
{
  calCtxDestroy(PyLong_AsUnsignedLong(arg));
  Py_RETURN_NONE;
}


//Copy memory from one CAL memory allocation to another.
//Arguments:
// context
// dst resource
// src resource
static PyObject* cal_copy_async(PyObject* self, PyObject* args)
{
  PyObject* dst_binding;
  PyObject* src_binding;
  PyObject* tuple;
  CALcontext ctx;
  CALresource dst_res;
  CALresource src_res;
  CALmem dst_mem;
  CALmem src_mem;
  CALevent event;

  if(!PyArg_ParseTuple(args, "IO!O!", &ctx,
      &PyList_Type, &dst_binding, &PyList_Type, &src_binding)) {
    return NULL;
  }

  dst_res = (CALresource)PyLong_AsLong(PyList_GetItem(dst_binding, 0));
  src_res = (CALresource)PyLong_AsLong(PyList_GetItem(src_binding, 0));

  if(calCtxGetMem(&src_mem, ctx, src_res) != CAL_RESULT_OK)
    CAL_ERROR("calCtxGetMem (src)", NULL);

  if(calCtxGetMem(&dst_mem, ctx, dst_res) != CAL_RESULT_OK)
    CAL_ERROR("calCtxGetMem (dst)", NULL);
    
  if(calMemCopy(&event, ctx, src_mem, dst_mem, 0) != CAL_RESULT_OK)
    CAL_ERROR("calMemCopy", NULL);

  //API requires that this be called to actually start the copy
  calCtxIsEventDone(ctx, event);

  tuple = PyTuple_New(3);
  PyTuple_SET_ITEM(tuple, 0, PyLong_FromUnsignedLong(event));
  PyTuple_SET_ITEM(tuple, 1, PyLong_FromUnsignedLong(dst_mem));
  PyTuple_SET_ITEM(tuple, 2, PyLong_FromUnsignedLong(src_mem));

  return tuple;
}


static PyObject* cal_join_copy(PyObject* self, PyObject* args)
{
  CALcontext ctx;
  CALmem dst_mem;
  CALmem src_mem;
  CALevent event;

  if(!PyArg_ParseTuple(args, "I(III)", &ctx, &event, &dst_mem, &src_mem)) {
    return NULL;
  }

  while(calCtxIsEventDone(ctx, event) == CAL_RESULT_PENDING);

  calCtxReleaseMem(ctx, dst_mem); 
  calCtxReleaseMem(ctx, src_mem); 
  Py_RETURN_NONE;
}


//Bind memory allocations to registers prior to kernel execution.
// Remote allocations are unmapped first.
static CALmem* cal_acquire_bindings(CALcontext ctx, CALmodule mod,
                                    PyObject* bind_dict)
{
  PyObject* key;
  PyObject* value;
  Py_ssize_t pos = 0;
  CALmem* mem;
  int ret;
  int i;

  mem = malloc(sizeof(CALmem) * PyDict_Size(bind_dict));

  for(i = 0; PyDict_Next(bind_dict, &pos, &key, &value); i++) {
    char* regname;
    CALresource res;
    CALvoid* ptr;
    CALname name;

    regname = PyString_AsString(key); 
    res = (CALresource)PyLong_AsLong(PyList_GetItem(value, 0));
    ptr = (CALvoid*)PyLong_AsVoidPtr(PyList_GetItem(value, 1));

#ifdef _DEBUG
    printf("binding memory: %s %d\n", regname, res);
#endif

    //Need to call unMap, GetName, SetMem
    if(ptr != NULL) {
      calResUnmap(res);
    }

    if(calCtxGetMem(&mem[i], ctx, res) != CAL_RESULT_OK)
      CAL_ERROR("calCtxGetMem", NULL);

    ret = calModuleGetName(&name, ctx, mod, regname);
    if(ret != CAL_RESULT_OK)
      CAL_ERROR("calModuleGetName", NULL);

    if(calCtxSetMem(ctx, name, mem[i]) != CAL_RESULT_OK)
      CAL_ERROR("calCtxSetMem", NULL);
  }

  return mem;
}


//Release bindings after kernel execution.
// Remote allocations are re-mapped and their pointer updated.
static int cal_release_bindings(CALcontext ctx, PyObject* bind_dict,
                                CALmem* mem)
{
  PyObject* key;
  PyObject* value;
  Py_ssize_t pos = 0;
  int i;

  for(i = 0; PyDict_Next(bind_dict, &pos, &key, &value); i++) {
    char* regname;
    CALvoid* ptr;
    CALvoid* oldptr;
    CALuint pitch;
    CALresource res;

    regname = PyString_AsString(key);
    res = (CALresource)PyLong_AsLong(PyList_GetItem(value, 0));
    oldptr = (CALvoid*)PyLong_AsVoidPtr(PyList_GetItem(value, 1));

    if(calResMap(&ptr, &pitch, res, 0) != CAL_RESULT_OK)
      CAL_ERROR("calResMap", -1);

    if(ptr != oldptr) {
      PyList_SetItem(value, 1, PyLong_FromVoidPtr(ptr));
    }

    calCtxReleaseMem(ctx, mem[i]);
  }

  free(mem);
  return 0;
}


static PyObject* cal_run_stream_async(PyObject* self, PyObject* args)
{
  //Execute a kernel.
  //Arguments:
  // kernel image
  // context
  // domain (x, y, w, h)
  // dictionary of memory to bind (regname -> [CALresource, void*])
  PyObject* bindings;
  CALmem* mem;
  CALimage img;
  CALcontext ctx;
  CALdomain dom;
  CALmodule mod;
  CALfunc entry;
  CALevent event;
  struct ThreadInfo* ti;

  if(!PyArg_ParseTuple(args, "lI(IIII)O!", (long int*)&img, &ctx,
      &dom.x, &dom.y, &dom.width, &dom.height,
      &PyDict_Type, &bindings)) {
    return NULL;
  }

#ifdef _DEBUG
  printf("executing on context %d domain %d %d -> %d %d\n",
      ctx, dom.x, dom.y, dom.width, dom.height);
#endif

  if(calModuleLoad(&mod, ctx, img) != CAL_RESULT_OK)
    CAL_ERROR("calModuleLoad", NULL);


  //Acquire the memory bindings
  mem = cal_acquire_bindings(ctx, mod, bindings);
  if(mem == NULL) {
    return NULL;
  }


  //Execute the kernel
  if(calModuleGetEntry(&entry, ctx, mod, "main") != CAL_RESULT_OK)
    CAL_ERROR("calModuleGetEntry", NULL);

  if(calCtxRunProgram(&event, ctx, entry, &dom) != CAL_RESULT_OK)
    CAL_ERROR("calCtxRunProgram", NULL);

  if(calCtxIsEventDone(ctx, event) == CAL_RESULT_BAD_HANDLE)
    CAL_ERROR("calCtxIsEventDone", NULL);


  //Set up the ThreadInfo struct to keep track of the handles
  ti = malloc(sizeof(struct ThreadInfo));
  ti->bindings = bindings;
  Py_INCREF(bindings);

  ti->mem = mem;
  ti->ctx = ctx;
  ti->mod = mod;
  ti->event = event;
  return PyLong_FromVoidPtr(ti);
}


static PyObject* cal_join_stream(PyObject* self, PyObject* args)
{
  struct ThreadInfo* ti;

  ti = PyLong_AsVoidPtr(args);

  //Wait on the kernel to complete
  while(calCtxIsEventDone(ti->ctx, ti->event) == CAL_RESULT_PENDING) {
    sched_yield();
  }

  //Remap/free memory bindings
  cal_release_bindings(ti->ctx, ti->bindings, ti->mem);
  Py_DECREF(ti->bindings);

  calModuleUnload(ti->ctx, ti->mod);

  free(ti);
  Py_RETURN_NONE;
}


static PyObject* cal_run_stream(PyObject* self, PyObject* args)
{
  //Execute a kernel.
  //Arguments:
  // kernel image
  // context
  // domain (x, y, w, h)
  // dictionary of memory to bind (regname -> [CALresource, void*])
  PyObject* bindings;
  CALmem* mem;
  CALimage img;
  CALdomain dom;
  CALcontext ctx;
  CALmodule mod;
  CALfunc entry;
  CALevent event;

  if(!PyArg_ParseTuple(args, "lI(IIII)O!", (long int*)&img, &ctx,
      &dom.x, &dom.y, &dom.width, &dom.height, &PyDict_Type, &bindings)) {
    return NULL;
  }

#ifdef _DEBUG
  printf("executing on context %d domain %d %d -> %d %d\n",
      ctx, dom.x, dom.y, dom.width, dom.height);
#endif

  if(calModuleLoad(&mod, ctx, img) != CAL_RESULT_OK)
    CAL_ERROR("calModuleLoad", NULL);


  //Acquire the memory bindings
  mem = cal_acquire_bindings(ctx, mod, bindings);
  if(mem == NULL) {
    return NULL;
  }

  //Execute the kernel
  if(calModuleGetEntry(&entry, ctx, mod, "main") != CAL_RESULT_OK)
    CAL_ERROR("calModuleGetEntry", NULL);

  if(calCtxRunProgram(&event, ctx, entry, &dom) != CAL_RESULT_OK)
    CAL_ERROR("calCtxRunProgram", NULL);

  while(calCtxIsEventDone(ctx, event) == CAL_RESULT_PENDING) {
    sched_yield();
  }


  //Remap/free memory bindings
  cal_release_bindings(ctx, bindings, mem);

  calModuleUnload(ctx, mod);

  Py_RETURN_NONE;
}


//
// Memory Allocation
//

static PyObject* cal_alloc_local(PyObject* self, PyObject* args)
{
  //Arguments:
  //format -- PyInt format constant
  //width in elements
  //height in elements (1 for 1d allocation)
  //flag indicating global or not
  CALuint devnum;
  CALformat fmt;
  CALuint width;
  CALuint height;
  CALresallocflags flag;
  CALresource res;
  PyObject* handle;

  //TODO - make the flag argument optional
  if(!PyArg_ParseTuple(args, "IIIII", &devnum, &fmt, &width, &height, &flag)) {
    return NULL;
  }

  if(height == 1) { //1d allocation
    if(calResAllocLocal1D(&res, cal_devices[devnum],
        width, fmt, flag) != CAL_RESULT_OK)
      CAL_ERROR("calResAllocLocal1D", NULL);
  } else {          //2d allocation
    if(calResAllocLocal2D(&res, cal_devices[devnum],
        width, height, fmt, flag) != CAL_RESULT_OK)
      CAL_ERROR("calResAllocLocal2D", NULL);
  }

  handle = PyList_New(2);
  PyList_SET_ITEM(handle, 0, PyLong_FromUnsignedLong((unsigned long)res));
  PyList_SET_ITEM(handle, 1, PyLong_FromVoidPtr(NULL));
  return handle;
}


static PyObject* cal_free_local(PyObject* self, PyObject* args)
{
  CALresource res;

  res = (CALresource)PyLong_AsLong(PyList_GetItem(args, 0));
  if(res == -1 && PyErr_Occurred()) {
    return NULL;
  }

  calResFree(res);
  
  Py_RETURN_NONE;
}


static PyObject* cal_alloc_remote(PyObject* self, PyObject* args)
{
  //Arguments:
  //format -- PyInt format constant
  //width in elements
  //height in elements (1 for 1d allocation)
  //flag indicating global or not
  CALuint devnum;
  CALformat fmt;
  CALuint width;
  CALuint height;
  CALresallocflags flag;
  CALresource res;
  CALvoid* ptr;
  CALuint pitch;
  PyObject* handle;

  //TODO - make the flag argument optional
  if(!PyArg_ParseTuple(args, "IIIII", &devnum, &fmt, &width, &height, &flag)) {
    return NULL;
  }

  if(height == 1) { //1d allocation
    if(calResAllocRemote1D(&res, &cal_devices[devnum], 1,
        width, fmt, flag) != CAL_RESULT_OK)
      CAL_ERROR("calResAllocRemote1D", NULL);
  } else {          //2d allocation
    if(calResAllocRemote2D(&res, &cal_devices[devnum], 1,
        width, height, fmt, flag) != CAL_RESULT_OK)
      CAL_ERROR("calResAllocRemote2D", NULL);
  }

  if(calResMap(&ptr, &pitch, res, 0) != CAL_RESULT_OK)
    CAL_ERROR("calResMap", NULL);

  //Create a 'binding' to pass back to Python.
  handle = PyList_New(3);
  PyList_SET_ITEM(handle, 0, PyLong_FromUnsignedLong((unsigned long)res));
  PyList_SET_ITEM(handle, 1, PyLong_FromVoidPtr(ptr));
  PyList_SET_ITEM(handle, 2, PyInt_FromLong(pitch));
  return handle;
}


static PyObject* cal_free_remote(PyObject* self, PyObject* args)
{
  CALresource res;

  res = (CALresource)PyLong_AsLong(PyList_GetItem(args, 0));
  if(res == -1 && PyErr_Occurred()) {
    return NULL;
  }

  calResUnmap(res);
  calResFree(res);
  
  Py_RETURN_NONE;
}


#ifdef HAS_NUMPY
static PyObject* cal_set_ndarray_ptr(PyObject* self, PyObject* args)
{
  void* ptr;
  PyArrayObject* arr;

  if(!PyArg_ParseTuple(args, "O!l", &PyArray_Type, &arr, (void**)&ptr)) {
    return NULL;
  }

  arr->data = ptr;

  //TODO - update the pointer in the base CALMemBuffer also

  Py_INCREF(Py_None);
  return Py_None;
}
#else
static PyObject* cal_set_ndarray_ptr(PyObject* self, PyObject* args)
{
  PyErr_SetString(PyExc_NotImplementedError, "NumPy support not enabled");
  return NULL;
}
#endif


static PyMethodDef module_methods[] = {
  {"compile", cal_compile, METH_O, "Compile a CAL IL kernel, return an image"},
  {"free_image", cal_free_image, METH_O, "Free a compiled kernel image"},
  {"get_num_gpus", cal_get_num_gpus, METH_NOARGS, "Return number of GPUs"},
  {"alloc_ctx", cal_alloc_ctx, METH_O, "Allocate a CAL context"},
  {"free_ctx", cal_free_ctx, METH_O, "Release a CAL context"},
  {"copy_async", cal_copy_async, METH_VARARGS, "Start a GPU memory copy"},
  {"join_copy", cal_join_copy, METH_VARARGS, "Finish a GPU memory copy"},
  {"run_stream", cal_run_stream, METH_VARARGS, "Run a kernel on a GPU"},
  {"run_stream_async", cal_run_stream_async, METH_VARARGS, "Run a kernel on a GPU"},
  {"join_stream", cal_join_stream, METH_O, "Join a running kernel"},
  {"alloc_local", cal_alloc_local, METH_VARARGS, "Allocate local memory"},
  {"free_local", cal_free_local, METH_O, "Free local memory"},
  {"alloc_remote", cal_alloc_remote, METH_VARARGS, "Allocate remote memory"},
  {"free_remote", cal_free_remote, METH_O, "Free Remote Memory"},
  {"set_ndarray_ptr", cal_set_ndarray_ptr, METH_VARARGS, "Set ndarray pointer"},
  {NULL}  /* Sentinel */
};



//
// CALMemBuffer
//

static int
CALMemBuffer_init(CALMemBuffer* self, PyObject* args, PyObject* kwds)
{
  CALuint devnum;
  CALresallocflags flag;
  int i;

  //TODO - make the flag argument optional
  if(!PyArg_ParseTuple(args, "IIIII", &devnum, &self->fmt, &self->width, &self->height, &flag)) {
    return -1;
  }


  if(self->height == 1) { //1d allocation
    if(calResAllocRemote1D(&self->res, &cal_devices[devnum], 1,
        self->width, self->fmt, flag) != CAL_RESULT_OK)
      CAL_ERROR("calResAllocRemote1D", -1);
  } else {          //2d allocation
    if(calResAllocRemote2D(&self->res, &cal_devices[devnum], 1,
        self->width, self->height, self->fmt, flag) != CAL_RESULT_OK)
      CAL_ERROR("calResAllocRemote2D", -1);
  }

  if(calResMap(&self->ptr, &self->pitch, self->res, 0) != CAL_RESULT_OK)
    CAL_ERROR("calResMap", -1);

  //Calculate the length
  self->length = self->pitch * self->height;
  switch(self->fmt) {
  case CAL_FORMAT_FLOAT32_4:
  case CAL_FORMAT_SIGNED_INT32_4:
  case CAL_FORMAT_UNSIGNED_INT32_4:
    self->components = 4;
    self->length <<= 4;
    break;
  case CAL_FORMAT_FLOAT32_2:
  case CAL_FORMAT_SIGNED_INT32_2:
  case CAL_FORMAT_UNSIGNED_INT32_2:
    self->length <<= 3;
    self->components = 2;
    break;
  case CAL_FORMAT_FLOAT32_1:
  case CAL_FORMAT_SIGNED_INT32_1:
  case CAL_FORMAT_UNSIGNED_INT32_1:
    self->components = 1;
    self->length <<= 2;
  }

  for(i = 0; i < self->length / 4; i++) {
    ((float*)(self->ptr))[i] = (float)i;
  }

  return 0; 
}


static void
CALMemBuffer_dealloc(CALMemBuffer* self)
{
  calResUnmap(self->res);
  calResFree(self->res);
  
  self->ob_type->tp_free((PyObject*)self);
}


Py_ssize_t CALMemBuffer_readbuffer(PyObject* self, Py_ssize_t seg, void** ptr)
{
  CALMemBuffer* buf = (CALMemBuffer*)self;
  *ptr = buf->ptr;
  return buf->length;
}

Py_ssize_t CALMemBuffer_writebuffer(PyObject* self, Py_ssize_t seg, void** ptr)
{
  CALMemBuffer* buf = (CALMemBuffer*)self;
  *ptr = buf->ptr;
  return buf->length;
}

Py_ssize_t CALMemBuffer_segcount(PyObject* self, Py_ssize_t* len)
{
  CALMemBuffer* buf = (CALMemBuffer*)self;

  if(len != NULL) {
    *len = buf->length;
  }

  return 1;
}


static PyBufferProcs CALMemBuffer_bufferprocs = {
  CALMemBuffer_readbuffer,
  CALMemBuffer_writebuffer,
  CALMemBuffer_segcount,
  NULL
};


static PyMemberDef CALMemBuffer_members[] = {
  {"width", T_INT, offsetof(CALMemBuffer, width), 0, "width"},
  {"height", T_INT, offsetof(CALMemBuffer, height), 0, "height"},
  {"pitch", T_INT, offsetof(CALMemBuffer, pitch), 0, "pitch"},
  {"length", T_INT, offsetof(CALMemBuffer, length), 0, "length"},
  {"format", T_INT, offsetof(CALMemBuffer, fmt), 0, "format"},
  {"pointer", T_LONG, offsetof(CALMemBuffer, ptr), 0, "pointer"},
  {"res", T_INT, offsetof(CALMemBuffer, res), 0, "res"},
  {NULL}
};




static PyTypeObject CALMemBufferType = {
  PyObject_HEAD_INIT(NULL)
  0,                              /*ob_size*/
  "cal_exec.calmembuffer",            /*tp_name*/
  sizeof(CALMemBuffer),               /*tp_basicsize*/
  0,                              /*tp_itemsize*/
  (destructor)CALMemBuffer_dealloc,   /*tp_dealloc*/
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
  &CALMemBuffer_bufferprocs,          /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT,             /*tp_flags*/
  "CALMemBuffer",                     /*tp_doc */
  0,                              /* tp_traverse */
  0,                              /* tp_clear */
  0,                              /* tp_richcompare */
  0,                              /* tp_weaklistoffset */
  0,                              /* tp_iter */
  0,                              /* tp_iternext */
  0,                              /* tp_methods */
  CALMemBuffer_members,               /* tp_members */
  0,
  0,                              /* tp_base */
  0,                              /* tp_dict */
  0,                              /* tp_descr_get */
  0,                              /* tp_descr_set */
  0,  /* tp_dictoffset */
  (initproc)CALMemBuffer_init,        /* tp_init */
  0,                              /* tp_alloc */
  0,                              /* tp_new */
};


PyMODINIT_FUNC initcal_exec(void)
{
  PyObject* mod;
  CALuint i;

  CALMemBufferType.tp_new = PyType_GenericNew;
  if(PyType_Ready(&CALMemBufferType) < 0) {
    return;
  }

  mod = Py_InitModule("cal_exec", module_methods);

  Py_INCREF(&CALMemBufferType);
  PyModule_AddObject(mod, "calmembuffer", (PyObject*)&CALMemBufferType);

#ifdef HAS_NUMPY
  import_array();

  Py_INCREF(Py_True);
  PyModule_AddObject(mod, "HAS_NUMPY", Py_True);
#else
  Py_INCREF(Py_False);
  PyModule_AddObject(mod, "HAS_NUMPY", Py_False);
#endif

  PyModule_AddIntConstant(mod, "FMT_FLOAT32_1", CAL_FORMAT_FLOAT32_1);
  PyModule_AddIntConstant(mod, "FMT_FLOAT32_2", CAL_FORMAT_FLOAT32_2);
  PyModule_AddIntConstant(mod, "FMT_FLOAT32_4", CAL_FORMAT_FLOAT32_4);
  PyModule_AddIntConstant(mod, "FMT_SIGNED_INT32_1", CAL_FORMAT_SIGNED_INT32_1);
  PyModule_AddIntConstant(mod, "FMT_SIGNED_INT32_2", CAL_FORMAT_SIGNED_INT32_2);
  PyModule_AddIntConstant(mod, "FMT_SIGNED_INT32_4", CAL_FORMAT_SIGNED_INT32_4);
  PyModule_AddIntConstant(mod, "FMT_UNSIGNED_INT32_1", CAL_FORMAT_UNSIGNED_INT32_1);
  PyModule_AddIntConstant(mod, "FMT_UNSIGNED_INT32_2", CAL_FORMAT_UNSIGNED_INT32_2);
  PyModule_AddIntConstant(mod, "FMT_UNSIGNED_INT32_4", CAL_FORMAT_UNSIGNED_INT32_4);
  PyModule_AddIntConstant(mod, "GLOBAL_BUFFER", CAL_RESALLOC_GLOBAL_BUFFER);

  calInit();

  calDeviceGetCount(&cal_device_count);
  cal_devices = malloc(sizeof(CALdevice) * cal_device_count);
  cal_device_info = malloc(sizeof(CALdeviceinfo) * cal_device_count);

  for(i = 0; i < cal_device_count; i++) {
    calDeviceGetInfo(&cal_device_info[i], 0);
    calDeviceOpen(&cal_devices[i], i);
  }
}

