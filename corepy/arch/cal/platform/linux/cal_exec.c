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


//Internal structure used for keeping track of various handles for copy_local
// bindings.  An array of this struct gets created prior to executing code,
// then used and released immediately after in the run_stream() calls.

struct CopyBindingRecord {
  CALmem remotemem;
  CALmem localmem;
  CALresource localres;

  PyObject* regname;
  PyObject* binding;
};


CALuint cal_device_count = 0;
CALdevice* cal_devices = NULL;
CALdeviceinfo* cal_device_info = NULL;


//
// CAL kernel compilation
//

static PyObject* cal_compile(PyObject* self, PyObject* arg)
{
  char* kernel = NULL;
  CALobject obj = NULL;
  CALimage img = NULL;
  //Take a string, compile it, and return a kernel image ready to execute

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


static PyObject* cal_free_image(PyObject* self, PyObject* arg)
{
  CALimage img = NULL;

  img = (CALimage)PyLong_AsUnsignedLong(arg);
  calclFreeImage(img);

  Py_INCREF(Py_None);
  return Py_None;
}


static PyObject* cal_get_num_gpus(PyObject* self, PyObject* arg)
{
  return PyInt_FromLong((unsigned int)cal_device_count);
}


//
// CAL kernel execution
//

//Allocate local memory, copy remote memory to the local, and bind the local
//memory instead of the user's remote memory.
static struct CopyBindingRecord* cal_bind_copy_memory(PyObject* bind_dict,
    CALuint dev_num, CALcontext ctx, CALmodule mod)
{
  PyObject* key;
  PyObject* value;
  Py_ssize_t pos = 0;
  struct CopyBindingRecord* recs;
  int ret;
  int i;

  recs = malloc(sizeof(struct CopyBindingRecord) * PyDict_Size(bind_dict));

  for(i = 0; PyDict_Next(bind_dict, &pos, &key, &value); i++) {
    char* regname;
    CALresource res;
    CALname name;
    CALuint width;
    CALuint height;
    CALformat fmt;
    CALresallocflags flag = 0;
    CALevent event;

    regname = PyString_AsString(key); 

    width = PyInt_AsLong(PyList_GetItem(value, 1));
    height = PyInt_AsLong(PyList_GetItem(value, 2));
    fmt = PyLong_AsUnsignedLong(PyList_GetItem(value, 3));
    res = (CALresource)PyLong_AsLong(PyList_GetItem(value, 4));

    if(!strcmp("g[]", regname)) {
      flag = CAL_RESALLOC_GLOBAL_BUFFER;
    }

#ifdef _DEBUG
    printf("binding remote memory: %s\n", regname);
#endif

    //Need to call unMap, GetName, SetMem
    calResUnmap(res);

    if(calCtxGetMem(&recs[i].remotemem, ctx, res) != CAL_RESULT_OK)
      CAL_ERROR("calCtxGetMem", NULL);

    //if(calModuleGetName(&name, ctx, mod, regname) != CAL_RESULT_OK) {
    ret = calModuleGetName(&name, ctx, mod, regname);
    if(ret != CAL_RESULT_OK)
      CAL_ERROR("calModuleGetName", NULL);

    //Allocate the memory first
    if(height == 1) { //1d allocation
      if(calResAllocLocal1D(&recs[i].localres,
          cal_devices[dev_num], width, fmt, flag) != CAL_RESULT_OK)
        CAL_ERROR("calResAllocLocal1D", NULL);
    } else {  //2d allocation
      if(calResAllocLocal2D(&recs[i].localres,
          cal_devices[dev_num], width, height, fmt, flag) != CAL_RESULT_OK)
        CAL_ERROR("calResAllocLocal2D", NULL);
    }

    if(calCtxGetMem(&recs[i].localmem, ctx, recs[i].localres) != CAL_RESULT_OK)
      CAL_ERROR("calCtxGetMem", NULL);

    if(calCtxSetMem(ctx, name, recs[i].localmem) != CAL_RESULT_OK)
      CAL_ERROR("calCtxSetMem", NULL);

    //TODO - other register types to copy?  cb?
    if(!strcmp("g[]", regname) || regname[0] == 'i') {
      //puts("copying remote memory to local memory");
      //OK, copy the remote memory to the local memory.
      if(calMemCopy(&event, ctx, recs[i].remotemem, recs[i].localmem, 0) != CAL_RESULT_OK)
        CAL_ERROR("calMemCopy", NULL);

      while(calCtxIsEventDone(ctx, event) == CAL_RESULT_PENDING);
    }

    //TODO - need to save the mem, localmem, and localres handles for later!
    //Could modify the bind_dict -- replace each value with a tuple containing
    // the old value and the extra information
    //Build and return a new dictionary or C array of handles to use later.
    // Could create a C struct to hold the handles directly..
    //  How do elements get related back to the bind_dict?
    //   Keep a PyObject reference to the value (binding) in the struct, and
    //    refer to that for updating the re-mapped memory pointer.
    Py_INCREF(key);
    Py_INCREF(value);
    recs[i].regname = key;
    recs[i].binding = value;
  }

  return recs;
}


//After executing, copy local memory back to its remote memory location
//(depending on the register type), and remap the remote memory allocations.

static int cal_remap_copy_memory(struct CopyBindingRecord* recs,
                                 int len, CALcontext ctx)
{
  int i;
  char* regname;
  CALvoid* ptr;
  CALvoid* oldptr;
  CALuint pitch;
  CALuint height;
  CALformat fmt;
  CALresource res;
  CALevent event;
  PyObject* tuple;

  for(i = 0; i < len; i++) {
    regname = PyString_AsString(recs[i].regname);
    tuple = PyList_AsTuple(recs[i].binding); 
    if(!PyArg_ParseTuple(tuple, 
        "liill;cal_remap_copy_memory(): remote bindings must have 5 components",
        &oldptr, &pitch, &height, &fmt, &res)) {
      return -1;
    }

    Py_DECREF(tuple);

    //Copy the local memory back to the remote memory
    if(!strcmp("g[]", regname) || regname[0] == 'o') {
      //puts("copying local memory back out to remote memory");
      if(calMemCopy(&event, ctx, recs[i].localmem, recs[i].remotemem, 0) != CAL_RESULT_OK)
        CAL_ERROR("calMemCopy", -1);
      
      while(calCtxIsEventDone(ctx, event) == CAL_RESULT_PENDING);
    }

    //Free the local memory
    calCtxReleaseMem(ctx, recs[i].remotemem);
    calCtxReleaseMem(ctx, recs[i].localmem);
    calResFree(recs[i].localres);

    //Re-map remote memory and set the pointer in the binding
    if(calResMap(&ptr, &pitch, res, 0) != CAL_RESULT_OK)
      CAL_ERROR("calResMap", -1);

    if(ptr != oldptr) {
      PyList_SetItem(recs[i].binding, 0, PyLong_FromVoidPtr(ptr));
      //PyList_SetItem(recs[i].binding, 1, PyInt_FromLong(pitch));
    }

    Py_DECREF(recs[i].regname);
    Py_DECREF(recs[i].binding);
  }

  free(recs);
  return 0;
}


//Unmap remote memory allocations, and bind them to their registers prior to
//kernel execution.
static int cal_bind_remote_memory(PyObject* bind_dict,
                                  CALcontext ctx, CALmodule mod)
{
  PyObject* key;
  PyObject* value;
  Py_ssize_t pos = 0;
  int ret;

  while(PyDict_Next(bind_dict, &pos, &key, &value)) {
    char* regname;
    CALresource res;
    CALmem mem;
    CALname name;

    regname = PyString_AsString(key); 
    res = (CALresource)PyLong_AsLong(PyList_GetItem(value, 4));

#ifdef _DEBUG
    printf("binding remote memory: %s\n", regname);
#endif

    //Need to call unMap, GetName, SetMem
    calResUnmap(res);

    if(calCtxGetMem(&mem, ctx, res) != CAL_RESULT_OK)
      CAL_ERROR("calCtxGetMem", -1);

    //if(calModuleGetName(&name, ctx, mod, regname) != CAL_RESULT_OK) {
    ret = calModuleGetName(&name, ctx, mod, regname);
    if(ret != CAL_RESULT_OK)
      CAL_ERROR("calModuleGetName", -1);

    if(calCtxSetMem(ctx, name, mem) != CAL_RESULT_OK)
      CAL_ERROR("calCtxSetMem", -1);
  }

  return 0;
}


//Remap remote memory allocations after kernel execution.
static int cal_remap_remote_memory(PyObject* bind_dict)
{
  PyObject* key;
  PyObject* value;
  Py_ssize_t pos = 0;
  char* regname;
  CALvoid* ptr;
  CALvoid* oldptr;
  CALuint pitch;
  CALuint height;
  CALformat fmt;
  CALresource res;
  PyObject* tuple;

  while(PyDict_Next(bind_dict, &pos, &key, &value)) {
    regname = PyString_AsString(key);
    tuple = PyList_AsTuple(value); 
    if(!PyArg_ParseTuple(tuple, 
        "liill;cal_remap_remote_memory(): remote bindings must have 5 components",
        &oldptr, &pitch, &height, &fmt, &res)) {
      return -1;
    }

    Py_DECREF(tuple);

    if(calResMap(&ptr, &pitch, res, 0) != CAL_RESULT_OK)
      CAL_ERROR("calResMap", -1);

    if(ptr != oldptr) {
      PyList_SetItem(value, 0, PyLong_FromVoidPtr(ptr));
      //PyList_SetItem(value, 1, PyInt_FromLong(pitch)); //TODO - needed?
    }
  }


  return 0;
}


//Allocate and bind local memory allocations prior to kernel execution
static CALresource* cal_bind_local_memory(PyObject* bind_dict,
    CALuint dev_num, CALcontext ctx, CALmodule mod)
{
  PyObject* key;
  PyObject* value;
  Py_ssize_t pos = 0;
  Py_ssize_t size = PyDict_Size(bind_dict);
  CALresource* res;
  int i;

  if(size == 0) {
    return NULL;
  }
 
  res = malloc(sizeof(CALresource) * size); 

  // dictionary of local memory to bind (regname -> (w, h, fmt))
  for(i = 0; PyDict_Next(bind_dict, &pos, &key, &value); i++) {
    char* regname;
    CALuint width;
    CALuint height;
    CALformat fmt;
    CALmem mem;
    CALname name;
    CALresallocflags flag = 0;

    regname = PyString_AsString(key); 
    if(!PyArg_ParseTuple(value, "iii", &width, &height, &fmt)) {
      free(res);
      return NULL;
    }

#ifdef _DEBUG
    printf("binding local memory %s %d %d %d\n", regname, width, height, fmt);
#endif

    if(!strcmp("g[]", regname)) {
      flag = CAL_RESALLOC_GLOBAL_BUFFER;
    }

    //Allocate the memory first
    if(height == 1) { //1d allocation
      if(calResAllocLocal1D(&res[i], cal_devices[dev_num], width, fmt, flag)
          != CAL_RESULT_OK) {
        free(res);
        CAL_ERROR("calResAllocLocal1D", NULL);
      }
    } else {  //2d allocation
      if(calResAllocLocal2D(&res[i],
          cal_devices[dev_num], width, height, fmt, flag) != CAL_RESULT_OK) {
        free(res);
        CAL_ERROR("calResAllocLocal2D", NULL);
      }
    }

    if(calCtxGetMem(&mem, ctx, res[i]) != CAL_RESULT_OK) {
      free(res);
      CAL_ERROR("calCtxGetMem", NULL);
    }

    if(calModuleGetName(&name, ctx, mod, regname) != CAL_RESULT_OK) {
      free(res);
      CAL_ERROR("calModuleGetName", NULL);
    }    

    if(calCtxSetMem(ctx, name, mem) != CAL_RESULT_OK) {
      free(res);
      CAL_ERROR("calCtxSetMem", NULL);
    }
  }

  return res;
}


//Free local memory allocations after kernel executions
static void cal_free_local_memory(CALresource* res, Py_ssize_t num_res)
{
  int i;

  if(num_res == 0) {
    return;
  }
 
  for(i = 0; i < num_res; i++) {
    calResFree(res[i]);
  }

  free(res);
}


struct ThreadInfo {
  PyObject* remote_bindings;
  CALresource* local_res;
  struct CopyBindingRecord* recs;

  CALcontext ctx;
  CALmodule mod;
  CALevent event;

  int num_local_res;
  int num_recs;
};


static PyObject* cal_run_stream_async(PyObject* self, PyObject* args)
{
  //Execute a kernel.
  //Arguments:
  // kernel image
  // device number
  // domain (x, y, w, h)
  // dictionary of local memory to bind (regname -> (w, h, fmt))
  // dictionary of remote memory to bind (regname -> memhandle)
  PyObject* remote_bindings;
  PyObject* local_bindings;
  PyObject* copy_bindings;
  CALresource* local_res;
  struct CopyBindingRecord* recs;
  CALimage img;
  CALuint dev_num;
  CALdomain dom;
  CALcontext ctx;
  CALmodule mod;
  CALfunc entry;
  CALevent event;
  struct ThreadInfo* ti;

  if(!PyArg_ParseTuple(args, "li(iiii)O!O!O!", (long int*)&img, &dev_num,
      &dom.x, &dom.y, &dom.width, &dom.height,
      &PyDict_Type, &local_bindings,
      &PyDict_Type, &remote_bindings,
      &PyDict_Type, &copy_bindings)) {
    return NULL;
  }

#ifdef _DEBUG
  printf("executing on device %d domain %d %d -> %d %d\n",
      dev_num, dom.x, dom.y, dom.width, dom.height);
#endif

  if(calCtxCreate(&ctx, cal_devices[dev_num]) != CAL_RESULT_OK)
    CAL_ERROR("calCtxCreate", NULL);

  if(calModuleLoad(&mod, ctx, img) != CAL_RESULT_OK)
    CAL_ERROR("calModuleLoad", NULL);


  //Set up all the memory bindings
  recs = cal_bind_copy_memory(copy_bindings, dev_num, ctx, mod);
  if(recs == NULL) {
    return NULL;
  }

  if(cal_bind_remote_memory(remote_bindings, ctx, mod) != 0) {
    return NULL;
  }

  local_res = cal_bind_local_memory(local_bindings, dev_num, ctx, mod);
  if(local_res == NULL && PyDict_Size(local_bindings) != 0) {
    return NULL;
  }


  //Execute the kernel
  if(calModuleGetEntry(&entry, ctx, mod, "main") != CAL_RESULT_OK)
    CAL_ERROR("calModuleGetEntry", NULL);

  if(calCtxRunProgram(&event, ctx, entry, &dom) != CAL_RESULT_OK)
    CAL_ERROR("calCtxRunProgram", NULL);

  if(calCtxIsEventDone(ctx, event) == CAL_RESULT_BAD_HANDLE)
    CAL_ERROR("calCtxIsEventDone", NULL);

  ti = malloc(sizeof(struct ThreadInfo));
  ti->remote_bindings = remote_bindings;
  Py_INCREF(remote_bindings);

  ti->local_res = local_res;
  ti->recs = recs;
  ti->ctx = ctx;
  ti->mod = mod;
  ti->event = event;
  ti->num_local_res = PyDict_Size(local_bindings);
  ti->num_recs = PyDict_Size(copy_bindings);
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
  cal_remap_copy_memory(ti->recs, ti->num_recs, ti->ctx);
  cal_remap_remote_memory(ti->remote_bindings);
  cal_free_local_memory(ti->local_res, ti->num_local_res);

  calModuleUnload(ti->ctx, ti->mod);
  calCtxDestroy(ti->ctx);

  Py_DECREF(ti->remote_bindings);

  free(ti);

  Py_INCREF(Py_None);
  return Py_None;
}


static PyObject* cal_run_stream(PyObject* self, PyObject* args)
{
  //Execute a kernel.
  //Arguments:
  // kernel image
  // device number
  // domain (x, y, w, h)
  // dictionary of local memory to bind (regname -> (w, h, fmt))
  // dictionary of remote memory to bind (regname -> memhandle)
  // dictionary of remote memory to copy local and bind (regname -> memhandle)
  PyObject* copy_bindings;
  PyObject* remote_bindings;
  PyObject* local_bindings;
  CALresource* local_res;
  struct CopyBindingRecord* recs;
  CALimage img;
  CALuint dev_num;
  CALdomain dom;
  CALcontext ctx;
  CALmodule mod;
  CALfunc entry;
  CALevent event;

  if(!PyArg_ParseTuple(args, "li(iiii)O!O!O!", (long int*)&img, &dev_num,
      &dom.x, &dom.y, &dom.width, &dom.height,
      &PyDict_Type, &local_bindings,
      &PyDict_Type, &remote_bindings,
      &PyDict_Type, &copy_bindings)) {
    return NULL;
  }

#ifdef _DEBUG
  printf("executing on device %d domain %d %d -> %d %d\n",
      dev_num, dom.x, dom.y, dom.width, dom.height);
#endif

  if(calCtxCreate(&ctx, cal_devices[dev_num]) != CAL_RESULT_OK)
    CAL_ERROR("calCtxCreate", NULL);

  if(calModuleLoad(&mod, ctx, img) != CAL_RESULT_OK)
    CAL_ERROR("calModuleLoad", NULL);


  //Set up all the memory bindings
  recs = cal_bind_copy_memory(copy_bindings, dev_num, ctx, mod);
  if(recs == NULL) {
    return NULL;
  }

  if(cal_bind_remote_memory(remote_bindings, ctx, mod) != 0) {
    return NULL;
  }

  local_res = cal_bind_local_memory(local_bindings, dev_num, ctx, mod);
  if(local_res == NULL && PyDict_Size(local_bindings) != 0) {
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
  cal_remap_copy_memory(recs, PyDict_Size(copy_bindings), ctx);
  cal_remap_remote_memory(remote_bindings);
  cal_free_local_memory(local_res, PyDict_Size(local_bindings));

  calGetErrorString();
  calModuleUnload(ctx, mod);
  calCtxDestroy(ctx);

  Py_INCREF(Py_None);
  return Py_None;
}


//
// Memory Allocation
//

// Remote memory has the same issue, but specific to N devices.
static PyObject* cal_alloc_remote(PyObject* self, PyObject* args)
{
  //Arguments:
  //format -- PyInt format constant
  //width in elements
  //height in elements (1 for 1d allocation)
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
  if(!PyArg_ParseTuple(args, "iiiii", &devnum, &fmt, &width, &height, &flag)) {
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

  handle = PyList_New(5);
  PyList_SET_ITEM(handle, 0, PyLong_FromVoidPtr(ptr));
  PyList_SET_ITEM(handle, 1, PyInt_FromLong(pitch));
  PyList_SET_ITEM(handle, 2, PyInt_FromLong(height));
  PyList_SET_ITEM(handle, 3, PyLong_FromUnsignedLong((unsigned long)fmt));
  PyList_SET_ITEM(handle, 4, PyLong_FromUnsignedLong((unsigned long)res));
  return handle;
}


static PyObject* cal_free_remote(PyObject* self, PyObject* args)
{
  CALvoid* ptr;
  CALuint pitch;
  CALuint height;
  CALformat fmt;
  CALresource res;
  PyObject* tuple;

  tuple = PyList_AsTuple(args);
  if(!PyArg_ParseTuple(tuple, "liill",
      (CALvoid**)&ptr, (CALuint*)&pitch, (CALuint*)&height,
      (CALformat*)&fmt, (CALresource*)&res)) {
    return NULL;
  }
  Py_DECREF(tuple);

  calResUnmap(res);
  calResFree(res);
  
  Py_INCREF(Py_None);
  return Py_None;
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
  {"run_stream", cal_run_stream, METH_VARARGS, "Run a kernel on a GPU"},
  {"run_stream_async", cal_run_stream_async, METH_VARARGS, "Run a kernel on a GPU"},
  {"join_stream", cal_join_stream, METH_O, "Join a running kernel"},
  {"alloc_remote", cal_alloc_remote, METH_VARARGS, "Allocate Remote Memory"},
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
  if(!PyArg_ParseTuple(args, "iiiii", &devnum, &self->fmt, &self->width, &self->height, &flag)) {
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

