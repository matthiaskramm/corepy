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

//#ifndef _DEBUG
//#define _DEBUG 0
//#endif

#define CAL_ERROR(fn, ret)                                                \
do { PyErr_Format(PyExc_RuntimeError, "%s: %s", fn, calGetErrorString()); \
     return ret; } while(0)

#define CALCL_ERROR(fn, ret)                                                \
do { PyErr_Format(PyExc_RuntimeError, "%s: %s", fn, calclGetErrorString()); \
     return ret; } while(0)


CALuint cal_device_count = 0;
CALdevice* cal_devices = NULL;
CALdeviceinfo* cal_device_info = NULL;


void cal_init(void)
{
  CALuint i;

  if(cal_devices != 0) {
    return;
  }

  calInit();

  calDeviceGetCount(&cal_device_count);
  cal_devices = malloc(sizeof(CALdevice) * cal_device_count);
  cal_device_info = malloc(sizeof(CALdeviceinfo) * cal_device_count);

  for(i = 0; i < cal_device_count; i++) {
    calDeviceGetInfo(&cal_device_info[i], 0);
    calDeviceOpen(&cal_devices[i], i);
  }
}


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

  cal_init();
  
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
  cal_init();

  return PyInt_FromLong((unsigned int)cal_device_count);
}


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
    res = (CALresource)PyLong_AsLong(PyList_GetItem(value, 2));

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


static int cal_remap_remote_memory(PyObject* bind_dict)
{
  PyObject* key;
  PyObject* value;
  Py_ssize_t pos = 0;

  while(PyDict_Next(bind_dict, &pos, &key, &value)) {
    char* regname;
    CALvoid* ptr;
    CALvoid* oldptr;
    CALuint pitch;
    CALresource res;
    PyObject* tuple;
   
    regname = PyString_AsString(key);
    tuple = PyList_AsTuple(value); 
    if(!PyArg_ParseTuple(tuple, "lil", &oldptr, &pitch, &res))
      return -1;

    Py_DECREF(tuple);

    if(calResMap(&ptr, &pitch, res, 0) != CAL_RESULT_OK)
      CAL_ERROR("calResMap", -1);

    if(ptr != oldptr) {
      //tuple = PyTuple_Pack(3, PyLong_FromVoidPtr(ptr),
      //                        PyInt_FromLong(pitch),
      //                        PyLong_FromUnsignedLong((unsigned long)res));
      //PyDict_SetItem(bind_dict, key, tuple);
      //Py_DECREF(tuple);
      PyList_SetItem(value, 0, PyLong_FromVoidPtr(ptr));
      PyList_SetItem(value, 1, PyInt_FromLong(pitch)); //TODO - needed?
    }
  }

  return 0;
}


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

  CALcontext ctx;
  CALmodule mod;
  CALevent event;

  int num_local_res;
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
  CALresource* local_res;
  CALimage img;
  CALuint dev_num;
  CALdomain dom;
  CALcontext ctx;
  CALmodule mod;
  CALfunc entry;
  CALevent event;
  struct ThreadInfo* ti;

  if(!PyArg_ParseTuple(args, "li(iiii)O!O!", (long int*)&img, &dev_num,
      &dom.x, &dom.y, &dom.width, &dom.height,
      &PyDict_Type, &local_bindings,
      &PyDict_Type, &remote_bindings)) {
    return NULL;
  }

#ifdef _DEBUG
  printf("executing on device %d domain %d %d -> %d %d\n",
      dev_num, dom.x, dom.y, dom.width, dom.height);
#endif

  cal_init();

  if(calCtxCreate(&ctx, cal_devices[dev_num]) != CAL_RESULT_OK)
    CAL_ERROR("calCtxCreate", NULL);

  if(calModuleLoad(&mod, ctx, img) != CAL_RESULT_OK)
    CAL_ERROR("calModuleLoad", NULL);


  if(cal_bind_remote_memory(remote_bindings, ctx, mod) != 0) {
    return NULL;
  }

  local_res = cal_bind_local_memory(local_bindings, dev_num, ctx, mod);
  if(local_res == NULL && PyDict_Size(local_bindings) != 0) {
    return NULL;
  }

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
  ti->ctx = ctx;
  ti->mod = mod;
  ti->event = event;
  ti->num_local_res = PyDict_Size(local_bindings);
  return PyLong_FromVoidPtr(ti);
}


static PyObject* cal_join_stream(PyObject* self, PyObject* args)
{
  struct ThreadInfo* ti;

  ti = PyLong_AsVoidPtr(args);

  while(calCtxIsEventDone(ti->ctx, ti->event) == CAL_RESULT_PENDING) {
    sched_yield();
  }

  calModuleUnload(ti->ctx, ti->mod);
  calCtxDestroy(ti->ctx);

  cal_remap_remote_memory(ti->remote_bindings);
  Py_DECREF(ti->remote_bindings);
  cal_free_local_memory(ti->local_res, ti->num_local_res);

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
  PyObject* remote_bindings;
  PyObject* local_bindings;
  CALresource* local_res;
  CALimage img;
  CALuint dev_num;
  CALdomain dom;
  CALcontext ctx;
  CALmodule mod;
  CALfunc entry;
  CALevent event;

  if(!PyArg_ParseTuple(args, "li(iiii)O!O!", (long int*)&img, &dev_num,
      &dom.x, &dom.y, &dom.width, &dom.height,
      &PyDict_Type, &local_bindings,
      &PyDict_Type, &remote_bindings)) {
    return NULL;
  }

#ifdef _DEBUG
  printf("executing on device %d domain %d %d -> %d %d\n",
      dev_num, dom.x, dom.y, dom.width, dom.height);
#endif

  cal_init();

  if(calCtxCreate(&ctx, cal_devices[dev_num]) != CAL_RESULT_OK)
    CAL_ERROR("calCtxCreate", NULL);

  if(calModuleLoad(&mod, ctx, img) != CAL_RESULT_OK)
    CAL_ERROR("calModuleLoad", NULL);


  if(cal_bind_remote_memory(remote_bindings, ctx, mod) != 0) {
    return NULL;
  }

  local_res = cal_bind_local_memory(local_bindings, dev_num, ctx, mod);
  if(local_res == NULL && PyDict_Size(local_bindings) != 0) {
    return NULL;
  }

  if(calModuleGetEntry(&entry, ctx, mod, "main") != CAL_RESULT_OK)
    CAL_ERROR("calModuleGetEntry", NULL);

  if(calCtxRunProgram(&event, ctx, entry, &dom) != CAL_RESULT_OK)
    CAL_ERROR("calCtxRunProgram", NULL);

  while(calCtxIsEventDone(ctx, event) == CAL_RESULT_PENDING) {
    sched_yield();
  }

  calModuleUnload(ctx, mod);
  calCtxDestroy(ctx);

  cal_remap_remote_memory(remote_bindings);
  cal_free_local_memory(local_res, PyDict_Size(local_bindings));

  Py_INCREF(Py_None);
  return Py_None;
}


//Memory mgmt API:
// Allocate local memory (1d, 2d, type, components, wxh)
// Allocate remote memory.. how can it be mapped into python?
//  Always map the memory, then run_stream can unmap all resources before
//  executing.

//TODO - local memory is device specific.  Maybe i should delay allocating it
// until execution time?
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

  handle = PyList_New(3);
  PyList_SET_ITEM(handle, 0, PyLong_FromVoidPtr(ptr));
  PyList_SET_ITEM(handle, 1, PyInt_FromLong(pitch));
  PyList_SET_ITEM(handle, 2, PyLong_FromUnsignedLong((unsigned long)res));
  return handle;
}


static PyObject* cal_free_remote(PyObject* self, PyObject* args)
{
  CALvoid* ptr;
  CALuint pitch;
  CALresource res;
  PyObject* tuple;

  tuple = PyList_AsTuple(args);
  if(!PyArg_ParseTuple(tuple, "lil",
      (CALvoid**)&ptr, (CALuint*)&pitch, (CALresource*)&res)) {
    return NULL;
  }
  Py_DECREF(tuple);

  calResUnmap(res);
  calResFree(res);
  
  Py_INCREF(Py_None);
  return Py_None;
}


static PyMethodDef module_methods[] = {
  {"compile", cal_compile, METH_O, "Compile a CAL IL kernel, return an image"},
  {"free_image", cal_free_image, METH_O, "Free a compiled kernel image"},
  {"get_num_gpus", cal_get_num_gpus, METH_NOARGS, "Return number of GPUs"},
  {"run_stream", cal_run_stream, METH_VARARGS, "Run a kernel on a GPU"},
  {"run_stream_async", cal_run_stream_async, METH_VARARGS, "Run a kernel on a GPU"},
  {"join_stream", cal_join_stream, METH_O, "Join a running kernel"},
  {"alloc_remote", cal_alloc_remote, METH_VARARGS, "Allocate Remote Memory"},
  {"free_remote", cal_free_remote, METH_O, "Free Remote Memory"},
  {NULL}  /* Sentinel */
};


PyMODINIT_FUNC initcal_exec(void)
{
  PyObject* mod;

  mod = Py_InitModule("cal_exec", module_methods);

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
}

