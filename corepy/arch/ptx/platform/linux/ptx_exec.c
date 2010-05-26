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
#include "cuda.h"

// TODO: take this out by changing ptx_get_error_string
#include "cuda_runtime_api.h" // UH OH!

#ifdef HAS_NUMPY
#include <numpy/arrayobject.h>
#endif

#ifndef _DEBUG
#define _DEBUG 0
#endif

enum ptx_types {s8, s16, s32, s64, u8, u16, u32, u64, f16, f32, f64, b8, b16, b32, b64, pred};

//Buffer object containing CUDA-allocated remote memory for integration with
// NumPy or anything else using the buffer interface.
typedef struct ptxMemBuffer {
  PyObject_HEAD;

  CUarray_format fmt;
  uint width;
  uint pitch;
  uint height;
  uint length;
  uint components;

  CUarray* ptr;
} ptxMemBuffer;

static PyTypeObject ptxMemBufferType;

//Internal structure for maintaining handles during asynchronous kernel
//execution.  A pointer to this is returned back to python when the kernel
//is started.  Python passes the pointer back in to the join call to finish
//the kernel execution.

//struct ThreadInfo {
//  //Dictionary of memory bindings (regname -> [CALresource, void*])
//  PyObject* bindings;
//  CALmem* mem;
//
//  CUcontext ctx;
//  CUmodule mod;
//  CALevent event;
//};


int cu_device_count = 0;
CUdevice* cu_devices = NULL;

//
// CU error handling
//
const char* ptx_get_error_string(int error)
{
  // TODO: handle this ourselves so we don't have to link to the cuda runtime api
  return cudaGetErrorString(error);
  //return "";
}

//
// CU kernel compilation
//

//Take a string, compile it, and return a kernel module pointer
static PyObject* ptx_compile(PyObject* self, PyObject* arg)
{
  CUresult result;
  char* kernel = NULL;
  CUmodule mod;

  //TODO - allow options?
  //int numOptions = 0;
  //CUptxas_option options[numOptions];
  //void* values[numOptions];

  //Does argument type checking
  kernel = PyString_AsString(arg);
  if(kernel == NULL) {
    return NULL;
  }

  // Creating a CUcontext POINTER and passing that in to cuCtxCreate causes a seg fault Python exit, for some reason
  CUcontext Ctx;
//  printf("Creating context...\n");
//  result = cuCtxCreate(&Ctx, CU_CTX_MAP_HOST, 0);
//  printf("Created context %d\n", Ctx);
//  if (result != CUDA_SUCCESS)
//    printf("Error in cuCtxCreate: %d - %s\n", result, ptx_get_error_string(result)); // do error stuff

  //if (cuModuleLoadDataEx(&mod, kernel, numOptions, options, values) != CUDA_SUCCESS)
  //  printf "Oops!\n"; // do error stuff
  result = cuModuleLoadData(&mod, kernel);
  if (result != CUDA_SUCCESS)
    printf("Error in cuModuleLoadData: %d - %s\n", result, ptx_get_error_string(result)); // do error stuff

  //printf("mod in ptx_compile: %d\n", mod);

  return PyLong_FromVoidPtr(mod);
  //return PyLong_FromUnsignedLong((unsigned long)mod);
}


//Free a compiled kernel image
static PyObject* ptx_unload_module(PyObject* self, PyObject* arg)
{
  CUmodule mod = NULL;

  mod = (CUmodule)PyLong_AsUnsignedLong(arg);
  cuModuleUnload(mod);

  Py_RETURN_NONE;
}


//Return the number of available GPUs
static PyObject* ptx_get_num_gpus(PyObject* self, PyObject* arg)
{
  return PyInt_FromLong((unsigned int)cu_device_count);
}


//
// CU kernel execution
//

int ctx_count = 0;

//Allocate a CUDA Driver API context on a particular device.
static PyObject* ptx_alloc_ctx(PyObject* self, PyObject* arg)
{
  int dev_num = PyLong_AsUnsignedLong(arg);
  CUcontext ctx;
  CUresult result;

  ctx_count += 1;
  printf("ctx_count: %d\n", ctx_count);

  result = cuCtxCreate(&ctx, CU_CTX_SCHED_AUTO, cu_devices[dev_num]);
  if (result != CUDA_SUCCESS)
    printf("Error in cuCtxCreate: %d - %s\n", result, ptx_get_error_string(result)); // do error stuff

  return PyLong_FromUnsignedLong((unsigned long)ctx);
}

//Release a ptx context.
static PyObject* ptx_free_ctx(PyObject* self, PyObject* arg)
{
  CUresult result;
  result = cuCtxDestroy((CUcontext)PyLong_AsUnsignedLong(arg));
  if (result != CUDA_SUCCESS)
    printf("Error in cuCtxDestroy: %d - %s\n", result, ptx_get_error_string(result)); // do error stuff

  ctx_count -= 1;
  printf("ctx_count: %d\n", ctx_count);

  Py_RETURN_NONE;
}

//Copy memory from one host memory allocation to device memory
//Arguments:
// device ptr
// host ptr
// size in BYTES
static PyObject* ptx_copy_htod(PyObject* self, PyObject* args)
{
  CUdeviceptr device_ptr;
  void* host_ptr;
  long unsigned int size;
  enum ptx_types type;

  CUresult result;

  if(!PyArg_ParseTuple(args, "kkk",
		       &device_ptr, &host_ptr, &size)) {
    return NULL;
  }

  result = cuMemcpyHtoD(device_ptr, host_ptr, size);
  if (result != CUDA_SUCCESS)
    printf("Error in cuMemcpyHtoD: %d - %s\n", result, ptx_get_error_string(result)); // do error stuff

  return Py_None;
}

static PyObject* ptx_copy_dtoh(PyObject* self, PyObject* args)
{
  CUdeviceptr device_ptr;
  void* host_ptr;
  long unsigned int size;
  enum ptx_types type;

  CUresult result;

  if(!PyArg_ParseTuple(args, "kkk",
		       &host_ptr, &device_ptr, &size)) {
    return NULL;
  }

  result = cuMemcpyDtoH(host_ptr, device_ptr, size);
  if (result != CUDA_SUCCESS)
    printf("Error in cuMemcpyDtoH: %d - %s\n", result, ptx_get_error_string(result)); // do error stuff

  return Py_None;
}


union param_data {
  unsigned int u32_value;
  int s32_value;
  float f32_value;
  unsigned long int u64_value;
  long int s64_value;
  double f64_value;
} param_data;

struct run_param {
  union param_data data;
  unsigned long int ptr_value;
} run_param;

// See Section 3.3 of the CUDA programming guide for information on this
#define ALIGN_OFFSET(offset, alignment) \
  offset = (offset + alignment - 1) & ~(alignment - 1)

static PyObject* ptx_run_stream(PyObject* self, PyObject* args)
{
  //Execute a kernel.
  //Arguments:
  // kernel module
  // threads: (x, y, z, w, h) (where x, y, z specify block shape)
  // parameter tuple
  //PyObject* parameter_types;
  //PyObject* parameters;
  //CUcontext* ctx;
  CUmodule mod;
  int x, y, z, w, h;
  PyObject* param_types;
  PyObject* params;

  struct run_param* c_params;
  Py_ssize_t num_params;
  long int param_type;
  PyObject* param_type_object;
  PyObject* param_object;
  // WARNING, we're assuming 64 bit
  // TODO: make these work for 32 bit systems also...
  int s32_param;
  unsigned int u32_param;
  float f32_param;
  long int s64_param;
  long unsigned int u64_param;
  double f64_param;
  int offset = 0;

  Py_ssize_t i;

  CUfunction entry;

  CUresult result;

  //printf("Entering run_stream\n");

  if(!PyArg_ParseTuple(args, "l(IIIII)O!O!", &mod,
		       &x, &y, &z, &w, &h, 
		       &PyTuple_Type, &param_types, &PyList_Type, &params)) {
    return NULL;
  }

  //#ifdef _DEBUG
  //printf("executing <<< %d %d %d ,  %d %d >>>\n",
  //	 x, y, z, w, h);
  //#endif

  //printf("mod in run_stream: %d\n", mod);
  // Get a function pointer
  result = cuModuleGetFunction(&entry, mod, "_main");
  if (result != CUDA_SUCCESS)
    printf("Errorin cuModuleGetFunction: %d - %s\n", result, ptx_get_error_string(result)); // do error stuff

  //printf("Setting up parameters\n");
  // setup parameters
  num_params = PyTuple_Size(param_types);
  if (num_params > 0)
  {
    c_params = malloc(sizeof(run_param)*num_params);
    //result = cuMemAllocHost((void**)&c_params, sizeof(run_param)*num_params);
    //result = cuMemHostAlloc((void**)&c_params, sizeof(run_param)*num_params, CU_MEMHOSTALLOC_DEVICEMAP);
    //if (result != CUDA_SUCCESS)
    //  printf("Errorin cuMemAllocHost: %d - %s\n", result, ptx_get_error_string(result)); // do error stuff
    for (i = 0; i < num_params; i++)
    {
      //printf("offset : %d\n", offset);

      param_type_object = PyTuple_GetItem(param_types, i);
      param_object = PyList_GetItem(params, i);
      param_type = (enum ptx_types) PyInt_AsLong(param_type_object);
      //printf("param_type: %d\n", param_type);
      
      if (param_type == u32 || param_type == b32)
      {
	c_params[i].data.u32_value = (unsigned int)PyInt_AsLong(param_object);
	ALIGN_OFFSET(offset, __alignof(u32_param));
	cuParamSetv(entry, offset, &(c_params[i].data.u32_value), sizeof(u32_param));
	offset += sizeof(u32_param);
      }
      else if (param_type == s32)
      {
	c_params[i].data.s32_value = (int)PyInt_AsLong(param_object);
	ALIGN_OFFSET(offset, __alignof(s32_param));
	cuParamSetv(entry, offset, &(c_params[i].data.s32_value), sizeof(s32_param));
	offset += sizeof(s32_param);
      }
      else if (param_type == f32)
      {
	c_params[i].data.f32_value = (float)PyFloat_AsDouble(param_object);
	ALIGN_OFFSET(offset, __alignof(f32_param));
	cuParamSetf(entry, offset, c_params[i].data.f32_value);
	offset += sizeof(f32_param);
      }
      else if (param_type == u64 || param_type == b64)
      {
	c_params[i].data.u64_value = (long unsigned int)(PyInt_AsLong(param_object));
	//printf("value from Python: %lx\n",PyInt_AsLong(param_object));
	ALIGN_OFFSET(offset, __alignof(u64_param));
	cuParamSetv(entry, offset, &(c_params[i].data.u64_value), sizeof(u64_param));
	//printf("%lx\n",&c_params);
	//printf("%lx\n",c_params);
	//printf("%lx\n",&c_params[i]);
	//printf("%lx\n",&(c_params[i].data.u64_value));
	//printf("value: %lx\n",c_params[i].data.u64_value);
	//printf("%lx\n",param_object);
	offset += sizeof(u64_param);
      }
      else if (param_type == s64)
      {
	c_params[i].data.s64_value = (long int)PyInt_AsLong(param_object);
	ALIGN_OFFSET(offset, __alignof(s64_param));
	cuParamSetv(entry, offset, &(c_params[i].data.s64_value), sizeof(s64_param));
	offset += sizeof(s32_param);
      }
      else if (param_type == f64)
      {
	c_params[i].data.f64_value = (double)PyFloat_AsDouble(param_object);
	ALIGN_OFFSET(offset, __alignof(f64_param));
	cuParamSetv(entry, offset, &(c_params[i].data.f64_value), sizeof(f64_param));
	offset += sizeof(f64_param);
      }
    }
  }

  //printf("Done with parameters\n");
  
  cuParamSetSize(entry, offset);

  cuFuncSetBlockShape(entry, x, y, z);
  //printf("cuLaunchGrid()\n");
  printf("%d, offset: %d, x: %d, y: %d, z: %d w: %d h: %d\n", entry, offset, x, y, z, w, h);
  result = cuLaunchGrid(entry, w, h);

  if (result != CUDA_SUCCESS)
    {

    printf("Errorin cuLaunchGrid: %d - %s\n", result, ptx_get_error_string(result));
    }
  //printf("cuLaunchGrid() done\n");

  // cuModuleUnload(mod);

  Py_RETURN_NONE;
}


//
// Memory Allocation
//

static PyObject* ptx_alloc_device(PyObject* self, PyObject* args)
{
  //Arguments:
  // size in bytes
  CUdeviceptr dptr;
  unsigned int bytesize;
  PyObject* handle;
  CUresult result;

  //TODO - make the flag argument optional
  if(!PyArg_ParseTuple(args, "I", &bytesize)) {
    return NULL;
  }

  result = cuMemAlloc(&dptr, bytesize);
  if (result != CUDA_SUCCESS)
    printf("Errorin cuMemAlloc: %d - %s\n", result, ptx_get_error_string(result));

  handle = PyLong_FromUnsignedLong((unsigned long)dptr);
  return handle;
}


static PyObject* ptx_free_device(PyObject* self, PyObject* args)
{
  CUdeviceptr dptr;

  if(!PyArg_ParseTuple(args, "I", &dptr)) {
    return NULL;
  }

  cuMemFree(dptr);
  
  Py_RETURN_NONE;
}


static PyObject* ptx_alloc_host(PyObject* self, PyObject* args)
{
  //Arguments:
  // size in bytes
  void* ptr;
  unsigned int bytesize;
  PyObject* handle;
  CUresult result;

  //TODO - make the flag argument optional
  if(!PyArg_ParseTuple(args, "I", &bytesize)) {
    return NULL;
  }

  result = cuMemAllocHost(&ptr, bytesize);
  if (result != CUDA_SUCCESS)
    printf("Errorin cuMemAllocHost: %d - %s\n", result, ptx_get_error_string(result));

  handle = PyLong_FromUnsignedLong((unsigned long)ptr);
  return handle;
}

static PyObject* ptx_free_host(PyObject* self, PyObject* args)
{
  void* ptr;

  if(!PyArg_ParseTuple(args, "I", &ptr)) {
    return NULL;
  }

  cuMemFreeHost(ptr);
  
  Py_RETURN_NONE;
}

#if 0
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

#endif


static PyMethodDef module_methods[] = {
  {"compile", ptx_compile, METH_O, "Compile a CAL IL kernel, return an image"},
  {"get_num_gpus", ptx_get_num_gpus, METH_NOARGS, "Return number of GPUs"},
  {"alloc_ctx", ptx_alloc_ctx, METH_O, "Allocate a CAL context"},
  {"free_ctx", ptx_free_ctx, METH_O, "Release a CAL context"},
  //{"copy_async", cal_copy_async, METH_VARARGS, "Start a GPU memory copy"},
  {"copy_htod", ptx_copy_htod, METH_VARARGS, "Synchronous copy from host to device"},
  {"copy_dtoh", ptx_copy_dtoh, METH_VARARGS, "Synchronous copy from device to host"},
  //{"join_copy", cal_join_copy, METH_VARARGS, "Finish a GPU memory copy"},
  {"run_stream", ptx_run_stream, METH_VARARGS, "Run a kernel on a GPU"},
  //{"run_stream_async", cal_run_stream_async, METH_VARARGS, "Run a kernel on a GPU"},
  //{"join_stream", cal_join_stream, METH_O, "Join a running kernel"},
  {"alloc_device", ptx_alloc_device, METH_VARARGS, "Allocate local memory"},
  {"free_device", ptx_free_device, METH_O, "Free local memory"},
  {"alloc_host", ptx_alloc_host, METH_VARARGS, "Allocate remote memory"},
  {"free_host", ptx_free_host, METH_O, "Free Remote Memory"},
  //{"set_ndarray_ptr", cal_set_ndarray_ptr, METH_VARARGS, "Set ndarray pointer"},
  {NULL}  /* Sentinel */
};

PyMODINIT_FUNC initptx_exec(void)
{
  PyObject* mod;

  //CALMemBufferType.tp_new = PyType_GenericNew;
  //if(PyType_Ready(&CALMemBufferType) < 0) {
  //  return;
  //}

  mod = Py_InitModule("ptx_exec", module_methods);

  //Py_INCREF(&CALMemBufferType);
  //PyModule_AddObject(mod, "calmembuffer", (PyObject*)&CALMemBufferType);

#ifdef HAS_NUMPY
  import_array();

  Py_INCREF(Py_True);
  PyModule_AddObject(mod, "HAS_NUMPY", Py_True);
#else
  Py_INCREF(Py_False);
  PyModule_AddObject(mod, "HAS_NUMPY", Py_False);
#endif

  // Most of these types are probably unnecessary, but just in case...
  PyModule_AddIntConstant(mod, "s8", s8);
  PyModule_AddIntConstant(mod, "s16", s16);
  PyModule_AddIntConstant(mod, "s32", s32);
  PyModule_AddIntConstant(mod, "s64", s64);
  PyModule_AddIntConstant(mod, "u8", u8);
  PyModule_AddIntConstant(mod, "u16", u16);
  PyModule_AddIntConstant(mod, "u32", u32);
  PyModule_AddIntConstant(mod, "u64", u64);
  PyModule_AddIntConstant(mod, "f16", f16);
  PyModule_AddIntConstant(mod, "f32", f32);
  PyModule_AddIntConstant(mod, "f64", f64);
  PyModule_AddIntConstant(mod, "b8", b8);
  PyModule_AddIntConstant(mod, "b16", b16);
  PyModule_AddIntConstant(mod, "b32", b32);
  PyModule_AddIntConstant(mod, "b64", b64);
  PyModule_AddIntConstant(mod, "pred", pred);

  int i;
  CUresult result;

  if(cu_devices != 0) {
    return;
  }

  result = cuInit(0);
  if (result != CUDA_SUCCESS)
    printf("Errorin cuInit: %d - %s\n", result, ptx_get_error_string(result)); // do error stuff

  result = cuDeviceGetCount(&cu_device_count);
  if (result != CUDA_SUCCESS)
    printf("Error in cuDeviceGetCount: %d - %s\n", result, ptx_get_error_string(result)); // do error stuff
  
  cu_devices = malloc(sizeof(CUdevice) * cu_device_count);
  //cu_device_info = malloc(sizeof(CUdevice_attribute) * cu_device_count);

  for(i = 0; i < cu_device_count; i++)
    cuDeviceGet(&cu_devices[i], i);

}
