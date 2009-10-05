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
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <malloc.h>
#include <pthread.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <signal.h>

#include <sched.h>
#include "spufs.h"

#ifdef HAS_LIBSPE2
#include <libspe2.h>
#endif


//Define _DEBUG to get informational debug output
//#ifndef _DEBUG
//#define _DEBUG 0
//#endif

#ifdef _DEBUG
#define DEBUG(args) printf args
#else
#define DEBUG(args)
#endif

#define SPUFS_PATH "/spu"
#define SPULS_SIZE (256 * 1024)

// SPU status codes from spu_run(2) man page
#define SPU_STOP_SIGNAL 0x02
#define SPU_HALT        0x04
#define SPU_WAIT_CHAN   0x08
#define SPU_SINGLE_STEP 0x10
#define SPU_INVAL_INST  0x20
#define SPU_INVAL_CHAN  0x40

#define SPU_CODE_MASK   0xFFFF
#define SPU_STOP_MASK   0x3FFF0000

// MFC DMA commands from 'Cell Broadband Engine Architecture' v1.01
// (CBE_Architecture_v101.pdf)
#define MFC_PUT     0x20
#define MFC_PUTB    0x21
#define MFC_GET     0x40
#define MFC_GETB    0x41


// ------------------------------------------------------------
// Typedefs
// ------------------------------------------------------------

//Base address type -- integer that holds memory addresses
typedef unsigned long addr_t;


typedef struct SPUExecParams {
  PyObject_HEAD;

  //Wrap the actual params in another struct so they can be easily read or
  // written to a file descriptor in one go
  struct params {
    // $r3
    addr_t addr;   // address of syn code
    unsigned int p1;  
    unsigned int p2;  
    unsigned int p3;

    // $r4
    unsigned int size;       // size of syn code
    unsigned int p4;
    unsigned int p5;
    unsigned int p6;

    // $r5
    unsigned int p7;
    unsigned int p8;
    unsigned int p9;
    unsigned int p10;
  } __attribute__((aligned(16))) p;
} SPUExecParams;

static PyTypeObject SPUExecParamsType;


//TODO - update this to be a python object.
// Which of these variables are public and private (C only)?
// Information structure returned when executing in async mode
typedef struct SPUContext {
  PyObject_HEAD;

  pthread_t th;
  struct spufs_context* spu_ctx;

  addr_t spuls;        //Local store address 
  addr_t spups;        //Problem state address

  int spu_run_ret;

  //SPUExecParams params;

  int mode;   //Execution mode; allows python to determine return type
  int stop;   //Whether stop code should be included with return value
} SPUContext;

static PyTypeObject SPUContextType;


struct ThreadParams {
  SPUContext* ctx;       //Execution Context
  addr_t addr;           //Main memory addr of associated stream
  int len;               //Length of stream in bytes
  addr_t code_lsa;       //Local store addr of associated stream
  addr_t exec_lsa;       //Local store addr to start execution
};  


// ------------------------------------------------------------
// Code
// ------------------------------------------------------------

//
// SPUContext
//

static int
SPUContext_init(SPUContext* self, PyObject* args, PyObject* kwds)
{
  self->spu_ctx = spufs_open_context("corepy-spu");
  self->spuls = (addr_t)self->spu_ctx->mem_ptr;
  self->spups = (addr_t)self->spu_ctx->psmap_ptr;

  return 0;
}


static void
SPUContext_dealloc(SPUContext* self)
{
  spufs_close_context(self->spu_ctx);

  self->ob_type->tp_free((PyObject*)self);
}


//
// Module functions
//

#define MAKE_EXECUTABLE_DOC \
"Mark a memory region as executable."

static PyObject* spu_make_executable(PyObject* self, PyObject* args)
{
  // Note: At some point in the future, memory protection will 
  // be an issue on PPC Linux.  The commented out code sets
  // the execute bit on the memory pages.

  /* 
  if(mprotect((void *)(addr & 0xFFFF0000), 
              size * 4 + (addr & 0xFFFF), 
              PROT_READ | PROT_WRITE | PROT_EXEC) == -1)
    perror("Error setting memory protections");
  */

  Py_RETURN_NONE;
}


#define CANCEL_ASYNC_DOC \
"Cancel execution of an SPU thread."

//TODO - does this really work?
static PyObject* spu_cancel_async(PyObject* self, PyObject* args)
{
  SPUContext* ctx = (SPUContext*)args;
  int ret;

  ret = pthread_cancel(ctx->th);
  if(ret) {
    errno = ret;
    return PyErr_SetFromErrno(PyExc_OSError);
  }

  Py_RETURN_NONE;
}


#define ALLOC_CONTEXT_DOC \
"Allocate and initialize a new SPU execution context."

static PyObject* spu_alloc_context(PyObject* self, PyObject* args)
{
  return PyObject_CallObject((PyObject*)&SPUContextType, NULL);
}


#define FREE_CONTEXT_DOC \
"Free an allocated SPU execution context."

static PyObject* spu_free_context(PyObject* self, PyObject* args)
{
  //The destructor takes care of freeing the object.
  Py_RETURN_NONE;
}


#define RUN_STREAM_DOC \
"Start SPU execution at the given LSA address, blocking until " \
"the SPU stops execution.  Assumes len and code_lsa are " \
"16-byte aligned."

// Arguments:
//   *ti   Execution context to run
//   addr     Address to load code from, NULL if none
//   len      Length of code in bytes
//   code_lsa Local store address to load code at
//   exec_lsa Local store address to start execution at
// Return: LSA address following last SPU instruction executed

static PyObject* spu_run_stream(PyObject* self, PyObject* args)
{
  SPUContext* ctx;
  addr_t addr;
  int len;
  addr_t code_lsa;
  addr_t exec_lsa;
  unsigned char* spuls;

  if(!PyArg_ParseTuple(args, "O!kikk",
      &SPUContextType, &ctx, &addr, &len, &code_lsa, &exec_lsa)) {
    return NULL;
  }

  spuls = (unsigned char*)ctx->spuls;

  if(addr != 0) {
    //Memcpy the code from main memory to the SPU local store
    DEBUG(("Copying code from EA %p to LSA %p (%x) %d bytes\n",
        addr, &spuls[code_lsa], code_lsa, len));
    memcpy(&spuls[code_lsa], (unsigned char*)addr, len);
  }

  DEBUG(("Starting SPU %p\n", ctx->spu_ctx));
  ctx->spu_run_ret = spufs_run(ctx->spu_ctx, (unsigned int*)&exec_lsa);
  DEBUG(("SPU %p finished executing, code %d\n",
      ctx->spu_ctx, ctx->spu_run_ret));

  return PyInt_FromLong(exec_lsa);
}


//Thread main for asynchronous execution
void *run_stream_thread(void* arg) {
  struct ThreadParams* tp = (struct ThreadParams*)arg;
  SPUContext* ctx = tp->ctx;
  unsigned char* spuls = (unsigned char*)tp->ctx->spuls;
  addr_t ret;

  //rc = run_stream(tp->ti, tp->addr, tp->len, tp->code_lsa, tp->exec_lsa);
  if(tp->addr != 0) {
    //Memcpy the code from main memory to the SPU local store
    DEBUG(("Copying code from EA %p to LSA %p (%x) %d bytes\n",
        tp->addr, &spuls[code_lsa], tp->code_lsa, tp->len));
    memcpy(&spuls[tp->code_lsa], (unsigned char*)tp->addr, tp->len);
  }

  DEBUG(("Starting SPU %p\n", ctx->spu_ctx));
  ctx->spu_run_ret = spufs_run(ctx->spu_ctx, (unsigned int*)&tp->exec_lsa);
  DEBUG(("SPU %p finished executing, code %d\n",
      ctx->spu_ctx, ctx->spu_run_ret));

  ret = tp->exec_lsa;
  free(tp);
  return (void*)ret;
}


#define RUN_STREAM_ASYNC_DOC \
"Start asynchronous SPU execution at the given LSA address, returning as " \
"soon as execution begins.  Assumes len and code_lsa are " \
"16-byte aligned.  Use wait_stream() to block until completion and obtain" \
"return status codes."

// Arguments:
//   *ti   Execution context to run
//   addr     Address to load code from, NULL if none
//   len      Length of code in bytes
//   code_lsa Local store address to load code at
//   exec_lsa Local store address to start execution at
// Return: LSA address following last SPU instruction executed

static PyObject* spu_run_stream_async(PyObject* self, PyObject* args)
{
  struct ThreadParams* tp;
  SPUContext* ctx;
  addr_t addr;
  int len;
  addr_t code_lsa;
  addr_t exec_lsa;
  int ret;

  if(!PyArg_ParseTuple(args, "O!kikk",
      &SPUContextType, &ctx, &addr, &len, &code_lsa, &exec_lsa)) {
    return NULL;
  }


  tp = (struct ThreadParams*)malloc(sizeof(struct ThreadParams));

  Py_INCREF(ctx); //Decremented in wait_stream()

  tp->ctx = ctx;
  tp->addr = addr;
  tp->len = len;
  tp->code_lsa = code_lsa;
  tp->exec_lsa = exec_lsa;

  ret = pthread_create(&ctx->th, NULL, run_stream_thread, (void*)tp);
  if(ret) {
    free(tp);
    errno = ret;
    return PyErr_SetFromErrno(PyExc_OSError);
  }

  Py_RETURN_NONE;
}


#define WAIT_STREAM_DOC \
"Wait for an asynchronous SPU execution thread to complete."

static PyObject* spu_wait_stream(PyObject* self, PyObject* args)
{
  SPUContext* ctx = (SPUContext*)args;
  addr_t ret;
  
  pthread_join(ctx->th, (void**)&ret);
  Py_DECREF(ctx);

  return PyInt_FromLong(ret);
}


#define GET_RESULT_DOC \
"Check the SPU return value, and return the value if it was a stop signal.  " \
"If the SPU halted, 0 is returned."

static PyObject* spu_get_result(PyObject* self, PyObject* args)
{
  SPUContext* ctx = (SPUContext*)args;

  switch(ctx->spu_run_ret & SPU_CODE_MASK) {
  case SPU_STOP_SIGNAL:
    return PyInt_FromLong((ctx->spu_run_ret & SPU_STOP_MASK) >> 16);
  case SPU_HALT:
    return PyInt_FromLong(0);
  default:
    DEBUG(("SPU unexpected stop reason %x\n", ctx->spu_run_ret));
  }


  Py_RETURN_NONE;
}


#define GET_NUM_AVAIL_SPUS_DOC \
"Wait for an asynchronous SPU execution thread to complete."

static PyObject* spu_get_num_avail_spus(PyObject* self, PyObject* args)
{
  return PyInt_FromLong(6);
}


#define GET_SPU_REGISTERS_DOC \
"Read the 128 SPU registers into a memory location.  The memory location is" \
"assumed to have room for 128 * 16 bytes."

static PyObject* spu_get_spu_registers(PyObject* self, PyObject* args)
{
  SPUContext* ctx;
  unsigned long data;

  //TODO - use the read/write buffer interface instead of passing ptrs?
  if(!PyArg_ParseTuple(args, "O!k",
      &SPUContextType, &ctx, &data)) {
    return NULL;
  }

  if(read(ctx->spu_ctx->regs_fd, (void*)data, 128 * 16) != 128 * 16) {
    return PyErr_SetFromErrno(PyExc_OSError);
  }

  lseek(ctx->spu_ctx->regs_fd, 0, SEEK_SET);

  Py_RETURN_NONE;
}


#define PUT_SPU_REGISTERS_DOC \
"Write the 128 SPU registers from a memory location."

static PyObject* spu_put_spu_registers(PyObject* self, PyObject* args)
{
  SPUContext* ctx;
  unsigned long data;

  //TODO - use the read/write buffer interface instead of passing ptrs?
  if(!PyArg_ParseTuple(args, "O!k",
      &SPUContextType, &ctx, &data)) {
    return NULL;
  }

  if(write(ctx->spu_ctx->regs_fd, (void*)data, 128 * 16) != 128 * 16) {
    return PyErr_SetFromErrno(PyExc_OSError);
  }

  lseek(ctx->spu_ctx->regs_fd, 0, SEEK_SET);

  Py_RETURN_NONE;
}


#define PUT_SPU_PARAMS_DOC \
"Write parameters to the SPU registers."

static PyObject* spu_put_spu_params(PyObject* self, PyObject* args)
{
  SPUContext* ctx;
  SPUExecParams* params;

  if(!PyArg_ParseTuple(args, "O!O!",
      &SPUContextType, &ctx, &SPUExecParamsType, &params)) {
    return NULL;
  }

  // SPUFS is *RETARDED*!! 12 bytes into the file means seek to pos 3.
  lseek(ctx->spu_ctx->regs_fd, 3, SEEK_SET);

  if(write(ctx->spu_ctx->regs_fd, (void*)&params->p, 12 * 4) != 12 * 4) {
    perror("put_spu_params write");
  }

  lseek(ctx->spu_ctx->regs_fd, 0, SEEK_SET);
  return PyInt_FromLong(6);
}


#define GET_PHYS_ID_DOC \
"Return the physical ID number of an SPU context.  SPU contexts are not " \
"assigned a physical ID unless they are executing.  If an SPU context is not " \
"executing, -1 is returned."

static PyObject* spu_get_phys_id(PyObject* self, PyObject* args)
{
  SPUContext* ctx = (SPUContext*)args;

  return PyInt_FromLong(spufs_get_phys_id(ctx->spu_ctx));
}


#define READ_OUT_MBOX_DOC \
"Read a value from an SPU's outbound mailbox."

static PyObject* spu_read_out_mbox(PyObject* self, PyObject* args)
{
  SPUContext* ctx = (SPUContext*)args;
  volatile unsigned int* addr = (volatile unsigned int*)(ctx->spups + 0x4004);

  return PyLong_FromUnsignedLong(*addr);
}


#define STAT_OUT_MBOX_DOC \
"Return the number of valid entries in an SPU's outbound mailbox."

static PyObject* spu_stat_out_mbox(PyObject* self, PyObject* args)
{
  SPUContext* ctx = (SPUContext*)args;
  volatile unsigned int* addr = (volatile unsigned int*)(ctx->spups + 0x4014);

  return PyLong_FromUnsignedLong(*addr);
}


#define STAT_READ_OUT_MBOX_DOC \
"Poll an SPU's outbound mailbox until a value is available; return that value."

static PyObject* spu_stat_read_out_mbox(PyObject* self, PyObject* args)
{
  SPUContext* ctx = (SPUContext*)args;
  volatile unsigned int* addr = (volatile unsigned int*)(ctx->spups + 0x4014);
  volatile unsigned int status;

  //__asm__("eieio");

  //Poll the status register until a message is available
  do {
    status = *addr;
  } while((status & 0xFF) == 0);

  //Manual says to do this, is it necessary?
  __asm__("eieio");

  addr = (volatile unsigned int*)(ctx->spups + 0x4004);
  return PyLong_FromUnsignedLong(*((volatile unsigned int*)addr));
}


#define POLL_OUT_MBOX_DOC \
"Poll an SPU's outbound mailbox until a value is available."

static PyObject* spu_poll_out_mbox(PyObject* self, PyObject* args)
{
  SPUContext* ctx = (SPUContext*)args;
  volatile unsigned int* addr = (volatile unsigned int*)(ctx->spups + 0x4014);
  volatile unsigned int status;

  //__asm__("eieio");

  //Poll the status register until a message is available
  do {
    status = *addr;
  } while((status & 0xFF) == 0);

  //Manual says to do this, is it necessary?
  __asm__("eieio");

  Py_RETURN_NONE;
}


#define READ_OUT_IBOX_DOC \
"Read an SPU's outbound interrupt mailbox; blocks until a value is available."

static PyObject* spu_read_out_ibox(PyObject* self, PyObject* args)
{
  SPUContext* ctx = (SPUContext*)args;
  unsigned int data = 0;

  if(read(ctx->spu_ctx->ibox_fd, &data, 4) != 4) {
    perror("read_out_mbox read");
  }

  return PyLong_FromUnsignedLong(data);
}


#define STAT_OUT_IBOX_DOC \
"Return the number of valid entries in an SPU's outbound interrupt mailbox."

static PyObject* spu_stat_out_ibox(PyObject* self, PyObject* args)
{
  SPUContext* ctx = (SPUContext*)args;
  unsigned int* addr = (unsigned int*)(ctx->spups + 0x4014);
  return PyLong_FromUnsignedLong(((*addr & 0x10000) >> 16));
}


#define WRITE_IN_MBOX_DOC \
"Write a value to an SPU's inbound mailbox."

static PyObject* spu_write_in_mbox(PyObject* self, PyObject* args)
{
  //TODO - takes more than one arg, need parsetuple
  SPUContext* ctx;
  unsigned int* addr;
  unsigned int data;
  
  if(!PyArg_ParseTuple(args, "O!I",
      &SPUContextType, &ctx, &data)) {
    return NULL;
  }

  addr = (unsigned int*)(ctx->spups + 0x400C);
  *addr = data;

  Py_RETURN_NONE;
}


#define WRITE_IN_MBOX_LIST_DOC \
"Write a value to the inbound mailbox of a list of SPUs."

static PyObject* spu_write_in_mbox_list(PyObject* self, PyObject* args)
{
  PyObject* ctx_list;
  unsigned int data;
  Py_ssize_t len;
  Py_ssize_t i;

  if(!PyArg_ParseTuple(args, "OI", &ctx_list, &data)) {
    return NULL;
  }


  ctx_list = PySequence_Fast(ctx_list, 
      "Context list must be an iterable sequence");
  if(ctx_list == NULL) {
    return NULL;
  }

  len = PySequence_Length(ctx_list);
  for(i = 0; i < len; i++) {
    SPUContext* ctx = (SPUContext*)PySequence_Fast_GET_ITEM(ctx_list, i);
    unsigned int* addr = (unsigned int*)(ctx->spups + 0x400C);
    *addr = data;
  }

  Py_RETURN_NONE;
}


#define STAT_IN_MBOX_DOC \
"Return the number of available entries in an SPU's inbound mailbox."

static PyObject* spu_stat_in_mbox(PyObject* self, PyObject* args)
{
  SPUContext* ctx = (SPUContext*)args;
  volatile unsigned int* addr = (volatile unsigned int*)(ctx->spups + 0x4014);

  return PyLong_FromUnsignedLong(((*addr & 0x700) >> 8));
}


#define WRITE_SIGNAL_DOC \
"Write a value to the specified signal register (1 or 2) of an SPU."

static PyObject* spu_write_signal(PyObject* self, PyObject* args)
{
  SPUContext* ctx;
  unsigned int which;
  unsigned int data;
  unsigned int* addr;

  if(!PyArg_ParseTuple(args, "O!II", &SPUContextType, &ctx, &which, &data)) {
    return NULL;
  }

  if(which == 1) {
    addr = (unsigned int*)(ctx->spups + 0x1400C);
  } else { //if(which == 2) {
    addr = (unsigned int*)(ctx->spups + 0x1C00C);
  }

  *addr = data;
  Py_RETURN_NONE;
}


#define SET_SIGNAL_MODE_DOC \
"Set the mode (0 = overwrite, 1 = logical OR) of an SPU signal register " \
"(1 or 2)."

static PyObject* spu_set_signal_mode(PyObject* self, PyObject* args)
{
  SPUContext* ctx;
  unsigned int which;
  unsigned int mode;

  if(!PyArg_ParseTuple(args, "O!II", &SPUContextType, &ctx, &which, &mode)) {
    return NULL;
  }

  spufs_set_signal_mode(ctx->spu_ctx, which, mode);

  Py_RETURN_NONE;
}


// MFC Proxy DMA Functions

//Internal utility routine
static inline PyObject* write_mfc_cmd(
    PyObject* self, PyObject* args, uint16_t dma_cmd)
{
  SPUContext* ctx;
  unsigned int tid = 0;
  unsigned int rid = 0;
  struct mfc_dma_command cmd;
  unsigned int data;
  unsigned int* addr;

  if(!PyArg_ParseTuple(args, "O!IkII|II", &SPUContextType, &ctx,
      &cmd.lsa, &cmd.ea, &cmd.size, &cmd.tag, &tid, &rid)) {
    return NULL;
  }

  cmd.xclass = (tid << 8) | rid;
  cmd.cmd = dma_cmd;

  addr = (unsigned int*)(ctx->spups + 0x3000);
  memcpy(addr, &cmd, sizeof(struct mfc_dma_command));

  //Have to read the command status register to start the DMA
  addr = (unsigned int*)(ctx->spups + 0x3014);
  data = *addr;

  Py_RETURN_NONE;
} 


#define SPU_PUT_DOC \
"Issue a DMA PUT command."

static PyObject* spu_put(PyObject* self, PyObject* args)
{
  return write_mfc_cmd(self, args, MFC_PUT);
}


#define SPU_PUTB_DOC \
"Issue a DMA PUTB command."

static PyObject* spu_putb(PyObject* self, PyObject* args)
{
  return write_mfc_cmd(self, args, MFC_PUTB);
}


#define SPU_PUTF_DOC \
"Issue a DMA PUTF command."

static PyObject* spu_putf(PyObject* self, PyObject* args)
{
  return write_mfc_cmd(self, args, MFC_PUTF);
}


#define SPU_GET_DOC \
"Issue a DMA GET command."

static PyObject* spu_get(PyObject* self, PyObject* args)
{
  return write_mfc_cmd(self, args, MFC_GET);
}


#define SPU_GETB_DOC \
"Issue a DMA GETB command."

static PyObject* spu_getb(PyObject* self, PyObject* args)
{
  return write_mfc_cmd(self, args, MFC_GETB);
}


#define SPU_GETF_DOC \
"Issue a DMA GETF command."

static PyObject* spu_getf(PyObject* self, PyObject* args)
{
  return write_mfc_cmd(self, args, MFC_GETF);
}


#define READ_TAG_STATUS_DOC \
"Read and return an SPU's DMA tag status."

static PyObject* spu_read_tag_status(PyObject* self, PyObject* args)
{
  SPUContext* ctx;
  unsigned int mask;
  unsigned int* addr;

  if(!PyArg_ParseTuple(args, "O!I", &SPUContextType, &ctx, &mask)) {
    return NULL;
  }

  addr = (unsigned int*)(ctx->spups + 0x321C);
  *addr = mask;

  addr = (unsigned int*)(ctx->spups + 0x322C);
  return PyLong_FromUnsignedLong(*addr);
}


#define POLL_TAG_STATUS_DOC \
"Poll an SPU's DMA tag status until the groups indicated by the specified " \
"mask are complete."

static PyObject* spu_poll_tag_status(PyObject* self, PyObject* args)
{
  SPUContext* ctx;
  unsigned int mask;
  unsigned int status;
  unsigned int* addr;

  if(!PyArg_ParseTuple(args, "O!I", &SPUContextType, &ctx, &mask)) {
    return NULL;
  }

  addr = (unsigned int*)(ctx->spups + 0x321C);
  *addr = mask;

  addr = (unsigned int*)(ctx->spups + 0x322C);
  status = *addr;
  while(1) {
    if(status != 0 && (status ^ mask) == 0) {
      return PyLong_FromUnsignedLong(status);
    }
  }

  return PyLong_FromUnsignedLong(0);
}


// SPE Native Code execution
// This allows a user to execute an SPE program in a separate binary via CorePy
#ifdef HAS_LIBSPE2

//Internal function used to actually execute the native code
static int run_native_code(char* filename, SPUExecParams* params)
{
  spe_program_handle_t* prgm;
  spe_context_ptr_t ctx;
  spe_stop_info_t stopinfo;
  unsigned int entry = SPE_DEFAULT_ENTRY;

  ctx = spe_context_create(0, NULL);
  if(ctx == NULL) {
    perror("run_native_code spe_context_create");
    return -1;
  }

  prgm = spe_image_open(filename);
  if(prgm == NULL) {
    perror("run_native_code spe_image_open");
    return -1;
  }

  if(spe_program_load(ctx, prgm) == -1) {
    perror("run_native_code spe_program_load");
    return -1;
  }

  spe_context_run(ctx, &entry, 0, (void*)&params->p, NULL, &stopinfo);

  spe_image_close(prgm);  
  spe_context_destroy(ctx);

  return stopinfo.result.spe_exit_code;
}


static PyObject* spu_run_native_code(PyObject* self, PyObject* args)
{
  char* filename;
  SPUExecParams* params;
  int rc;

  if(!PyArg_ParseTuple(args, "sO!", &filename, &SPUExecParamsType, &params)) {
    return NULL;
  }

  rc = run_native_code(filename, params);
  if(rc == -1) {
    return PyErr_SetFromErrno(PyExc_OSError);
  }
  
  return PyLong_FromLong(rc);
}


struct NativeParams {
  char* filename;
  SPUExecParams* params;
  //PyObject* args;
  pthread_t th;
  int rc;
};


void *run_native_code_thread(void* arg) {
  struct NativeParams* np = arg;

  np->rc = run_native_code(np->filename, np->params);

  return (void*)NULL;
}


static PyObject* spu_run_native_code_async(PyObject* self, PyObject* args)
{
  char* filename;
  SPUExecParams* params;
  struct NativeParams* np;
  int rc;

  if(!PyArg_ParseTuple(args, "sO!", &filename, &SPUExecParamsType, &params)) {
    return NULL;
  }

  np = (struct NativeParams*)malloc(sizeof(struct NativeParams));

  np->filename = strdup(filename);
  np->params = params;
  Py_INCREF(params);

  rc = pthread_create(&np->th, NULL, run_native_code_thread, (void*)np);
  if(rc) {
    Py_DECREF(args);
    free(np->filename);
    free(np);
    errno = rc;
    return PyErr_SetFromErrno(PyExc_OSError);
  }

  return PyLong_FromVoidPtr(np);
}


//TODO - add these methods to the module, test them
static PyObject* spu_join_native_code(PyObject* self, PyObject* args)
{
  struct NativeParams* np;
  int rc;

  np = PyLong_AsVoidPtr(args);

  pthread_join(np->th, NULL);

  rc = np->rc;

  Py_DECREF(np->params);
  free(np->filename);
  free(np);
  return PyLong_FromLong(rc);
}


#else


static PyObject* spu_run_native_code(PyObject* self, PyObject* args)
{
  PyErr_SetString(PyExc_NotImplementedError,
      "Not compiled with libspe2 support");
  return NULL;
}


static PyObject* spu_run_native_code_async(PyObject* self, PyObject* args)
{
  PyErr_SetString(PyExc_NotImplementedError,
      "Not compiled with libspe2 support");
  puts("ERROR run_native_code_async() not available; compile with libspe2 support");
  return NULL;
}


static PyObject* spu_join_native_code(PyObject* self, PyObject* args)
{
  PyErr_SetString(PyExc_NotImplementedError,
      "Not compiled with libspe2 support");
  return NULL;
}


#endif //HAS_LIBSPE2

//TODO - document
static PyMemberDef SPUExecParams_members[] = {
  {"addr", T_UINT, offsetof(SPUExecParams, p.addr), 0, NULL},
  {"p1", T_UINT, offsetof(SPUExecParams, p.p1), 0, NULL},
  {"p2", T_UINT, offsetof(SPUExecParams, p.p2), 0, NULL},
  {"p3", T_UINT, offsetof(SPUExecParams, p.p3), 0, NULL},
  {"size", T_UINT, offsetof(SPUExecParams, p.size), 0, NULL},
  {"p4", T_UINT, offsetof(SPUExecParams, p.p4), 0, NULL},
  {"p5", T_UINT, offsetof(SPUExecParams, p.p5), 0, NULL},
  {"p6", T_UINT, offsetof(SPUExecParams, p.p6), 0, NULL},
  {"p7", T_UINT, offsetof(SPUExecParams, p.p7), 0, NULL},
  {"p8", T_UINT, offsetof(SPUExecParams, p.p8), 0, NULL},
  {"p9", T_UINT, offsetof(SPUExecParams, p.p9), 0, NULL},
  {"p10", T_UINT, offsetof(SPUExecParams, p.p10), 0, NULL},
  {NULL}
};


static PyTypeObject SPUExecParamsType = {
  PyObject_HEAD_INIT(NULL)
  0,                              /*ob_size*/
  "spu_exec.ExecParams",          /*tp_name*/
  sizeof(SPUExecParams),          /*tp_basicsize*/
  0,                              /*tp_itemsize*/
  0,                              /*tp_dealloc*/
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
  0,                              /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT,             /*tp_flags*/
  "ExecParams",                   /*tp_doc */
  0,                              /* tp_traverse */
  0,                              /* tp_clear */
  0,                              /* tp_richcompare */
  0,                              /* tp_weaklistoffset */
  0,                              /* tp_iter */
  0,                              /* tp_iternext */
  0,                              /* tp_methods */
  SPUExecParams_members,          /* tp_members */
  0,
  0,                              /* tp_base */
  0,                              /* tp_dict */
  0,                              /* tp_descr_get */
  0,                              /* tp_descr_set */
  0,                              /* tp_dictoffset */
  0,                              /* tp_init */
  0,                              /* tp_alloc */
  0,                              /* tp_new */
};


//TODO - document
static PyMemberDef SPUContext_members[] = {
  {"spu_ctx", T_ULONG, offsetof(SPUContext, spu_ctx), 0, "spu_ctx"},
  {"spuls", T_ULONG, offsetof(SPUContext, spuls), 0, "spuls"},
  {"spups", T_ULONG, offsetof(SPUContext, spups), 0, "spups"},
  //{"params", T_OBJECT, offsetof(SPUContext, params), 0, "params"},
  {"mode", T_INT, offsetof(SPUContext, mode), 0, "mode"},
  {"stop", T_INT, offsetof(SPUContext, stop), 0, "stop"},
  {NULL}
};


static PyTypeObject SPUContextType = {
  PyObject_HEAD_INIT(NULL)
  0,                              /*ob_size*/
  "spu_exec.Context",             /*tp_name*/
  sizeof(SPUContext),             /*tp_basicsize*/
  0,                              /*tp_itemsize*/
  (destructor)SPUContext_dealloc, /*tp_dealloc*/
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
  0,                              /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT,             /*tp_flags*/
  "SPUContext",                   /*tp_doc */
  0,                              /* tp_traverse */
  0,                              /* tp_clear */
  0,                              /* tp_richcompare */
  0,                              /* tp_weaklistoffset */
  0,                              /* tp_iter */
  0,                              /* tp_iternext */
  0,                              /* tp_methods */
  SPUContext_members,             /* tp_members */
  0,
  0,                              /* tp_base */
  0,                              /* tp_dict */
  0,                              /* tp_descr_get */
  0,                              /* tp_descr_set */
  0,  /* tp_dictoffset */
  (initproc)SPUContext_init,      /* tp_init */
  0,                              /* tp_alloc */
  0,                              /* tp_new */
};



static PyMethodDef module_methods[] = {
  {"make_executable", spu_make_executable, METH_VARARGS, MAKE_EXECUTABLE_DOC},
  {"cancel_async", spu_cancel_async, METH_O, CANCEL_ASYNC_DOC},
  {"alloc_context", spu_alloc_context, METH_NOARGS, ALLOC_CONTEXT_DOC},
  {"free_context", spu_free_context, METH_O, FREE_CONTEXT_DOC},
  {"run_stream", spu_run_stream, METH_VARARGS, RUN_STREAM_DOC},
  {"run_stream_async", spu_run_stream_async, METH_VARARGS, RUN_STREAM_ASYNC_DOC},
  {"wait_stream", spu_wait_stream, METH_O, WAIT_STREAM_DOC},
  {"get_result", spu_get_result, METH_O, GET_RESULT_DOC},
  {"get_num_avail_spus", spu_get_num_avail_spus, METH_NOARGS, GET_NUM_AVAIL_SPUS_DOC},
  {"get_spu_registers", spu_get_spu_registers, METH_VARARGS, GET_SPU_REGISTERS_DOC},
  {"put_spu_registers", spu_put_spu_registers, METH_VARARGS, PUT_SPU_REGISTERS_DOC},
  {"put_spu_params", spu_put_spu_params, METH_VARARGS, PUT_SPU_PARAMS_DOC},
  {"spu_get_phys_id", spu_get_phys_id, METH_O, GET_PHYS_ID_DOC},
  {"read_out_mbox", spu_read_out_mbox, METH_O, READ_OUT_MBOX_DOC},
  {"stat_out_mbox", spu_stat_out_mbox, METH_O, STAT_OUT_MBOX_DOC},
  {"stat_read_out_mbox", spu_stat_read_out_mbox, METH_O, STAT_OUT_MBOX_DOC},
  {"poll_out_mbox", spu_poll_out_mbox, METH_O, POLL_OUT_MBOX_DOC},
  {"read_out_ibox", spu_read_out_ibox, METH_O, READ_OUT_IBOX_DOC},
  {"stat_out_ibox", spu_stat_out_ibox, METH_O, STAT_OUT_IBOX_DOC},
  {"write_in_mbox", spu_write_in_mbox, METH_VARARGS, WRITE_IN_MBOX_DOC},
  {"write_in_mbox_list", spu_write_in_mbox_list, METH_VARARGS, WRITE_IN_MBOX_LIST_DOC},
  {"stat_in_mbox", spu_stat_in_mbox, METH_O, STAT_IN_MBOX_DOC},
  {"write_signal", spu_write_signal, METH_VARARGS, WRITE_SIGNAL_DOC},
  {"set_signal_mode", spu_set_signal_mode, METH_VARARGS, SET_SIGNAL_MODE_DOC},
  {"spu_put", spu_put, METH_VARARGS, SPU_PUT_DOC},
  {"spu_putb", spu_putb, METH_VARARGS, SPU_PUTB_DOC},
  {"spu_putf", spu_putf, METH_VARARGS, SPU_PUTF_DOC},
  {"spu_get", spu_get, METH_VARARGS, SPU_GET_DOC},
  {"spu_getb", spu_getb, METH_VARARGS, SPU_GETB_DOC},
  {"spu_getf", spu_getf, METH_VARARGS, SPU_GETF_DOC},
  {"read_tag_status", spu_read_tag_status, METH_VARARGS, READ_TAG_STATUS_DOC},
  {"poll_tag_status", spu_poll_tag_status, METH_VARARGS, POLL_TAG_STATUS_DOC},
  {"run_native_code", spu_run_native_code, METH_VARARGS, NULL},
  {"run_native_code_async", spu_run_native_code_async, METH_VARARGS, NULL},
  {"join_native_code", spu_join_native_code, METH_O, NULL},
  {NULL}  /* Sentinel */
};


PyMODINIT_FUNC initspu_exec(void)
{
  PyObject* mod;

  SPUContextType.tp_alloc = PyType_GenericAlloc;
  SPUContextType.tp_new = PyType_GenericNew;
  if(PyType_Ready(&SPUContextType) < 0) {
    return;
  }

  SPUExecParamsType.tp_new = PyType_GenericNew;
  if(PyType_Ready(&SPUExecParamsType) < 0) {
    return;
  }


  mod = Py_InitModule("spu_exec", module_methods);

  Py_INCREF(&SPUContextType);
  PyModule_AddObject(mod, "Context", (PyObject*)&SPUContextType);
  Py_INCREF(&SPUExecParamsType);
  PyModule_AddObject(mod, "ExecParams", (PyObject*)&SPUExecParamsType);

  //PyModule_AddIntConstant(mod, "FMT_FLOAT32_1", CAL_FORMAT_FLOAT32_1);

  //TODO - maybe expose available SPUs as a constant?
}

