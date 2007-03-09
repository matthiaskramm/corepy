# Copyright 2006-2007 The Trustees of Indiana University.

# This software is available for evaluation purposes only.  It may not be
# redistirubted or used for any other purposes without express written
# permission from the authors.

# Authors:
#   Christopher Mueller (chemuell@cs.indiana.edu)
#   Andrew Lumsdaine    (lums@cs.indiana.edu)

import os
import mmap
import thread
import spu_syscalls
import synbuffer
import pdb

import spu_exec
ExecParams = spu_syscalls.ExecParams
#from spu_exec import ExecParams
from spu_exec import aligned_memory

def execute_param_async(addr, params): #(buffer, parameters = None):
    size = params.size
    lsa = 256*1024 - size*4 - (16-((size*4)%16))

    (fd, filename) = _create_thread()

    success = _copy_program(filename, lsa, addr, params)

    err_ret = 0
    return thread.start_new_thread(spu_syscalls.spu_run, (fd, lsa, err_ret))

def wait_async(id, result):
    err = 0
    

def execute_param_int(addr, params):
    size = params.size
    lsa = 256*1024 - size*4 - (16-((size*4)%16))

    (fd, filename) = _create_thread()
    success = _copy_program(filename, lsa, addr, params)
    err_ret = 0
    retval = spu_syscalls.spu_run_sync(fd, lsa, err_ret)
    # return (retval & 0x3fff0000) >> 16
    return (retval >> 16) & 0xFF

def _create_thread():
    filename = "/spu/24680" # ideally this should be generated as a unique id
    error = "                                                                "

    fd = spu_syscalls.spu_create(filename, error)
    if fd == -1:
        print "Error creating spu thread. error = ", error
    return (fd, filename)

def _copy_program(filename, lsa, addr, params):
    size = params.size
    lsa = 256*1024 - size*4 - (16-((size*4)%16))

    mem_filename = filename + "/mem"
    regs_filename = filename + "/reg"

    # copy memory over
    pdb.set_trace()
    fd = os.open(mem_filename, os.O_RDWR)
    # f = os.fdopen(fd)
    #    os.ftruncate(fd, 1024*256)
    # os.lseek(fd, 1024*256, 0)
    # f.truncate(1024*256)
    # os.ftruncate(fd, 1024*256)
    # m = mmap.mmap(fd, 1) # os.fstat(fd).st_size)#, access=mmap.PROT_READ|mmap.PROT_WRITE)
    # m.resize(256*1024)
    # spu_syscalls.copy_memory(synbuffer.buffer_info(m)[0]+lsa, addr, size)
    spu_syscalls.copy_memory_to_file(fd, lsa, addr, size)
    os.close(fd)

    # parameters = None
    # this is most certainly not the correct way to do this -
    # at least if one believes in using official, stable interfaces...
    # if parameters != None:
    #    g = open(filename3, "w")
    #    parameters.tofile(g)
    #    g.close()

    return True

# def execute_param_async(addr, ExecParams params): #(buffer, parameters = None):

#     filename = "/spu/24680" # ideally this should be generated as a unique id
#     filename2 = filename + "/mem"
#     filename3 = filename + "/regs"
#     lsa = 256*1024 - len(buffer)*4 - (16-((len(buffer)*4)%16))
#     error = "                                                                "

#     fd = spu_syscalls.spu_create(filename, error)
#     if fd == -1:
#         print "Error. error = ", error

#     # pdb.set_trace()
#     # copy memory over
#     f = open(filename2, "w")
#     f.seek(lsa)
#     buffer.tofile(f)
#     f.close()

#     parameters = None
#     # this is most certainly not the correct way to do this -
#     # at least if one believes in using official, stable interfaces...
#     if parameters != None:
#         g = open(filename3, "w")
#         parameters.tofile(g)
#         g.close()

#     err_ret = 0
#     # status = spu_syscalls.spu_run(fd, lsa) # lsa WILL be modified
#     pdb.set_trace()
#     return thread.start_new_thread(spu_syscalls.spu_run, (fd, lsa, err_ret))
#     # pdb.set_trace()
#     # spu_syscalls.spu_run(fd, lsa, err_ret)
