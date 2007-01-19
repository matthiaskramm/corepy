#include <errno.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>
#include <sys/syscall.h>
#include <string.h>

// In an ideal future, this no longer exists...
// This is only here so we can duplicate the interface from 
// the old ugly way of doing things.
struct ExecParams {
  // $r3
  unsigned int addr;   // address of syn code
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
};

void copy_memory(char* dest, char* src, int length)
{
  memcpy(dest, src, length);
}

void copy_memory_to_file(int fd, int floc, int src, int length)
{
  char *src_addr = (char*)src;
  lseek(fd, floc, SEEK_SET);
  write(fd, src_addr, length);
}

int spu_create(char* filename, char* error_msg=(char*)0) 
{  
  int error;
  int fd = syscall(SYS_spu_create, filename, 0, S_IRWXU);

  if (fd == -1)
  {
    error = errno;
    printf("errno = %s\n", strerror(error));
    error_msg = strerror(error);
  }

  return fd;
}

void *spu_run(int fd, unsigned int npc, unsigned int status)  //PyTupleObject *info)
{
  int retval;
  /*
  int fd; 
  int npc;
  fd = (int)PyInt_AS_LONG(PyTuple_GET_ITEM(&info, 0));
  npc = (int)PyInt_AS_LONG(PyTuple_GET_ITEM(&info, 1));
  */

  //  syscall(SYS_spu_run, fd, (unsigned int *)&npc, (unsigned int *)0);
  printf("fd = %i npc = %x status = %i\n", fd, npc, status);
  retval = syscall(SYS_spu_run, fd, (unsigned int *)&npc, (unsigned int*)&status);	       
  printf("status = %i\n", status);
  return (void*)retval;
  
}

int spu_run_sync(int fd, unsigned int npc, unsigned int status)  //PyTupleObject *info)
{
  int retval;
  retval = syscall(SYS_spu_run, fd, (unsigned int *)&npc, (unsigned int*)&status);	       
  return retval;
}
