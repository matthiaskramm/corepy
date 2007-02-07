#include <stdio.h>
#include <spu_intrinsics.h>
#include <spu_mfcio.h>

#define SPU_EXIT_FAILURE 0x2001
#define SPU_READY 0x000D

// There are three ways the user can get parameters:
//    1) $r80-81 (non-volatile regs)
//    2) LS: 0x10, 0x20, 0x30 
//    3) $r3-5 (function parameters)

typedef void (*Stream_func_void)(vector unsigned int v1, 
                                 vector unsigned int v2,
                                 vector unsigned int v3);

int main(unsigned long long id) {

  // Preferred slot parameters
  register unsigned int p1 asm("$3");
  register unsigned int p2 asm("$4");
  register unsigned int p3 asm("$5");
  
  // Full vector parameters
  register vector unsigned int v1 asm("$3");
  register vector unsigned int v2 asm("$4");
  register vector unsigned int v3 asm("$5");

  // Non-volatile registers used as safe storage for parameters
  register vector unsigned int vaddr asm("$80");
  register vector unsigned int vsize asm("$81");
  register vector unsigned int vuser asm("$82");

  unsigned int addr = p1;
  unsigned int size = p2;
  unsigned int user = p3;

  // Save the paramaters into non-volitile registers
  vaddr = v1;
  vsize = v2;
  vuser = v3;

  // Save the parameters into memory
  *((vector unsigned int*)0x10) = v1;
  *((vector unsigned int*)0x20) = v2;
  *((vector unsigned int*)0x30) = v3;

  // Put the instructions at the end of memory (overwrite the stack),
  // rounded down to 128 byte alignment 
  unsigned int lsa = (0x3FFFF - size) & 0xFFF80;

  // Put a branch instruction at address 0 that branches to the new
  // code.  This didn't work using the function call trick after the
  // stop - blowing away the stack messed things up too much.
  // unsigned int ba = 805306368 | (lsa & 0x3FFFC) << 5;
  // *((unsigned int*)0x0) = ba;

  unsigned int stop = SPU_READY;
  *((unsigned int*)(lsa - 4)) = stop;
  
  // Keep this around...for some odd reason things end up here
  // sometimes... 
  //  *((unsigned int*)0x1BEF0) = ba;

  // Recompute the size for an aligned transfer
  size = size + (16 - size % 16);
  
  // printf("lsa: 0x%X addr: 0x%X, size: %d main: 0x%X ba: 0x%X\n", lsa, addr, size, main, ba);

  // Copy the synthetic program and transfer execution
  // mfc_barrier(0);
  // spu_mfcdma32((void*)lsa, addr, size, 12, MFC_GET_CMD);
  // mfc_barrier(0);
  
  // mfc_get(lsa, (void*)((unsigned long)addr), size, 12, 0, 0);

  // Set the tag bit to 12
  // mfc_write_tag_mask(1<<12);

  // Wait for the transfer to complete
  // mfc_read_tag_status_all();

  // Tell the PPU we're ready
  // spu_stop(SPU_READY); 

  // Call the branch instruction we created above.
  // ((Stream_func_void)0)(vaddr, vsize, vuser);
  ((Stream_func_void)(lsa - 4))(vaddr, vsize, vuser);

  // This should never be executed
  return SPU_EXIT_FAILURE;
}
