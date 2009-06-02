# Copyright (c) 2006-2009 The Trustees of Indiana University.                   
# All rights reserved.                                                          
#                                                                               
# Redistribution and use in source and binary forms, with or without            
# modification, are permitted provided that the following conditions are met:   
#                                                                               
# - Redistributions of source code must retain the above copyright notice, this 
#   list of conditions and the following disclaimer.                            
#                                                                               
# - Redistributions in binary form must reproduce the above copyright notice,   
#   this list of conditions and the following disclaimer in the documentation   
#   and/or other materials provided with the distribution.                      
#                                                                               
# - Neither the Indiana University nor the names of its contributors may be used
#   to endorse or promote products derived from this software without specific  
#   prior written permission.                                                   
#                                                                               
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"   
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE     
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE   
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL    
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR    
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER    
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, 
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.          

import corepy.lib.extarray as extarray
import corepy.arch.spu.isa as spu
import corepy.arch.spu.platform as env
import corepy.arch.spu.lib.dma as dma
from corepy.arch.spu.lib.util import load_word

# This example program demonstrates sending mailbox messages from one SPU to
# another.  In order for an SPU to send messages/signals to another SPU, the
# source SPU must know that base address of the memory-mapped problem state area
# of the target SPU.  However the addresses are not known until the SPUs have
# been started, so the addresses must be passed to the SPUs by the PPU.  The PPU
# builds one array of the addresses for the SPUs, then gives the address of this
# array to each SPU to DMA into local store and load into registers.

# A race condition is possible if mailboxes are used to send the address of the
# array.  What can happen is that an earlier SPU gets the message, loads the
# array into registers, and sends a mailbox message to a following SPU, before
# that following SPU receives the initial array address message from the PPU.
# The solution used in this example program is to use signal to send the array
# address instead of a mailbox.

if __name__ == '__main__':
  SPUS = 6

  proc = env.Processor()
  codes = [env.InstructionStream() for i in xrange(0, SPUS)]

  for rank, code in enumerate(codes):
    spu.set_active_code(code)

    # First all the SPUs should start up and wait for an mbox message.
    # The PPU will collect all the PS map addresses into an array for the SPUs.
    r_psinfo_mma = dma.spu_read_signal1(code)

    # DMA the PS info into local store
    dma.mem_get(code, 0x0, r_psinfo_mma, SPUS * 4 * 4, 17)
    dma.mem_complete(code, 17)

    # Load the PS info into some registers.. one register per address
    r_psinfo = code.acquire_registers(SPUS)
    for i in xrange(0, SPUS):
      spu.lqd(r_psinfo[i], code.r_zero, i)

    # Initialize a data register with this rank and store it at LSA 0
    r_send = code.acquire_register()
    load_word(code, r_send, rank)
    spu.stqd(r_send, code.r_zero, 0)
    code.release_register(r_send)

    # Send our rank as a mailbox message to the rank after this rank
    dma.mem_write_in_mbox(code, r_psinfo[(rank + 1) % SPUS], 12, 18)
    dma.mem_complete(code, 18)

    # Receive the message the preceding rank sent
    r_recv = dma.spu_read_in_mbox(code)

    # Write the value out the interrupt mailbox for the PPU
    dma.spu_write_out_intr_mbox(code, r_recv)
    code.release_register(r_recv)


  # Start the SPUs
  id = [proc.execute(codes[i], async = True) for i in xrange(0, SPUS)]

  # Set up an array of pointers to PS maps.
  psinfo = extarray.extarray('I', SPUS * 4)
  for i in xrange(0, SPUS * 4, 4):
    psinfo[i] = id[i / 4].spups
  psinfo.synchronize()

  # Send the psinfo address to all the SPUs.
  addr = psinfo.buffer_info()[0]
  for i in xrange(0, SPUS):
    env.spu_exec.write_signal(id[i], 1, addr)

  # Wait for a mailbox message from each SPU; the value should be the preceding
  # rank.  Join each SPU once the message is received, too.
  for i in xrange(0, SPUS):
    val = env.spu_exec.read_out_ibox(id[i])
    assert(val == (i - 1) % SPUS)

    proc.join(id[i])


