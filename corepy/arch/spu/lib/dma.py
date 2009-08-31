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

# Synthetic functions and components for DMA operations
# Based on /opt/IBM/cell-sdk-1.1/src/include/spu/spu_mfcio.h
#          /opt/IBM/cell-sdk-1.1/sysroot/usr/lib/gcc/spu/4.0.2/include/spu_internals.h
#          linux-2.6.16/include/asm-powerpc/spu.h

import corepy.arch.spu.isa as spu
import corepy.arch.spu.lib.util as util
import corepy.spre.spe as spe

# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------

SPE_SIG_NOTIFY_REG_1 = 1
SPE_SIG_NOTIFY_REG_2 = 2


# MFC Commands (spu.h)
MFC_PUT_CMD    = 0x20
MFC_PUTS_CMD   = 0x28
MFC_PUTR_CMD   = 0x30
MFC_PUTF_CMD   = 0x22
MFC_PUTB_CMD   = 0x21
MFC_PUTFS_CMD  = 0x2A
MFC_PUTBS_CMD  = 0x29
MFC_PUTRF_CMD  = 0x32
MFC_PUTRB_CMD  = 0x31
MFC_PUTL_CMD   = 0x24
MFC_PUTRL_CMD  = 0x34
MFC_PUTLF_CMD  = 0x26
MFC_PUTLB_CMD  = 0x25
MFC_PUTRLF_CMD = 0x36
MFC_PUTRLB_CMD = 0x35

MFC_GET_CMD    = 0x40
MFC_GETS_CMD   = 0x48
MFC_GETF_CMD   = 0x42
MFC_GETB_CMD   = 0x41
MFC_GETFS_CMD  = 0x4A
MFC_GETBS_CMD  = 0x49
MFC_GETL_CMD   = 0x44
MFC_GETLF_CMD  = 0x46
MFC_GETLB_CMD  = 0x45

# SPU Event Mask
SPU_WrEventMask  = 1
SPU_WrEventAck   = 2
SPU_RdEventMask  = 11

# Signals (ibid)
SPU_RdSigNotify1 = 3
SPU_RdSigNotify2 = 4

# SPU Decrementer Channels (BE Handbook, Ch. 17, p443-445)
SPU_WrDec = 7
SPU_RdDec = 8

# MFC Command Parameter Channels (BE Handbook, Ch. 17, p443-445)
MFC_LSA   = 16
MFC_EAH   = 17
MFC_EAL   = 18
MFC_Size  = 19
MFC_TagID = 20
MFC_Cmd   = 21

# MFC Tag Status (ibid)
MFC_WrTagMask   = 22
MFC_WrTagUpdate = 23
MFC_RdTagStat   = 24

# Mbox (ibid)
SPU_WrOutMbox   = 28
SPU_RdInMbox    = 29
SPU_WrOutIntrMbox = 30

# Mask values for SPU_RdEventStat, SPU_WrEventMask, and SPU_WrEventAck (BE Handbook 447,384,464)
MFC_DECREMENTER_EVENT = 0x0020

# Tag Status update conditions (BE Handbook, p458)
MFC_TAG_UPDATE_IMMEDIATE = 0
MFC_TAG_UPDATE_ANY = 1
MFC_TAG_UPDATE_ALL = 2


# ------------------------------------------------------------
# High-level DMA routines
# ------------------------------------------------------------

# TODO - routines for DMA lists

# TODO - should this be merged into existing routines like mfc_get and mfc_put?

def mem_get(code, lsa, mma, size, tag):
  """Start a single asynchronous DMA GET operation.  Parameters are as follows:
    
      lsa     Local store address of data destination
      mma     Main memory address of source data
      size    Size in bytes of data to transfer; hardware limit is 16Kb
      tag     DMA control tag to associate with this transfer
  """

  param_regs = {}

  r_lsa = util.get_param_reg(code, lsa, param_regs, copy = False)
  r_mma = util.get_param_reg(code, mma, param_regs, copy = False)
  r_size = util.get_param_reg(code, size, param_regs, copy = False)
  r_tag = util.get_param_reg(code, tag, param_regs, copy = False)

  mfc_getf(code, r_lsa, r_mma, r_size, r_tag)
  
  util.put_param_reg(code, r_lsa, param_regs)
  util.put_param_reg(code, r_mma, param_regs)
  util.put_param_reg(code, r_size, param_regs)
  util.put_param_reg(code, r_tag, param_regs)
  return


def mem_put(code, lsa, mma, size, tag):
  """Start a single asynchronous DMA PUT operation.  Parameters are as follows:
    
      lsa     Local store address of source data
      mma     Main memory address of data destination
      size    Size in bytes of data to transfer; hardware limit is 16Kb
      tag     DMA control tag to associate with this transfer
  """

  param_regs = {}

  r_lsa = util.get_param_reg(code, lsa, param_regs, copy = False)
  r_mma = util.get_param_reg(code, mma, param_regs, copy = False)
  r_size = util.get_param_reg(code, size, param_regs, copy = False)
  r_tag = util.get_param_reg(code, tag, param_regs, copy = False)

  mfc_putb(code, r_lsa, r_mma, r_size, r_tag)

  util.put_param_reg(code, r_lsa, param_regs)
  util.put_param_reg(code, r_mma, param_regs)
  util.put_param_reg(code, r_size, param_regs)
  util.put_param_reg(code, r_tag, param_regs)
  return


def mem_write_in_mbox(code, psmap, lsa, tag, cache = False):
  """Write a 32bit message at a local LSA from this SPU to another.
     psmap must contain the base address of the target SPU's PS map.
     lsa must be 12 mod 16 for DMA alignment purposes.

     This is a DMA operation; it must be completed using mem_complete() or
     similar method."""

  if isinstance(lsa, (int, long)):
    if (lsa % 16) != 12:
      print "ERROR LSA for mem_write_mbox() must be 12 mod 16"
      assert(0)

#  r_mbox_mma_cached = True
#  ref = "__mem_write_in_mbox_mma_reg_%s" % (str(psmap))
#  r_mbox_mma = code.prgm.get_storage(ref)
#  if not isinstance(r_mbox_mma, spu.Register):
#    r_size_cached = False
#    r_mbox_mma = code.acquire_register()
#    if isinstance(psmap, (int, long)):
#      util.load_word(code, r_mbox_mma, psmap + 0x400C)
#    else:
#      util.load_word(code, r_mbox_mma, 0x400C)
#      code.add(spu.a(r_mbox_mma, r_mbox_mma, psmap))
#
#    if cache == True:
#      r_mbox_mma_cached = True
#      code.prgm.add_storage(ref, r_mbox_mma)

  r_mbox_mma = code.prgm.acquire_register()
  if isinstance(psmap, (int, long)):
    util.load_word(code, r_mbox_mma, psmap + 0x400C)
  else:
    util.load_word(code, r_mbox_mma, 0x400C)
    code.add(spu.a(r_mbox_mma, r_mbox_mma, psmap))

  r_size_cached = True
  ref = "_const_val_4"
  r_size = code.prgm.get_storage(ref)
  if not isinstance(r_size, spu.Register):
    r_size_cached = False
    r_size = code.prgm.acquire_register()
    util.load_word(code, r_size, 4)
    if cache == True:
      r_size_cached = True
      code.prgm.add_storage(ref, r_size)

  mem_put(code, lsa, r_mbox_mma, r_size, tag)

  code.prgm.release_register(r_mbox_mma)
  if cache == False:
    #if not isinstance(psmap, (int, long)) and r_mbox_mma_cached == False:
    if r_size_cached == False:
      code.prgm.release_register(r_size)
  return


def mem_write_signal(code, which, psmap, lsa, tag, cache = False):
  """Write a 32bit message at a local LSA from this SPU to another.
     psmap must contain the base address of the target SPU's PS map.
     lsa must be 12 mod 16 for DMA alignment purposes.

     This is a DMA operation; it must be completed using mem_complete() or
     similar method."""

  if isinstance(lsa, (int, long)):
    if (lsa % 16) != 12:
      print "ERROR LSA for mem_write_signal() must be 12 mod 16"
      assert(0)

  addr = 0x1400C
  if which == 2:
    addr = 0x1C00C

#  r_sig_mma_cached = True
#  ref = "__mem_write_signal_mma_reg_%d_%s" % (which, str(psmap))
#  r_sig_mma = code.prgm.get_storage(ref)
#  if not isinstance(r_sig_mma, spu.Register):
#    r_sig_mma_cached = False
#    r_sig_mma = code.acquire_register()
#    if isinstance(psmap, (int, long)):
#      util.load_word(code, r_sig_mma, psmap + addr)
#    else:
#      util.load_word(code, r_sig_mma, addr)
#      code.add(spu.a(r_sig_mma, r_sig_mma, psmap))

#    if cache == True:
#      r_sig_mma_cached = True
#      code.prgm.add_storage(ref, r_sig_mma)

  r_sig_mma = code.prgm.acquire_register()
  if isinstance(psmap, (int, long)):
    util.load_word(code, r_sig_mma, psmap + addr)
  else:
    util.load_word(code, r_sig_mma, addr)
    code.add(spu.a(r_sig_mma, r_sig_mma, psmap))

  #r_size_cached = True
  #ref = "_const_val_4"
  #r_size = code.prgm.get_storage(ref)
  #if not isinstance(r_size, spu.Register):
  #  r_size_cached = False
  #  r_size = code.acquire_register()
  #  util.load_word(code, r_size, 4)
  #  if cache == True:
  #    r_size_cached = True
  #    code.prgm.add_storage(ref, r_size)

  r_size = code.prgm.acquire_register()
  util.load_word(code, r_size, 4)

  mem_put(code, lsa, r_sig_mma, r_size, tag)
  code.prgm.release_register(r_size)
  code.prgm.release_register(r_sig_mma)

  #if cache == False:
    #if not isinstance(psmap, (int, long)):
    #if r_size_cached == False:
    #  code.release_register(r_size)
  return


def mem_complete(code, tag):
  """Complete a set of asynchronous DMA operations.

     If tag is a literal number, all DMA operations on that tag are completed.
     If a register provided, that register is used as-is as the tag mask.  The
     tag mask is a bit field indicating which tags should be completed.  For
     example, to complete only tag number 12, the tag mask should have the
     value 1 << 12, or have the bit in position 12 set.  Multiple tags may be
     completed by setting multiple bits.
  """

  if isinstance(tag, (spe.Register, spe.Variable)):
    mfc_write_tag_mask(code, tag)
  else:
    mfc_write_tag_mask(code, 1 << tag)
    
  r_status = mfc_read_tag_status_all(code)
  code.prgm.release_register(r_status)
  return



# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def MFC_CMD_WORD(tid, rid, cmd):
  return (tid << 24) | (rid << 16) | cmd


# ------------------------------------------------------------
# Core dma functions
# ------------------------------------------------------------

def spu_mfcdma32(code, r_ls, r_ea, r_size, r_tagid, cmd):
#  print "spu_mfcdma32 cmd", cmd, str(cmd)
#  ref = "__spu_mfcdma32_cmd_%s" % str(cmd)
#  r_cmd = code.prgm.get_storage(ref)
#  if not isinstance(r_cmd, spu.Register):
#    r_cmd = code.acquire_register()
#    util.load_word(code, r_cmd, cmd)
#    code.prgm.add_storage(ref, r_cmd)
  
  r_cmd = code.prgm.acquire_register()
  util.load_word(code, r_cmd, cmd)

  code.add(spu.wrch(r_ls, MFC_LSA))
  code.add(spu.wrch(r_ea, MFC_EAL))
  code.add(spu.wrch(r_size, MFC_Size))
  code.add(spu.wrch(r_tagid, MFC_TagID))
  last = code.add(spu.wrch(r_cmd, MFC_Cmd))

  code.prgm.release_register(r_cmd)
  return last

def spu_mfcdma64(code, r_ls, r_eah, r_eal, r_size, r_tagid, cmd):
  r_cmd = code.prgm.acquire_register()
  util.load_word(code, r_cmd, cmd)

  code.add(spu.wrch(r_ls, MFC_LSA))
  code.add(spu.wrch(r_eah, MFC_EAH))
  code.add(spu.wrch(r_eal, MFC_EAL))
  code.add(spu.wrch(r_size, MFC_Size))
  code.add(spu.wrch(r_tagid, MFC_TagID))
  last = code.add(spu.wrch(r_cmd, MFC_Cmd))

  code.release_register(r_cmd)
  return last

def spu_writech(code, ch, msg):
  # msg may be either a literal value, or a register containing the value
  if isinstance(msg, (spe.Register, spe.Variable)):
    last = code.add(spu.wrch(msg, ch))
  else:
    r_msg = code.prgm.acquire_register()
    util.load_word(code, r_msg, msg)
    last = code.add(spu.wrch(r_msg, ch))
    code.prgm.release_register(r_msg)

  return last

def spu_readch(code, ch, r_msg = None):
  if r_msg is None:
    r_msg = code.prgm.acquire_register()

  code.add(spu.rdch(r_msg, ch))
  return r_msg


# ------------------------------------------------------------
# MFC DMA Commands
# ------------------------------------------------------------

def mfc_put(code, r_ls, r_ea, r_size, r_tagid, r_tid = 0, r_rid = 0):
  return spu_mfcdma32(code, r_ls, r_ea, r_size, r_tagid, MFC_CMD_WORD(r_tid, r_rid,MFC_PUT_CMD))

def mfc_putf(code, r_ls, r_ea, r_size, r_tagid, r_tid = 0, r_rid = 0):
  return spu_mfcdma32(code, r_ls, r_ea, r_size, r_tagid, MFC_CMD_WORD(r_tid, r_rid,MFC_PUTF_CMD))

def mfc_putb(code, r_ls, r_ea, r_size, r_tagid, r_tid = 0, r_rid = 0):
  return spu_mfcdma32(code, r_ls, r_ea, r_size, r_tagid, MFC_CMD_WORD(r_tid, r_rid,MFC_PUTB_CMD))

def mfc_get(code, r_ls, r_ea, r_size, r_tagid, r_tid = 0, r_rid = 0):
  return spu_mfcdma32(code, r_ls, r_ea, r_size, r_tagid, MFC_CMD_WORD(r_tid, r_rid,MFC_GET_CMD))

def mfc_getf(code, r_ls, r_ea, r_size, r_tagid, r_tid = 0, r_rid = 0):
  return spu_mfcdma32(code, r_ls, r_ea, r_size, r_tagid, MFC_CMD_WORD(r_tid, r_rid,MFC_GETF_CMD))

def mfc_getb(code, r_ls, r_ea, r_size, r_tagid, r_tid = 0, r_rid = 0):
  return spu_mfcdma32(code, r_ls, r_ea, r_size, r_tagid, MFC_CMD_WORD(r_tid, r_rid,MFC_GETB_CMD))


# ------------------------------------------------------------
# MFC Tag-Status
# ------------------------------------------------------------

def mfc_write_tag_mask(code, mask):
  return spu_writech(code, MFC_WrTagMask, mask)

# define mfc_read_tag_mask()              spu_readch(MFC_RdTagMask)

def mfc_write_tag_update(code, ts):
  return spu_writech(code, MFC_WrTagUpdate, ts)

# define mfc_write_tag_update_immediate() mfc_write_tag_update(MFC_TAG_UPDATE_IMMEDIATE)
# define mfc_write_tag_update_any()       mfc_write_tag_update(MFC_TAG_UPDATE_ANY)

def mfc_write_tag_update_all(code):
  return mfc_write_tag_update(code, MFC_TAG_UPDATE_ALL)

# define mfc_stat_tag_update()            spu_readchcnt(MFC_WrTagUpdate)

def mfc_read_tag_status(code):
  return spu_readch(code, MFC_RdTagStat)

# define mfc_read_tag_status_immediate()  mfc_write_tag_update_immediate();    \
#                                         mfc_read_tag_status()
# define mfc_read_tag_status_any()        mfc_write_tag_update_any();        \
#                                         mfc_read_tag_status()

def mfc_read_tag_status_all(code):
  mfc_write_tag_update_all(code)
  return mfc_read_tag_status(code)

# define mfc_stat_tag_status()            spu_readchcnt(MFC_RdTagStat)


# ------------------------------------------------------------
# Signals
# ------------------------------------------------------------

def spu_read_signal1(code): return spu_readch(code, SPU_RdSigNotify1)
def spu_stat_signal1(code): return spu_readchcnt(code, SPU_RdSigNotify1)
def spu_read_signal2(code): return spu_readch(code, SPU_RdSigNotify2)
def spu_stat_signal2(code): return spu_readchcnt(code, SPU_RdSigNotify2)

# ------------------------------------------------------------
# Mail Box
# ------------------------------------------------------------

def spu_read_in_mbox(code):
  return spu_readch(code, SPU_RdInMbox)

def spu_stat_in_mbox(code):
  return spu_readchcnt(code, SPU_RdInMbox)

def spu_write_out_mbox(code, data):
  return spu_writech(code, SPU_WrOutMbox, data)

def spu_stat_out_mbox(code):
  return spu_readchcnt(code, SPU_WrOutMbox)

def spu_write_out_intr_mbox(code, data):
  return spu_writech(code, SPU_WrOutIntrMbox, data)

def spu_stat_out_intr_mbox(code):
  return spu_readchcnt(code, SPU_WrOutIntrMbox)

# ------------------------------------------------------------
# Performance Decrementer
# ------------------------------------------------------------

def spu_read_decr(code, reg = None):
  return spu_readch(code, SPU_RdDec, reg)

def spu_write_decr(code, data):
  return spu_writech(code, SPU_WrDec, data)

def spu_start_decr(code):
  return spu_writech(code, SPU_WrEventMask, MFC_DECREMENTER_EVENT)

def spu_stop_decr(code):
  spu_writech(code, SPU_WrEventMask, 0)
  return spu_writech(code, SPU_WrEventAck, MFC_DECREMENTER_EVENT)

# ------------------------------------------------------------
# Unit Tests
# ------------------------------------------------------------

def TestMFC():
  import corepy.lib.extarray as extarray
  import corepy.arch.spu.platform as synspu 

  size = 32
  #data_array = array.array('I', range(size))
  #data = synspu.aligned_memory(size, typecode = 'I')
  #data.copy_to(data_array.buffer_info()[0], len(data_array))
  data = extarray.extarray('I', range(size))
  code = synspu.InstructionStream()

  r_zero    = code.acquire_register()
  r_ea_data = code.acquire_register()
  r_ls_data = code.acquire_register()
  r_size    = code.acquire_register()
  r_tag     = code.acquire_register()  

  # Load zero
  util.load_word(code, r_zero, 0)

  print 'array ea: %X' % (data.buffer_info()[0])
  print 'r_zero = %s, ea_data = %s, ls_data = %s, r_size = %s, r_tag = %s' % (
    str(r_zero), str(r_ea_data), str(r_ls_data), str(r_size), str(r_tag))
  
  # Load the effective address
  print 'test ea: %X' % data.buffer_info()[0]
  util.load_word(code, r_ea_data, data.buffer_info()[0])

  # Load the size
  code.add(spu.ai(r_size, r_zero, size * 4))

  # Load the tag
  code.add(spu.ai(r_tag, r_zero, 2))

  # Load the lsa
  code.add(spu.ai(r_ls_data, r_zero, 0))

  # Load the data into address 0
  mfc_get(code, r_ls_data, r_ea_data, r_size, r_tag)

  # Set the tag bit to 2
  mfc_write_tag_mask(code, 1<<2);

  # Wait for the transfer to complete
  mfc_read_tag_status_all(code);

  # Increment the data values by 1 using an unrolled loop (no branches)
  r_current = code.acquire_register()

  for lsa in range(0, size * 4, 16):
    code.add(spu.lqa(r_current, (lsa >> 2)))
    code.add(spu.ai(r_current, r_current, 1))
    code.add(spu.stqa(r_current, (lsa >> 2)))

  code.release_register(r_current)
             
  # Store the values back to main memory

  # Load the data into address 0
  mfc_put(code, r_ls_data, r_ea_data, r_size, r_tag)

  # Set the tag bit to 2
  mfc_write_tag_mask(code, 1<<2);

  # Wait for the transfer to complete
  mfc_read_tag_status_all(code);

  # Cleanup
  code.release_register(r_zero)
  code.release_register(r_ea_data)
  code.release_register(r_ls_data)  
  code.release_register(r_size)
  code.release_register(r_tag)  

  # Stop for debugging
  # code.add(spu.stop(0xA))

  # Execute the code
  proc = synspu.Processor()
  # code.print_code()
  #print data_array
  proc.execute(code)

  #data.copy_from(data_array.buffer_info()[0], len(data_array))

  for i in range(size):
    assert(data[i] == i + 1)
  
  return


def TestMbox():
  import corepy.arch.spu.platform as synspu 

  code = synspu.InstructionStream()

  # Send a message to the PPU
  spu_write_out_mbox(code, 0xDEADBEEFl)

  # Get a message from the PPU
  reg = spu_read_in_mbox(code)

  # And send it back
  code.add(spu.wrch(reg, SPU_WrOutMbox))
  
  proc = synspu.Processor()

  spe_id = proc.execute(code, async=True)
  synspu.spu_exec.write_in_mbox(spe_id, 0x88CAFE)

  while synspu.spu_exec.stat_out_mbox(spe_id) == 0: pass
  print 'spe said: 0x%X' % (synspu.spu_exec.read_out_mbox(spe_id))
  while synspu.spu_exec.stat_out_mbox(spe_id) == 0: pass
  print 'spe said: 0x%X' % (synspu.spu_exec.read_out_mbox(spe_id))

  proc.join(spe_id)
  
  return


def TestSignal():
  import corepy.arch.spu.platform as synspu 

  code = synspu.InstructionStream()

  # Get a signal from the PPU
  reg = spu_read_signal1(code)

  # And send it back
  code.add(spu.wrch(reg, SPU_WrOutMbox))
  
  proc = synspu.Processor()

  spe_id = proc.execute(code, async=True)
  synspu.spu_exec.write_signal(spe_id, 1, 0xCAFEBABEl)
  
  while synspu.spu_exec.stat_out_mbox(spe_id) == 0: pass

  print 'sig said: 0x%X' % (synspu.spu_exec.read_out_mbox(spe_id))

  proc.join(spe_id)
  
  return


def TestDecrementer():
  import corepy.arch.spu.platform as synspu 
  import time

  code = synspu.InstructionStream()

  spu_write_decr(code, 0x7FFFFFFFl)
  spu_start_decr(code)

  # Get a message from the PPU
  spu_read_in_mbox(code)

  reg = spu_read_decr(code)
  spu_write_out_mbox(code, reg)
  spu_stop_decr(code)

  proc = synspu.Processor()

  spe_id = proc.execute(code, async=True)

  print 'test is sleeping for 1 second'
  time.sleep(1)
  synspu.spu_exec.write_in_mbox(spe_id, 0x44CAFE)

  while synspu.spu_exec.stat_out_mbox(spe_id) == 0: pass

  print 'spu said: 0x%X' % (synspu.spu_exec.read_out_mbox(spe_id))

  proc.join(spe_id)

  return


if __name__=='__main__':
  TestMFC()
  TestMbox()
  TestSignal()
  TestDecrementer()

