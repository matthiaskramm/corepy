# Synthetic functions and components for DMA operations
# Based on /opt/IBM/cell-sdk-1.1/src/include/spu/spu_mfcio.h
#          /opt/IBM/cell-sdk-1.1/sysroot/usr/lib/gcc/spu/4.0.2/include/spu_internals.h
#          linux-2.6.16/include/asm-powerpc/spu.h

import array
import corepy.arch.spu.platform as synspu 
import corepy.arch.spu.isa as spu
import corepy.arch.spu.lib.util as util

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


# MFC Command Parameter Channels (BE Handbook, Ch. 17, p443-445)
MFC_LSA   = 16
MFC_EAH   = 17
MFC_EAL   = 18
MFC_Size  = 19
MFC_TagID = 20
MFC_Cmd   = 21

# Signals (ibid)
SPU_RdSigNotify1 = 3
SPU_RdSigNotify2 = 4

# MFC Tag Status (ibid)
MFC_WrTagMask   = 22
MFC_WrTagUpdate = 23
MFC_RdTagStat   = 24

# Mbox (ibid)
SPU_WrOutMbox   = 28
SPU_RdInMbox    = 29
SPU_WrOutIntrMbox = 30

# Tag Status update conditions (BE Handbook, p458)
MFC_TAG_UPDATE_IMMEDIATE = 0
MFC_TAG_UPDATE_ANY = 1
MFC_TAG_UPDATE_ALL = 2

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def MFC_CMD_WORD(tid, rid, cmd):
  return (tid << 24) | (rid << 16) | cmd


# ------------------------------------------------------------
# Core dma functions
# ------------------------------------------------------------

def spu_mfcdma32(code, r_ls, r_ea, r_size, r_tagid, cmd):
  r_cmd = code.acquire_register()
  util.load_word(code, r_cmd, cmd)
  
  code.add(spu.wrch(r_ls, MFC_LSA))
  code.add(spu.wrch(r_ea, MFC_EAL))
  code.add(spu.wrch(r_size, MFC_Size))
  code.add(spu.wrch(r_tagid, MFC_TagID))
  last = code.add(spu.wrch(r_cmd, MFC_Cmd))

  code.release_register(r_cmd)
  return last

def spu_mfcdma64(code, r_ls, r_eah, r_eal, r_size, r_tagid, cmd):
  r_cmd = code.acquire_register()
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
  r_msg = code.acquire_register()

  util.load_word(code, r_msg, msg)
  last = code.add(spu.wrch(r_msg, ch))

  code.release_register(r_msg)
  return last

def spu_readch(code, ch, r_msg = None):
  if r_msg is None:
    r_msg = code.acquire_register()

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
  r_status = spu_readch(code, MFC_RdTagStat)
  return r_status

# define mfc_read_tag_status_immediate()  mfc_write_tag_update_immediate();    \
#                                         mfc_read_tag_status()
# define mfc_read_tag_status_any()        mfc_write_tag_update_any();        \
#                                         mfc_read_tag_status()

def mfc_read_tag_status_all(code):
  mfc_write_tag_update_all(code)
  r_status = mfc_read_tag_status(code)
  return r_status

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

def spu_write_out_mbox(code, data):
  return spu_writech(code, SPU_WrOutMbox,data)

def spu_stat_out_mbox(code):
  spu_readchcnt(code, SPU_WrOutMbox)

def spu_write_out_intr_mbox(code, data):
  return spu_writech(code, SPU_WrOutIntrMbox, data)

def spu_stat_out_intr_mbox(code):
  return spu_readchcnt(code, SPU_WrOutIntrMbox)


# ------------------------------------------------------------
# Unit Tests
# ------------------------------------------------------------

def TestMFC():
  size = 32
  data = array.array('I', range(size))
  
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
  util.load_word(code, r_ea_data, data.buffer_info()[0])

  # Load the size
  code.add(spu.ai(r_size, r_zero, size * 4))

  # Load the tag
  code.add(spu.ai(r_tag, r_zero, 12))

  # Load the lsa
  code.add(spu.ai(r_ls_data, r_zero, 0))

  # Load the data into address 0
  mfc_get(code, r_ls_data, r_ea_data, r_size, r_tag)

  # Set the tag bit to 12
  mfc_write_tag_mask(code, 1<<12);

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

  # Set the tag bit to 12
  mfc_write_tag_mask(code, 1<<12);

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

  print data
  proc.execute(code)
  print data
  
  return


def TestMbox():

  code = synspu.InstructionStream()

  # Send a message to the PPU
  spu_write_out_mbox(code, 0xDEADBEAFl)

  # Get a message from the PPU
  reg = spu_read_in_mbox(code)

  # And send it back
  code.add(spu.wrch(reg, SPU_WrOutMbox))
  
  proc = synspu.Processor()

  spe_id = proc.execute(code, mode='async')
  synspu.spu_exec.write_in_mbox(spe_id, 0x88CAFE)
  
  while synspu.spu_exec.stat_out_mbox(spe_id) != 0:
    print 'spe said: 0x%X' % (synspu.spu_exec.read_out_mbox(spe_id))

  proc.join(spe_id)
  
  return


def TestSignal():

  code = synspu.InstructionStream()

  # Get a signal from the PPU
  reg = spu_read_signal1(code)

  # And send it back
  code.add(spu.wrch(reg, SPU_WrOutMbox))
  
  proc = synspu.Processor()

  spe_id = proc.execute(code, mode='async')
  synspu.spu_exec.write_signal(spe_id, 1, 0xCAFEBABEl)
  
  while synspu.spu_exec.stat_out_mbox(spe_id) != 0:
    print 'sig said: 0x%X' % (synspu.spu_exec.read_out_mbox(spe_id))

  proc.join(spe_id)
  
  return


if __name__=='__main__':
  TestMFC()
  TestMbox()
  TestSignal()
