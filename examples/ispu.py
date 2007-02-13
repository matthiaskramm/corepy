# Playing around with debugging and interactive SPE development...

import corepy.arch.spu.platform as synspu
import corepy.arch.spu.isa as spu
import array
import sys

class SPUCLI:
  """
  Simple Command line interface to the SPUs.
  """

  def __init__(self):
    
    # Code and memory buffers
    self.code = synspu.InstructionStream()
    self.regs = synspu.aligned_memory(128 * 4, typecode='I')
    
    # Runtime parameters
    self.speid = None
    self.reg_lsa = None
    self.proc = None
    
    self.synthesize()
    
    return

  def synthesize(self):
    code = self.code
    
    spu.set_active_code(code)
    
    # Reload the instructions
    spu.sync(1)              

    # Next instruction to execute
    lbl_op = code.size()
    spu.nop(0)    

    # Placeholders for register store instructions
    lbl_store = code.size()
    for i in range(128):
      spu.nop(0)

    # Stop for next command
    lbl_stop = code.size()
    spu.stop(0xC) 

    # Loop back to the beginning - this is triggered by SIGCONT
    lbl_br = code.size()
    spu.br(- (lbl_br + 1))

    # Error stop guard - this should never execute
    spu.stop(0xE)

    # Storage space for saved registers
    # Align to a 16-byte boundary
    while (code.size() % 4) != 0:
      spu.nop(0)

    lbl_regs = code.size()
    
    # Create space for the saved registers
    for i in range(128):
      # 16 bytes/register
      spu.nop(0)
      spu.nop(0)
      spu.nop(0)
      spu.nop(0)

    # Insert the store commands
    for i in range(128):
      st_offset = ((lbl_regs - lbl_store) - i) + i * 4
      code[lbl_store + i] = spu.stqr(i, st_offset)

    code.cache_code()
    code.print_code()
    code_size = len(code._prologue._code) * 4
    self.xfer_size = code_size  + (16 - (code_size) % 16);
    print 'xfer_size:', self.xfer_size
    self.code_lsa = (0x3FFFF - code_size) & 0xFFF80;
    self.reg_lsa = self.code_lsa + lbl_regs * 4

    self.lbl_op = lbl_op
    return
  

  def load_regs(self):
    reg_bytes = 128 * 16
    tag = 2
    print 'loading %d bytes from 0x%X-0x%X to 0x%X' % (reg_bytes, self.reg_lsa,
                                                       self.reg_lsa + reg_bytes,
                                                       self.regs.buffer_info()[0])

    synspu.spu_exec.spu_putb(self.speid, self.reg_lsa, self.regs.buffer_info()[0],
                             reg_bytes, tag, 0, 0)
    print 'waiting for regs...'
    synspu.spu_exec.read_tag_status_all(self.speid, 1 << tag)
    print 'got regs.'
    return

  def get_regs(self):
    self.load_regs()
    regs = []
    for reg in range(128):
      regs.append((self.regs.word_at(reg * 4),
                   self.regs.word_at(reg * 4 + 1),
                   self.regs.word_at(reg * 4 + 2),
                   self.regs.word_at(reg * 4 + 3)))

    return regs

    
  def print_regs(self):
    self.load_regs()
    for i in range(64):
      reg = i
      print 'r%03d: 0x%08X 0x%08X 0x%08X 0x%08X' % (
        reg, self.regs.word_at(reg * 4),
        self.regs.word_at(reg * 4 + 1),
        self.regs.word_at(reg * 4 + 2),
        self.regs.word_at(reg * 4 + 3)),

      reg = i + 64
      print 'r%03d: 0x%08X 0x%08X 0x%08X 0x%08X' % (
        reg, self.regs.word_at(reg * 4),
        self.regs.word_at(reg * 4 + 1),
        self.regs.word_at(reg * 4 + 2),
        self.regs.word_at(reg * 4 + 3))
      
    return

  def start(self):

    self.proc = synspu.Processor()
    self.speid = self.proc.execute(self.code, mode='async', debug = True)

    r = synspu.spu_exec.wait_stop_event(self.speid)

    return

  def stop(self):
    self.proc.cancel(self.speid)
    return

  def execute(self, cmd):
    self.code._prologue[self.lbl_op] = cmd

    tag = 1
    # print 'a', self.speid, self.code_lsa, '0x%X' % (self.code._prologue._code.buffer_info()[0]), \
    #       self.xfer_size, tag
    # print 'xfer_size:', self.xfer_size
    synspu.spu_exec.spu_getb(self.speid, self.code_lsa,
                             # self.code._prologue._code.buffer_info()[0],
                             self.code._prologue.inst_addr(),
                             self.xfer_size, tag, 0, 0)
    # print 'b'
    synspu.spu_exec.read_tag_status_all(self.speid, 1 << tag)
    # print 'c'
    self.proc.resume(self.speid)
    synspu.spu_exec.wait_stop_event(self.speid)
    return


import wx

class RegisterWindow(wx.Panel):

  def __init__(self, parent, id = -1):
    wx.Panel.__init__(self, parent, id)
    
    self._buildGUI()
    return

  def _buildGUI(self):
    listRegs = wx.ListCtrl(self, -1, style=wx.LC_REPORT)

    listRegs.InsertColumn(0, 'Register')
    listRegs.InsertColumn(1, 'Value')
    
    for i in range(128):
      idx = listRegs.InsertStringItem(i, '%d' % (i))
      listRegs.SetStringItem(idx, 1, '0x???????? 0x???????? 0x???????? 0x????????')
      listRegs.SetItemData(idx, i)

    listRegs.SetColumnWidth(0, wx.LIST_AUTOSIZE)
    listRegs.SetColumnWidth(1, wx.LIST_AUTOSIZE)
    
    sizer = wx.BoxSizer(wx.VERTICAL)
    sizer.Add(listRegs, 1, wx.EXPAND)

    sizer.Layout()
    self.SetSizer(sizer)

    self.listRegs = listRegs

    return

  def HighlightReg(self, reg, highlight):
    if not highlight:
      self.listRegs.SetItemBackgroundColour(reg, wx.WHITE)
    else:
      self.listRegs.SetItemBackgroundColour(reg, wx.RED)

    self.listRegs.EnsureVisible(reg)
    return

  def SetRegValue(self, reg, value):
    self.listRegs.SetStringItem(reg, 1, '0x%08X 0x%08X 0x%08X 0x%08X' % value)
    return
  
class SPUApp(wx.App):

  def OnInit(self):
    self.lastDiffs = []
    self.regDiffs = []
    
    self._buildGUI()
    self._startSPU()

    return True

  def _buildGUI(self):
    frame = wx.Frame(None, -1, 'Interactive SPU')

    txtCmd = wx.TextCtrl(frame, -1, style = wx.TE_PROCESS_ENTER)
    listRegs = RegisterWindow(frame)

    cmdSizer = wx.BoxSizer(wx.VERTICAL)
    cmdSizer.Add(listRegs, 1, wx.EXPAND)
    cmdSizer.Add(txtCmd, 0, wx.EXPAND)
    
    cmdSizer.Layout()
    
    frame.SetSizer(cmdSizer)
    frame.Show(True)

    self.Bind(wx.EVT_TEXT_ENTER, self.OnExecute, id=txtCmd.GetId())

    self.txtCmd = txtCmd
    self.listRegs = listRegs
    self.frame = frame
    return

  def _startSPU(self):
    cli = SPUCLI()
    cli.start()

    # self.lastRegs = cli.get_regs()
    self.lastRegs = [(0,0,0,0) for i in range(128)]
    self._updateRegView(True)
    self.cli = cli
    return

  def _updateRegs(self):
    regs = self.cli.get_regs()
    self.lastDiffs = self.regDiffs
    
    diffs = []
    for i in range(128):
      if regs[i] != self.lastRegs[i]:
        diffs.append(i)
    
    self.regDiffs = diffs
    self.lastRegs = regs
    return

  def _updateRegView(self, all = False):
    if all:
      for reg in range(128):
        self.listRegs.SetRegValue(reg, self.lastRegs[reg])
        self.listRegs.HighlightReg(reg, False)
      
    for diff in self.lastDiffs:
      self.listRegs.HighlightReg(diff, False)

    for diff in self.regDiffs:
      self.listRegs.SetRegValue(diff, self.lastRegs[diff])
      self.listRegs.HighlightReg(diff, True)

    return

  
  def _executeSPU(self, cmd):
    try:
      inst = eval('spu.%s' % cmd)
    except:
      print 'Error creating command: %s' % cmd
    else:
      self.cli.execute(inst)
      self._updateRegs()
      self._updateRegView()
      
    return
  
  def OnExecute(self, event):
    cmd = self.txtCmd.GetValue()

    if cmd == 'quit':
      self.frame.Close()
    else:
      self._executeSPU(cmd)
      self.txtCmd.Clear()
    
    return

if __name__=='__main__':

  # cli = SPUCLI()
  
  # cli.start()
  
  # cli.execute(spu.ai(11, 0, 127))
  # cli.execute(spu.ai(31, 0, 126))
  # cli.execute(spu.a(127, 126, 125))
  
  # cli.print_regs()
  
  # cli.stop()
  app = SPUApp(0)
  app.MainLoop()
  
