# Copyright 2006-2007 The Trustees of Indiana University.

# This software is available for evaluation purposes only.  It may not be
# redistirubted or used for any other purposes without express written
# permission from the authors.

# Authors:
#   Christopher Mueller (chemuell@cs.indiana.edu)
#   Andrew Lumsdaine    (lums@cs.indiana.edu)

# Playing around with debugging and interactive SPE development...

__doc__ = """

ispu is an interactive SPU program that lets the user execute commands
one at a time on the SPU and view the results in Python.  

ispu has command line and GUI modes.  The GUI used wxPython.  To run
the GUI, make sure wxPython is installed and simply run ispu.py from
the command line:

% pythonw ispu.py

The command line mode lets the user run ispu in the Python
interpreter. The following is a a simple SPU session:

% python
...
>>> import corepy.arch.spu.isa as spu
>>> import ispu

>>> cli = ispu.ISPU()

>>> cli.start()
>>> cli.execute(spu.iohl(127, 0xB8CA))
>>> cli.execute(spu.iohl(126, 0x1234))
>>> cli.execute(spu.a(125, 126, 127))

>>> regs = cli.get_regs()
>>> print '%X' % regs[125][0]

>>> cli.stop()

When running, ispu reserves an SPU.  When used interactively, make
sure to call the stop() method to free the SPU when done.

"""

import corepy.arch.spu.platform as synspu
import corepy.arch.spu.isa as spu
import array
import sys

class ISPU:
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
    # code.print_code()
    code_size = len(code._prologue._code) * 4
    self.xfer_size = code_size  + (16 - (code_size) % 16);
    # print 'xfer_size:', self.xfer_size
    self.code_lsa = (0x3FFFF - code_size) & 0xFFF80;
    self.reg_lsa = self.code_lsa + lbl_regs * 4

    self.lbl_op = lbl_op
    return
  

  def load_regs(self):
    reg_bytes = 128 * 16
    tag = 2
    # print 'loading %d bytes from 0x%X-0x%X to 0x%X' % (reg_bytes, self.reg_lsa,
    # self.reg_lsa + reg_bytes,
    # self.regs.buffer_info()[0])

    synspu.spu_exec.spu_putb(self.speid, self.reg_lsa, self.regs.buffer_info()[0],
                             reg_bytes, tag, 0, 0)
    # print 'waiting for regs...'
    synspu.spu_exec.read_tag_status_all(self.speid, 1 << tag)
    # print 'got regs.'
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


try:
  import wx
except:
  print 'Warning: wx not found.  GUI is not available'
  wx = None

class RegisterWindow(wx.Panel):

  def __init__(self, parent, id = -1):
    wx.Panel.__init__(self, parent, id)
    
    self._buildGUI()
    return

  def _buildGUI(self):
    listRegs = wx.ListCtrl(self, -1, style=wx.LC_REPORT | wx.SUNKEN_BORDER)

    listRegs.InsertColumn(0, 'Register')
    listRegs.InsertColumn(1, 'Value')

    fixedFont = wx.Font(11, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
    for i in range(128):
      idx = listRegs.InsertStringItem(i, '%d' % (i))
      listRegs.SetStringItem(idx, 1, '0x???????? 0x???????? 0x???????? 0x????????')
      listRegs.SetItemData(idx, i)
      listRegs.SetItemFont(idx, fixedFont)

    listRegs.SetColumnWidth(0, wx.LIST_AUTOSIZE_USEHEADER)
    listRegs.SetColumnWidth(1, 350) # wx.LIST_AUTOSIZE)
    
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
    print """
    *** Thank you for using the wxPython Interactive SPU *** 

    To use, simply type any SPU ISA command into the command box using 
    the CorePy ISA syntax and with integer values for registers.  For
    example, to create and add two vectors, enter the following
    commands one at time followed by a return:

      ai(11, 0, 127) 
      ai(31, 0, 126)
      a(127, 126, 125)

    ISPU will highlight registers as they change.

    Previous instructions can be accessed from the history list using
    the up/down arrow keys.

    Type 'quit' or close the window to exit.
    
    *** Email chemuell@cs.indiana.edu with any questions/comments ***  
    """

    self.lastDiffs = []
    self.regDiffs = []

    self.history = []
    self.currentCmd = -1
    
    self._buildGUI()
    self._startSPU()

    return True

  def _buildGUI(self):
    frame = wx.Frame(None, -1, 'Interactive SPU')

    stcCmd = wx.StaticText(frame, -1, 'Command:')
    txtCmd = wx.TextCtrl(frame, -1, style = wx.TE_PROCESS_ENTER)

    txtSizer = wx.BoxSizer(wx.HORIZONTAL)
    txtSizer.Add((5,-1))
    txtSizer.Add(stcCmd, flag = wx.ALIGN_CENTER)
    txtSizer.Add(txtCmd, 1)
    txtSizer.Add((5,-1))    
    txtSizer.Layout()
    
    listRegs = RegisterWindow(frame)

    cmdSizer = wx.BoxSizer(wx.VERTICAL)
    cmdSizer.Add(listRegs, 1, wx.EXPAND)
    cmdSizer.Add((-1,2))        
    cmdSizer.Add(txtSizer, 0, wx.EXPAND)
    cmdSizer.Add((-1,2))            
    cmdSizer.Layout()

    lstHistory = wx.ListCtrl(frame, -1, size = (150, -1),
                             style = (wx.LC_REPORT | wx.LC_NO_HEADER | wx.LC_SINGLE_SEL | 
                                      wx.SUNKEN_BORDER))
    lstHistory.InsertColumn(0, 'Command History', -1)
    lstHistory.SetColumnWidth(0, 120)
    
    mainSizer = wx.BoxSizer(wx.HORIZONTAL)
    mainSizer.Add(cmdSizer, 1, wx.EXPAND)
    mainSizer.Add(lstHistory, 0, wx.EXPAND)

    mainSizer.Layout()
    
    frame.SetSizer(mainSizer)
    frame.Show(True)

    self.Bind(wx.EVT_TEXT_ENTER, self.OnExecute, id=txtCmd.GetId())
    self.Bind(wx.EVT_CHAR, self.OnChar, id=txtCmd.GetId())    

    self.txtCmd = txtCmd
    self.lstHistory = lstHistory
    self.listRegs = listRegs
    self.frame = frame
    return

  def _startSPU(self):
    cli = ISPU()
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

  def _setCurrent(self, idx):

    if self.currentCmd != -1:
      self.lstHistory.SetItemBackgroundColour(self.currentCmd, wx.WHITE)
      self.lstHistory.SetItemTextColour(self.currentCmd, wx.BLACK)

    self.currentCmd = idx

    if idx == -1:
      self.lstHistory.SetItemBackgroundColour(self.currentCmd, wx.WHITE)
      self.lstHistory.SetItemTextColour(self.currentCmd, wx.BLACK)
    else:      
      self.lstHistory.SetItemBackgroundColour(self.currentCmd, wx.BLUE)
      self.lstHistory.SetItemTextColour(self.currentCmd, wx.WHITE)

    self.lstHistory.EnsureVisible(self.currentCmd)    
    return
  
  def OnChar(self, event):

    key = event.GetKeyCode()

    if len(self.history) == 0:
      pass
    elif key == wx.WXK_UP:
      # print 'up'
      
      idx = self.currentCmd
      if idx == -1: idx = len(self.history) - 1
      else:         idx -= 1

      self._setCurrent(idx)
      self.txtCmd.SetValue(self.history[self.currentCmd])
      
    elif key == wx.WXK_DOWN and (self.currentCmd + 1) < len(self.history):
      # print 'down'
      idx = self.currentCmd
      if idx == -1: idx = len(self.history) - 1
      else:         idx += 1
      self._setCurrent(idx)
      self.txtCmd.SetValue(self.history[self.currentCmd])

    event.Skip()
    return 
  
  def OnExecute(self, event):
    cmd = self.txtCmd.GetValue()

    if cmd == 'quit':
      self.frame.Close()
    else:
      self._executeSPU(cmd)
      self.txtCmd.Clear()

      cmdIdx = len(self.history)
      self.history.append(cmd)
      self.lstHistory.InsertStringItem(cmdIdx, cmd)
      
      self.lstHistory.EnsureVisible(cmdIdx)
      self._setCurrent(-1)
    return

if __name__=='__main__':

  # cli = ISPU()
  
  # cli.start()
  
  # cli.execute(spu.ai(11, 0, 127))
  # cli.execute(spu.ai(31, 0, 126))
  # cli.execute(spu.a(127, 126, 125))
  
  # cli.print_regs()
  
  # cli.stop()
  app = SPUApp(0)
  app.MainLoop()
