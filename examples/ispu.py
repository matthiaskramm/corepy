# Copyright (c) 2006-2008 The Trustees of Indiana University.                   
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

import corepy.arch.spu.platform as env
import corepy.arch.spu.isa as spu
import corepy.arch.spu.lib.util as util
import corepy.lib.extarray as extarray


# Feature TODO:
#  Allow for an instruction stream to be passed in
#   Breakpoints?
#   Watch variables?
#  Local store inspection
#  Memory inspection
#  Changing of register/local store/memory contents via GUI
#   Insert the instructions to do this into the stream?
#   Or just do it all behind the scenes?
#  Stick with always executing one instruction at a time, or allow for executing
#  more than once before stopping?
#   Always doing one at a time is simpler, but slower
#  What should the GUI look like?
#   Separate windows for the instruction list, registers, local store, mem, etc


class ISPU:
  """
  Simple Command line interface to the SPUs.
  """

  def __init__(self):
    
    # Code and memory buffers
    self.code = env.InstructionStream()
    self.regs = extarray.extarray('I', 128 * 4)
    self.regs.clear()
    
    # Runtime parameters
    self.speid = None
    self.reg_lsa = None
    self.proc = None
    
    self.synthesize()
    
    return

  def synthesize(self):
    # Okay.  This code is not going to exceed 256 instructions (1kb).  Knowing that,
    # the register contents can be safely placed at 0x3F400 in localstore, 3kb from
    # the top.  The SPRE will place the instruction stream as close to the top as
    # possible.  But since it is not going to be more than 1kb worth of instructions,
    # it will not overlap with the register contents.

    code = self.code
    spu.set_active_code(code)
    
    # Reload the instructions
    spu.sync(1)

    # Next instruction to execute
    lbl_op = code.size()
    spu.nop(0)    

    # Placeholders for register store instructions
    for i in range(128):
       spu.stqa(i, 0xFD00 + (i * 4))
    #  spu.stqa(i, 0xFE00 + (i * 4))

    # Stop for next command
    spu.stop(0x0FFF) 

    lbl_regs = code.size()
    
    # Create space for the saved registers
    #for i in range(128):
    #  # 16 bytes/register
    #  spu.nop(0)
    #  spu.lnop()
    #  spu.nop(0)
    #  spu.lnop()

    # Clearing active code here is important!
    spu.set_active_code(None)
    code.cache_code()

    code_size = len(code._prologue._code) * 4
    self.xfer_size = code_size  + (16 - (code_size) % 16);
    print 'xfer_size:', self.xfer_size

    self.code_lsa = (0x3FFFF - code_size) & 0xFFF80;
    self.lbl_op = lbl_op
    return
  

  def load_regs(self):
    env.spu_exec.spu_putb(self.speid, 0x3F400, self.regs.buffer_info()[0],
                          128 * 16, 2, 0, 0)
    env.spu_exec.read_tag_status_all(self.speid, 1 << 2)
    return

  def get_regs(self):
    self.load_regs()
    regs = []
    for reg in range(128):
      regs.append((self.regs[reg * 4],
                   self.regs[reg * 4 + 1],
                   self.regs[reg * 4 + 2],
                   self.regs[reg * 4 + 3]))

    return regs

    
  def print_regs(self):
    self.load_regs()
    for i in range(64):
      reg = i
      print 'r%03d: 0x%08X 0x%08X 0x%08X 0x%08X' % (
        reg, self.regs[reg * 4],
        self.regs[reg * 4 + 1],
        self.regs[reg * 4 + 2],
        self.regs[reg * 4 + 3]),

      reg = i + 64
      print 'r%03d: 0x%08X 0x%08X 0x%08X 0x%08X' % (
        reg, self.regs[reg * 4],
        self.regs[reg * 4 + 1],
        self.regs[reg * 4 + 2],
        self.regs[reg * 4 + 3])
      
    return


  def start(self):
    self.started = True
    #self.proc = env.Processor()

    #self.speid = self.proc.execute(self.code, async = True, debug = False)
    #env.spu_exec.wait(self.speid)

    self.code_len = len(self.code._code_array) * self.code._code_array.itemsize
    if self.code_len % 16 != 0:
      self.code_len += 16 - (self.code_len % 16)
    self.code_lsa = 0x40000 - self.code_len

    self.ctx = env.spu_exec.alloc_context()
    self.code.cache_code()
    env.spu_exec.run_stream(self.ctx, self.code.start_addr(), self.code_len, self.code_lsa, self.code_lsa)
    return

  def stop(self):
    env.spu_exec.free_context(self.ctx)
    self.ctx = None
    self.started = False
    return

  def execute(self, cmd):
    if self.started != True:
      print "ERROR ISPU not started; do ISPU.start() first"
      return

    self.code[self.lbl_op] = cmd
    self.code.cache_code()

    env.spu_exec.run_stream(self.ctx, self.code.start_addr(), self.code_len, self.code_lsa, self.code_lsa)
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

