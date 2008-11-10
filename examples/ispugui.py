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

import wx
import wx.lib.mixins.listctrl as listmix
import wx.stc as stc
import string
import sys
import os
import re
import StringIO

import corepy.lib.extarray as extarray
import corepy.lib.printer as printer
import corepy.arch.spu.platform as env
import corepy.arch.spu.isa as spu
import corepy.spre.spe as spe


class EditorCtrl(stc.StyledTextCtrl):
  def __init__(self, app, parent, id):
    stc.StyledTextCtrl.__init__(self, parent, id)

    mask = stc.STC_MOD_INSERTTEXT | stc.STC_MOD_DELETETEXT
    mask |= stc.STC_MOD_BEFOREINSERT | stc.STC_MOD_BEFOREDELETE
    mask |= stc.STC_PERFORMED_USER|stc.STC_PERFORMED_UNDO|stc.STC_PERFORMED_REDO
    self.SetModEventMask(mask)

    self.Bind(stc.EVT_STC_MARGINCLICK, self.OnMarginClick)
    self.Bind(stc.EVT_STC_CHANGE, self.OnChange)

    self.exec_mark = None
    self.line_count = 0
    return

  def OnMarginClick(self, event):
    line = self.LineFromPosition(event.GetPosition())
    if event.GetControl():
      # TODO - move the BP down to the first non-label line
      if self.IsBreakSet(line):
        self.MarkerDelete(line, 0)
      else:
        self.MarkerAdd(line, 0)
    else:
      self.SetExecMark(self.LineFromPosition(event.GetPosition()))

  def OnChange(self, event):
    lc = self.GetLineCount()
    if lc != self.line_count:
      self.line_count = lc
      cl = self.GetCurrentLine()
      
      # Update exec_mark
      if self.exec_mark != None and self.exec_mark >= cl - 1:
        for i in xrange(cl - 1, lc):
          if (self.MarkerGet(i) & 2) != 0:
            print "old exec_mark, new", self.exec_mark, i
            self.SetExecMark(i)
            break

    return

  def SetExecMark(self, line):
    if self.exec_mark != None:
        self.MarkerDelete(self.exec_mark, 1)
    if line != None:
        self.MarkerAdd(line, 1)

    self.exec_mark = line
    return

  def IsBreakSet(self, line):
    return (self.MarkerGet(line) & 1) != 0


class EditorWindow(wx.Frame):
  def __init__(self, app, parent, id):
    wx.Frame.__init__(self, parent, id, "SPU Debugger -- Editor")

    editCtrl = EditorCtrl(app, self, -1)
    self.editCtrl = editCtrl

    # Set up a status bar
    statusCtrl = wx.StatusBar(self, -1)
    self.statusCtrl = statusCtrl

    statusCtrl.SetFieldsCount(1)
    statusCtrl.SetStatusText("")

    mainSizer = wx.BoxSizer(wx.VERTICAL)
    mainSizer.Add(editCtrl, 1, wx.EXPAND | wx.ALL)
    mainSizer.Add(statusCtrl, 0, wx.EXPAND)

    mainSizer.Layout()

    self.SetSizer(mainSizer)

    # make some styles
    editCtrl.StyleSetSpec(stc.STC_STYLE_DEFAULT, "face:Courier")
    editCtrl.StyleClearAll()

    editCtrl.SetMarginType(0, stc.STC_MARGIN_NUMBER)
    editCtrl.SetMarginWidth(0, 22)

    # setup some markers
    editCtrl.SetMarginType(1, stc.STC_MARGIN_SYMBOL)
    editCtrl.SetMarginSensitive(1, 1)
    editCtrl.MarkerDefine(0, stc.STC_MARK_CIRCLE, "#FF0000", "#FF0000")
    editCtrl.MarkerDefine(1, stc.STC_MARK_SHORTARROW, "#00AF00", "#00AF00")
    editCtrl.MarkerDefine(2, stc.STC_MARK_SHORTARROW, "#AFAF00", "#AFAF00")

    # Toolbox stuff
    tsize = (16, 16)
    quitbmp = wx.ArtProvider.GetBitmap('gtk-quit', wx.ART_TOOLBAR, tsize)
    execbmp = wx.ArtProvider.GetBitmap('gtk-execute', wx.ART_TOOLBAR, tsize)
    stepbmp = wx.ArtProvider.GetBitmap('gtk-go-forward', wx.ART_TOOLBAR, tsize)
    contbmp = wx.ArtProvider.GetBitmap('gtk-goto-last', wx.ART_TOOLBAR, tsize)
    cancelbmp = wx.ArtProvider.GetBitmap('gtk-cancel', wx.ART_TOOLBAR, tsize)
    clearbmp = wx.ArtProvider.GetBitmap('gtk-clear', wx.ART_TOOLBAR, tsize)
    revertbmp = wx.ArtProvider.GetBitmap('gtk-revert-to-saved', wx.ART_TOOLBAR, tsize)
    dndbmp = wx.ArtProvider.GetBitmap('gtk-dnd', wx.ART_TOOLBAR, tsize)

    tb = self.CreateToolBar(wx.NO_BORDER | wx.TB_FLAT | wx.TB_HORIZONTAL)
    tb.SetToolBitmapSize(tsize)

    tb.AddLabelTool(0x01, "Quit Debugger", quitbmp, shortHelp="Quit Debugger")
    self.Bind(wx.EVT_TOOL, self.OnMiscClick, id=0x01)

    tb.AddSeparator()

    tb.AddLabelTool(0x10, "Execute", execbmp, shortHelp="Execute")
    self.Bind(wx.EVT_TOOL, self.OnExecClick, id=0x10)

    tb.AddLabelTool(0x20, "Step", stepbmp, shortHelp="Step")
    self.Bind(wx.EVT_TOOL, self.OnExecClick, id=0x20)

    tb.AddLabelTool(0x30, "Continue", contbmp, shortHelp="Continue")
    self.Bind(wx.EVT_TOOL, self.OnExecClick, id=0x30)

    tb.AddSeparator()

    tb.AddLabelTool(0x02, "Clear Breakpoints",
        cancelbmp, shortHelp="Clear Breakpoints")
    self.Bind(wx.EVT_TOOL, self.OnMiscClick, id=0x02)

    tb.AddSeparator()

    tb.AddLabelTool(0x57, "Reset All",
        dndbmp, shortHelp="Reset code, registers, and local store")
    self.Bind(wx.EVT_TOOL, self.OnResetClick, id=0x57)

    tb.AddLabelTool(0x51, "Reset Code",
        revertbmp, shortHelp="Reset code to initial stream")
    self.Bind(wx.EVT_TOOL, self.OnResetClick, id=0x51)

    tb.AddLabelTool(0x58, "Clear code",
        clearbmp, shortHelp="Clear all code")
    self.Bind(wx.EVT_TOOL, self.OnResetClick, id=0x58)

    #tb.AddLabelTool(0x52, "Reset Registers",
    #    clearbmp, shortHelp="Reset all register values to 0")
    #self.Bind(wx.EVT_TOOL, self.OnResetClick, id=0x52)

    #tb.AddLabelTool(0x54, "Reset Local Store",
    #    clearbmp, shortHelp="Reset all local store values to 0")
    #self.Bind(wx.EVT_TOOL, self.OnResetClick, id=0x54)

    self.app = app

    #self.Update()
    self.Show(True)
    return


  def OnMiscClick(self, event):
    id = event.GetId()

    if id == 0x01:
      # Exit the debugger
      self.Close()
    elif id == 0x02:
      # Clear breakpoints
      for i in xrange(0, self.editCtrl.GetLineCount()):
        self.editCtrl.MarkerDelete(i, 0)
    return
        

  def OnResetClick(self, event):
    id = event.GetId()

    if id & 0xF0 != 0x50:
      return

    if id & 0x1:  # Reset code
      self.ResetCode()
    if id & 0x2:  # Reset registers
      self.app.reg_frame.listCtrl.ResetRegisters()
    if id & 0x4:  # Reset local store
      self.app.ls_frame.listCtrl.ResetLocalStore()
    if id & 0x8:
      self.editCtrl.ClearAll()
    return

  def OnExecClick(self, event):
    id = event.GetId()

    step = None
    if id == 0x20: # Step
      # Make every instruction but the current one be a debug stop
      start = 0
      if self.editCtrl.exec_mark != None:
        start = self.editCtrl.exec_mark
      step = start
    elif id == 0x10: # Execute
      # Execute from the beginning.. easy
      start = 0
    elif id == 0x30: # Continue
      # Execute from the current instruction
      start = 0
      if self.editCtrl.exec_mark != None:
        start = self.editCtrl.exec_mark


    # Bail if we're on a blank line
    line = self.editCtrl.GetLine(start).strip()
    if line == '':
      return

    self.statusCtrl.SetStatusText("Executing...")

    # Move the start point past any labels
    while line[-1] == ':':
      start += 1
      line = self.editCtrl.GetLine(start).strip()

    if step != None:
      step = start

    # Starting on a line with a breakpoint on it? 
    if self.editCtrl.IsBreakSet(start):
      # If we're starting at a breakpoint, need to be able to step once,
      # redo the IS, then continue.

      # Generate the code, making sure the start inst is enabled even if BP set
      self.editCtrl.MarkerDelete(start, 0)
      code = self.GenerateStream(start)
      self.editCtrl.MarkerAdd(start, 0)

      stop = self.app.ExecuteStream(code, start)

      codelen = len(code) - 1
      while stop < codelen and isinstance(code[stop + 1], spe.Label):
        stop += 1

      # Once the breakpoint is stepped, do a continue if we weren't already
      # stepping.  Careful not to execute again if only a step was clicked!
      if step == None:
        start = stop
        # Generate & execute the stream
        code = self.GenerateStream()
        stop = self.app.ExecuteStream(code, start)

    else:
      # Generate & execute the stream
      code = self.GenerateStream(step)
      stop = self.app.ExecuteStream(code, start)

    # Update the execution mark
    codelen = len(code) - 1
    while stop < codelen and isinstance(code[stop + 1], spe.Label):
      stop += 1

    self.editCtrl.SetExecMark(stop)
    self.app.Update()
    self.statusCtrl.SetStatusText("")
    return


  def GenerateStream(self, step = None):
    code = env.InstructionStream()
    txt = self.editCtrl.GetText().split('\n')
    txtlen = len(txt)

    for i in xrange(0, txtlen):
      # For the stop case, want all instructions except the current one to be
      # STOP instructions.
      cmd = txt[i].strip()
      if step != None and i != step:
        if cmd == "" or cmd[0] == '#':
          continue
        if cmd[-1] == ":":
          # Label - better parsing?
          code.add(code.get_label(cmd[:-1]))
        else:
          code.add(spu.stop(0x2FFF))
        continue

      if self.editCtrl.IsBreakSet(i):
        code.add(spu.stop(0x2FFF))
        continue

      if cmd != "" and cmd[0] != '#':
        if cmd[-1] == ":":
          # Label - better parsing?
          inst = code.get_label(cmd[:-1])
        else:
          # Instruction
          strcmd = re.sub("Label\((.*?)\)", "code.get_label('\\1')", cmd)
          try:
            inst = eval('spu.%s' % strcmd)
          except:
            print 'Error creating instruction: %s' % cmd

        code.add(inst)
    code.cache_code()
    return code


  def ResetCode(self):
    self.editCtrl.ClearAll()

    fd = StringIO.StringIO()
    printer.PrintInstructionStream(self.app.code, printer.Default(), fd = fd)

    for line in fd.getvalue().split('\n'):
       if line != "" and line != "BODY:":
         self.editCtrl.AddText("%s\n" % line)
    fd.close()

    return


class RegisterListCtrl(wx.ListCtrl, listmix.TextEditMixin):
  def __init__(self, app, parent, id, style, size = (-1, -1)):
    wx.ListCtrl.__init__(self, parent, id, size = size, style = style)
    listmix.TextEditMixin.__init__(self)

    self.attr = wx.ListItemAttr()
    self.attr.SetFont(wx.Font(11,
        wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))

    #self.attr_red = wx.ListItemAttr()
    #self.attr_red.SetFont(wx.Font(11,
    #    wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
    #self.attr_red.SetTextColour(wx.RED)

    self.Bind(wx.EVT_LIST_BEGIN_LABEL_EDIT, self.OnBeginEdit)

    self.app = app
    self._cur_regs = extarray.extarray('I', 128 * 4)
    #self._prev_regs = extarray.extarray('I', 128 * 4)
    #self._prev_regs.clear()
    return


  def OnBeginEdit(self, event):
    if event.GetColumn() == 0:
      event.Veto()
    return


  def SetVirtualData(self, item, column, data):
    self._cur_regs[item * 4 + (column - 1)] = int(data, self.base)

    env.spu_exec.put_spu_registers(self.app.ctx, self._cur_regs.buffer_info()[0])

    self.app.reg_frame.Update()
    self.app.ls_frame.Update()  # TODO - why?
    return


  def ResetRegisters(self):
    self._cur_regs.clear()
    env.spu_exec.put_spu_registers(self.app.ctx, self._cur_regs.buffer_info()[0])
    self.app.reg_frame.Update()


  def OnGetItemText(self, item, column):
    if column == 0:
      return "%d" % (item)
    elif column > 0 and column < 5:
      if self.base == 16:
        return "%08X" % self._cur_regs[item * 4 + (column - 1)]
      elif self.base == 10:
        return "%010d" % self._cur_regs[item * 4 + (column - 1)]
    #elif column == 1:
    #  return "0x%08X %08X %08X %08X" % (self._cur_regs[item * 4],
    #                                    self._cur_regs[item * 4 + 1],
    #                                    self._cur_regs[item * 4 + 2],
    #                                    self._cur_regs[item * 4 + 3])


  def OnGetItemAttr(self, item):
    #idx = item * 4
    #if self._prev_regs[idx] != self._cur_regs[idx] or (
    #   self._prev_regs[idx + 1] != self._cur_regs[idx + 1]) or (
    #   self._prev_regs[idx + 2] != self._cur_regs[idx + 2]) or (
    #   self._prev_regs[idx + 3] != self._cur_regs[idx + 3]):
    #  self._prev_regs[idx] = self._cur_regs[idx]
    #  self._prev_regs[idx + 1] = self._cur_regs[idx + 1]
    #  self._prev_regs[idx + 2] = self._cur_regs[idx + 2]
    #  self._prev_regs[idx + 3] = self._cur_regs[idx + 3]
    #  return self.attr_red
    #else:
    #  return self.attr
    return self.attr


class RegisterWindow(wx.Frame):
  def __init__(self, app, parent, id):
    wx.Frame.__init__(self, parent, id, "SPU Debugger -- Registers")

    listCtrl = RegisterListCtrl(app, self, -1, style = wx.LC_REPORT | wx.LC_VIRTUAL)
    self.listCtrl = listCtrl
    
    listCtrl.InsertColumn(0, "Register")
    listCtrl.InsertColumn(1, "Value[0]")
    listCtrl.InsertColumn(2, "Value[1]")
    listCtrl.InsertColumn(3, "Value[2]")
    listCtrl.InsertColumn(4, "Value[3]")

    listCtrl.SetColumnWidth(0, 80)
    listCtrl.SetColumnWidth(1, 80) #wx.LIST_AUTOSIZE)
    listCtrl.SetColumnWidth(2, 80) #wx.LIST_AUTOSIZE)
    listCtrl.SetColumnWidth(3, 80) #wx.LIST_AUTOSIZE)
    listCtrl.SetColumnWidth(4, 80) #wx.LIST_AUTOSIZE)

    listCtrl.SetItemCount(128)

    tsize = (16, 16)
    clearbmp = wx.ArtProvider.GetBitmap('gtk-clear', wx.ART_TOOLBAR, tsize)
    convertbmp = wx.ArtProvider.GetBitmap('gtk-convert', wx.ART_TOOLBAR, tsize)

    tb = self.CreateToolBar(wx.NO_BORDER | wx.TB_FLAT | wx.TB_HORIZONTAL)
    tb.SetToolBitmapSize(tsize)

    tb.AddLabelTool(0x52, "Reset Registers",
        clearbmp, shortHelp="Reset all register values to 0")
    self.Bind(wx.EVT_TOOL, self.OnToolClick, id=0x52)

    tb.AddSeparator()

    tb.AddRadioLabelTool(0x100, "Hexadecimal",
        convertbmp, shortHelp="Show values in hexadecimal")
    self.Bind(wx.EVT_TOOL, self.OnToolClick, id=0x100)

    tb.AddRadioLabelTool(0x101, "Decimal",
        convertbmp, shortHelp="Show values in decimal")
    self.Bind(wx.EVT_TOOL, self.OnToolClick, id=0x101)

    self.listCtrl.base = 16
    self.app = app
    self.Update()
    self.Show(True)
    return


  def OnToolClick(self, event):
    id = event.GetId()

    if id == 0x52:
      self.listCtrl.ResetRegisters()
    elif id == 0x100:
      self.listCtrl.base = 16
      self.listCtrl.RefreshItems(0, 128)
    elif id == 0x101:
      self.listCtrl.base = 10
      self.listCtrl.RefreshItems(0, 128)
    return


  def Update(self):
    env.spu_exec.get_spu_registers(self.app.ctx, self.listCtrl._cur_regs.buffer_info()[0])
    self.listCtrl.RefreshItems(0, 128)
    return


class LocalStoreListCtrl(wx.ListCtrl, listmix.TextEditMixin):
  def __init__(self, app, parent, id, style, size = (-1, -1)):
    wx.ListCtrl.__init__(self, parent, id, size = size, style = style)
    listmix.TextEditMixin.__init__(self)

    self.attr = wx.ListItemAttr()
    self.attr.SetFont(wx.Font(11,
        wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))

    #self.attr_red = wx.ListItemAttr()
    #self.attr_red.SetFont(wx.Font(11,
    #    wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
    #self.attr_red.SetTextColour(wx.RED)

    self.Bind(wx.EVT_LIST_BEGIN_LABEL_EDIT, self.OnBeginEdit)

    self.base = 16
    self._cur_ls = app.localstore
    self.app = app
    #self._prev_ls = extarray.extarray('I', 16384 * 4)
    #self._prev_ls.clear()
    return

  def OnBeginEdit(self, event):
    if event.GetColumn() == 0:
      event.Veto()
    return


  def SetVirtualData(self, item, column, data):
    self._cur_ls[item * 4 + (column - 1)] = int(data, self.base)
    self.app.mem_frame.Update()
    return


  def ResetLocalStore(self):
    self._cur_ls.clear()
    self.app.ls_frame.Update()
    self.app.mem_frame.Update()


  def OnGetItemText(self, item, column):
    if column == 0:
      return "0x%06X" % (item * 16)
    elif column > 0 and column < 5:
      if self.base == 16:
        return "%08X" % self._cur_ls[item * 4 + (column - 1)]
      elif self.base == 10:
        return "%010d" % self._cur_ls[item * 4 + (column - 1)]


  def OnGetItemAttr(self, item):
    #idx = item * 4
    #if self._prev_ls[idx] != self._cur_ls[idx] or (
    #   self._prev_ls[idx + 1] != self._cur_ls[idx + 1]) or (
    #   self._prev_ls[idx + 2] != self._cur_ls[idx + 2]) or (
    #   self._prev_ls[idx + 3] != self._cur_ls[idx + 3]):
    #  self._prev_ls[idx] = self._cur_ls[idx]
    #  self._prev_ls[idx + 1] = self._cur_ls[idx + 1]
    #  self._prev_ls[idx + 2] = self._cur_ls[idx + 2]
    #  self._prev_ls[idx + 3] = self._cur_ls[idx + 3]
    #  return self.attr_red
    #else:
    #  return self.attr
    return self.attr


class LocalStoreWindow(wx.Frame):
  def __init__(self, app, parent, id):
    wx.Frame.__init__(self, parent, id, "SPU Debugger -- Local Store")

    listCtrl = LocalStoreListCtrl(app, self, -1,
        style = wx.LC_REPORT | wx.LC_VIRTUAL | wx.LC_EDIT_LABELS)
    self.listCtrl = listCtrl

    listCtrl.InsertColumn(0, "Address")
    listCtrl.InsertColumn(1, "Value[0]")
    listCtrl.InsertColumn(2, "Value[1]")
    listCtrl.InsertColumn(3, "Value[2]")
    listCtrl.InsertColumn(4, "Value[3]")

    listCtrl.SetColumnWidth(0, 80)
    listCtrl.SetColumnWidth(1, 80) #wx.LIST_AUTOSIZE)
    listCtrl.SetColumnWidth(2, 80) #wx.LIST_AUTOSIZE)
    listCtrl.SetColumnWidth(3, 80) #wx.LIST_AUTOSIZE)
    listCtrl.SetColumnWidth(4, 80) #wx.LIST_AUTOSIZE)

    listCtrl.SetItemCount(16384)

    tsize = (16, 16)
    clearbmp = wx.ArtProvider.GetBitmap('gtk-clear', wx.ART_TOOLBAR, tsize)
    convertbmp = wx.ArtProvider.GetBitmap('gtk-convert', wx.ART_TOOLBAR, tsize)

    tb = self.CreateToolBar(wx.NO_BORDER | wx.TB_FLAT | wx.TB_HORIZONTAL)
    tb.SetToolBitmapSize(tsize)

    tb.AddLabelTool(0x52, "Reset local store",
        clearbmp, shortHelp="Reset all local store values to 0")
    self.Bind(wx.EVT_TOOL, self.OnToolClick, id=0x52)

    tb.AddSeparator()

    tb.AddRadioLabelTool(0x100, "Hexadecimal",
        convertbmp, shortHelp="Show values in hexadecimal")
    self.Bind(wx.EVT_TOOL, self.OnToolClick, id=0x100)

    tb.AddRadioLabelTool(0x101, "Decimal",
        convertbmp, shortHelp="Show values in decimal")
    self.Bind(wx.EVT_TOOL, self.OnToolClick, id=0x101)

    self.listCtrl.base = 16
    self.Show(True)
    return


  def OnToolClick(self, event):
    id = event.GetId()

    if id == 0x52:
      self.listCtrl.ResetLocalStore()
    elif id == 0x100:
      self.listCtrl.base = 16
      self.listCtrl.RefreshItems(0, 16384)
    elif id == 0x101:
      self.listCtrl.base = 10
      self.listCtrl.RefreshItems(0, 16384)
    return


  def Update(self):
    # Update the GUI with most recent local store contents
    self.listCtrl.RefreshItems(0, 16384)
    return


class MemoryListCtrl(wx.ListCtrl, listmix.TextEditMixin):
  def __init__(self, app, parent, id, style, size = (-1, -1)):
    wx.ListCtrl.__init__(self, parent, id, size = size, style = style)
    listmix.TextEditMixin.__init__(self)

    self.attr = wx.ListItemAttr()
    self.attr.SetFont(wx.Font(11,
        wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))

    self.attr_gray = wx.ListItemAttr()
    self.attr_gray.SetFont(wx.Font(11,
        wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
    self.attr_gray.SetTextColour('#800517')
   

    self.Bind(wx.EVT_LIST_BEGIN_LABEL_EDIT, self.OnBeginEdit)

    self.app = app

    self._cur_ls = app.localstore
    self._array = extarray.extarray('I', 1)
    self._filename = "/proc/%d/maps" % os.getpid()
    self._map_cache = 0
    return

  def UpdateMaps(self):
    fd = open(self._filename, "r")

    maps = []
    prev = None
    bytes = 0

    for line in fd:
      split = line.split()
      mode = split[1]

      if mode[0] != 'r':
        continue

      w = False
      if mode[1] == 'w':
        w = True

      addrs = split[0].split('-')

      map = (bytes >> 4, int(addrs[0], 16), int(addrs[1], 16), w)
      if prev != None and prev[2] == map[1] and prev[3] == map[3]:
        # Maps touch and have same write perm, merge them
        maps[-1] = (prev[0], prev[1], map[2], map[3])
      else:
        maps.append(map)

      bytes += map[2] - map[1]
      prev = map

    fd.close()
    self.maps = maps
    self.bytes = bytes
    self._map_len = len(maps)
    return


  def GetMap(self, item):
    map = self.maps[self._map_cache]
    if item >= map[0]:
      if self._map_cache == self._map_len - 1 or item < self.maps[self._map_cache + 1][0]:
        return map

    self.UpdateMaps()

    maps = self.maps
    for i in xrange(1, self._map_len):
      if item < maps[i][0]:
        self._map_cache = i - 1
        return maps[i - 1]
    self._map_cache = self._map_len - 1
    return maps[-1]


  def OnBeginEdit(self, event):
    column = event.GetColumn()
    if column == 0:
      event.Veto()
    elif column < 5:
      map = self.GetMap(event.GetItem())
      if map[3] == False:
        event.Veto()
    return


  def SetVirtualData(self, item, column, data):
    map = self.GetMap(item)
    addr = map[1] + ((item - map[0]) * 16)

    self._array.set_memory(addr + (4 * (column - 1)))
    self._array[0] = int(data, self.base)
    self.app.reg_frame.Update()
    self.app.ls_frame.Update()
    return


  def OnGetItemText(self, item, column):
    map = self.GetMap(item)
    addr = map[1] + ((item - map[0]) * 16)
    if column == 0:
      return "0x%08X" % (addr)
    elif column < 5:
      self._array.set_memory(addr + (4 * (column - 1)))
      if self.base == 16:
        return "%08X" % (self._array[0])
      elif self.base == 10:
        return "%010d" % (self._array[0])
    return ""


  def OnGetItemAttr(self, item):
    map = self.GetMap(item)
    if map[3] == True:
      return self.attr
    else:
      return self.attr_gray


class MemoryWindow(wx.Frame):
  def __init__(self, app, parent, id):
    wx.Frame.__init__(self, parent, id, "SPU Debugger -- Main Memory")

    listCtrl = MemoryListCtrl(app, self, -1,
        style = wx.LC_REPORT | wx.LC_VIRTUAL | wx.LC_EDIT_LABELS)
    self.listCtrl = listCtrl

    listCtrl.InsertColumn(0, "Address")
    listCtrl.InsertColumn(1, "Value[0]")
    listCtrl.InsertColumn(2, "Value[1]")
    listCtrl.InsertColumn(3, "Value[2]")
    listCtrl.InsertColumn(4, "Value[3]")

    listCtrl.SetColumnWidth(0, 80)
    listCtrl.SetColumnWidth(1, 80) #wx.LIST_AUTOSIZE)
    listCtrl.SetColumnWidth(2, 80) #wx.LIST_AUTOSIZE)
    listCtrl.SetColumnWidth(3, 80) #wx.LIST_AUTOSIZE)
    listCtrl.SetColumnWidth(4, 80) #wx.LIST_AUTOSIZE)

    listCtrl.UpdateMaps()
    listCtrl.SetItemCount(listCtrl.bytes / 16)

    stcCmd = wx.StaticText(self, -1, 'Go To:')
    txtCmd = wx.TextCtrl(self, -1, style = wx.TE_PROCESS_ENTER)
 
    self.stcCmd = stcCmd
    self.txtCmd = txtCmd

    cmdSizer = wx.BoxSizer(wx.HORIZONTAL)
    cmdSizer.Add((5,-1))
    cmdSizer.Add(stcCmd, 0, flag = wx.ALIGN_CENTER)
    cmdSizer.Add((5,-1))
    cmdSizer.Add(txtCmd, 1)
    cmdSizer.Layout()

    mainSizer = wx.BoxSizer(wx.VERTICAL)
    mainSizer.Add(listCtrl, 1, wx.EXPAND | wx.ALL)
    mainSizer.Add(cmdSizer, 0, wx.EXPAND)

    mainSizer.Layout()

    self.SetSizer(mainSizer)

    self.Bind(wx.EVT_TEXT_ENTER, self.OnExecute, id=txtCmd.GetId())

    # Set up the tool bar
    tsize = (16, 16)
    convertbmp = wx.ArtProvider.GetBitmap('gtk-convert', wx.ART_TOOLBAR, tsize)

    tb = self.CreateToolBar(wx.NO_BORDER | wx.TB_FLAT | wx.TB_HORIZONTAL)
    tb.SetToolBitmapSize(tsize)

    tb.AddRadioLabelTool(0x100, "Hexadecimal",
        convertbmp, shortHelp="Show values in hexadecimal")
    self.Bind(wx.EVT_TOOL, self.OnToolClick, id=0x100)

    tb.AddRadioLabelTool(0x101, "Decimal",
        convertbmp, shortHelp="Show values in decimal")
    self.Bind(wx.EVT_TOOL, self.OnToolClick, id=0x101)

    self.listCtrl.base = 16
    self.Show(True)
    return


  def OnToolClick(self, event):
    id = event.GetId()

    if id == 0x100:
      self.listCtrl.base = 16
      self.Update()
    elif id == 0x101:
      self.listCtrl.base = 10
      self.Update()
    return


  def OnExecute(self, event):
    try:
      addr = int(self.txtCmd.GetValue(), 16)
    except:
      print "Malformed go-to address, ignoring"
      return

    for i in xrange(1, len(self.listCtrl.maps)):
      if addr < self.listCtrl.maps[i][1]:
        map = self.listCtrl.maps[i - 1]
        if addr >= map[2]:
          print "Invalid address, ignoring"
        else:
          item = map[0] + ((addr - map[1]) / 16)
          self.listCtrl.UpdateMaps()
          self.listCtrl.EnsureVisible(item)
        return

    map = self.listCtrl.maps[-1]
    if addr >= map[2]:
      print "Invalid address, ignoring"
    else:
      item = map[0] + ((addr - map[1]) / 16)
      self.listCtrl.UpdateMaps()
      self.listCtrl.EnsureVisible(item)
    return

  def Update(self):
    # Update the GUI with most recent local store contents
    self.listCtrl.UpdateMaps()
    self.listCtrl.SetItemCount(self.listCtrl.bytes / 16)
    self.listCtrl.RefreshItems(0, self.listCtrl.bytes / 16)
    return


class SPUApp(wx.App):
  def __init__(self, code):
    self.code = code
    wx.App.__init__(self)
    return


  def OnInit(self):
    self._startSPU()
    self._buildGUI()

    self.edit_frame.ResetCode()
    return True


  def _buildGUI(self):
    edit_frame = EditorWindow(self, None, -1)
    reg_frame = RegisterWindow(self, edit_frame, -1)
    ls_frame = LocalStoreWindow(self, edit_frame, -1)
    mem_frame = MemoryWindow(self, edit_frame, -1)

    edit_frame.SetFocus()

    self.edit_frame = edit_frame
    self.reg_frame = reg_frame
    self.ls_frame = ls_frame
    self.mem_frame = mem_frame
    return


  def _startSPU(self):
    self.ctx = ctx = env.spu_exec.alloc_context()

    # Execute a no-op instruction stream so the prolog is executed
    code = env.InstructionStream()
    code.add(spu.nop(code.r_zero))

    code.cache_code()
    itemsize = code.render_code.itemsize 
    code_len = len(code.render_code) * itemsize
    if code_len % 16 != 0:
      code_len += 16 - (code_len % 16)
    code_lsa = 0x40000 - code_len

    env.spu_exec.run_stream(ctx, code.inst_addr(), code_len, code_lsa, code_lsa)

    self.localstore = extarray.extarray('I', 262144 / 4)
    self.localstore.set_memory(ctx.spuls)
    return


  def ExecuteStream(self, code, start):
    """Start executing code at instruction number start, and return the stop
       instruction number"""

    code.cache_code()
    itemsize = code.render_code.itemsize 
    code_len = len(code.render_code) * itemsize
    if code_len % 16 != 0:
      code_len += 16 - (code_len % 16)
    code_lsa = 0x40000 - code_len

    offset = start
    for i in xrange(1, start + 1):
      if isinstance(code[i], spe.Label):
        offset -= 1
    
    exec_lsa = code_lsa + ((offset + len(code._prologue) - 1) * itemsize)

    ret = env.spu_exec.run_stream(self.ctx, code.inst_addr(), code_len, code_lsa, exec_lsa)

    offset = ((ret - code_lsa) / 4) - len(code._prologue)

    if offset == 0:
      return 0

    off = 0
    for i in xrange(1, len(code)):
      inst = code[i]
      if not isinstance(inst, spe.Label):
        off += 1
        if off == offset:
          return i

    return 0

  def Update(self):
    self.reg_frame.Update()
    self.ls_frame.Update()
    self.mem_frame.Update()
    return


if __name__=='__main__':
  code = env.InstructionStream()
  reg = code.acquire_register()
  foo = code.acquire_register(reg = 1)

  code.add(code.get_label("FOO"))
  code.add(spu.il(foo, 0xCAFE))
  code.add(spu.ilhu(reg, 0xDEAD))
  code.add(spu.iohl(reg, 0xBEEF))
  code.add(spu.stqd(reg, code.r_zero, 4))

  lbl_loop = code.get_label("LOOP")
  lbl_break = code.get_label("BREAK")

  r_cnt = code.gp_return
  r_stop = code.acquire_register(reg = 9)
  r_cmp = code.acquire_register()

  code.add(spu.ori(r_cnt, code.r_zero, 0))
  code.add(spu.il(r_stop, 5))

  code.add(lbl_loop)
  code.add(spu.ceq(r_cmp, r_cnt, r_stop))
  code.add(spu.brnz(r_cmp, code.get_label("BREAK")))
  code.add(spu.ai(r_cnt, r_cnt, 1))
  code.add(spu.br(code.get_label("LOOP")))
  code.add(lbl_break)

  app = SPUApp(code)
  app.MainLoop()

