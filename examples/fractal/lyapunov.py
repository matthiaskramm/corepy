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

# Lyapunoc Fractal Demo

import Numeric
import math

SIZE = (100, 100)


# ------------------------------------------------------------
# Python + (some) Numeric
# ------------------------------------------------------------

def lyapunov(pattern, r1_range, r2_range, max_init, max_n, size = SIZE):
  results = Numeric.zeros(size, typecode = Numeric.Float)
  r1_inc = (r1_range[1] - r1_range[0]) / size[0]
  r2_inc = (r2_range[1] - r2_range[0]) / size[1]

  r1 = r1_range[0] + r1_inc
  r2 = r2_range[0] + r2_inc

  print r1_range, r2_range, r1_inc, r2_inc, r1, r2

  for x in range(size[0]):
    r2 = r2_range[0] + r2_inc
    print 'col:', x, r1, r2
  
    for y in range(size[1]):
      results[y, x] = lyapunov_point(pattern, r1, r2, max_init, max_n)
      r2 += r2_inc      
    r1 += r1_inc      
    
  return Numeric.where(Numeric.less(results, 0), results, 0)

def lyapunov_point(pattern, r1, r2, max_init, max_n, x0 = 0.5):
  r_idx = 0
  r_max = len(pattern)
  x = x0

  rs = (r1, r2)
  r = [rs[i] for i in pattern]

  # Init
  for i in range(max_init):
    x = r[i % r_max] * x * (1.0 - x)
    # print (r[i % r_max], x),
  # print
  if x == float('-infinity'):
    return -10.0
  
  # Derive Exponent
  total = 0.0
  try:
  
    for i in range(max_n):
      ri = r[i % r_max]
      x = ri * x * (1.0 - x)
      # print ri, x, math.log(abs(ri - 2 * ri * x)), math.log(2)
      total = total + math.log(abs(ri - 2.0 * ri * x), 2) # / math.log(2)
  except:
    print 'oops:', ri, x

  # print total, float(max_n), total / float(max_n)
  return total / float(max_n)


# ------------------------------------------------------------
# AltiVec
# ------------------------------------------------------------

# import corepy.arch.ppc.isa as ppc
# import corepy.arch.ppc.platform as synppc
# import corepy.arch.ppc.lib.iterators as ppc_iter
# import corepy.arch.ppc.types.ppc_types as ppc_types

# import corepy.arch.vmx.isa as vmx
# import corepy.arch.vmx.types.vmx_types as vmx_types

# def synthesize_lyapunov_point_vmx(code, r_vecs, r1, r2, x0, result, max_init, max_n):
#   x = vmx_types.SingleFloat()
#   r = vmx_types.SingleFloat()  
#   t1 = vmx_types.SingleFloat()  
#   t2 = vmx_types.SingleFloat()  

#   # Init
#   x.v = x0
  
#   for i in ppc_iter.syn_ite(code, max_init, mode = ppc_iter.CTR):
#     x.v = r * x * (1.0 - x)

#   # if x == float('-infinity'):
#   #   return -10.0

#   # Derive Exponent
#   total = vmx_types.SingleFloat()
#   total = total * 0.0
  
#   for i in ppc_iter.syn_ite(code, max_n, mode = ppc_iter.CTR):
#     r.v = load_r()
#     x.v = r * x * (1.0 - x)

#     t1.v = r - 2.0 * r * x
#     t2.v = t1 * -1.0
    
#     total.v = total + vmx.log.ex(vmx.vmaxfp.ex(t1, t2))

#   result.v = total / max_n

#   return result

# def synthesize_lyapunov_vmx(code, pattern, rx_range, ry_range, max_init, max_n, size = SIZE):
#   old_code = ppc.get_active_code()
#   ppc.set_active_code(code)
  
#   # Create Numeric arrays for the results, r values, and pattern
#   results = Numeric.zeros(size, typecode = Numeric.Float32)

#   rx_inc = (rx_range[1] - rx_range[0]) / size[0]
#   ry_inc = (ry_range[1] - ry_range[0]) / size[1]
#   r_inc  =  Numeric.array((rx_inc, rx_inc, rx_inc, rx_inc,
#                            ry_inc, ry_inc, ry_inc, ry_inc),
#                           typecode = Numeric.Float32)

#   rx = rx_range[0] + rx_inc
#   ry = ry_range[0] + ry_inc
#   r_init =  Numeric.array((rx, rx + rx_inc, rx + rx_inc * 2, rx + rx_inc * 3,
#                            ry, ry, ry, ry),
#                           typecode = Numeric.Float32)
                          
#   rs = (rx, ry)
#   r_vecs = [[rs[i]] * 4 for i in pattern]
#   r_vecs = reduce(lambda a, b: a + b, r_vecs, [])
#   r_vecs = Numeric.array(r_vecs, typecode = Numeric.Float32)

#   x0_array = Numeric.array((.5, .5, .5, .5), typecode = Numeric.Float32)
  
#   # Synthetic Variables
#   temp  = ppc_types.UnsigedWord(0)
#   results_addr = ppc_types.UnsigedWord(synppc.array_address(results))
#   r_inc_addr   = ppc_types.UnsigedWord(synppc.array_address(r_inc))
#   r_init_addr  = ppc_types.UnsigedWord(synppc.array_address(r_init))
#   r_vecs_addr  = ppc_types.UnsigedWord(synppc.array_address(r_vecs))
#   x0_addr      = ppc_types.UnsigedWord(synppc.array_address(x0_array))  
  
#   rx = vmx_types.SingleFloat()
#   ry = vmx_types.SingleFloat()
#   x0 = vmx_types.SingleFloat()  
#   result = vmx_types.SingleFloat()

#   rx_init = vmx_types.SingleFloat()
#   ry_init = vmx_types.SingleFloat()

#   rx_inc = vmx_types.SingleFloat()
#   ry_inc = vmx_types.SingleFloat()

#   # Load the values values for r into registers
#   ppc.lvx(rx_init, 0, r_init_addr)
#   ppc.lvx(rx_inc,  0, r_inc_addr)

#   temp.v = 16
#   ppc.lvx(ry_init, temp, r_init_addr)
#   ppc.lvx(ry_inc,  temp, r_inc_addr)

#   ppc.lvx(x0,  0, x0_addr)
  
#   # Main loop
#   for y in ppc_iter.syn_range(size[1]):
#     rx.v = rx_init
#     for x in ppc_iter.syn_range(size[0] / 4, 4):
#       synthesize_lyapunov_point_vmx(code, r_vecs, r1, r2, x0, result, max_init, max_n):
#       rx.v = rx + rx_inc

#       # TODO: STORE RESULT
#       # results[y, x] = lyapunov_point(pattern, rx, ry, max_init, max_n)
#     ry.v = ry + ry_inc
    
#   return 

import wx

class FractalData:
  def __init__(self, data = None):
    self._data = data
    return

  def SetData(self, data):
    self._data = data
    return
  
  def Draw(self, dc):
    if self._data is None: return


    h, w = self._data.shape

    self._data.shape = (h*w,)
    mn = -1.0 # min(self._data)
    self._data.shape = (h,w)

    # print mn
    shaded = self._data / mn * 255.0

    for y in range(h):
      for x in range(w):
        # print self._data[y, x]
        if self._data[y, x] < 0.0:
          if self._data[y, x] > -1.0:
            shade = int(shaded[y, x]) % 255
          if self._data[y, x] > -10.0:
            shade = 255 - int(shaded[y, x]) % 255
          else:
            shade = 1.0
          # print shade
          dc.SetPen(wx.Pen(wx.Colour(shade, shade, shade)))
          dc.DrawPoint(x, y)
      # print '------------------------------'
    return
  
class FractalPlot(wx.Window):
  """
  Simple 2D plot window.  Data and axes are set by the user and managed by
  this class.  The Draw() method calls the draw methods on the axis and data.

  The display can be copied to the clipboard using ctl-c.
  """

  PublicMethods = ('GetXAxis', 'SetXAxis', 'GetYAxis', 'SetYAxis',
                   'GetData', 'AddData', 'ClearData',
                   'ShowAxisLabels', 'Clear', 'CopyToClipboard',
                   'IsInViewport', 'BoundXY', 'BoundX', 'BoundY',
                   'MouseToData', 'Draw',
                   'SetStatusFrame',
                   '_lastDataViewport')
    
  def __init__(self, parent, id = -1, style = 0, catchMotion = 1, catchChar = 1):
    wx.Window.__init__(self, parent, id, style = style)
    
    self.SetBackgroundColour(wx.WHITE)
    self._data = []
    wx.EVT_PAINT(self, self.OnPaint)
    
    return

  def AddData(self, data):
    self._data.append(data)
    return
  
  def CopyToClipboard(self):
    """
    Copy the drawing to the clipboard as a bitmap.
    """

    w, h = self.GetSize()
    bmp = wx.EmptyBitmap(w, h)
    memDC = wx.MemoryDC()
    memDC.SelectObject(bmp)

    plotDC = wx.ClientDC(self)
    memDC.Blit(0, 0, w, h, plotDC, 0, 0)

    if wx.TheClipboard.Open():
      data = wx.BitmapDataObject(bmp)
      wx.TheClipboard.SetData(data)
      wx.TheClipboard.Close()
    return

  def Draw(self, dc):
    """
    Draw everything!
    """

    print 'Drawing...'
    for data in self._data:
      data.Draw(dc)
    print 'Done.'
    return

  # Event handlers
  def OnPaint(self, event):
    dc = wx.PaintDC(self)
    dc.BeginDrawing()
    dc.SetBrush(wx.TRANSPARENT_BRUSH)
    self.Draw(dc)
    dc.EndDrawing()
    return


if __name__=='__main__':

  class App(wx.App):

    def OnInit(self):
      self.ShowLyapunovPlot()            
      return True

    def ShowLyapunovPlot(self):
      
      frame = wx.Frame(None, -1, 'Lyapunov')
      frame.SetSize(SIZE)

      plot = FractalPlot(frame)

      if True:
        raw_data = lyapunov([0,1], [2.0, 4.0], [2.0, 4.0], 200, 400)
      else:
        raw_data = Numeric.zeros(SIZE, typecode = Numeric.Float)
        for i in range(100):
          i_start = i - 100
          row = Numeric.arange(i_start, i)
          raw_data[i, :] = row
          
      fractal_data = FractalData(raw_data)
      plot.AddData(fractal_data)
      frame.Show(True)
      return 


  app = App(0)
  app.MainLoop()






