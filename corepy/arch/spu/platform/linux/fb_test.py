import cell_fb

fb = cell_fb.framebuffer()
cell_fb.fb_open(fb)

idx = 0


xoff = 0
xinc = 5
try:
  # if True:
  while True:
    cell_fb.fb_clear(fb, idx)
    for i in range(100):
      for j in range(100):
        cell_fb.fb_write_pixel(fb, idx, i + xoff, j, 0xFFFFFFFF)

    cell_fb.fb_wait_vsync(fb)
    cell_fb.fb_flip(fb, idx)

    idx = 1 - idx
    xoff += xinc
    if xoff == 300 or xoff == 0:
      xinc *= -1
except: pass

print fb.w, fb.h, hex(cell_fb.fb_addr(fb, 0)), hex(cell_fb.fb_addr(fb, 1))
cell_fb.fb_close(fb)
