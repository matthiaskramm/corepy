import corepy.lib.extarray as extarray
import corepy.arch.cal.isa as cal
import corepy.arch.cal.platform as env
import math
import time
import random
import ctypes
import sys

import nbody

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *


glrender = ctypes.CDLL("glrender.so")
glrender.render.argtypes = (ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_int)
glrender.render.restype = None
glrender.render2.argtypes = (ctypes.c_ulong, ctypes.c_ulong, ctypes.c_int)
glrender.render2.restype = None
glrender.render3.argtypes = (ctypes.c_ulong, ctypes.c_int)
glrender.render3.restype = None

SQRT_BODIES = 64
N_BODIES = SQRT_BODIES ** 2
print "total bodies", N_BODIES


def InitGL(Width, Height): 
  glClearColor(0.0, 0.0, 0.0, 0.0)
  glClearDepth(1.0)
  glDepthFunc(GL_LESS)
  glEnable(GL_DEPTH_TEST)
  #glShadeModel(GL_SMOOTH)
  #glEnable(GL_POINT_SMOOTH);
  #glHint(GL_POINT_SMOOTH_HINT, GL_FASTEST);
  glMatrixMode(GL_PROJECTION)
  glLoadIdentity() 

  gluPerspective(45.0, float(Width)/float(Height), 0.1, 100.0)

  glMatrixMode(GL_MODELVIEW)
  return


def ReshapeGL(Width, Height):
  if Height == 0:
    Height = 1

  glViewport(0, 0, Width, Height)
  glMatrixMode(GL_PROJECTION)
  glLoadIdentity()
  gluPerspective(45.0, float(Width)/float(Height), 0.1, 100.0)
  glMatrixMode(GL_MODELVIEW)


def DrawGL():
  pass
  #global glrender
  #global fps_time, fps_count

  #glrender.render(x.buffer_info()[0], y.buffer_info()[0], m.buffer_info()[0], N_BODIES)

  #fps_count += 1
  #cur_time = time.time()
  #if cur_time - fps_time >= 1.0:
  #  print "FPS:", fps_count
  #  fps_time = cur_time
  #  fps_count = 0


def ProcessStep():
  #global x, y, vx, vy, m
  global pos, vel, code, step, proc
  global fps_time, fps_count

  for i in xrange(0, 4):
    inp = step % 2
    out = (step + 1) % 2
    code.set_remote_binding('i0', pos[inp], copy_local = False)
    code.set_remote_binding('i1', vel[inp], copy_local = False)
    code.set_remote_binding('o0', pos[out])
    code.set_remote_binding('o1', vel[out])
    proc.execute(code, (0, 0, SQRT_BODIES, SQRT_BODIES))

    fps_count += 1
    step += 1

  glrender.render3(pos[out].buffer_info()[0], N_BODIES)
  #nbody.cal_nb_exec(code, pos[inp], vel[inp], pos[out], vel[out], mass)


  #nbody.py_nb_step3(x, y, vx, vy, m, 0.0001)
  #glrender.render(x.buffer_info()[0], y.buffer_info()[0], m.buffer_info()[0], N_BODIES)

  cur_time = time.time()
  if cur_time - fps_time >= 1.0:
    print "FPS:", fps_count
    #print "Vel", vel[out][0], vel[out][1], vel[out][16], vel[out][17]
    #print "Vel", vx[0], vy[0], vx[2], vy[2]
    fps_time = cur_time
    fps_count = 0

  return


def py_nbody():
  global x, y, vx, vy, m

  x = extarray.extarray('f', N_BODIES)
  y = extarray.extarray('f', N_BODIES)
  vx = extarray.extarray('f', N_BODIES)
  vy = extarray.extarray('f', N_BODIES)
  m = extarray.extarray('f', N_BODIES)
 
  for i in xrange(0, N_BODIES): 
    x[i] = random.uniform(-1.0, 1.0)
    y[i] = random.uniform(-1.0, 1.0)
    #vx[i] = random.uniform(-1.0, 1.0)
    #vy[i] = random.uniform(-1.0, 1.0)
    vx[i] = 0.0
    vy[i] = 0.0
    m[i] = random.uniform(1e9, 1e10)

  return

def cal_nbody():
  global pos, vel, mass, code, step, proc
  step = 0
  proc = env.Processor(1)

  pos = [proc.alloc_remote('f', 4, SQRT_BODIES, SQRT_BODIES) for i in xrange(0, 2)]
  vel = [proc.alloc_remote('f', 4, SQRT_BODIES, SQRT_BODIES) for i in xrange(0, 2)]

  pos[0].clear()
  vel[0].clear()
  pos[1].clear()
  vel[1].clear()

  for i in xrange(0, N_BODIES):
    pos[0][i * 4] = random.uniform(-20.0, 20.0)
    pos[0][i * 4 + 1] = random.uniform(-12.0, 12.0)
    pos[0][i * 4 + 2] = 0.0
    pos[0][i * 4 + 3] = random.uniform(1e9, 1e12)
    #vel[0][i * 4] = random.uniform(-1.0, 1.0)
    #vel[0][i * 4 + 1] = random.uniform(-1.0, 1.0)
    vel[0][i * 4] = 0.0
    vel[0][i * 4 + 1] = 0.0
    vel[0][i * 4 + 2] = 0.0
    vel[0][i * 4 + 3] = 0.0

  pos[0][0] = -4.0
  pos[0][1] = 0.0
  pos[0][2] = 0.0
  pos[0][3] = 1e16

  ind = 1
  pos[0][ind * 4] = 4.0
  pos[0][ind * 4 + 1] = 0.0
  pos[0][ind * 4 + 2] = 0.0
  pos[0][ind * 4 + 3] = 1e16

  ind = 2
  pos[0][ind * 4] = 0.0
  pos[0][ind * 4 + 1] = 4.0
  pos[0][ind * 4 + 2] = 0.0
  pos[0][ind * 4 + 3] = 1e16

  ind = 3
  pos[0][ind * 4] = 0.0
  pos[0][ind * 4 + 1] = -4.0
  pos[0][ind * 4 + 2] = 0.0
  pos[0][ind * 4 + 3] = 1e16

  code = nbody.cal_nb_generate_2d(SQRT_BODIES, 0.000002)
  code.cache_code()
  return


if __name__ == '__main__':
  global fps_time, fps_count
  fps_time = time.time()
  fps_count = 0

  #random.seed(1)

  #py_nbody()
  cal_nbody()

  glutInit(sys.argv)

  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH)
  glutInitWindowSize(800, 600)
  glutInitWindowPosition(0, 0)
  window = glutCreateWindow("nbody")

  glutDisplayFunc(DrawGL)

  glutIdleFunc(ProcessStep)
  #glutIdleFunc(DrawGL)

  glutReshapeFunc(ReshapeGL)
  glutFullScreen()

  InitGL(800, 600)

  glutMainLoop()

